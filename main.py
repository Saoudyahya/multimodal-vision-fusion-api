from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
from faster_whisper import WhisperModel
from ultralytics import YOLO
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    ViTModel
)
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict
from collections import defaultdict, Counter
import os
import io
import torch
import torch.nn as nn
import uvicorn
import threading
import time


# Multi-modal Fusion Components
class MultiModalFusionModule(nn.Module):
    """Fusion module: Combines YOLO spatial + ViT semantic features"""

    def __init__(self, spatial_dim=8, semantic_dim=768, hidden_dim=256, num_heads=4):
        super().__init__()

        self.spatial_projection = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.semantic_projection = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, spatial_features, semantic_features):
        """
        Args:
            spatial_features: [N_objects, 8]
            semantic_features: [1, 768]
        Returns:
            fused_features: [1, 256]
            attention_weights: attention map
        """
        spatial_proj = self.spatial_projection(spatial_features)
        semantic_proj = self.semantic_projection(semantic_features)

        if spatial_proj.dim() == 2:
            spatial_proj = spatial_proj.unsqueeze(0)
        if semantic_proj.dim() == 2:
            semantic_proj = semantic_proj.unsqueeze(0)

        attn_output, attn_weights = self.cross_attention(
            query=spatial_proj,
            key=semantic_proj,
            value=semantic_proj
        )

        spatial_attended = self.layer_norm1(spatial_proj + attn_output)
        spatial_pooled = torch.mean(spatial_attended, dim=1)
        semantic_norm = self.layer_norm2(semantic_proj.squeeze(1))

        concatenated = torch.cat([spatial_pooled, semantic_norm], dim=-1)
        fused_features = self.fusion_layer(concatenated)

        return fused_features, attn_weights


class ObjectDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image):
        """Detect objects and extract spatial features"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        results = self.model(img)
        detected_objects = []
        spatial_features_list = []

        img_height, img_width = img.shape[:2]

        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            label = self.model.names[int(cls)]

            # Normalize spatial coordinates for fusion
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            area = width * height
            aspect_ratio = width / (height + 1e-6)

            spatial_feature = [center_x, center_y, width, height, area, aspect_ratio, conf, cls]
            spatial_features_list.append(spatial_feature)

            detected_objects.append({
                "label": label,
                "confidence": round(conf, 2),
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
            })

        # Convert to tensor for fusion
        if spatial_features_list:
            spatial_features = torch.tensor(spatial_features_list, dtype=torch.float32)
        else:
            spatial_features = torch.zeros((1, 8), dtype=torch.float32)

        return img, detected_objects, spatial_features

    @staticmethod
    def annotate_image(img, detected_objects):
        """Annotate image with bounding boxes"""
        annotated = img.copy()
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["bounding_box"]
            label = f"{obj['label']} {obj['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated


class ImageCaptioning:
    def __init__(self, model_name):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_caption(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption


class SemanticFeatureExtractor:
    """Extract semantic features from ViT for fusion"""

    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.vit_model = ViTModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.vit_model.eval()

    def extract_features(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        inputs = self.processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            semantic_features = outputs.last_hidden_state[:, 0, :]  # CLS token [1, 768]

        return semantic_features


class GPT2DescriptionGenerator:
    def __init__(self, model_name="gpt2", fusion_hidden_dim=256):
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Adapter for fusion features
        self.feature_adapter = nn.Linear(fusion_hidden_dim, self.model.config.n_embd)

    def generate_description(self, detected_objects, caption, fused_features=None, use_fusion=False):
        if not detected_objects:
            return "No objects detected in the image."

        object_list = [f"{obj['label']} ({obj['confidence'] * 100:.0f}%)"
                       for obj in detected_objects]

        if use_fusion and fused_features is not None:
            prompt = f"Caption: {caption}\nObjects: {', '.join(object_list)}\nDetailed description:"
        else:
            prompt = (
                f"Image caption: {caption}\n"
                f"Detected objects: {', '.join(object_list)}.\n"
                "Describe the scene including spatial relationships and context:"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=200)

        if use_fusion and fused_features is not None:
            # Use visual features through adapter
            adapted_features = self.feature_adapter(fused_features)
            input_embeds = self.model.transformer.wte(inputs.input_ids)
            combined_embeds = torch.cat([adapted_features.unsqueeze(1), input_embeds], dim=1)

            outputs = self.model.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=120,
                num_beams=4,
                temperature=0.8,
                top_p=0.9,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        else:
            # Baseline without fusion
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                num_beams=3,
                temperature=0.7,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class UserGuidance:
    @staticmethod
    def generate_guidance(detected_objects: List[Dict], caption: str = "",
                          image_width: int = 640) -> str:
        if not detected_objects:
            return "No objects detected."

        position_groups = {"left": [], "front": [], "right": []}
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["bounding_box"]
            label = obj["label"]
            center_x = (x1 + x2) / 2

            # Camera view is mirrored, so we swap left/right
            if center_x < image_width / 3:
                position_groups["right"].append(label)
            elif center_x > 2 * image_width / 3:
                position_groups["left"].append(label)
            else:
                position_groups["front"].append(label)

        guidance_parts = []
        if caption:
            guidance_parts.append(f"Scene: {caption}.")

        for position in ["left", "front", "right"]:
            if position_groups[position]:
                counter = Counter(position_groups[position])
                descriptions = []
                for obj, count in counter.most_common():
                    if count > 1:
                        descriptions.append(f"{count} {obj}s")
                    else:
                        descriptions.append(f"a {obj}")

                if descriptions:
                    if len(descriptions) > 1:
                        objects_text = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
                    else:
                        objects_text = descriptions[0]

                    guidance_parts.append(f"To your {position}: {objects_text}.")

        return " ".join(guidance_parts) if guidance_parts else "No objects in specific positions."


class CameraManager:
    def __init__(self):
        self.camera = None
        self.latest_frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.frame_count = 0
        self.error_count = 0
        self.last_error = None
        self.backend_name = None

    def start_camera(self, camera_index=1):
        """Start camera capture - optimized for external USB camera"""
        if self.is_running:
            return {"success": True, "message": "Camera already running", "backend": self.backend_name}

        # Release any existing camera
        if self.camera:
            self.camera.release()
            time.sleep(0.3)

        print(f"\nOpening camera at index {camera_index}...")

        # Try DirectShow first (best for USB cameras on Windows)
        try:
            self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            if self.camera.isOpened():
                # Set optimal properties for USB camera
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.camera.set(cv2.CAP_PROP_FPS, 30)

                # Give camera time to initialize
                time.sleep(0.5)

                # Test frame capture
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    self.backend_name = "DirectShow"
                    self.is_running = True
                    self.frame_count = 0
                    self.error_count = 0

                    # Start capture thread
                    threading.Thread(target=self._capture_frames, daemon=True).start()

                    # Wait and verify
                    time.sleep(0.5)

                    if self.frame_count > 0:
                        print(f"✓ Camera {camera_index} started successfully with DirectShow")
                        return {
                            "success": True,
                            "message": f"Camera {camera_index} started successfully",
                            "backend": "DirectShow"
                        }

                self.camera.release()
        except Exception as e:
            print(f"DirectShow failed: {e}")
            if self.camera:
                self.camera.release()

        # If DirectShow failed, try other backends
        backends = [(cv2.CAP_ANY, "Auto"), (cv2.CAP_V4L2, "V4L2")]
        for backend, name in backends:
            try:
                print(f"Trying {name}...")
                self.camera = cv2.VideoCapture(camera_index, backend)
                if self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        self.backend_name = name
                        self.is_running = True
                        threading.Thread(target=self._capture_frames, daemon=True).start()
                        time.sleep(0.5)
                        return {"success": True, "message": f"Camera started with {name}", "backend": name}
                self.camera.release()
            except:
                continue

        self.last_error = "Failed to open camera with any backend"
        return {
            "success": False,
            "error": self.last_error,
            "suggestion": f"Make sure camera is at index {camera_index} and not in use by another app"
        }

    def _capture_frames(self):
        """Capture frames continuously"""
        consecutive_errors = 0
        max_errors = 30

        while self.is_running:
            try:
                ret, frame = self.camera.read()

                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.frame_count += 1
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    self.error_count += 1

                    if consecutive_errors >= max_errors:
                        print(f"Too many errors, stopping camera")
                        self.is_running = False
                        break

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"Capture error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    self.is_running = False
                    break
                time.sleep(0.1)

    def get_frame(self):
        """Get latest frame"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_stats(self):
        """Get statistics"""
        return {
            "is_running": self.is_running,
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "has_frame": self.latest_frame is not None,
            "backend": self.backend_name
        }

    def stop_camera(self):
        """Stop camera"""
        self.is_running = False
        time.sleep(0.2)

        if self.camera:
            self.camera.release()
            self.camera = None

        self.latest_frame = None
        print("Camera stopped")


class MainApp:
    def __init__(self, yolo_path, caption_model_name, use_fusion=False):
        self.detector = ObjectDetection(yolo_path)
        self.captioner = ImageCaptioning(caption_model_name)
        self.guidance_generator = UserGuidance()
        self.use_fusion = use_fusion

        # Initialize both GPT-2 and GPT-2 mini
        print("Loading GPT-2 models...")
        self.gpt2_generator = GPT2DescriptionGenerator("gpt2")
        self.gpt2_mini_generator = GPT2DescriptionGenerator("distilgpt2")  # DistilGPT2 is smaller and faster
        print("✓ GPT-2 models loaded!")

        # Initialize fusion components if needed
        if use_fusion:
            print("Loading fusion components...")
            self.semantic_extractor = SemanticFeatureExtractor()
            self.fusion_module = MultiModalFusionModule()
            print("✓ Fusion components loaded!")
        else:
            self.semantic_extractor = None
            self.fusion_module = None

    def process_image(self, image, model_type="none", use_fusion=False):
        """
        Process image with different model options

        Args:
            image: Input image
            model_type: "none", "gpt2", or "gpt2-mini"
            use_fusion: Whether to use multi-modal fusion
        """
        # Detect objects and get spatial features
        img, detected_objects, spatial_features = self.detector.detect_objects(image)

        # Generate caption
        image_caption = self.captioner.generate_caption(img)

        # Generate guidance (non-LLM)
        guidance = self.guidance_generator.generate_guidance(
            detected_objects,
            image_caption,
            image_width=img.shape[1]
        )

        result = {
            "caption": image_caption,
            "detected_objects": detected_objects,
            "guidance": guidance,
            "model_used": model_type,
            "fusion_enabled": False
        }

        # Generate LLM description if requested
        if model_type in ["gpt2", "gpt2-mini"]:
            fused_features = None

            # Select the appropriate model
            if model_type == "gpt2":
                generator = self.gpt2_generator
            else:  # gpt2-mini
                generator = self.gpt2_mini_generator

            # Use fusion if requested and available
            if use_fusion and self.semantic_extractor and self.fusion_module:
                try:
                    semantic_features = self.semantic_extractor.extract_features(img)
                    fused_features, attention_weights = self.fusion_module(
                        spatial_features,
                        semantic_features
                    )
                    result["fusion_enabled"] = True

                    # Handle NaN for single object case
                    mean_attn = attention_weights.mean().item()
                    max_attn = attention_weights.max().item()
                    std_attn = attention_weights.std().item()

                    # Replace NaN with 0.0
                    if torch.isnan(torch.tensor(std_attn)):
                        std_attn = 0.0

                    result["attention_stats"] = {
                        "mean": float(mean_attn) if not torch.isnan(torch.tensor(mean_attn)) else 0.0,
                        "max": float(max_attn) if not torch.isnan(torch.tensor(max_attn)) else 0.0,
                        "std": float(std_attn)
                    }
                except Exception as e:
                    print(f"Fusion failed: {e}")
                    fused_features = None

            # Generate description
            llm_description = generator.generate_description(
                detected_objects,
                image_caption,
                fused_features=fused_features,
                use_fusion=(fused_features is not None)
            )
            result["llm_description"] = llm_description

        return result, img


# Initialize
camera_manager = CameraManager()

print("\n" + "=" * 60)
print("Loading AI Models...")
print("=" * 60)

# Enable fusion globally (set to True to enable)
ENABLE_FUSION = True

main_app = MainApp(
    yolo_path="yolov9c.pt",
    caption_model_name="nlpconnect/vit-gpt2-image-captioning",
    use_fusion=ENABLE_FUSION
)
print("✓ All models loaded successfully!")
print(f"✓ Fusion mode: {'ENABLED' if ENABLE_FUSION else 'DISABLED'}")

# Whisper setup
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'whisper_models')
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Whisper model...")
model_path = os.path.join(MODEL_DIR, "tiny")
if os.path.exists(model_path):
    whisper_model = WhisperModel(model_path, device="cpu")
else:
    whisper_model = WhisperModel("tiny", device="cpu", download_root=MODEL_DIR)
print("✓ Whisper model loaded!")
print("=" * 60 + "\n")

# FastAPI app
app = FastAPI(title="Vision & Audio Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Vision & Audio Processing API",
        "status": "online",
        "camera_status": camera_manager.get_stats()
    }


@app.post("/camera/start")
async def start_camera(camera_index: int = Query(1, description="Camera index")):
    """Start camera"""
    result = camera_manager.start_camera(camera_index)

    if result["success"]:
        return {
            "status": "success",
            "message": result["message"],
            "backend": result.get("backend"),
            "camera_index": camera_index
        }
    else:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": result["error"],
                "suggestion": result.get("suggestion")
            }
        )


@app.post("/camera/stop")
async def stop_camera():
    """Stop camera"""
    camera_manager.stop_camera()
    return {"status": "success", "message": "Camera stopped"}


@app.get("/camera/status")
async def camera_status():
    """Get camera status"""
    stats = camera_manager.get_stats()
    return {
        "is_running": stats["is_running"],
        "has_frame": stats["has_frame"],
        "frames_captured": stats["frame_count"],
        "errors": stats["error_count"],
        "backend": stats["backend"]
    }


@app.get("/camera/frame")
async def get_camera_frame():
    """Get latest frame as JPEG"""
    frame = camera_manager.get_frame()

    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")

    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )


@app.post("/process_camera/basic")
async def process_camera_basic(
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process camera frame - Basic mode (no LLM)"""
    frame = camera_manager.get_frame()

    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available. Start camera first.")

    try:
        result, processed_img = main_app.process_image(frame, model_type="none", use_fusion=False)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_camera/gpt2")
async def process_camera_gpt2(
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process camera frame - GPT-2 (no fusion)"""
    frame = camera_manager.get_frame()

    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available. Start camera first.")

    try:
        result, processed_img = main_app.process_image(frame, model_type="gpt2", use_fusion=False)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_camera/gpt2-mini")
async def process_camera_gpt2_mini(
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process camera frame - GPT-2 Mini (no fusion)"""
    frame = camera_manager.get_frame()

    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available. Start camera first.")

    try:
        result, processed_img = main_app.process_image(frame, model_type="gpt2-mini", use_fusion=False)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_camera/gpt2-fusion")
async def process_camera_gpt2_fusion(
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process camera frame - GPT-2 + Multi-Modal Fusion"""
    frame = camera_manager.get_frame()

    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available. Start camera first.")

    try:
        result, processed_img = main_app.process_image(frame, model_type="gpt2", use_fusion=True)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_camera/gpt2-mini-fusion")
async def process_camera_gpt2_mini_fusion(
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process camera frame - GPT-2 Mini + Multi-Modal Fusion"""
    frame = camera_manager.get_frame()

    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available. Start camera first.")

    try:
        result, processed_img = main_app.process_image(frame, model_type="gpt2-mini", use_fusion=True)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Same endpoints for uploaded images
@app.post("/process_image/basic")
async def process_image_basic(
        file: UploadFile = File(...),
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process uploaded image - Basic mode (no LLM)"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result, processed_img = main_app.process_image(img, model_type="none", use_fusion=False)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_image/gpt2")
async def process_image_gpt2(
        file: UploadFile = File(...),
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process uploaded image - GPT-2 (no fusion)"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result, processed_img = main_app.process_image(img, model_type="gpt2", use_fusion=False)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_image/gpt2-mini")
async def process_image_gpt2_mini(
        file: UploadFile = File(...),
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process uploaded image - GPT-2 Mini (no fusion)"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result, processed_img = main_app.process_image(img, model_type="gpt2-mini", use_fusion=False)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_image/gpt2-fusion")
async def process_image_gpt2_fusion(
        file: UploadFile = File(...),
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process uploaded image - GPT-2 + Multi-Modal Fusion"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result, processed_img = main_app.process_image(img, model_type="gpt2", use_fusion=True)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_image/gpt2-mini-fusion")
async def process_image_gpt2_mini_fusion(
        file: UploadFile = File(...),
        annotate: bool = Query(False, description="Return annotated image")
):
    """Process uploaded image - GPT-2 Mini + Multi-Modal Fusion"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result, processed_img = main_app.process_image(img, model_type="gpt2-mini", use_fusion=True)

        response = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if annotate:
            annotated_img = ObjectDetection.annotate_image(processed_img, result["detected_objects"])
            _, buffer = cv2.imencode('.jpg', annotated_img)
            response["annotated_image_base64"] = buffer.tobytes().hex()

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def allowed_file(filename):
    """Check audio file type"""
    allowed_extensions = {'wav', 'mp3', 'm4a', 'flac'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio"""
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid audio format")

    try:
        audio_bytes = await file.read()
        segments, info = whisper_model.transcribe(io.BytesIO(audio_bytes))

        transcribed_text = ""
        transcription_segments = []

        for segment in segments:
            transcribed_text += f"{segment.text} "
            transcription_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })

        return JSONResponse(content={
            'status': 'success',
            'data': {
                'filename': file.filename,
                'duration': info.duration,
                'transcription': transcribed_text.strip(),
                'segments': transcription_segments
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Server on http://localhost:5000")
    print("=" * 60)
    print("\nCamera Endpoints:")
    print("  POST /camera/start?camera_index=1")
    print("  POST /camera/stop")
    print("  GET  /camera/status")
    print("  GET  /camera/frame")
    print("\nCamera Processing Endpoints:")
    print("  POST /process_camera/basic           - No LLM")
    print("  POST /process_camera/gpt2            - GPT-2")
    print("  POST /process_camera/gpt2-mini       - GPT-2 Mini")
    print("  POST /process_camera/gpt2-fusion     - GPT-2 + Fusion")
    print("  POST /process_camera/gpt2-mini-fusion - GPT-2 Mini + Fusion")
    print("\nImage Upload Processing Endpoints:")
    print("  POST /process_image/basic")
    print("  POST /process_image/gpt2")
    print("  POST /process_image/gpt2-mini")
    print("  POST /process_image/gpt2-fusion")
    print("  POST /process_image/gpt2-mini-fusion")
    print("\nAudio:")
    print("  POST /transcribe")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=5000)