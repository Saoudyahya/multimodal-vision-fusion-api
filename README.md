# FastAPI Vision & Audio Processing System

A complete AI-powered vision processing API with multiple language models, multi-modal fusion, and audio transcription capabilities.

## 🌟 Features

- **5 Processing Modes**: From basic detection to advanced multi-modal fusion
- **Multiple Models**: GPT-2, DistilGPT-2 (GPT-2 Mini)
- **Multi-Modal Fusion**: Combines YOLO spatial features + ViT semantic features
- **Object Detection**: YOLOv9 for accurate detection
- **Image Captioning**: ViT-GPT2 for scene understanding
- **Audio Transcription**: Whisper for speech-to-text
- **USB Camera Support**: Works with external USB cameras
- **Real-time Processing**: Continuous camera stream processing

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
- [Processing Modes](#processing-modes)
- [Usage Examples](#usage-examples)
- [Performance](#performance)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

## 🚀 Quick Start

### 1. Find Your Camera
```bash
python find_camera.py
```
This will show you which camera index to use (likely 0 or 1 for USB cameras).

### 2. Start the Server
```bash
python app.py
```

### 3. Test All Endpoints
```bash
python test_all_endpoints.py
```

That's it! Your API is now running at `http://localhost:5000`

## 📦 Installation

### Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended for fusion)
- USB Camera or built-in webcam

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download YOLO Model
```bash
# Download YOLOv9 model (~50MB)
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9c.pt

# Or use browser to download from:
# https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9c.pt
```

### Step 3: First Run
On first run, the following models will be downloaded automatically:
- ViT-GPT2 Image Captioning (~500MB)
- GPT-2 (~500MB)
- DistilGPT-2 (~319MB)
- ViT Base (~350MB) - for fusion
- Whisper Tiny (~75MB)

**Total download: ~1.7GB** (one-time only)

## 🎯 API Endpoints

### Camera Control

```http
POST   /camera/start?camera_index=1    # Start camera
POST   /camera/stop                    # Stop camera
GET    /camera/status                  # Get camera status
GET    /camera/frame                   # Get latest frame (JPEG)
```

### Camera Processing (5 Modes)

```http
POST   /process_camera/basic                # No LLM - Fast
POST   /process_camera/gpt2                 # GPT-2
POST   /process_camera/gpt2-mini            # DistilGPT-2
POST   /process_camera/gpt2-fusion          # GPT-2 + Multi-Modal Fusion
POST   /process_camera/gpt2-mini-fusion     # DistilGPT-2 + Fusion ⭐
```

### Image Upload Processing

```http
POST   /process_image/basic                 # Upload + Basic
POST   /process_image/gpt2                  # Upload + GPT-2
POST   /process_image/gpt2-mini             # Upload + DistilGPT-2
POST   /process_image/gpt2-fusion           # Upload + GPT-2 + Fusion
POST   /process_image/gpt2-mini-fusion      # Upload + DistilGPT-2 + Fusion ⭐
```

### Audio

```http
POST   /transcribe                          # Transcribe audio file
```

### Query Parameters

All processing endpoints support:
- `?annotate=true` - Return annotated image with bounding boxes

## 🎨 Processing Modes

| Mode | Model | Fusion | Speed | Quality | Use Case |
|------|-------|--------|-------|---------|----------|
| **Basic** | None | ❌ | ⚡ 0.5-1s | Good | Real-time detection |
| **GPT-2** | GPT-2 | ❌ | 🐢 3-5s | Better | Detailed descriptions |
| **GPT-2 Mini** | DistilGPT-2 | ❌ | 🏃 2-3s | Better | Fast descriptions |
| **GPT-2 Fusion** | GPT-2 | ✅ | 🐌 5-8s | Best | Highest quality |
| **GPT-2 Mini Fusion** | DistilGPT-2 | ✅ | 🏃 4-6s | Best | **Recommended** ⭐ |

### What Each Mode Provides

#### 1. Basic Mode
- ✅ Object detection (YOLO)
- ✅ Image captioning (ViT-GPT2)
- ✅ Spatial guidance (left/right/front)

#### 2. GPT-2 / GPT-2 Mini Mode
- ✅ Everything in Basic
- ✅ Detailed text descriptions from LLM

#### 3. Fusion Mode (GPT-2 or GPT-2 Mini + Fusion)
- ✅ Everything in GPT-2 mode
- ✅ Multi-modal feature fusion
- ✅ Visual context for LLM
- ✅ Attention statistics
- ✅ Spatially-aware descriptions

### Multi-Modal Fusion Explained

Fusion combines:
1. **Spatial Features** from YOLO (positions, sizes, confidence)
2. **Semantic Features** from ViT (visual understanding)
3. **Cross-Attention** mechanism to merge both

Result: **Richer, more spatially-aware descriptions**

```
Image → YOLO (spatial) ┐
                        ├→ Cross-Attention → Fused Features → GPT-2 → Rich Description
Image → ViT (semantic) ┘
```

## 💻 Usage Examples

### Python

```python
import requests
import time

BASE_URL = "http://localhost:5000"

# Start camera
response = requests.post(f"{BASE_URL}/camera/start?camera_index=1")
print(response.json())

time.sleep(2)  # Wait for camera to initialize

# Process with different modes
# Basic mode
r = requests.post(f"{BASE_URL}/process_camera/basic")
data = r.json()['data']
print(f"Objects: {len(data['detected_objects'])}")
print(f"Guidance: {data['guidance']}")

# GPT-2 mode
r = requests.post(f"{BASE_URL}/process_camera/gpt2")
data = r.json()['data']
print(f"Description: {data['llm_description']}")

# GPT-2 Mini + Fusion (Recommended)
r = requests.post(f"{BASE_URL}/process_camera/gpt2-mini-fusion")
data = r.json()['data']
print(f"Description: {data['llm_description']}")
print(f"Attention: {data['attention_stats']}")

# Get annotated image
r = requests.post(f"{BASE_URL}/process_camera/basic?annotate=true")
if 'annotated_image_base64' in r.json():
    import binascii
    img_data = binascii.unhexlify(r.json()['annotated_image_base64'])
    with open("annotated.jpg", "wb") as f:
        f.write(img_data)

# Upload and process image
with open("myimage.jpg", "rb") as f:
    files = {"file": f}
    r = requests.post(
        f"{BASE_URL}/process_image/gpt2-mini-fusion",
        files=files
    )
    print(r.json()['data']['llm_description'])

# Stop camera
requests.post(f"{BASE_URL}/camera/stop")
```

### cURL

```bash
# Start camera (use your camera index)
curl -X POST "http://localhost:5000/camera/start?camera_index=1"

# Basic processing
curl -X POST "http://localhost:5000/process_camera/basic"

# GPT-2 Mini + Fusion (recommended)
curl -X POST "http://localhost:5000/process_camera/gpt2-mini-fusion"

# Get annotated image
curl -X POST "http://localhost:5000/process_camera/basic?annotate=true" \
  -o response.json

# Upload and process
curl -X POST -F "file=@image.jpg" \
  "http://localhost:5000/process_image/gpt2-mini-fusion"

# Save camera frame
curl "http://localhost:5000/camera/frame" -o frame.jpg

# Transcribe audio
curl -X POST -F "file=@audio.wav" \
  "http://localhost:5000/transcribe"

# Stop camera
curl -X POST "http://localhost:5000/camera/stop"
```

### JavaScript

```javascript
const BASE_URL = 'http://localhost:5000';

// Start camera
await fetch(`${BASE_URL}/camera/start?camera_index=1`, {
  method: 'POST'
});

// Wait for initialization
await new Promise(resolve => setTimeout(resolve, 2000));

// Process with GPT-2 Mini + Fusion
const response = await fetch(
  `${BASE_URL}/process_camera/gpt2-mini-fusion`,
  { method: 'POST' }
);
const data = await response.json();

console.log('Caption:', data.data.caption);
console.log('Description:', data.data.llm_description);
console.log('Attention:', data.data.attention_stats);

// Upload image
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch(
  `${BASE_URL}/process_image/gpt2-mini-fusion`,
  {
    method: 'POST',
    body: formData
  }
);
const result = await uploadResponse.json();
console.log(result.data.llm_description);

// Stop camera
await fetch(`${BASE_URL}/camera/stop`, { method: 'POST' });
```

## 📊 Performance

### Processing Times (CPU, Single Frame)

| Endpoint | Time | Description Length | Memory |
|----------|------|-------------------|--------|
| Basic | 0.5-1s | N/A | ~2GB |
| GPT-2 | 3-5s | 100-200 chars | ~4GB |
| GPT-2 Mini | 2-3s | 100-200 chars | ~3GB |
| GPT-2 Fusion | 5-8s | 150-300 chars | ~6GB |
| GPT-2 Mini Fusion | 4-6s | 150-300 chars | ~5GB |

### Model Sizes

| Model | Size | Speed vs GPT-2 | Quality vs GPT-2 |
|-------|------|---------------|------------------|
| GPT-2 | 548MB | 1.0x (baseline) | 100% |
| DistilGPT-2 | 319MB | 1.6x faster | 97% |

### Recommendations

- **Real-time apps**: Use `/basic`
- **Balanced**: Use `/gpt2-mini` or `/gpt2-mini-fusion` ⭐
- **Best quality**: Use `/gpt2-fusion`
- **Production**: Use `/gpt2-mini-fusion` (best balance)

## 🔧 Configuration

### Enable/Disable Fusion Globally

Edit `app.py` around line 365:
```python
ENABLE_FUSION = True  # Set to False to disable fusion
```

When disabled, fusion endpoints will return an error.

### Change Camera Index

Edit when starting camera:
```python
requests.post("http://localhost:5000/camera/start?camera_index=1")
# Change 1 to your camera index from find_camera.py
```

### Use Smaller YOLO Model

For faster processing, replace `yolov9c.pt` with `yolov8n.pt`:

```bash
# Download YOLOv8 Nano (smaller, faster)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Edit `app.py` line ~365:
```python
main_app = MainApp(
    yolo_path="yolov8n.pt",  # Changed from yolov9c.pt
    ...
)
```

## 🐛 Troubleshooting

### Camera Not Found

**Problem**: Camera fails to start

**Solution**:
```bash
python find_camera.py
```
- Note which camera index works (usually 0 or 1)
- Use that index: `POST /camera/start?camera_index=1`
- Make sure no other app is using the camera (Zoom, Teams, Skype, etc.)

### Out of Memory

**Problem**: Server crashes or runs out of memory

**Solutions**:
1. Disable fusion: Set `ENABLE_FUSION = False` in `app.py`
2. Use GPT-2 Mini instead of GPT-2
3. Use smaller YOLO: `yolov8n.pt` instead of `yolov9c.pt`
4. Close other applications

### Slow Processing

**Problem**: Endpoints take too long

**Solutions**:
1. Use `/gpt2-mini` instead of `/gpt2`
2. Use `/basic` for real-time apps
3. Disable fusion for faster processing
4. Use GPU if available (install `torch` with CUDA)

### NaN Error in Fusion

**Problem**: `Out of range float values are not JSON compliant: nan`

**Solution**: This is fixed in the latest version. If you still see it:
- Update to the latest `app.py`
- The fix handles single-object detection properly

### First Run Timeout

**Problem**: First request times out

**Solution**: First run downloads models (~1.7GB). This can take 2-5 minutes depending on internet speed. Just wait and try again.

### Models Not Downloading

**Problem**: Models fail to download

**Solution**:
```bash
# Pre-download models manually
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, VisionEncoderDecoderModel, ViTModel
from ultralytics import YOLO

# Download models
AutoTokenizer.from_pretrained('gpt2')
AutoModelForCausalLM.from_pretrained('gpt2')
AutoTokenizer.from_pretrained('distilgpt2')
AutoModelForCausalLM.from_pretrained('distilgpt2')
VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
ViTModel.from_pretrained('google/vit-base-patch16-224')
print('Models downloaded successfully!')
"
```

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                       │
├─────────────────────────────────────────────────────────┤
│  Camera Manager (USB Camera Support)                    │
│  ├─ DirectShow Backend (Windows)                        │
│  ├─ V4L2 Backend (Linux)                               │
│  └─ AVFoundation Backend (macOS)                        │
├─────────────────────────────────────────────────────────┤
│  Object Detection (YOLO)                                │
│  ├─ Bounding boxes                                      │
│  ├─ Class labels                                        │
│  └─ Confidence scores                                   │
├─────────────────────────────────────────────────────────┤
│  Image Captioning (ViT-GPT2)                           │
│  └─ Natural language scene description                  │
├─────────────────────────────────────────────────────────┤
│  Spatial Guidance Generator                             │
│  └─ Left/Right/Front object positioning                │
├─────────────────────────────────────────────────────────┤
│  Multi-Modal Fusion (Optional)                          │
│  ├─ YOLO Spatial Features Extractor                    │
│  ├─ ViT Semantic Features Extractor                    │
│  └─ Cross-Attention Fusion Module                      │
├─────────────────────────────────────────────────────────┤
│  Language Models                                         │
│  ├─ GPT-2 (548MB)                                      │
│  └─ DistilGPT-2 (319MB)                                │
├─────────────────────────────────────────────────────────┤
│  Audio Transcription (Whisper)                          │
│  └─ Speech-to-text                                      │
└─────────────────────────────────────────────────────────┘
```

### Multi-Modal Fusion Architecture

```
         ┌─────────┐
         │  Image  │
         └────┬────┘
              │
    ┌─────────┴─────────┐
    │                   │
┌───▼────┐         ┌───▼────┐
│  YOLO  │         │  ViT   │
│Spatial │         │Semantic│
│ [N,8]  │         │ [768]  │
└───┬────┘         └───┬────┘
    │                  │
    └────────┬─────────┘
             │
      ┌──────▼──────┐
      │   Cross-    │
      │  Attention  │
      │  Fusion     │
      │   [256]     │
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │   Adapter   │
      │   Layer     │
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │   GPT-2/    │
      │ DistilGPT-2 │
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │    Rich     │
      │ Description │
      └─────────────┘
```

## 📝 Response Format

### Basic/GPT-2/GPT-2 Mini Response

```json
{
  "status": "success",
  "data": {
    "caption": "a person sitting at a desk with a laptop",
    "detected_objects": [
      {
        "label": "person",
        "confidence": 0.95,
        "bounding_box": [100, 150, 400, 600]
      },
      {
        "label": "laptop",
        "confidence": 0.89,
        "bounding_box": [200, 300, 450, 500]
      }
    ],
    "guidance": "Scene: a person sitting at a desk with a laptop. To your front: a person and a laptop.",
    "model_used": "gpt2",
    "fusion_enabled": false,
    "llm_description": "Image caption: a person sitting at a desk with a laptop\nDetected objects: person (95%), laptop (89%).\nDescribe the scene..."
  },
  "timestamp": "2024-12-02 15:30:45"
}
```

### Fusion Mode Response

```json
{
  "status": "success",
  "data": {
    "caption": "a person sitting at a desk with a laptop",
    "detected_objects": [...],
    "guidance": "...",
    "model_used": "gpt2-mini",
    "fusion_enabled": true,
    "llm_description": "Caption: a person sitting at a desk with a laptop\nObjects: person (95%), laptop (89%)\nDetailed description: The image shows a person...",
    "attention_stats": {
      "mean": 0.5234,
      "max": 0.8912,
      "std": 0.1567
    }
  },
  "timestamp": "2024-12-02 15:30:45"
}
```

## 📚 Files Included

- `app.py` - Main FastAPI server
- `test_all_endpoints.py` - Comprehensive test script
- `find_camera.py` - Camera diagnostic tool
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `QUICKSTART.md` - Quick reference guide
- `FIXES.md` - Troubleshooting and fixes
- `API_DOCS.md` - Detailed API documentation

## 🎓 Use Cases

### 1. Assistive Technology
```python
# Real-time guidance for visually impaired
response = requests.post(f"{BASE_URL}/process_camera/basic")
guidance = response.json()['data']['guidance']
# Convert to speech: "To your left: a chair. To your front: a door."
```

### 2. Content Analysis
```python
# Analyze uploaded images
with open("product_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/process_image/gpt2-mini-fusion",
        files={"file": f}
    )
description = response.json()['data']['llm_description']
# Use for product descriptions, alt text, etc.
```

### 3. Security Monitoring
```python
# Continuous monitoring
requests.post(f"{BASE_URL}/camera/start?camera_index=1")
while monitoring:
    response = requests.post(f"{BASE_URL}/process_camera/basic")
    objects = response.json()['data']['detected_objects']
    if any(obj['label'] == 'person' for obj in objects):
        trigger_alert()
```

### 4. Video Meeting Enhancement
```python
# Real-time scene understanding
response = requests.post(f"{BASE_URL}/process_camera/gpt2-mini")
caption = response.json()['data']['caption']
# Add as overlay or metadata to video stream
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional language models
- More fusion architectures
- Performance optimizations
- Additional camera backends
- WebSocket support for streaming

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Credits

- **YOLOv9** by WongKinYiu
- **Transformers** by Hugging Face
- **Faster-Whisper** by guillaumekln
- **FastAPI** by Sebastián Ramírez
- **OpenCV** for computer vision
- **PyTorch** for deep learning

## 📞 Support

Having issues? Check:
1. `FIXES.md` - Common problems and solutions
2. `QUICKSTART.md` - Quick reference
3. `API_DOCS.md` - Detailed API documentation

## ⭐ Quick Commands Reference

```bash
# Setup
python find_camera.py              # Find camera
python app.py                      # Start server
python test_all_endpoints.py      # Test everything

# Quick test
curl -X POST "http://localhost:5000/camera/start?camera_index=1"
curl -X POST "http://localhost:5000/process_camera/gpt2-mini-fusion"
curl -X POST "http://localhost:5000/camera/stop"
```

## 🎯 Recommended Setup

For the best experience:
1. Use **GPT-2 Mini + Fusion** (`/gpt2-mini-fusion`)
2. Enable fusion: `ENABLE_FUSION = True`
3. Use USB camera at index 1
4. 16GB RAM recommended

This gives you the best balance of speed, quality, and visual awareness!

---

**Ready to get started?** Run `python find_camera.py` then `python app.py`!