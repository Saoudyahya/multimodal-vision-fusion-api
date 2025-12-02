import requests
import time
import binascii

BASE_URL = "http://localhost:5000"


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_endpoint(endpoint_name, url, method="POST", timeout=60):
    """Test a specific endpoint and display results"""
    print(f"\n{endpoint_name}")
    print("-" * 70)

    try:
        if method == "POST":
            response = requests.post(url, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)

        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})

            # Display results
            print(f"✓ Status: {response.status_code}")
            print(f"  Model: {data.get('model_used', 'N/A')}")
            print(f"  Fusion: {data.get('fusion_enabled', False)}")
            print(f"  Caption: {data.get('caption', 'N/A')[:60]}...")
            print(f"  Objects: {len(data.get('detected_objects', []))}")

            if 'llm_description' in data:
                desc = data['llm_description']
                print(f"  Description: {desc[:80]}...")

            if 'attention_stats' in data:
                stats = data['attention_stats']
                print(f"  Attention - Mean: {stats['mean']:.4f}, Max: {stats['max']:.4f}")

            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            print(f"  Error: {response.json()}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_all_endpoints():
    print_section("FastAPI Vision System - Complete Test")
    print("\nMake sure:")
    print("  1. Server is running: python app.py")
    print("  2. Camera at index 1 is available")

    input("\nPress Enter to start testing...")

    # 1. Start camera
    print_section("Step 1: Starting Camera")
    response = requests.post(f"{BASE_URL}/camera/start?camera_index=1", timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code != 200:
        print("\n❌ Camera failed to start. Exiting...")
        return False

    print("\n⏳ Waiting 2 seconds for camera to stabilize...")
    time.sleep(2)

    # 2. Check camera status
    print_section("Step 2: Camera Status")
    response = requests.get(f"{BASE_URL}/camera/status")
    status = response.json()
    print(f"Running: {status['is_running']}")
    print(f"Frames captured: {status['frames_captured']}")
    print(f"Backend: {status.get('backend', 'Unknown')}")

    if not status.get('has_frame'):
        print("\n⚠️ No frames available yet")
        return False

    # 3. Save a frame
    print_section("Step 3: Saving Camera Frame")
    response = requests.get(f"{BASE_URL}/camera/frame")
    if response.status_code == 200:
        with open("camera_frame.jpg", "wb") as f:
            f.write(response.content)
        print("✓ Frame saved as 'camera_frame.jpg'")

    # 4. Test all processing endpoints
    print_section("Step 4: Testing All Processing Endpoints")

    endpoints = [
        ("Basic (No LLM)", f"{BASE_URL}/process_camera/basic"),
        ("GPT-2", f"{BASE_URL}/process_camera/gpt2"),
        ("GPT-2 Mini", f"{BASE_URL}/process_camera/gpt2-mini"),
        ("GPT-2 + Fusion", f"{BASE_URL}/process_camera/gpt2-fusion"),
        ("GPT-2 Mini + Fusion", f"{BASE_URL}/process_camera/gpt2-mini-fusion"),
    ]

    results = {}
    for name, url in endpoints:
        success = test_endpoint(name, url)
        results[name] = success
        time.sleep(1)  # Small delay between requests

    # 5. Get annotated image
    print_section("Step 5: Getting Annotated Image")
    response = requests.post(f"{BASE_URL}/process_camera/basic?annotate=true")
    if response.status_code == 200:
        result = response.json()
        if 'annotated_image_base64' in result:
            img_data = binascii.unhexlify(result['annotated_image_base64'])
            with open("annotated_frame.jpg", "wb") as f:
                f.write(img_data)
            print("✓ Annotated frame saved as 'annotated_frame.jpg'")

    # 6. Stop camera
    print_section("Step 6: Stopping Camera")
    response = requests.post(f"{BASE_URL}/camera/stop")
    if response.status_code == 200:
        print("✓ Camera stopped")

    # Summary
    print_section("Test Summary")
    print("\nEndpoint Results:")
    for name, success in results.items():
        status_icon = "✓" if success else "✗"
        print(f"  {status_icon} {name}")

    print("\nGenerated Files:")
    print("  • camera_frame.jpg")
    print("  • annotated_frame.jpg")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")

    return all_passed


def test_comparison():
    """Quick comparison test of all models"""
    print_section("Quick Comparison Test")

    # Start camera
    requests.post(f"{BASE_URL}/camera/start?camera_index=1")
    time.sleep(2)

    endpoints = {
        "Basic": f"{BASE_URL}/process_camera/basic",
        "GPT-2": f"{BASE_URL}/process_camera/gpt2",
        "GPT-2 Mini": f"{BASE_URL}/process_camera/gpt2-mini",
        "GPT-2 + Fusion": f"{BASE_URL}/process_camera/gpt2-fusion",
        "GPT-2 Mini + Fusion": f"{BASE_URL}/process_camera/gpt2-mini-fusion",
    }

    print("\nTesting all endpoints...")
    print("(This may take a minute)\n")

    for name, url in endpoints.items():
        print(f"Testing {name}...", end=" ")
        start_time = time.time()

        try:
            response = requests.post(url, timeout=60)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()['data']
                desc_len = len(data.get('llm_description', ''))
                print(f"✓ ({elapsed:.1f}s, {desc_len} chars)")
            else:
                print(f"✗ Failed")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Stop camera
    requests.post(f"{BASE_URL}/camera/stop")
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        test_comparison()
    else:
        try:
            test_all_endpoints()
        except KeyboardInterrupt:
            print("\n\n⚠️ Test interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Unexpected error: {e}")