# Graduation Project App (dgis-mobile)

This Android application serves as a real-time computer vision assistant designed for edge devices. Built using Kotlin, the app processes a live camera feed to perform sophisticated Object Detection (YOLO) and Monocular Depth Estimation simultaneously. Furthermore, the application features an automatic USB Serial integration (for ESP32 and other microcontrollers) to transmit environmental data, acting as a powerful perception module for assistive navigation or autonomous robotics.

## Key Features

- **Live Object Detection (Edge AI)**:
  - Runs **YOLOv11** and **YOLOv26** object detection models utilizing **TensorFlow Lite**.
  - Supports multiple quantizations (INT8, FP16) and seamlessly accommodates GPU delegation for enhanced performance.
  - Automatically parses bounding boxes, scales them to the screen coordinate system, and handles Non-Maximum Suppression (NMS).
  
- **Monocular Depth Estimation**:
  - Employs **Depth Anything V2** running via **ONNX Runtime Android** to calculate relative depth per frame.
  - Produces an 8-bit raw alpha depth map used for "Hazard Analysis" (e.g., detecting proximity to walls or obstacles).
  - Translates depth estimations into a visual, *Inferno*-styled colormapped image presented in a Picture-in-Picture (PiP) view.

- **USB Host & Microcontroller Integration**:
  - Uses the `usb-serial-for-android` library to establish a 115200 baud connection over USB (OTG / Host Mode).
  - Transmits serialized hazard detections, YOLO signals, and IMU data externally, perfect for triggering actuators or motors on an ESP32.
  - The app auto-launches upon detecting an authorized USB device.

- **Sensor Data (IMU)**:
  - Subscribes to the device's default `TYPE_ROTATION_VECTOR` sensor to track spatial orientation (Gyroscope data output).

- **CameraX Pipeline**:
  - Leverages Android **CameraX** for robust and responsive camera lifecycle management.
  - Frames are ingested via an `ImageAnalysis` use-case, converted to Bitmaps, and scheduled continuously to ML workers.

## Project Structure & Notable Components

- `MainActivity.kt`: Central module handling CameraX orchestration, USB Serial connections, Sensor management, multithreaded ML dispatching, and permissions.
- `YOLOv11Detector.kt` / `YOLOv26Detector.kt`: Handlers covering inference, quantization-aware de-scaling, layout transposing, and NMS for their respective model versions.
- `DepthEstimator.kt`: Encapsulates the configuration of the ONNX session to run the `fused_model_uint8_256.onnx` model asynchronously.
- `Colormap.kt`: Maps raw intensity matrices into an RGBA Inferno heatmap for real-time visualization.
- `BoxOverlayView.kt`: Custom un-letterboxing View scaling ML bounding boxes onto the device's active live preview accurately.

## Requirements

- **Android SDK:** Min SDK 26 (Android 8.0 Oreo), Target SDK 36.
- **Hardware:** Android arm64-v8a / armeabi-v7a / x86_64 device equipped with a back-facing camera and OTG support (if using serial features).
- **ML Models:** Must be placed in the `app/src/main/assets/` directory.
  - Required ONNX: `fused_model_uint8_256.onnx`
  - Required TFLite: `Venom_v5YOLOv11n_FP16.tflite` (or designated YOLO models referenced in MainActivity).
  - Required Labels: `yolo_labels.txt`

## Build and Run

1. Open the project in **Android Studio** or **VS Code**.
2. Sync Project with Gradle Files to fetch AndroidX, CameraX, TensorFlow Lite, and ONNX Runtime dependencies.
3. Build the application and deploy it via USB or Wireless debugging:
   ```bash
   ./gradlew installDebug
   ```
4. Upon launching for the first time, grant the **Camera** permission.
5. *(Optional)* Connect your ESP32/microcontroller via USB. The system will prompt you for authorization, and the app's hazard text updates will be relayed to your device.