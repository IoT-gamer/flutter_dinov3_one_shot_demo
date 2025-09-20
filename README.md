# DINOv3 One-Shot Segmentation Flutter Demo

This repository contains a Flutter application demonstrating real-time, one-shot object segmentation using a DINOv3 feature extractor model. The app uses the device's camera feed, allowing users to define a target object with a single reference image and see it segmented live.

## 📋 Features

* **One-Shot Learning:** Define an object to segment using just one example image (the "prototype").

* **Real-Time Segmentation:** Segments objects directly from the live camera feed.

* **Largest Area Filtering:** An option to display only the largest contiguous segmented object, removing smaller, potentially noisy detections.

* **ONNX Runtime:** Utilizes the `flutter_onnxruntime` package for efficient, cross-platform model inference.

* **Hardware Acceleration:** Leverages platform-specific accelerators like NNAPI on Android and Core ML on iOS  for better performance.

* **Responsive UI:** Performs heavy model inference on a separate isolate to prevent UI jank and ensure a smooth user experience.

## ⚙️ How It Works

The application follows a simple but powerful workflow for one-shot segmentation:

* **Create a Prototype:** The user selects a reference image from their gallery. This image must be a PNG with a transparent background (RGBA), where the object of interest is opaque.

* **Extract Features:** The DINOv3 model processes this reference image and extracts a feature vector (the "object prototype") from the non-transparent parts of the image.

* **Process Camera Feed:** The app continuously receives frames from the camera. To optimize performance, it only processes a fraction of these frames, determined by the `frameSkipCount`.

* **Segment and Compare:** For each processed frame, the DINOv3 model extracts features for small patches of the image. The app then calculates the cosine similarity between the reference prototype and the feature vector of each patch.

* **Visualize the Mask:** Patches with a similarity score above a defined `similarityThreshold` are considered part of the target object. These patches are colored to create a segmentation mask, which is overlaid on the camera preview in real-time.

## 🚀 Getting Started

Follow these steps to get the demo up and running on your local machine.

1. Prerequisites
    * Flutter SDK (stable channel)
    * An IDE like VS Code or Android Studio
    * A physical device (Android or iOS) for testing camera features.

2. Clone the Repository
    ```bash
    git clone https://github.com/IoT-gamer/flutter_dinov3_one_shot_demo.git
    cd flutter_dinov3_one_shot_demo
    ```

3. Get the ONNX Model
    This project requires a `dinov3_feature_extractor.onnx` model.
    * You can export your own model by following the steps in this Jupyter Notebook:
[DINOv3 ONNX Export Notebook](https://github.com/IoT-gamer/segment-anything-dinov3-onnx/blob/main/notebooks/dinov3_onnx_export.ipynb)
    * Place the exported `dinov3_feature_extractor.onnx` file inside the `assets/` folder in the root of the project.

4. Install Dependencies
    ```bash
    flutter pub get
    ```

4. Run the App
    Connect your device and run:
    ```bash
    flutter run
    ```

##  📱 App Usage

1. Create a Reference Image
You need a reference image of the object you want to segment. The image must be a PNG with a transparent background (RGBA format), where only the object is visible.

    * You can easily create such images using the Flutter Segment Anything App:
    https://github.com/IoT-gamer/flutter_segment_anything_app

2. Set the Reference Prototype
    1. Launch the app.

    2. Tap the **Upload File** icon on the floating action button.

    3. Select the RGBA reference image you created from your device's gallery.

    4. A success message "✅ Reference prototype created!" will appear if the image is valid. If the image is invalid (e.g., not a PNG with transparency), an error will be shown.


3. Start Segmentation
    1. After setting a prototype, tap the **Play** icon.

    2. Point your camera at different scenes. The app will highlight any objects it recognizes as being similar to your reference prototype.

    3. Tap the **Stop** icon to pause the segmentation process.

4. Filter for Largest Area
    1. While segmentation is active, you may see multiple disconnected areas highlighted.

    2. To focus only on the main object, tap the Filter icon (`filter_center_focus`).

    3. This will toggle a mode that processes the mask and displays only the largest single area, helping to reduce noise.

## 🔧 Configuration

You can easily tweak the model's behavior by modifying the constants in `lib/constants.dart`:

* `inputSize`: The input resolution for the model. Smaller sizes are faster but may be less accurate. For example, try comparing 320, 400, 768.

* `similarityThreshold`: A value between 0.0 and 1.0 that determines how similar a patch must be to the prototype to be included in the mask. Higher values are stricter.

* `frameSkipCount`: The number of camera frames to skip between each processing cycle. Increasing this value improves performance but reduces the real-time feel.

## 🙏 Acknowledgements
This work builds upon the official implementations and research from the following projects:

**DINOv3:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.