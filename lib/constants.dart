/// Constants used throughout the application
class AppConstants {
  static const String modelFilename = 'dinov3_feature_extractor.onnx';
  static const int inputSize =
      320; // Try comparing 320, 400, 768 for speed vs accuracy
  static const double similarityThreshold = 0.7; // Threshold for classification
  static const int frameSkipCount =
      5; // Number of frames to skip between processing for speed
}
