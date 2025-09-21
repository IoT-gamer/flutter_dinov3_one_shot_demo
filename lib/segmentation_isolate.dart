import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_dinov3_one_shot_demo/constants.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'dart:math';

import 'package:opencv_dart/opencv_dart.dart' as cv;

// Constants
const int imageSize = AppConstants.inputSize;
const int patchSize = 16;
final imagenetMean = [0.485, 0.456, 0.406];
final imagenetStd = [0.229, 0.224, 0.225];

Future<OrtSession> initializeSession(Map<String, dynamic> args) async {
  BackgroundIsolateBinaryMessenger.ensureInitialized(
    args['token'] as RootIsolateToken,
  );
  final String modelPath = args['path'];
  final ort = OnnxRuntime();
  final providers = Platform.isAndroid
      ? [OrtProvider.NNAPI, OrtProvider.CPU]
      : Platform.isIOS
      ? [OrtProvider.CORE_ML, OrtProvider.CPU]
      : [OrtProvider.CPU];
  final options = OrtSessionOptions(providers: providers);
  final session = await ort.createSession(modelPath, options: options);
  print('✅ ONNX Session Initialized in Isolate with NNAPI provider.');
  return session;
}

Future<List<double>> createPrototype(Map<String, dynamic> args) async {
  final OrtSession session = args['session'];
  final Uint8List rgbaBytes = args['bytes'];
  final image = img.decodeImage(rgbaBytes)!;
  final rgbImage = image.convert(numChannels: 3);
  final maskImage = img.Image(
    width: image.width,
    height: image.height,
    numChannels: 1,
  );
  for (final pixel in image) {
    maskImage.setPixelR(pixel.x, pixel.y, pixel.a);
  }

  num maxVal = 0;
  num minVal = 255;
  double avgVal = 0;
  if (maskImage.isNotEmpty) {
    for (final pixel in maskImage) {
      final pVal = pixel.r;
      // In a single-channel image, R is the value
      if (pVal > maxVal) maxVal = pVal;
      if (pVal < minVal) minVal = pVal;
      avgVal += pVal;
    }
    avgVal /= maskImage.length;
  } else {
    minVal = 0;
  }
  // DEBUG: Print mask statistics for verification
  print(
    'Isolate Debug: Mask Stats -> Max: $maxVal, Min: $minVal, Avg: $avgVal',
  );
  final preprocessedData = _preprocessForPrototyping(rgbImage, maskImage);
  final inputTensor = await OrtValue.fromList(
    preprocessedData['input_tensor'] as Float32List,
    preprocessedData['shape'] as List<int>,
  );
  final inputs = {'input_image': inputTensor};
  final outputs = await session.run(inputs);
  final featuresTensor = outputs.values.first;
  final List<dynamic> flattenedList = await featuresTensor.asFlattenedList();
  final allFeatures = Float32List.fromList(flattenedList.cast<double>());
  final featureDim = allFeatures.length ~/ preprocessedData['num_patches'];
  final patchMask = preprocessedData['patch_mask'] as List<bool>;
  final foregroundFeatures = <List<double>>[];
  for (int i = 0; i < patchMask.length; i++) {
    if (patchMask[i]) {
      final feature = allFeatures.sublist(i * featureDim, (i + 1) * featureDim);
      foregroundFeatures.add(feature.cast<double>());
    }
  }
  if (foregroundFeatures.isEmpty) return [];
  final objectPrototype = List.filled(featureDim, 0.0);
  for (final feature in foregroundFeatures) {
    for (int i = 0; i < featureDim; i++) {
      objectPrototype[i] += feature[i];
    }
  }
  for (int i = 0; i < featureDim; i++) {
    objectPrototype[i] /= foregroundFeatures.length;
  }
  await inputTensor.dispose();
  await featuresTensor.dispose();
  print('✅ Prototype created in Isolate.');
  return objectPrototype;
}

Future<Map<String, dynamic>> runSegmentation(Map<String, dynamic> args) async {
  final OrtSession session = args['session'];
  final List<double> objectPrototype = args['prototype'];
  final List<Uint8List> planes = args['planes'];
  final int width = args['width'];
  final int height = args['height'];
  // Get threshold from args, with a default fallback.
  final double similarityThreshold = args['threshold'] ?? 0.7;
  final bool showLargestOnly = args['showLargestOnly'] ?? false;

  // Mats will be created, so we use a try/finally to ensure they are disposed.
  cv.Mat? yuvMat, rgbMat, rotatedMat, resizedMat;
  // Mats for post-processing
  cv.Mat? maskMat, labels, stats, centroids;
  try {
    // We assume an I420 format, which is common for Flutter's CameraImage.
    final int yuvSize = width * height * 3 ~/ 2;
    final yuvBytes = Uint8List(yuvSize);
    yuvBytes.setRange(0, width * height, planes[0]);
    yuvBytes.setRange(width * height, width * height * 5 ~/ 4, planes[1]);
    yuvBytes.setRange(width * height * 5 ~/ 4, yuvSize, planes[2]);
    yuvMat = cv.Mat.fromList(
      height * 3 ~/ 2,
      width,
      cv.MatType.CV_8UC1,
      yuvBytes,
    );
    rgbMat = cv.cvtColor(yuvMat, cv.COLOR_YUV2RGB_I420);

    rotatedMat = cv.rotate(rgbMat, cv.ROTATE_90_CLOCKWISE);

    final int hPatches = imageSize ~/ patchSize;
    int wPatches =
        (rotatedMat.cols * imageSize) ~/ (rotatedMat.rows * patchSize);
    if (wPatches % 2 != 0) wPatches -= 1;
    final int newH = hPatches * patchSize;
    final int newW = wPatches * patchSize;

    resizedMat = cv.resize(rotatedMat, (
      newW,
      newH,
    ), interpolation: cv.INTER_CUBIC);
    final preprocessed = _preprocessImageFromBytes(resizedMat.data, newW, newH);

    final inputTensor = await OrtValue.fromList(
      preprocessed['input_tensor'] as Float32List,
      preprocessed['shape'] as List<int>,
    );
    final inputs = {'input_image': inputTensor};
    final outputs = await session.run(inputs);
    final featuresTensor = outputs.values.first;
    final List<dynamic> flattenedList = await featuresTensor.asFlattenedList();
    final testFeatures = Float32List.fromList(flattenedList.cast<double>());
    final numPatches = preprocessed['num_patches'] as int;
    final featureDim = testFeatures.length ~/ numPatches;
    final similarityScores = List<double>.filled(numPatches, 0.0);
    for (int i = 0; i < numPatches; i++) {
      final feature = testFeatures.sublist(
        i * featureDim,
        (i + 1) * featureDim,
      );
      similarityScores[i] = _cosineSimilarity(feature, objectPrototype);
    }
    await inputTensor.dispose();
    await featuresTensor.dispose();

    List<double> finalScores = similarityScores;
    if (showLargestOnly) {
      final w = preprocessed['w_patches'] as int;
      final h = preprocessed['h_patches'] as int;
      final maskData = Uint8List.fromList(
        similarityScores
            // Use the dynamic threshold here
            .map((s) => s > similarityThreshold ? 255 : 0)
            .toList(),
      );
      maskMat = cv.Mat.fromList(h, w, cv.MatType.CV_8UC1, maskData);

      labels = cv.Mat.empty();
      stats = cv.Mat.empty();
      centroids = cv.Mat.empty();
      cv.connectedComponentsWithStats(
        maskMat,
        labels,
        stats,
        centroids,
        8,
        cv.MatType.CV_32S,
        cv.CCL_DEFAULT,
      );
      if (stats.rows > 1) {
        int maxArea = 0;
        int largestComponentLabel = 0;
        // Start from 1 to skip background component
        for (int i = 1; i < stats.rows; i++) {
          final area = stats.at<int>(i, cv.CC_STAT_AREA);
          if (area > maxArea) {
            maxArea = area;
            largestComponentLabel = i;
          }
        }

        if (largestComponentLabel != 0) {
          final filteredScores = List<double>.filled(numPatches, 0.0);
          final labelsData = labels.data.buffer.asInt32List();
          for (int i = 0; i < numPatches; i++) {
            if (labelsData[i] == largestComponentLabel) {
              filteredScores[i] = similarityScores[i];
            }
          }

          finalScores = filteredScores;
        }
      }
    }

    return {
      'scores': finalScores,
      'width': preprocessed['w_patches'],
      'height': preprocessed['h_patches'],
    };
  } finally {
    // IMPORTANT - Dispose all Mats to prevent memory leaks.
    yuvMat?.dispose();
    rgbMat?.dispose();
    rotatedMat?.dispose();
    resizedMat?.dispose();
    // Dispose new Mats
    maskMat?.dispose();
    labels?.dispose();
    stats?.dispose();
    centroids?.dispose();
  }
}

Map<String, dynamic> _preprocessForPrototyping(img.Image rgb, img.Image mask) {
  final preprocessedImage = _preprocessImage(rgb);
  final wPatches = preprocessedImage['w_patches'] as int;
  final hPatches = preprocessedImage['h_patches'] as int;
  final resizedMask = img.copyResize(
    mask,
    width: wPatches,
    height: hPatches,
    interpolation: img.Interpolation.nearest,
  );
  // Use '.red' for single-channel mask data.
  final maskBytes = resizedMask.getBytes(order: img.ChannelOrder.red);
  final patchMask = maskBytes.map((e) => e > 127).toList();

  return {...preprocessedImage, 'patch_mask': patchMask};
}

Map<String, dynamic> _preprocessImageFromBytes(
  Uint8List imageBytes,
  int newW,
  int newH,
) {
  final int hPatches = newH ~/ patchSize;
  final int wPatches = newW ~/ patchSize;
  final int numPatches = wPatches * hPatches;
  final inputTensor = Float32List(1 * 3 * newH * newW);
  int bufferIndex = 0;
  // This loop rearranges the RGB data into the NCHW format required by the model
  // and applies the ImageNet normalization constants.
  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < newH; y++) {
      for (int x = 0; x < newW; x++) {
        final int pixelIndex = (y * newW + x) * 3;
        final double val = imageBytes[pixelIndex + c] / 255.0;
        inputTensor[bufferIndex++] = (val - imagenetMean[c]) / imagenetStd[c];
      }
    }
  }

  return {
    'input_tensor': inputTensor,
    'shape': [1, 3, newH, newW],
    'w_patches': wPatches,
    'h_patches': hPatches,
    'num_patches': numPatches,
  };
}

Map<String, dynamic> _preprocessImage(img.Image image) {
  final w = image.width;
  final h = image.height;
  final hPatches = imageSize ~/ patchSize;
  var wPatches = (w * imageSize) ~/ (h * patchSize);
  if (wPatches % 2 != 0) wPatches -= 1;
  final newH = hPatches * patchSize;
  final newW = wPatches * patchSize;
  final resizedImg = img.copyResize(
    image,
    width: newW,
    height: newH,
    interpolation: img.Interpolation.cubic,
  );
  final numPatches = wPatches * hPatches;
  final inputTensor = Float32List(1 * 3 * newH * newW);
  int bufferIndex = 0;
  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < newH; y++) {
      for (int x = 0; x < newW; x++) {
        final pixel = resizedImg.getPixel(x, y);
        double val;
        if (c == 0) {
          val = pixel.rNormalized.toDouble();
        } else if (c == 1) {
          val = pixel.gNormalized.toDouble();
        } else {
          val = pixel.bNormalized.toDouble();
        }
        inputTensor[bufferIndex++] = (val - imagenetMean[c]) / imagenetStd[c];
      }
    }
  }
  return {
    'input_tensor': inputTensor,
    'shape': [1, 3, newH, newW],
    'w_patches': wPatches,
    'h_patches': hPatches,
    'num_patches': numPatches,
  };
}

double _cosineSimilarity(List<double> vec1, List<double> vec2) {
  double dotProduct = 0.0;
  double mag1 = 0.0;
  double mag2 = 0.0;
  for (int i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    mag1 += vec1[i] * vec1[i];
    mag2 += vec2[i] * vec2[i];
  }
  mag1 = sqrt(mag1);
  mag2 = sqrt(mag2);
  if (mag1 == 0 || mag2 == 0) return 0.0;
  return dotProduct / (mag1 * mag2);
}
