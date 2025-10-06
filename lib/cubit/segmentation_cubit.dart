import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:bloc/bloc.dart';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dinov3_one_shot_demo/constants.dart';
import 'package:image_picker/image_picker.dart';
import 'package:integral_isolates/integral_isolates.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path/path.dart' as p;
import '../segmentation_isolate.dart';
import 'package:image/image.dart' as img;

part 'segmentation_state.dart';

class SegmentationCubit extends Cubit<SegmentationState> {
  SegmentationCubit() : super(const SegmentationState());

  late final StatefulIsolate _isolate;
  OrtSession? _session;
  bool _isProcessing = false;

  Future<void> initialize() async {
    emit(state.copyWith(status: SegmentationStatus.loadingModel));
    _isolate = StatefulIsolate();
    try {
      final byteData = await rootBundle.load(
        'assets/${AppConstants.modelFilename}',
      );
      final directory = await getApplicationDocumentsDirectory();
      final modelPath = p.join(directory.path, AppConstants.modelFilename);
      await File(modelPath).writeAsBytes(byteData.buffer.asUint8List());
      final token = RootIsolateToken.instance!;
      _session = await _isolate.compute(initializeSession, {
        'path': modelPath,
        'token': token,
      });
      emit(state.copyWith(status: SegmentationStatus.modelReady));
    } catch (e) {
      emit(
        state.copyWith(
          status: SegmentationStatus.error,
          errorMessage: 'Failed to initialize model: $e',
        ),
      );
    }
  }

  Future<void> setReferencePrototype() async {
    if (_session == null ||
        state.status == SegmentationStatus.creatingPrototype)
      return;
    emit(state.copyWith(status: SegmentationStatus.creatingPrototype));

    try {
      final XFile? pickedFile = await ImagePicker().pickImage(
        source: ImageSource.gallery,
      );
      if (pickedFile == null) {
        emit(state.copyWith(status: SegmentationStatus.modelReady));
        return;
      }

      final fileBytes = await pickedFile.readAsBytes();
      final image = img.decodePng(fileBytes);
      if (image == null || image.numChannels != 4) {
        emit(
          state.copyWith(
            status: SegmentationStatus.error,
            errorMessage:
                'Invalid file. Please select a PNG with transparency (RGBA).',
          ),
        );
        emit(
          state.copyWith(
            status: SegmentationStatus.modelReady,
            errorMessage: null,
          ),
        );
        return;
      }

      final prototype = await _isolate.compute(createPrototype, {
        'session': _session!,
        'bytes': fileBytes,
        'inputSize': state.selectedInputSize,
      });

      if (prototype.isNotEmpty) {
        emit(
          state.copyWith(
            status: SegmentationStatus.prototypeReady,
            objectPrototype: prototype,
          ),
        );
      } else {
        emit(
          state.copyWith(
            status: SegmentationStatus.error,
            errorMessage: 'Could not create prototype from the selected image.',
          ),
        );
        emit(
          state.copyWith(
            status: SegmentationStatus.modelReady,
            errorMessage: null,
          ),
        );
      }
    } catch (e) {
      emit(
        state.copyWith(
          status: SegmentationStatus.error,
          errorMessage: 'An error occurred: $e',
        ),
      );
      emit(
        state.copyWith(
          status: SegmentationStatus.modelReady,
          errorMessage: null,
        ),
      );
    }
  }

  void processCameraImage(CameraImage cameraImage) {
    if (!state.isSegmenting ||
        _isProcessing ||
        state.objectPrototype == null ||
        _session == null) {
      return;
    }
    _isProcessing = true;

    _isolate
        .compute(runSegmentation, {
          'session': _session!,
          'prototype': state.objectPrototype!,
          'planes': cameraImage.planes.map((p) => p.bytes).toList(),
          'width': cameraImage.width,
          'height': cameraImage.height,
          'format': cameraImage.format.group,
          'threshold': state.similarityThreshold,
          'inputSize': state.selectedInputSize,
          'showLargestOnly': state.showLargestAreaOnly,
        })
        .then((result) {
          if (state.isSegmenting && result.isNotEmpty) {
            _updateOverlay(
              result['scores'] as List<double>,
              result['width'] as int,
              result['height'] as int,
            );
          }
          _isProcessing = false;
        });
  }

  void _updateOverlay(List<double> scores, int width, int height) {
    final pixels = Uint8List(width * height * 4);
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > state.similarityThreshold) {
        pixels[i * 4 + 0] = 30; // R
        pixels[i * 4 + 1] = 255; // G
        pixels[i * 4 + 2] = 150; // B
        pixels[i * 4 + 3] = 170; // A
      }
    }

    ui.decodeImageFromPixels(pixels, width, height, ui.PixelFormat.rgba8888, (
      img,
    ) {
      emit(state.copyWith(overlayImage: img));
    });
  }

  void toggleSegmentation() {
    if (state.status != SegmentationStatus.prototypeReady) return;
    final newIsSegmenting = !state.isSegmenting;
    emit(
      state.copyWith(
        isSegmenting: newIsSegmenting,
        clearOverlay: !newIsSegmenting,
      ),
    );
  }

  void updateInputSize(int newSize) {
    if (newSize != state.selectedInputSize) {
      emit(
        state.copyWith(
          selectedInputSize: newSize,
          isSegmenting: false,
          status: SegmentationStatus.modelReady,
          clearPrototype: true,
          clearOverlay: true,
        ),
      );
    }
  }

  void updateThreshold(double value) =>
      emit(state.copyWith(similarityThreshold: value));
  void toggleLargestAreaOnly() =>
      emit(state.copyWith(showLargestAreaOnly: !state.showLargestAreaOnly));
  void toggleSliderVisibility() =>
      emit(state.copyWith(showSlider: !state.showSlider));

  @override
  Future<void> close() {
    _session?.close();
    _isolate.dispose();
    return super.close();
  }
}
