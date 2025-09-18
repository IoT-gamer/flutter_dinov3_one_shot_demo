// lib/camera_screen.dart

import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dinov3_one_shot_demo/constants.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:integral_isolates/integral_isolates.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'segmentation_isolate.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.camera});
  final CameraDescription camera;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  late final StatefulIsolate _isolate;
  static const double similarityThreshold = AppConstants.similarityThreshold;
  int _frameCounter = 0;
  static const int frameSkipCount = AppConstants.frameSkipCount;

  OrtSession? _session;
  List<double>? _objectPrototype;

  bool _isProcessing = false;
  bool _isSegmenting = false;
  ui.Image? _overlayImage;

  bool _isCreatingPrototype = false;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
    );
    _initializeControllerFuture = _controller.initialize().then((_) {
      _controller.startImageStream(_processCameraImage);
    });

    _isolate = StatefulIsolate();
    _initIsolate();
  }

  Future<void> _initIsolate() async {
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
    print('🚀 Isolate initialized and session returned to main thread.');
  }

  void _processCameraImage(CameraImage cameraImage) {
    _frameCounter++;
    if (_frameCounter % frameSkipCount != 0) {
      return;
    }
    if (!_isSegmenting ||
        _isProcessing ||
        _objectPrototype == null ||
        _session == null) {
      return;
    }
    setState(() => _isProcessing = true);

    final image = _convertCameraImage(cameraImage);
    _isolate
        .compute(runSegmentation, {
          'session': _session!,
          'prototype': _objectPrototype!,
          'image': image,
        })
        .then((result) {
          if (result.isNotEmpty) {
            _updateOverlay(
              result['scores'] as List<double>,
              result['width'] as int,
              result['height'] as int,
            );
          }
          setState(() => _isProcessing = false);
        });
  }

  Future<void> _setReferencePrototype() async {
    if (_session == null || _isCreatingPrototype) return;

    setState(() => _isCreatingPrototype = true);

    try {
      final ImagePicker picker = ImagePicker();
      final XFile? pickedFile = await picker.pickImage(
        source: ImageSource.gallery,
      );

      if (pickedFile == null) {
        // User canceled the picker
        return;
      }

      final Uint8List fileBytes = await pickedFile.readAsBytes();
      final img.Image? image = img.decodePng(fileBytes);

      if (image == null || image.numChannels != 4) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              '❌ Invalid file. Please select a PNG image with transparency (RGBA).',
            ),
            backgroundColor: Colors.red,
          ),
        );
        return;
      }

      final prototype = await _isolate.compute(createPrototype, {
        'session': _session!,
        'bytes': fileBytes,
      });

      if (!mounted) return;

      if (prototype.isNotEmpty) {
        setState(() => _objectPrototype = prototype);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('✅ Reference prototype created!')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              '❌ Could not create prototype from the selected image.',
            ),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isCreatingPrototype = false);
      }
    }
  }

  void _updateOverlay(List<double> scores, int width, int height) {
    final pixels = Uint8List(width * height * 4);
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > similarityThreshold) {
        pixels[i * 4 + 0] = 30; // Red
        pixels[i * 4 + 1] = 255; // Green
        pixels[i * 4 + 2] = 150; // Blue
        pixels[i * 4 + 3] = 170; // Alpha
      }
    }

    ui.decodeImageFromPixels(pixels, width, height, ui.PixelFormat.rgba8888, (
      img,
    ) {
      if (mounted) {
        setState(() => _overlayImage = img);
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _session?.close();
    _isolate.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    bool isReady = _objectPrototype != null;
    return Scaffold(
      appBar: AppBar(title: const Text('DINOv3 One-Shot Segmentation')),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_controller),
                if (_overlayImage != null)
                  FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: _overlayImage!.width.toDouble(),
                      height: _overlayImage!.height.toDouble(),
                      child: CustomPaint(
                        painter: OverlayPainter(_overlayImage!),
                      ),
                    ),
                  ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            onPressed: _isCreatingPrototype ? null : _setReferencePrototype,
            tooltip: 'Set Reference',
            child: _isCreatingPrototype
                ? const CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                  )
                : const Icon(Icons.upload_file),
          ),
          const SizedBox(height: 16),
          FloatingActionButton(
            onPressed: _isCreatingPrototype
                ? null
                : () {
                    if (!isReady) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Please set a reference first!'),
                        ),
                      );
                      return;
                    }
                    setState(() => _isSegmenting = !_isSegmenting);
                  },
            tooltip: 'Toggle Segmentation',
            backgroundColor: _isCreatingPrototype ? Colors.grey : null,
            child: Icon(_isSegmenting ? Icons.stop : Icons.play_arrow),
          ),
        ],
      ),
    );
  }

  img.Image _convertCameraImage(CameraImage image) {
    img.Image resultImage;
    if (image.format.group == ImageFormatGroup.yuv420) {
      resultImage = img.Image.fromBytes(
        width: image.width,
        height: image.height,
        bytes: image.planes[0].bytes.buffer,
        order: img.ChannelOrder.red,
      );
    } else {
      resultImage = img.Image.fromBytes(
        width: image.width,
        height: image.height,
        bytes: image.planes[0].bytes.buffer,
        order: img.ChannelOrder.bgra,
      );
    }

    return img.copyRotate(resultImage, angle: 90);
  }
}

class OverlayPainter extends CustomPainter {
  final ui.Image image;
  OverlayPainter(this.image);
  @override
  void paint(Canvas canvas, Size size) {
    paintImage(
      canvas: canvas,
      rect: Rect.fromLTWH(0, 0, size.width, size.height),
      image: image,
      fit: BoxFit.fill,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
