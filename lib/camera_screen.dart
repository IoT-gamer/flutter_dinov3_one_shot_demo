import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_dinov3_one_shot_demo/constants.dart';
import 'cubit/segmentation_cubit.dart';

class CameraScreen extends StatelessWidget {
  const CameraScreen({super.key, required this.camera});
  final CameraDescription camera;

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => SegmentationCubit()..initialize(),
      child: CameraView(camera: camera),
    );
  }
}

class CameraView extends StatefulWidget {
  const CameraView({super.key, required this.camera});
  final CameraDescription camera;

  @override
  State<CameraView> createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  int _frameCounter = 0;

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
  }

  void _processCameraImage(CameraImage cameraImage) {
    _frameCounter++;
    if (_frameCounter % AppConstants.frameSkipCount != 0) {
      return;
    }
    // Forward the image to the cubit for processing
    if (mounted) {
      context.read<SegmentationCubit>().processCameraImage(cameraImage);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    // Cubit is closed by BlocProvider, no need to call close here.
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<SegmentationCubit, SegmentationState>(
      // Only listen when the status changes.
      listenWhen: (previous, current) => previous.status != current.status,
      listener: (context, state) {
        if (state.status == SegmentationStatus.error &&
            state.errorMessage != null) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('❌ ${state.errorMessage}'),
              backgroundColor: Colors.red,
            ),
          );
        } else if (state.status == SegmentationStatus.prototypeReady) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('✅ Reference prototype created!')),
          );
        }
      },
      builder: (context, state) {
        final cubit = context.read<SegmentationCubit>();
        final bool isPrototypeReady =
            state.status == SegmentationStatus.prototypeReady;
        final bool isCreatingPrototype =
            state.status == SegmentationStatus.creatingPrototype;

        return Scaffold(
          appBar: AppBar(
            title: const Text('DINOv3 One-Shot Segmentation'),
            actions: [
              PopupMenuButton<int>(
                initialValue: state.selectedInputSize,
                tooltip: 'Select Input Size',
                onSelected: cubit.updateInputSize,
                itemBuilder: (BuildContext context) {
                  return [320, 400, 512, 768].map((int size) {
                    return PopupMenuItem<int>(
                      value: size,
                      child: Text('Input Size: $size'),
                    );
                  }).toList();
                },
                icon: const Icon(Icons.aspect_ratio),
              ),
              IconButton(
                icon: Icon(
                  Icons.tune,
                  color: state.showSlider
                      ? Theme.of(context).colorScheme.secondary
                      : null,
                ),
                tooltip: 'Toggle Threshold Slider',
                onPressed: cubit.toggleSliderVisibility,
              ),
            ],
          ),
          body: FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(_controller),
                    if (state.overlayImage != null)
                      FittedBox(
                        fit: BoxFit.cover,
                        child: SizedBox(
                          width: state.overlayImage!.width.toDouble(),
                          height: state.overlayImage!.height.toDouble(),
                          child: CustomPaint(
                            painter: OverlayPainter(state.overlayImage!),
                          ),
                        ),
                      ),
                    if (state.showSlider)
                      Positioned(
                        bottom: 120,
                        left: 20,
                        right: 20,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.5),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.tune, color: Colors.white),
                              Expanded(
                                child: Slider(
                                  value: state.similarityThreshold,
                                  min: 0.5,
                                  max: 0.9,
                                  divisions: 8,
                                  label: state.similarityThreshold
                                      .toStringAsFixed(2),
                                  onChanged: cubit.updateThreshold,
                                ),
                              ),
                            ],
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
                onPressed: isCreatingPrototype
                    ? null
                    : cubit.setReferencePrototype,
                tooltip: 'Set Reference',
                child: isCreatingPrototype
                    ? const CircularProgressIndicator(
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      )
                    : const Icon(Icons.upload_file),
              ),
              const SizedBox(height: 16),
              FloatingActionButton(
                onPressed: isCreatingPrototype
                    ? null
                    : () {
                        if (!isPrototypeReady) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Please set a reference first!'),
                            ),
                          );
                          return;
                        }
                        cubit.toggleSegmentation();
                      },
                tooltip: 'Toggle Segmentation',
                backgroundColor: isCreatingPrototype ? Colors.grey : null,
                child: Icon(state.isSegmenting ? Icons.stop : Icons.play_arrow),
              ),
              const SizedBox(height: 16),
              FloatingActionButton(
                onPressed: cubit.toggleLargestAreaOnly,
                tooltip: 'Toggle Largest Area Only',
                backgroundColor: state.showLargestAreaOnly
                    ? Theme.of(context).primaryColor
                    : Colors.grey,
                child: const Icon(Icons.filter_center_focus),
              ),
            ],
          ),
        );
      },
    );
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
