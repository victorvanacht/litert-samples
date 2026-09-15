# LiteRT Core Samples

Basic usage samples and end-to-end demonstrations for LiteRT APIs.

## Overview

- **CompiledModel API**: Code examples in C++, C, Python, Kotlin, Swift, and Rust.
- **End-to-End Demos**: Showcases for ASR, TTS, Vision, and other model capabilities.
- **Feature Demos**: Miscellaneous demonstrations highlighting LiteRT features.

## victortest shell benchmarks

The `victortest` and `victortestcpp` Android apps expose a self-targeting instrumentation entry point for automated shell-driven latency tests:

```sh
adb shell mkdir -p /sdcard/Android/data/com.google.ai.edge.examples.victortest/files/models
adb push yourmodel.tflite \
	/sdcard/Android/data/com.google.ai.edge.examples.victortest/files/models/yourmodel.tflite

adb shell am instrument -w \
	-e model /sdcard/Android/data/com.google.ai.edge.examples.victortest/files/models/yourmodel.tflite \
	-e model_display_name yourmodel.tflite \
	-e runs 20 \
	-e warmup_runs 5 \
	com.google.ai.edge.examples.victortest/.ShellBenchmarkInstrumentation

adb shell mkdir -p /sdcard/Android/data/com.google.ai.edge.examples.victortestcpp/files/models
adb push yourmodel.tflite \
	/sdcard/Android/data/com.google.ai.edge.examples.victortestcpp/files/models/yourmodel.tflite

adb shell am instrument -w \
	-e model /sdcard/Android/data/com.google.ai.edge.examples.victortestcpp/files/models/yourmodel.tflite \
	-e model_display_name yourmodel.tflite \
	-e runs 20 \
	-e warmup_runs 5 \
	com.google.ai.edge.examples.victortestcpp/.ShellBenchmarkInstrumentation
```

The command prints the resolved option values, `average_inference_time_ms`, `total_inference_time_ms`, and `inference_times_ms` in the instrumentation result output. Supported extras mirror the UI controls: `model`, `model_display_name`, `input`, `output`, `accelerator`, `gpu_precision`, `gpu_backend`, `gpu_priority`, `gpu_buffer_storage_type`, `gpu_prefer_texture_weights`, `gpu_constant_tensor_sharing`, `gpu_infinite_float_capping`, and `run_mode`. Push shell-provided files into the app-specific external files directory, because app processes cannot reliably open raw paths under shared folders such as `/sdcard/Download` on modern Android. The default model is `asset://selfie_multiclass.tflite`. Latency benchmarking currently supports `run_mode=SYNCHRONOUS`.