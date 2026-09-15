package com.google.ai.edge.examples.victortestcpp

enum class AcceleratorChoice(val displayName: String) {
  CPU("CPU"),
  GPU("GPU"),
}

enum class RunMode {
  SYNCHRONOUS,
  ASYNCHRONOUS,
}

enum class GpuPrecision {
  DEFAULT,
  FP16,
  FP32,
  FP16_WITH_FP32_ACCUM,
}

enum class GpuBackend {
  AUTOMATIC,
  OPENCL,
  WEBGPU,
  OPENGL,
}

enum class GpuPriority {
  DEFAULT,
  LOW,
  NORMAL,
  HIGH,
}

enum class GpuBufferStorageType {
  DEFAULT,
  BUFFER,
  TEXTURE2D,
}

data class ModelRunResult(
  val displayName: String,
  val inferenceTimeMillis: Long,
  val tensorDescriptions: List<String>,
)

data class ThroughputResult(
  val inferencesPerSecond: Double,
  val tensorDescriptions: List<String>,
)
