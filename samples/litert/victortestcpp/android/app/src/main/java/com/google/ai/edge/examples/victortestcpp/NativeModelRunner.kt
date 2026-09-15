package com.google.ai.edge.examples.victortestcpp

import android.content.Context
import android.net.Uri
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

class NativeModelRunner(private val context: Context) {
  companion object {
    init {
      System.loadLibrary("victortestcpp_jni")
    }
  }

  suspend fun runSynchronous(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    gpuPrecision: GpuPrecision,
    gpuBackend: GpuBackend,
    gpuPriority: GpuPriority,
    gpuBufferStorageType: GpuBufferStorageType,
    gpuPreferTextureWeights: Boolean,
    gpuConstantTensorSharing: Boolean,
    gpuInfiniteFloatCapping: Boolean,
    onLog: suspend (String) -> Unit,
    onResult: suspend (ModelRunResult) -> Unit,
  ): Unit = withContext(Dispatchers.IO) {
    val modelFile = copyToCache(uri, displayName)
    val handle = nativePrepare(modelFile.absolutePath, if (accelerator == AcceleratorChoice.GPU) 1 else 0)
    try {
      val tensorDescriptions = nativeTensorDescriptions(handle).toList()
      onLog("Compiled model in C++")
      for (description in tensorDescriptions) onLog(description)
      var iteration = 0
      while (currentCoroutineContext().isActive) {
        iteration++
        val elapsedMillis = nativeRun(handle)
        onLog("Inference #$iteration: $elapsedMillis ms")
        onResult(ModelRunResult(displayName, elapsedMillis, tensorDescriptions))
      }
    } finally {
      nativeClose(handle)
      modelFile.delete()
    }
  }

  suspend fun runAsynchronous(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    gpuPrecision: GpuPrecision,
    gpuBackend: GpuBackend,
    gpuPriority: GpuPriority,
    gpuBufferStorageType: GpuBufferStorageType,
    gpuPreferTextureWeights: Boolean,
    gpuConstantTensorSharing: Boolean,
    gpuInfiniteFloatCapping: Boolean,
    concurrency: Int,
    onLog: suspend (String) -> Unit,
    onThroughput: suspend (ThroughputResult) -> Unit,
  ): Unit = withContext(Dispatchers.IO) {
    val modelFile = copyToCache(uri, displayName)
    val handle = nativePrepare(modelFile.absolutePath, if (accelerator == AcceleratorChoice.GPU) 1 else 0)
    try {
      val tensorDescriptions = nativeTensorDescriptions(handle).toList()
      onLog("Compiled model in C++")
      for (description in tensorDescriptions) onLog(description)
      while (currentCoroutineContext().isActive) {
        val rate = nativeRunConcurrent(handle, concurrency)
        onLog("Throughput: %.1f inferences/s".format(rate))
        onThroughput(ThroughputResult(rate, tensorDescriptions))
      }
    } finally {
      nativeClose(handle)
      modelFile.delete()
    }
  }

  private external fun nativePrepare(modelPath: String, accelerator: Int): Long

  private external fun nativeTensorDescriptions(handle: Long): Array<String>

  private external fun nativeRun(handle: Long): Long

  private external fun nativeRunConcurrent(handle: Long, concurrency: Int): Double

  private external fun nativeClose(handle: Long)

  private fun copyToCache(uri: Uri, displayName: String): File {
    val safeName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
    val file = File.createTempFile("victortestcpp_", "_$safeName", context.cacheDir)
    val assetName = uri.schemeSpecificPart.removePrefix("//")
    val input = if (uri.scheme == "asset") context.assets.open(assetName)
    else context.contentResolver.openInputStream(uri)
    input.use {
      requireNotNull(input) { "Unable to open $displayName" }
      file.outputStream().use { output -> input.copyTo(output) }
    }
    return file
  }
}