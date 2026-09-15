package com.google.ai.edge.examples.victortestcpp

import android.content.Context
import android.net.Uri
import com.google.ai.edge.examples.modelrunner.common.AcceleratorChoice
import com.google.ai.edge.examples.modelrunner.common.InferenceRunner
import com.google.ai.edge.examples.modelrunner.common.ModelRunResult
import com.google.ai.edge.examples.modelrunner.common.ThroughputResult
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

class NativeModelRunner(private val context: Context) : InferenceRunner {
  companion object {
    init {
      System.loadLibrary("victortestcpp_jni")
    }
  }

  override suspend fun runSynchronous(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    gpuPrecision: CompiledModel.GpuOptions.Precision,
    gpuBackend: CompiledModel.GpuOptions.Backend,
    gpuPriority: CompiledModel.GpuOptions.Priority,
    gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
    gpuPreferTextureWeights: Boolean,
    gpuConstantTensorSharing: Boolean,
    gpuInfiniteFloatCapping: Boolean,
    inputFileUri: Uri?,
    outputFileUri: Uri?,
    onLog: suspend (String) -> Unit,
    onResult: suspend (ModelRunResult) -> Unit,
  ): Unit = withContext(Dispatchers.IO) {
    val modelFile = copyToCache(uri, displayName)
    val inputFile = inputFileUri?.let { copyToCache(it, "input_$displayName") }
    val outputFile = outputFileUri?.let { File.createTempFile("victortestcpp_out_", "_$displayName", context.cacheDir) }
    val handle = nativePrepare(
      modelFile.absolutePath,
      if (accelerator == AcceleratorChoice.GPU) 1 else 0,
      gpuPrecision.ordinal,
      gpuBackend.ordinal,
      gpuPriority.ordinal,
      gpuBufferStorageType.ordinal,
      gpuPreferTextureWeights,
      gpuConstantTensorSharing,
      gpuInfiniteFloatCapping,
      inputFile?.absolutePath,
      outputFile?.absolutePath,
    )
    try {
      val tensorDescriptions = nativeTensorDescriptions(handle).toList()
      onLog("Compiled model in C++")
      for (description in tensorDescriptions) onLog(description)
      var iteration = 0
      while (currentCoroutineContext().isActive) {
        iteration++
        val elapsedMillis = nativeRun(handle)
        if (outputFileUri != null && outputFile != null) copyOutputToUri(outputFile, outputFileUri)
        onLog("Inference #$iteration: $elapsedMillis ms")
        onResult(ModelRunResult(displayName, elapsedMillis, tensorDescriptions))
      }
    } finally {
      nativeClose(handle)
      modelFile.delete()
      inputFile?.delete()
      outputFile?.delete()
    }
  }

  override suspend fun runAsynchronous(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    gpuPrecision: CompiledModel.GpuOptions.Precision,
    gpuBackend: CompiledModel.GpuOptions.Backend,
    gpuPriority: CompiledModel.GpuOptions.Priority,
    gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
    gpuPreferTextureWeights: Boolean,
    gpuConstantTensorSharing: Boolean,
    gpuInfiniteFloatCapping: Boolean,
    inputFileUri: Uri?,
    outputFileUri: Uri?,
    concurrency: Int,
    onLog: suspend (String) -> Unit,
    onThroughput: suspend (ThroughputResult) -> Unit,
  ): Unit = withContext(Dispatchers.IO) {
    val modelFile = copyToCache(uri, displayName)
    val inputFile = inputFileUri?.let { copyToCache(it, "input_$displayName") }
    val outputFile = outputFileUri?.let { File.createTempFile("victortestcpp_out_", "_$displayName", context.cacheDir) }
    val handle = nativePrepare(
      modelFile.absolutePath,
      if (accelerator == AcceleratorChoice.GPU) 1 else 0,
      gpuPrecision.ordinal,
      gpuBackend.ordinal,
      gpuPriority.ordinal,
      gpuBufferStorageType.ordinal,
      gpuPreferTextureWeights,
      gpuConstantTensorSharing,
      gpuInfiniteFloatCapping,
      inputFile?.absolutePath,
      outputFile?.absolutePath,
    )
    try {
      val tensorDescriptions = nativeTensorDescriptions(handle).toList()
      onLog("Compiled model in C++")
      for (description in tensorDescriptions) onLog(description)
      while (currentCoroutineContext().isActive) {
        val rate = nativeRunConcurrent(handle, concurrency)
        if (outputFileUri != null && outputFile != null) copyOutputToUri(outputFile, outputFileUri)
        onLog("Throughput: %.1f inferences/s".format(rate))
        onThroughput(ThroughputResult(rate, tensorDescriptions))
      }
    } finally {
      nativeClose(handle)
      modelFile.delete()
      inputFile?.delete()
      outputFile?.delete()
    }
  }

  private fun copyOutputToUri(outputFile: File, destination: Uri) {
    context.contentResolver.openOutputStream(destination, "wt")?.use { output ->
      outputFile.inputStream().use { input -> input.copyTo(output) }
    }
  }

  private external fun nativePrepare(
    modelPath: String,
    accelerator: Int,
    precision: Int,
    backend: Int,
    priority: Int,
    storageType: Int,
    preferTextureWeights: Boolean,
    constantTensorSharing: Boolean,
    infiniteFloatCapping: Boolean,
    inputFilePath: String?,
    outputFilePath: String?,
  ): Long

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