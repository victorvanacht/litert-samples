package com.google.ai.edge.examples.victortest

import android.content.Context
import android.net.Uri
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi

class ModelRunner(private val context: Context) {
  suspend fun runContinuously(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    onLog: suspend (String) -> Unit,
    onResult: suspend (ModelRunResult) -> Unit,
  ): ModelRunResult = withContext(Dispatchers.IO) {
    onLog("Loading $displayName")
    onLog("Accelerator: ${accelerator.name}")
    val modelFile = copyToCache(uri, displayName)
    onLog("Copied model to cache: ${modelFile.name} (${modelFile.length()} bytes)")

    // Use the TFLite Interpreter API purely to introspect generic tensor shapes/types;
    // it is bundled transitively with litert-api and never used for actual inference.
    onLog("Inspecting tensors with TFLite Interpreter")
    val (inputShapes, outputShapes) = inspectTensorShapes(modelFile, onLog)
    onLog("Found ${inputShapes.size} input tensor(s), ${outputShapes.size} output tensor(s)")

    onLog("Compiling model for ${accelerator.name}")
    val model = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(accelerator.litertAccelerator))
    onLog("Compiled model")
    val inputBuffers = model.createInputBuffers()
    val outputBuffers = model.createOutputBuffers()
    onLog("Allocated ${inputBuffers.size} input buffer(s), ${outputBuffers.size} output buffer(s)")

    try {
      val inputDescriptions = inputBuffers.mapIndexed { index, buffer ->
        val shape = inputShapes[index]
        writeRandomInput(buffer, shape)
        val description = "input $index: ${shape.dataType}, ${shape.dimensions.toList()}"
        onLog("Wrote random $description")
        description
      }
      val outputDescriptions = outputShapes.mapIndexed { index, shape ->
        "output $index: ${shape.dataType}, ${shape.dimensions.toList()}"
      }
      var lastResult = ModelRunResult(displayName, 0, inputDescriptions + outputDescriptions)
      var iteration = 0
      onLog("Starting inference loop")
      while (currentCoroutineContext().isActive) {
        iteration++
        val startNanos = System.nanoTime()
        model.run(inputBuffers, outputBuffers)
        val elapsedMillis = (System.nanoTime() - startNanos) / 1_000_000
        lastResult = ModelRunResult(displayName, elapsedMillis, inputDescriptions + outputDescriptions)
        onLog("Inference #$iteration: $elapsedMillis ms")
        onResult(lastResult)
      }
      lastResult
    } finally {
      onLog("Releasing model resources")
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
      modelFile.delete()
      onLog("Released model resources")
    }
  }

  /** Loads the model with the TFLite Interpreter solely to read tensor shapes/types, then closes it. */
  private fun inspectTensorShapes(
    modelFile: File,
    onLog: suspend (String) -> Unit,
  ): Pair<List<TensorShape>, List<TensorShape>> {
    InterpreterApi.create(modelFile, InterpreterApi.Options()).use { interpreter ->
      val inputs = (0 until interpreter.inputTensorCount).map { index ->
        val tensor = interpreter.getInputTensor(index)
        TensorShape(tensor.shape(), tensor.dataType(), tensor.numElements())
      }
      val outputs = (0 until interpreter.outputTensorCount).map { index ->
        val tensor = interpreter.getOutputTensor(index)
        TensorShape(tensor.shape(), tensor.dataType(), tensor.numElements())
      }
      return inputs to outputs
    }
  }

  private fun writeRandomInput(buffer: TensorBuffer, shape: TensorShape) {
    when (shape.dataType) {
      DataType.FLOAT32 -> buffer.writeFloat(FloatArray(shape.elementCount) { Random.nextFloat() })
      DataType.INT32 -> buffer.writeInt(IntArray(shape.elementCount) { Random.nextInt() })
      DataType.UINT8, DataType.INT8 -> buffer.writeInt8(ByteArray(shape.elementCount) { Random.nextInt().toByte() })
      DataType.BOOL -> buffer.writeBoolean(BooleanArray(shape.elementCount) { Random.nextBoolean() })
      DataType.INT64 -> buffer.writeLong(LongArray(shape.elementCount) { Random.nextLong() })
      else -> error("Unsupported input tensor type: ${shape.dataType}")
    }
  }

  private fun copyToCache(uri: Uri, displayName: String): File {
    val safeName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
    val file = File.createTempFile("victortest_", "_$safeName", context.cacheDir)
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

enum class AcceleratorChoice(val litertAccelerator: com.google.ai.edge.litert.Accelerator) {
  CPU(com.google.ai.edge.litert.Accelerator.CPU),
  GPU(com.google.ai.edge.litert.Accelerator.GPU),
}

private data class TensorShape(val dimensions: IntArray, val dataType: DataType, val elementCount: Int)

data class ModelRunResult(
  val displayName: String,
  val inferenceTimeMillis: Long,
  val tensorDescriptions: List<String>,
)