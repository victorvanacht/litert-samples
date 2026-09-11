package com.google.ai.edge.examples.victortest

import android.content.Context
import android.net.Uri
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi

class ModelRunner(private val context: Context) {
  /** Runs one inference at a time, waiting for each result before starting the next. */
  suspend fun runSynchronous(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    onLog: suspend (String) -> Unit,
    onResult: suspend (ModelRunResult) -> Unit,
  ): Unit = withContext(Dispatchers.IO) {
    val prepared = prepareModel(uri, displayName, accelerator, onLog)
    try {
      val inputBuffers = prepared.model.createInputBuffers()
      val outputBuffers = prepared.model.createOutputBuffers()
      onLog("Allocated ${inputBuffers.size} input buffer(s), ${outputBuffers.size} output buffer(s)")
      inputBuffers.forEachIndexed { index, buffer -> writeRandomInput(buffer, prepared.inputShapes[index]) }

      var iteration = 0
      onLog("Starting synchronous inference loop")
      while (currentCoroutineContext().isActive) {
        iteration++
        val startNanos = System.nanoTime()
        prepared.model.run(inputBuffers, outputBuffers)
        // Read back every output, like the working image_segmentation sample does. Skipping this
        // let the GPU backend enqueue work unbounded with no synchronization.
        outputBuffers.forEachIndexed { index, buffer -> readAndDiscardOutput(buffer, prepared.outputShapes[index]) }
        val elapsedMillis = (System.nanoTime() - startNanos) / 1_000_000
        onLog("Inference #$iteration: $elapsedMillis ms")
        onResult(ModelRunResult(displayName, elapsedMillis, prepared.tensorDescriptions))
      }
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
    } finally {
      prepared.cleanup()
    }
  }

  /**
   * Keeps [concurrency] inferences in flight at all times: as soon as one slot's run completes and
   * its output is read, that slot immediately starts its next run. Reports aggregate throughput
   * (inferences/second) instead of a single latency, since individual calls now overlap.
   */
  suspend fun runAsynchronous(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    concurrency: Int,
    onLog: suspend (String) -> Unit,
    onThroughput: suspend (ThroughputResult) -> Unit,
  ): Unit = withContext(Dispatchers.IO) {
    val prepared = prepareModel(uri, displayName, accelerator, onLog)
    try {
      val slots =
        List(concurrency) {
          val inputBuffers = prepared.model.createInputBuffers()
          val outputBuffers = prepared.model.createOutputBuffers()
          inputBuffers.forEachIndexed { index, buffer -> writeRandomInput(buffer, prepared.inputShapes[index]) }
          inputBuffers to outputBuffers
        }
      onLog("Allocated $concurrency concurrent buffer set(s)")

      val completedCount = AtomicLong(0)
      val startNanos = System.nanoTime()
      onLog("Starting asynchronous inference loop with $concurrency in-flight execution(s)")

      coroutineScope {
        launch {
          while (isActive) {
            delay(200)
            val elapsedSeconds = (System.nanoTime() - startNanos) / 1_000_000_000.0
            val rate = completedCount.get() / elapsedSeconds
            onLog("Throughput: %.1f inferences/s".format(rate))
            onThroughput(ThroughputResult(rate, prepared.tensorDescriptions))
          }
        }
        slots.forEach { (inputBuffers, outputBuffers) ->
          launch {
            while (isActive) {
              prepared.model.run(inputBuffers, outputBuffers)
              outputBuffers.forEachIndexed { index, buffer ->
                readAndDiscardOutput(buffer, prepared.outputShapes[index])
              }
              completedCount.incrementAndGet()
            }
          }
        }
      }

      slots.forEach { (inputBuffers, outputBuffers) ->
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
      }
    } finally {
      prepared.cleanup()
    }
  }

  /** Copies the model, inspects its tensor shapes, and compiles it; shared by both run modes. */
  private suspend fun prepareModel(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    onLog: suspend (String) -> Unit,
  ): PreparedModel {
    onLog("Loading $displayName")
    onLog("Accelerator: ${accelerator.name}")
    val modelFile = copyToCache(uri, displayName)
    onLog("Copied model to cache: ${modelFile.name} (${modelFile.length()} bytes)")

    // Use the TFLite Interpreter API purely to introspect generic tensor shapes/types;
    // it is bundled transitively with litert-api and never used for actual inference.
    onLog("Inspecting tensors with TFLite Interpreter")
    val (inputShapes, outputShapes) = inspectTensorShapes(modelFile)
    onLog("Found ${inputShapes.size} input tensor(s), ${outputShapes.size} output tensor(s)")

    onLog("Compiling model for ${accelerator.name}")
    val model = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(accelerator.litertAccelerator))
    onLog("Compiled model")

    val tensorDescriptions =
      inputShapes.mapIndexed { index, shape -> "input $index: ${shape.dataType}, ${shape.dimensions.toList()}" } +
        outputShapes.mapIndexed { index, shape -> "output $index: ${shape.dataType}, ${shape.dimensions.toList()}" }

    return PreparedModel(model, inputShapes, outputShapes, tensorDescriptions) {
      onLog("Releasing model resources")
      model.close()
      modelFile.delete()
      onLog("Released model resources")
    }
  }

  /** Loads the model with the TFLite Interpreter solely to read tensor shapes/types, then closes it. */
  private fun inspectTensorShapes(modelFile: File): Pair<List<TensorShape>, List<TensorShape>> {
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

  /** Forces GPU readback/sync every iteration, mirroring the working image_segmentation sample. */
  private fun readAndDiscardOutput(buffer: TensorBuffer, shape: TensorShape) {
    when (shape.dataType) {
      DataType.FLOAT32 -> buffer.readFloat()
      DataType.INT32 -> buffer.readInt()
      DataType.UINT8, DataType.INT8 -> buffer.readInt8()
      DataType.BOOL -> buffer.readBoolean()
      DataType.INT64 -> buffer.readLong()
      else -> error("Unsupported output tensor type: ${shape.dataType}")
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

enum class RunMode {
  SYNCHRONOUS,
  ASYNCHRONOUS,
}

private data class TensorShape(val dimensions: IntArray, val dataType: DataType, val elementCount: Int)

private class PreparedModel(
  val model: CompiledModel,
  val inputShapes: List<TensorShape>,
  val outputShapes: List<TensorShape>,
  val tensorDescriptions: List<String>,
  val cleanup: suspend () -> Unit,
)

data class ModelRunResult(
  val displayName: String,
  val inferenceTimeMillis: Long,
  val tensorDescriptions: List<String>,
)

data class ThroughputResult(val inferencesPerSecond: Double, val tensorDescriptions: List<String>)
