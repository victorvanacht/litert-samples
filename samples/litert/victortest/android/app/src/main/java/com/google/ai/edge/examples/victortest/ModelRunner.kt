package com.google.ai.edge.examples.victortest

import android.content.Context
import android.net.Uri
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
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
    val prepared = prepareModel(uri, displayName, accelerator, gpuPrecision, gpuBackend, gpuPriority, gpuBufferStorageType, gpuPreferTextureWeights, gpuConstantTensorSharing, gpuInfiniteFloatCapping, onLog)
    try {
      val inputBuffers = prepared.model.createInputBuffers()
      val outputBuffers = prepared.model.createOutputBuffers()
      onLog("Allocated ${inputBuffers.size} input buffer(s), ${outputBuffers.size} output buffer(s)")
      writeInputs(inputBuffers, prepared.inputShapes, inputFileUri)

      var iteration = 0
      onLog("Starting synchronous inference loop")
      while (currentCoroutineContext().isActive) {
        iteration++
        val captureOutput = outputFileUri != null
        val startNanos = System.nanoTime()
        prepared.model.run(inputBuffers, outputBuffers)
        // Read back every output, like the working image_segmentation sample does. Skipping this
        // let the GPU backend enqueue work unbounded with no synchronization.
        val outputBytes = outputBuffers.mapIndexed { index, buffer -> readOutput(buffer, prepared.outputShapes[index], captureOutput) }
        val elapsedMillis = (System.nanoTime() - startNanos) / 1_000_000
        if (outputFileUri != null) writeOutputsToUri(outputBytes.filterNotNull(), outputFileUri)
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
    val prepared = prepareModel(uri, displayName, accelerator, gpuPrecision, gpuBackend, gpuPriority, gpuBufferStorageType, gpuPreferTextureWeights, gpuConstantTensorSharing, gpuInfiniteFloatCapping, onLog)
    try {
      val captureOutput = outputFileUri != null
      val slots =
        List(concurrency) {
          val inputBuffers = prepared.model.createInputBuffers()
          val outputBuffers = prepared.model.createOutputBuffers()
          writeInputs(inputBuffers, prepared.inputShapes, inputFileUri)
          inputBuffers to outputBuffers
        }
      onLog("Allocated $concurrency concurrent buffer set(s)")

      val completedCount = AtomicLong(0)
      val lastOutputBytes = AtomicReference<List<ByteArray>?>(null)
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
            // Write outside the per-inference loop so disk I/O never counts against throughput.
            if (outputFileUri != null) lastOutputBytes.get()?.let { writeOutputsToUri(it, outputFileUri) }
          }
        }
        slots.forEach { (inputBuffers, outputBuffers) ->
          launch {
            while (isActive) {
              prepared.model.run(inputBuffers, outputBuffers)
              val outputBytes = outputBuffers.mapIndexed { index, buffer ->
                readOutput(buffer, prepared.outputShapes[index], captureOutput)
              }
              completedCount.incrementAndGet()
              if (captureOutput) lastOutputBytes.set(outputBytes.filterNotNull())
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
    gpuPrecision: CompiledModel.GpuOptions.Precision,
    gpuBackend: CompiledModel.GpuOptions.Backend,
    gpuPriority: CompiledModel.GpuOptions.Priority,
    gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
    gpuPreferTextureWeights: Boolean,
    gpuConstantTensorSharing: Boolean,
    gpuInfiniteFloatCapping: Boolean,
    onLog: suspend (String) -> Unit,
  ): PreparedModel {
    onLog("Loading $displayName")
    onLog("Accelerator: ${accelerator.displayName}")
    val modelFile = copyToCache(uri, displayName)
    onLog("Copied model to cache: ${modelFile.name} (${modelFile.length()} bytes)")

    // Use the TFLite Interpreter API purely to introspect generic tensor shapes/types;
    // it is bundled transitively with litert-api and never used for actual inference.
    onLog("Inspecting tensors with TFLite Interpreter")
    val (inputShapes, outputShapes) = inspectTensorShapes(modelFile)
    onLog("Found ${inputShapes.size} input tensor(s), ${outputShapes.size} output tensor(s)")

    onLog("Compiling model for ${accelerator.displayName}")
    val model = CompiledModel.create(modelFile.absolutePath, accelerator.toCompiledModelOptions(gpuPrecision, gpuBackend, gpuPriority, gpuBufferStorageType, gpuPreferTextureWeights, gpuConstantTensorSharing, gpuInfiniteFloatCapping))
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

  /**
   * Forces GPU readback/sync every iteration, mirroring the working image_segmentation sample.
   * Skipping the read let the GPU backend enqueue work unbounded with no synchronization. When
   * [captureBytes] is true (an output file was selected), also returns the raw bytes so they can
   * be written to disk outside the timed/throughput region.
   */
  private fun readOutput(buffer: TensorBuffer, shape: TensorShape, captureBytes: Boolean): ByteArray? {
    return when (shape.dataType) {
      DataType.FLOAT32 -> buffer.readFloat().let { if (captureBytes) floatsToBytes(it) else null }
      DataType.INT32 -> buffer.readInt().let { if (captureBytes) intsToBytes(it) else null }
      DataType.UINT8, DataType.INT8 -> buffer.readInt8().let { if (captureBytes) it else null }
      DataType.BOOL -> buffer.readBoolean().let { if (captureBytes) booleansToBytes(it) else null }
      DataType.INT64 -> buffer.readLong().let { if (captureBytes) longsToBytes(it) else null }
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

  /** Fills each buffer from [inputFileUri]'s raw bytes (zero-padded if short), or with random data if null. */
  private fun writeInputs(buffers: List<TensorBuffer>, shapes: List<TensorShape>, inputFileUri: Uri?) {
    if (inputFileUri == null) {
      buffers.forEachIndexed { index, buffer -> writeRandomInput(buffer, shapes[index]) }
      return
    }
    val fileBytes = readAllBytes(inputFileUri)
    var offset = 0
    buffers.forEachIndexed { index, buffer ->
      offset = writeInputFromFile(buffer, shapes[index], fileBytes, offset)
    }
  }

  private fun writeInputFromFile(buffer: TensorBuffer, shape: TensorShape, fileBytes: ByteArray, offset: Int): Int {
    val elementSize = elementByteSize(shape.dataType)
    val byteSize = shape.elementCount * elementSize
    val chunk = ByteArray(byteSize)
    val available = (fileBytes.size - offset).coerceAtLeast(0)
    val toCopy = minOf(available, byteSize)
    if (toCopy > 0) System.arraycopy(fileBytes, offset, chunk, 0, toCopy)
    val bytes = ByteBuffer.wrap(chunk).order(ByteOrder.nativeOrder())
    when (shape.dataType) {
      DataType.FLOAT32 -> buffer.writeFloat(FloatArray(shape.elementCount).also { bytes.asFloatBuffer().get(it) })
      DataType.INT32 -> buffer.writeInt(IntArray(shape.elementCount).also { bytes.asIntBuffer().get(it) })
      DataType.UINT8, DataType.INT8 -> buffer.writeInt8(chunk)
      DataType.BOOL -> buffer.writeBoolean(BooleanArray(shape.elementCount) { chunk[it] != 0.toByte() })
      DataType.INT64 -> buffer.writeLong(LongArray(shape.elementCount).also { bytes.asLongBuffer().get(it) })
      else -> error("Unsupported input tensor type: ${shape.dataType}")
    }
    return offset + byteSize
  }

  private fun elementByteSize(dataType: DataType): Int = when (dataType) {
    DataType.FLOAT32, DataType.INT32 -> 4
    DataType.UINT8, DataType.INT8, DataType.BOOL -> 1
    DataType.INT64 -> 8
    else -> error("Unsupported tensor type: $dataType")
  }

  private fun floatsToBytes(values: FloatArray) =
    ByteBuffer.allocate(values.size * 4).order(ByteOrder.nativeOrder()).apply { asFloatBuffer().put(values) }.array()

  private fun intsToBytes(values: IntArray) =
    ByteBuffer.allocate(values.size * 4).order(ByteOrder.nativeOrder()).apply { asIntBuffer().put(values) }.array()

  private fun longsToBytes(values: LongArray) =
    ByteBuffer.allocate(values.size * 8).order(ByteOrder.nativeOrder()).apply { asLongBuffer().put(values) }.array()

  private fun booleansToBytes(values: BooleanArray) = ByteArray(values.size) { if (values[it]) 1 else 0 }

  /** Overwrites [uri] with the concatenated raw bytes of every output tensor. */
  private fun writeOutputsToUri(outputs: List<ByteArray>, uri: Uri) {
    context.contentResolver.openOutputStream(uri, "wt")?.use { output ->
      outputs.forEach { output.write(it) }
    }
  }

  private fun readAllBytes(uri: Uri): ByteArray {
    val assetName = uri.schemeSpecificPart.removePrefix("//")
    val input = if (uri.scheme == "asset") context.assets.open(assetName)
    else context.contentResolver.openInputStream(uri)
    return requireNotNull(input) { "Unable to open input file" }.use { it.readBytes() }
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

enum class AcceleratorChoice(
  val displayName: String,
  val litertAccelerator: com.google.ai.edge.litert.Accelerator,
) {
  CPU("CPU", com.google.ai.edge.litert.Accelerator.CPU),
  GPU("GPU", com.google.ai.edge.litert.Accelerator.GPU);

  fun toCompiledModelOptions(
    gpuPrecision: CompiledModel.GpuOptions.Precision,
    gpuBackend: CompiledModel.GpuOptions.Backend,
    gpuPriority: CompiledModel.GpuOptions.Priority,
    gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
    gpuPreferTextureWeights: Boolean,
    gpuConstantTensorSharing: Boolean,
    gpuInfiniteFloatCapping: Boolean,
  ): CompiledModel.Options {
    val options = CompiledModel.Options(litertAccelerator)
    if (litertAccelerator == com.google.ai.edge.litert.Accelerator.GPU) {
      options.gpuOptions = CompiledModel.GpuOptions(
        precision = gpuPrecision,
        backend = gpuBackend,
        priority = gpuPriority,
        bufferStorageType = gpuBufferStorageType,
        preferTextureWeights = gpuPreferTextureWeights,
        constantTensorSharing = gpuConstantTensorSharing,
        infiniteFloatCapping = gpuInfiniteFloatCapping,
      )
    }
    return options
  }
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
