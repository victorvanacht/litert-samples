/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.modelrunner.common

import android.app.Activity
import android.app.Instrumentation
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.util.Log
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking

abstract class ShellBenchmarkInstrumentation : Instrumentation() {
  abstract fun createModelRunner(context: Context): InferenceRunner

  override fun onCreate(arguments: Bundle) {
    super.onCreate(arguments)
    start()
    val results = Bundle()
    val resultCode = try {
      val config = ShellBenchmarkConfig.from(arguments)
      val benchmark = runBlocking {
        runShellBenchmark(createModelRunner(targetContext.applicationContext), config)
      }
      results.putString(REPORT_KEY_STREAMRESULT, formatSuccessResult(config, benchmark))
      Activity.RESULT_OK
    } catch (error: Exception) {
      Log.e(TAG, "Shell benchmark failed", error)
      results.putString(REPORT_KEY_STREAMRESULT, formatErrorResult(error))
      Activity.RESULT_CANCELED
    }
    finish(resultCode, results)
  }

  private companion object {
    const val TAG = "ShellBenchmark"

    fun formatSuccessResult(config: ShellBenchmarkConfig, benchmark: ShellBenchmarkResult): String {
      return buildString {
        appendResult("model", config.modelUri.toString())
        appendResult("model_display_name", config.modelDisplayName)
        appendResult("input", config.inputFileUri?.toString().orEmpty())
        appendResult("output", config.outputFileUri?.toString().orEmpty())
        appendResult("runs", config.runs.toString())
        appendResult("warmup_runs", config.warmupRuns.toString())
        appendLine()

        appendResult("run_mode", config.runMode.name)
        appendResult("accelerator", config.accelerator.name)
        appendResult("gpu_precision", config.gpuPrecision.name)
        appendResult("gpu_backend", config.gpuBackend.name)
        appendResult("gpu_priority", config.gpuPriority.name)
        appendResult("gpu_buffer_storage_type", config.gpuBufferStorageType.name)
        appendResult("gpu_prefer_texture_weights", config.gpuPreferTextureWeights.toString())
        appendResult("gpu_constant_tensor_sharing", config.gpuConstantTensorSharing.toString())
        appendResult("gpu_infinite_float_capping", config.gpuInfiniteFloatCapping.toString())
        appendLine()

        appendResult("status", "ok")
        appendResult("tensor_descriptions", benchmark.tensorDescriptions.joinToString(separator = "; "))
        appendResult("total_inference_time_ms", benchmark.totalInferenceTimeMillis.toString())
        appendResult("average_inference_time_ms", benchmark.averageInferenceTimeMillis.toString())
        appendResult("inference_times_ms", benchmark.inferenceTimesMillis.joinToString(prefix = "[", postfix = "]"))
      }
    }

    fun formatErrorResult(error: Exception): String {
      return buildString {
        appendResult("status", "error")
        appendResult("error", error.message ?: error.javaClass.simpleName)
      }
    }

    fun StringBuilder.appendResult(key: String, value: String) {
      append("INSTRUMENTATION_RESULT: ")
      append(key)
      append('=')
      appendLine(value)
    }
  }
}

data class ShellBenchmarkConfig(
  val modelUri: Uri,
  val modelDisplayName: String,
  val inputFileUri: Uri?,
  val outputFileUri: Uri?,
  val accelerator: AcceleratorChoice,
  val gpuPrecision: CompiledModel.GpuOptions.Precision,
  val gpuBackend: CompiledModel.GpuOptions.Backend,
  val gpuPriority: CompiledModel.GpuOptions.Priority,
  val gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
  val gpuPreferTextureWeights: Boolean,
  val gpuConstantTensorSharing: Boolean,
  val gpuInfiniteFloatCapping: Boolean,
  val runMode: RunMode,
  val runs: Int,
  val warmupRuns: Int,
) {
  companion object {
    fun from(arguments: Bundle): ShellBenchmarkConfig {
      val defaults = UiState()
      val modelValue = arguments.getString("model") ?: "asset://selfie_multiclass.tflite"
      val modelUri = parseShellUri(modelValue)
      val runMode = enumExtra<RunMode>(arguments, "run_mode", defaults.runMode)
      require(runMode == RunMode.SYNCHRONOUS) {
        "run_mode=ASYNCHRONOUS reports throughput in the UI and is not supported for average latency benchmarking"
      }
      val runs = arguments.getIntExtra("runs", 1)
      val warmupRuns = arguments.getIntExtra("warmup_runs", 0)
      require(runs > 0) { "runs must be greater than 0" }
      require(warmupRuns >= 0) { "warmup_runs must be at least 0" }
      return ShellBenchmarkConfig(
        modelUri = modelUri,
        modelDisplayName = arguments.getString("model_display_name") ?: displayNameFor(modelUri, modelValue),
        inputFileUri = arguments.getString("input")?.let(::parseShellUri),
        outputFileUri = arguments.getString("output")?.let(::parseShellUri),
        accelerator = enumExtra(arguments, "accelerator", defaults.accelerator),
        gpuPrecision = enumExtra(arguments, "gpu_precision", defaults.gpuPrecision),
        gpuBackend = enumExtra(arguments, "gpu_backend", defaults.gpuBackend),
        gpuPriority = enumExtra(arguments, "gpu_priority", defaults.gpuPriority),
        gpuBufferStorageType = enumExtra(arguments, "gpu_buffer_storage_type", defaults.gpuBufferStorageType),
        gpuPreferTextureWeights = arguments.getBooleanExtra("gpu_prefer_texture_weights", defaults.gpuPreferTextureWeights),
        gpuConstantTensorSharing = arguments.getBooleanExtra("gpu_constant_tensor_sharing", defaults.gpuConstantTensorSharing),
        gpuInfiniteFloatCapping = arguments.getBooleanExtra("gpu_infinite_float_capping", defaults.gpuInfiniteFloatCapping),
        runMode = runMode,
        runs = runs,
        warmupRuns = warmupRuns,
      )
    }

    private inline fun <reified T : Enum<T>> enumExtra(arguments: Bundle, name: String, default: T): T {
      val value = arguments.getString(name) ?: return default
      return enumValues<T>().firstOrNull { it.name.equals(value, ignoreCase = true) }
        ?: error("Invalid $name=$value. Expected one of: ${enumValues<T>().joinToString { it.name }}")
    }

    private fun Bundle.getIntExtra(name: String, default: Int): Int {
      return getString(name)?.toIntOrNull() ?: getInt(name, default)
    }

    private fun Bundle.getBooleanExtra(name: String, default: Boolean): Boolean {
      val stringValue = getString(name)?.lowercase(Locale.US)
      return when (stringValue) {
        null -> getBoolean(name, default)
        "1", "true", "yes", "y", "on" -> true
        "0", "false", "no", "n", "off" -> false
        else -> error("Invalid $name=$stringValue. Expected true or false")
      }
    }

    private fun parseShellUri(value: String): Uri {
      return if (value.contains("://")) Uri.parse(value) else Uri.fromFile(File(value))
    }

    private fun displayNameFor(uri: Uri, rawValue: String): String {
      return when (uri.scheme) {
        "asset" -> uri.schemeSpecificPart.removePrefix("//")
        "file" -> File(requireNotNull(uri.path)).name
        else -> uri.lastPathSegment
      } ?: rawValue.substringAfterLast('/')
    }
  }
}

data class ShellBenchmarkResult(
  val inferenceTimesMillis: List<Long>,
  val tensorDescriptions: List<String>,
) {
  val totalInferenceTimeMillis: Long = inferenceTimesMillis.sum()
  val averageInferenceTimeMillis: Double = totalInferenceTimeMillis.toDouble() / inferenceTimesMillis.size
}

suspend fun runShellBenchmark(
  modelRunner: InferenceRunner,
  config: ShellBenchmarkConfig,
): ShellBenchmarkResult {
  val inferenceTimes = mutableListOf<Long>()
  var completedRuns = 0
  var tensorDescriptions = emptyList<String>()
  try {
    modelRunner.runSynchronous(
      config.modelUri,
      config.modelDisplayName,
      config.accelerator,
      config.gpuPrecision,
      config.gpuBackend,
      config.gpuPriority,
      config.gpuBufferStorageType,
      config.gpuPreferTextureWeights,
      config.gpuConstantTensorSharing,
      config.gpuInfiniteFloatCapping,
      config.inputFileUri,
      config.outputFileUri,
      onLog = { Log.i("ShellBenchmark", it) },
    ) { result ->
      completedRuns++
      tensorDescriptions = result.tensorDescriptions
      if (completedRuns > config.warmupRuns) inferenceTimes += result.inferenceTimeMillis
      if (completedRuns >= config.warmupRuns + config.runs) throw ShellBenchmarkComplete()
    }
  } catch (complete: ShellBenchmarkComplete) {
    // Expected termination after the requested finite run count.
  }
  check(inferenceTimes.size == config.runs) {
    "Benchmark stopped after ${inferenceTimes.size} measured runs; expected ${config.runs}"
  }
  return ShellBenchmarkResult(inferenceTimes, tensorDescriptions)
}

private class ShellBenchmarkComplete : CancellationException("Shell benchmark complete")