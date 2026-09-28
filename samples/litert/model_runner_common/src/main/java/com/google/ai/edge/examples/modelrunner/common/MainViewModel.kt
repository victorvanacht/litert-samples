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

import android.content.Context
import android.net.Uri
import android.os.Build
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(
  private val modelRunner: InferenceRunner,
  private val logFileWriter: LogFileWriter,
) : ViewModel() {
  companion object {
    private const val ASYNC_CONCURRENCY = 5

    /** [createModelRunner] lets each app supply its own [InferenceRunner] implementation. */
    fun getFactory(context: Context, createModelRunner: (Context) -> InferenceRunner) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            val appContext = context.applicationContext
            return MainViewModel(createModelRunner(appContext), LogFileWriter(appContext)) as T
          }
          throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
        }
      }
  }

  private val _uiState = MutableStateFlow(UiState())
  private var runJob: Job? = null
  val uiState: StateFlow<UiState> =
    _uiState.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

  init {
    // Read the previous session's tail, then clear and seed the file, all on the IO dispatcher
    // so ordering is guaranteed and disk access never touches the main thread.
    viewModelScope.launch(Dispatchers.IO) {
      val previousLastLine = logFileWriter.readLastLine()
      logFileWriter.clear()

      val startupLines = buildList {
        add("Device: ${Build.MANUFACTURER} ${Build.MODEL} (${Build.HARDWARE}), Android ${Build.VERSION.RELEASE}")
        if (previousLastLine != null && previousLastLine != "Stopped" && !previousLastLine.startsWith("Error:")) {
          add("Previous session did not exit cleanly, last line was: \"$previousLastLine\" (possible crash - check adb logcat -b crash)")
        }
        add("Log file: ${logFileWriter.file.absolutePath}")
      }
      startupLines.forEach { logFileWriter.appendLine(it) }
      _uiState.update { it.copy(logLines = startupLines) }
    }
    addModel(Uri.parse("asset://selfie_multiclass.tflite"), "selfie_multiclass.tflite")
  }

  fun addModel(uri: Uri, displayName: String) {
    val option = ModelOption(uri.toString(), displayName, uri)
    _uiState.update { state ->
      state.copy(
        models = (state.models + option).distinctBy { it.id },
        selectedModelId = option.id,
        inferenceTime = null,
        recentInferenceTimes = emptyList(),
        inferencesPerSecond = null,
        tensorDescriptions = emptyList(),
      )
    }
  }

  fun selectModel(id: String) {
    _uiState.update {
      it.copy(
        selectedModelId = id,
        inferenceTime = null,
        recentInferenceTimes = emptyList(),
        inferencesPerSecond = null,
        tensorDescriptions = emptyList(),
        logLines = emptyList(),
      )
    }
    appendLog("Selected model: ${_uiState.value.models.firstOrNull { it.id == id }?.displayName}")
  }

  fun setInputFile(uri: Uri?, displayName: String?) {
    _uiState.update { it.copy(inputFileUri = uri, inputFileName = displayName) }
    appendLog(if (uri != null) "Selected input file: $displayName" else "Cleared input file, using random inputs")
  }

  fun setOutputFile(uri: Uri?, displayName: String?) {
    _uiState.update { it.copy(outputFileUri = uri, outputFileName = displayName) }
    appendLog(if (uri != null) "Selected output file: $displayName" else "Cleared output file, discarding outputs")
  }

  fun setCpuOutputFile(uri: Uri?, displayName: String?) {
    _uiState.update { it.copy(cpuOutputFileUri = uri, cpuOutputFileName = displayName) }
    appendLog(if (uri != null) "Selected CPU output file: $displayName" else "Cleared CPU output file")
  }

  fun setGpuOutputFile(uri: Uri?, displayName: String?) {
    _uiState.update { it.copy(gpuOutputFileUri = uri, gpuOutputFileName = displayName) }
    appendLog(if (uri != null) "Selected GPU output file: $displayName" else "Cleared GPU output file")
  }

  fun selectAccelerator(accelerator: AcceleratorChoice) {
    _uiState.update { it.copy(accelerator = accelerator, inferenceTime = null, inferencesPerSecond = null, logLines = emptyList()) }
    appendLog("Selected accelerator: ${accelerator.displayName}")
  }

  fun setCpuThreadCount(threadCount: Int) {
    require(threadCount > 0)
    _uiState.update { it.copy(cpuThreadCount = threadCount, inferenceTime = null, inferencesPerSecond = null) }
    appendLog("XNNPACK CPU threads: $threadCount")
  }

  fun selectCpuKernelMode(kernelMode: CpuKernelMode) {
    _uiState.update { it.copy(cpuKernelMode = kernelMode, inferenceTime = null, inferencesPerSecond = null) }
    appendLog("CPU kernel mode: ${kernelMode.displayName}")
  }

  fun setXnnpackFlag(flag: XnnpackFlag, enabled: Boolean) {
    _uiState.update { state ->
      var flags = if (enabled) state.xnnpackFlags or flag.bit else state.xnnpackFlags and flag.bit.inv()
      if (enabled && flag == XnnpackFlag.ENABLE_SUBGRAPH_RESHAPING) {
        flags = flags and XnnpackFlag.DISABLE_SUBGRAPH_RESHAPING.bit.inv()
      } else if (enabled && flag == XnnpackFlag.DISABLE_SUBGRAPH_RESHAPING) {
        flags = flags and XnnpackFlag.ENABLE_SUBGRAPH_RESHAPING.bit.inv()
      }
      state.copy(xnnpackFlags = flags, inferenceTime = null, inferencesPerSecond = null)
    }
    appendLog("XNNPACK flag ${flag.displayName}: $enabled")
  }

  fun selectGpuPrecision(precision: CompiledModel.GpuOptions.Precision) {
    _uiState.update { it.copy(gpuPrecision = precision, inferenceTime = null, inferencesPerSecond = null, logLines = emptyList()) }
    appendLog("Selected GPU precision: ${precision.name}")
  }

  fun selectGpuBackend(backend: CompiledModel.GpuOptions.Backend) {
    _uiState.update { it.copy(gpuBackend = backend, inferenceTime = null, inferencesPerSecond = null, logLines = emptyList()) }
    appendLog("Selected GPU backend: ${backend.name}")
  }

  fun selectGpuPriority(priority: CompiledModel.GpuOptions.Priority) {
    _uiState.update { it.copy(gpuPriority = priority, inferenceTime = null, inferencesPerSecond = null, logLines = emptyList()) }
    appendLog("Selected GPU priority: ${priority.name}")
  }

  fun selectGpuBufferStorageType(storageType: CompiledModel.GpuOptions.BufferStorageType) {
    _uiState.update { it.copy(gpuBufferStorageType = storageType, inferenceTime = null, inferencesPerSecond = null, logLines = emptyList()) }
    appendLog("Selected GPU buffer storage: ${storageType.name}")
  }

  fun setGpuPreferTextureWeights(enabled: Boolean) {
    _uiState.update { it.copy(gpuPreferTextureWeights = enabled, inferenceTime = null, inferencesPerSecond = null) }
    appendLog("Prefer texture weights: $enabled")
  }

  fun setGpuConstantTensorSharing(enabled: Boolean) {
    _uiState.update { it.copy(gpuConstantTensorSharing = enabled, inferenceTime = null, inferencesPerSecond = null) }
    appendLog("Constant tensor sharing: $enabled")
  }

  fun setGpuInfiniteFloatCapping(enabled: Boolean) {
    _uiState.update { it.copy(gpuInfiniteFloatCapping = enabled, inferenceTime = null, inferencesPerSecond = null) }
    appendLog("Infinite float capping: $enabled")
  }

  fun selectRunMode(mode: RunMode) {
    _uiState.update { it.copy(runMode = mode, inferenceTime = null, recentInferenceTimes = emptyList(), inferencesPerSecond = null, logLines = emptyList()) }
    appendLog("Selected run mode: ${mode.name}")
  }

  fun toggleModelRun() {
    if (_uiState.value.isRunning) {
      appendLog("Stop requested")
      runJob?.cancel()
      return
    }
    if (runJob?.isActive == true) return

    val state = _uiState.value
    val option = state.models.firstOrNull { it.id == state.selectedModelId } ?: return
    val isPaired = state.accelerator == AcceleratorChoice.CPU_GPU
    if (isPaired && ((state.cpuOutputFileUri == null) != (state.gpuOutputFileUri == null) ||
          (state.cpuOutputFileUri != null && state.cpuOutputFileUri == state.gpuOutputFileUri))) {
      _uiState.update { it.copy(errorMessage = "Choose distinct CPU and GPU output files, or neither") }
      return
    }

    lateinit var job: Job
    job = viewModelScope.launch(start = CoroutineStart.LAZY) {
      val mode = if (isPaired) "paired CPU+GPU throughput" else "${state.runMode} run"
      _uiState.update {
        it.copy(
          isRunning = true,
          errorMessage = null,
          inferenceTime = null,
          recentInferenceTimes = emptyList(),
          inferencesPerSecond = null,
          cpuCompletedRuns = 0,
          gpuCompletedRuns = 0,
        )
      }
      appendLog("Starting $mode")
      try {
        if (isPaired) {
          PairedInferenceRunner(modelRunner).run(
            config = state.toPairedInferenceConfig(option),
            onLog = ::appendLog,
          ) { progress ->
            _uiState.update {
              it.copy(
                inferencesPerSecond = progress.inferencesPerSecond,
                cpuCompletedRuns = progress.cpuCompletedRuns,
                gpuCompletedRuns = progress.gpuCompletedRuns,
                tensorDescriptions = progress.tensorDescriptions,
              )
            }
          }
        } else when (state.runMode) {
          RunMode.SYNCHRONOUS ->
            modelRunner.runSynchronous(
              option.uri,
              option.displayName,
              state.accelerator,
              state.cpuThreadCount,
              state.cpuKernelMode,
              state.xnnpackFlags,
              state.gpuPrecision,
              state.gpuBackend,
              state.gpuPriority,
              state.gpuBufferStorageType,
              state.gpuPreferTextureWeights,
              state.gpuConstantTensorSharing,
              state.gpuInfiniteFloatCapping,
              state.inputFileUri,
              state.outputFileUri,
              onLog = ::appendLog,
            ) { result ->
              _uiState.update {
                it.copy(
                  inferenceTime = result.inferenceTimeMillis,
                  recentInferenceTimes = (it.recentInferenceTimes + result.inferenceTimeMillis).takeLast(10),
                  tensorDescriptions = result.tensorDescriptions,
                )
              }
            }
          RunMode.ASYNCHRONOUS ->
            modelRunner.runAsynchronous(
              option.uri,
              option.displayName,
              state.accelerator,
              state.cpuThreadCount,
              state.cpuKernelMode,
              state.xnnpackFlags,
              state.gpuPrecision,
              state.gpuBackend,
              state.gpuPriority,
              state.gpuBufferStorageType,
              state.gpuPreferTextureWeights,
              state.gpuConstantTensorSharing,
              state.gpuInfiniteFloatCapping,
              state.inputFileUri,
              state.outputFileUri,
              concurrency = ASYNC_CONCURRENCY,
              onLog = ::appendLog,
            ) { result ->
              _uiState.update {
                it.copy(
                  inferencesPerSecond = result.inferencesPerSecond,
                  tensorDescriptions = result.tensorDescriptions,
                )
              }
            }
        }
      } catch (error: CancellationException) {
        throw error
      } catch (error: Exception) {
        appendLog("Error: ${error.message ?: error.javaClass.simpleName}")
        _uiState.update {
          it.copy(errorMessage = error.message ?: error.javaClass.simpleName)
        }
      } finally {
        _uiState.update { it.copy(isRunning = false) }
        appendLog("Stopped")
        if (runJob === job) runJob = null
      }
    }
    runJob = job
    job.start()
  }

  private fun UiState.toPairedInferenceConfig(option: ModelOption) = PairedInferenceConfig(
    uri = option.uri,
    displayName = option.displayName,
    inputFileUri = inputFileUri,
    cpuOutputFileUri = cpuOutputFileUri,
    gpuOutputFileUri = gpuOutputFileUri,
    cpuThreadCount = cpuThreadCount,
    cpuKernelMode = cpuKernelMode,
    xnnpackFlags = xnnpackFlags,
    gpuPrecision = gpuPrecision,
    gpuBackend = gpuBackend,
    gpuPriority = gpuPriority,
    gpuBufferStorageType = gpuBufferStorageType,
    gpuPreferTextureWeights = gpuPreferTextureWeights,
    gpuConstantTensorSharing = gpuConstantTensorSharing,
    gpuInfiniteFloatCapping = gpuInfiniteFloatCapping,
  )

  suspend fun runShellBenchmarkFromUi(config: ShellBenchmarkConfig): ShellBenchmarkResult {
    require(config.accelerator == AcceleratorChoice.CPU_GPU || config.runMode == RunMode.SYNCHRONOUS) {
      "run_mode=ASYNCHRONOUS reports throughput in the UI and is not supported for average latency benchmarking"
    }

    addModel(config.modelUri, config.modelDisplayName)
    setInputFile(config.inputFileUri, config.inputFileUri?.lastPathSegment)
    setOutputFile(config.outputFileUri, config.outputFileUri?.lastPathSegment)
    setCpuOutputFile(config.cpuOutputFileUri, config.cpuOutputFileUri?.lastPathSegment)
    setGpuOutputFile(config.gpuOutputFileUri, config.gpuOutputFileUri?.lastPathSegment)
    selectRunMode(config.runMode)
    selectAccelerator(config.accelerator)
    _uiState.update {
      it.copy(
        cpuThreadCount = config.cpuThreadCount,
        cpuKernelMode = config.cpuKernelMode,
        xnnpackFlags = config.xnnpackFlags,
      )
    }
    selectGpuPrecision(config.gpuPrecision)
    selectGpuBackend(config.gpuBackend)
    selectGpuPriority(config.gpuPriority)
    selectGpuBufferStorageType(config.gpuBufferStorageType)
    setGpuPreferTextureWeights(config.gpuPreferTextureWeights)
    setGpuConstantTensorSharing(config.gpuConstantTensorSharing)
    setGpuInfiniteFloatCapping(config.gpuInfiniteFloatCapping)

    val option = _uiState.value.models.firstOrNull { it.id == config.modelUri.toString() }
      ?: error("Selected model is unavailable: ${config.modelUri}")
    if (config.accelerator == AcceleratorChoice.CPU_GPU) {
      return runPairedShellBenchmark(option, config)
    }

    val inferenceTimes = mutableListOf<Long>()
    var completedRuns = 0
    var tensorDescriptions = emptyList<String>()
    _uiState.update { it.copy(isRunning = true, errorMessage = null) }
    appendLog("Starting UI-driven shell benchmark")
    try {
      modelRunner.runSynchronous(
        option.uri,
        option.displayName,
        _uiState.value.accelerator,
        _uiState.value.cpuThreadCount,
        _uiState.value.cpuKernelMode,
        _uiState.value.xnnpackFlags,
        _uiState.value.gpuPrecision,
        _uiState.value.gpuBackend,
        _uiState.value.gpuPriority,
        _uiState.value.gpuBufferStorageType,
        _uiState.value.gpuPreferTextureWeights,
        _uiState.value.gpuConstantTensorSharing,
        _uiState.value.gpuInfiniteFloatCapping,
        _uiState.value.inputFileUri,
        _uiState.value.outputFileUri,
        onLog = ::appendLog,
      ) { result ->
        completedRuns++
        tensorDescriptions = result.tensorDescriptions
        _uiState.update {
          it.copy(
            inferenceTime = result.inferenceTimeMillis,
            tensorDescriptions = result.tensorDescriptions,
          )
        }
        if (completedRuns > config.warmupRuns) inferenceTimes += result.inferenceTimeMillis
        if (completedRuns >= config.warmupRuns + config.runs) throw ShellBenchmarkComplete()
      }
    } catch (complete: ShellBenchmarkComplete) {
      // Expected termination after the requested finite run count.
    } catch (error: CancellationException) {
      throw error
    } catch (error: Exception) {
      appendLog("Error: ${error.message ?: error.javaClass.simpleName}")
      _uiState.update { it.copy(errorMessage = error.message ?: error.javaClass.simpleName) }
      throw error
    } finally {
      _uiState.update { it.copy(isRunning = false) }
      appendLog("Stopped")
    }
    check(inferenceTimes.size == config.runs) {
      "Benchmark stopped after ${inferenceTimes.size} measured runs; expected ${config.runs}"
    }
    return ShellBenchmarkResult(inferenceTimes, tensorDescriptions)
  }

  private suspend fun runPairedShellBenchmark(
    option: ModelOption,
    config: ShellBenchmarkConfig,
  ): ShellBenchmarkResult {
    _uiState.update {
      it.copy(
        isRunning = true,
        errorMessage = null,
        inferenceTime = null,
        inferencesPerSecond = null,
        cpuCompletedRuns = 0,
        gpuCompletedRuns = 0,
      )
    }
    appendLog("Starting UI-driven paired CPU+GPU shell benchmark")
    try {
      val result = PairedInferenceRunner(modelRunner).run(
        config = _uiState.value.toPairedInferenceConfig(option),
        warmupRuns = config.warmupRuns,
        maxMeasuredRuns = config.runs,
        onLog = ::appendLog,
      ) { progress ->
        _uiState.update {
          it.copy(
            inferencesPerSecond = progress.inferencesPerSecond,
            cpuCompletedRuns = progress.cpuCompletedRuns,
            gpuCompletedRuns = progress.gpuCompletedRuns,
            tensorDescriptions = progress.tensorDescriptions,
          )
        }
      }
      check(result.completedRuns == config.runs) {
        "Benchmark completed ${result.completedRuns} measured runs; expected ${config.runs}"
      }
      return ShellBenchmarkResult(emptyList(), result.tensorDescriptions, result)
    } catch (error: CancellationException) {
      throw error
    } catch (error: Exception) {
      appendLog("Error: ${error.message ?: error.javaClass.simpleName}")
      _uiState.update { it.copy(errorMessage = error.message ?: error.javaClass.simpleName) }
      throw error
    } finally {
      _uiState.update { it.copy(isRunning = false) }
      appendLog("Stopped")
    }
  }

  fun errorMessageShown() {
    _uiState.update { it.copy(errorMessage = null) }
  }

  private fun appendLog(line: String) {
    _uiState.update { it.copy(logLines = (it.logLines + line).takeLast(100)) }
    viewModelScope.launch(Dispatchers.IO) { logFileWriter.appendLine(line) }
  }

  private class ShellBenchmarkComplete : CancellationException("Shell benchmark complete")
}
