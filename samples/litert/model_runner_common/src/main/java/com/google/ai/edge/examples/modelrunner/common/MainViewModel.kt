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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
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

  fun selectAccelerator(accelerator: AcceleratorChoice) {
    _uiState.update { it.copy(accelerator = accelerator, inferenceTime = null, inferencesPerSecond = null, logLines = emptyList()) }
    appendLog("Selected accelerator: ${accelerator.displayName}")
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
    _uiState.update { it.copy(runMode = mode, inferenceTime = null, inferencesPerSecond = null, logLines = emptyList()) }
    appendLog("Selected run mode: ${mode.name}")
  }

  fun toggleModelRun() {
    if (_uiState.value.isRunning) {
      appendLog("Stop requested")
      runJob?.cancel()
      runJob = null
      _uiState.update { it.copy(isRunning = false) }
      return
    }

    val option = _uiState.value.models.firstOrNull { it.id == _uiState.value.selectedModelId } ?: return
    val mode = _uiState.value.runMode
    runJob = viewModelScope.launch {
      _uiState.update { it.copy(isRunning = true, errorMessage = null) }
      appendLog("Starting $mode run")
      try {
        when (mode) {
          RunMode.SYNCHRONOUS ->
            modelRunner.runSynchronous(
              option.uri,
              option.displayName,
              _uiState.value.accelerator,
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
              _uiState.update {
                it.copy(
                  inferenceTime = result.inferenceTimeMillis,
                  tensorDescriptions = result.tensorDescriptions,
                )
              }
            }
          RunMode.ASYNCHRONOUS ->
            modelRunner.runAsynchronous(
              option.uri,
              option.displayName,
              _uiState.value.accelerator,
              _uiState.value.gpuPrecision,
              _uiState.value.gpuBackend,
              _uiState.value.gpuPriority,
              _uiState.value.gpuBufferStorageType,
              _uiState.value.gpuPreferTextureWeights,
              _uiState.value.gpuConstantTensorSharing,
              _uiState.value.gpuInfiniteFloatCapping,
              _uiState.value.inputFileUri,
              _uiState.value.outputFileUri,
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
        runJob = null
      }
    }
  }

  fun errorMessageShown() {
    _uiState.update { it.copy(errorMessage = null) }
  }

  private fun appendLog(line: String) {
    _uiState.update { it.copy(logLines = (it.logLines + line).takeLast(100)) }
    viewModelScope.launch(Dispatchers.IO) { logFileWriter.appendLine(line) }
  }
}
