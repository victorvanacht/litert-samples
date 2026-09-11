package com.google.ai.edge.examples.victortest

import android.content.Context
import android.net.Uri
import android.os.Build
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
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
  private val modelRunner: ModelRunner,
  private val logFileWriter: LogFileWriter,
) : ViewModel() {
  companion object {
    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            val appContext = context.applicationContext
            return MainViewModel(ModelRunner(appContext), LogFileWriter(appContext)) as T
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
        tensorDescriptions = emptyList(),
      )
    }
  }

  fun selectModel(id: String) {
    _uiState.update {
      it.copy(selectedModelId = id, inferenceTime = null, tensorDescriptions = emptyList(), logLines = emptyList())
    }
    appendLog("Selected model: ${_uiState.value.models.firstOrNull { it.id == id }?.displayName}")
  }

  fun selectAccelerator(accelerator: AcceleratorChoice) {
    _uiState.update { it.copy(accelerator = accelerator, inferenceTime = null, logLines = emptyList()) }
    appendLog("Selected accelerator: ${accelerator.name}")
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
    runJob = viewModelScope.launch {
      _uiState.update { it.copy(isRunning = true, errorMessage = null) }
      appendLog("Starting continuous run")
      try {
        modelRunner.runContinuously(
          option.uri,
          option.displayName,
          _uiState.value.accelerator,
          onLog = ::appendLog,
        ) { result ->
          _uiState.update {
            it.copy(
              inferenceTime = result.inferenceTimeMillis,
              tensorDescriptions = result.tensorDescriptions,
            )
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
