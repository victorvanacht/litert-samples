package com.google.ai.edge.examples.victortest

import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.background
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.Button
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.Switch
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.examples.victortest.view.ApplicationTheme

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
    setContent {
      val uiState by viewModel.uiState.collectAsStateWithLifecycle()
      val modelPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
          val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
            ?: uri.lastPathSegment
            ?: "selected_model.tflite"
          if (name.endsWith(".tflite", ignoreCase = true)) {
            contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
            viewModel.addModel(uri, name)
          } else {
            Toast.makeText(this, "Choose a .tflite model", Toast.LENGTH_SHORT).show()
          }
        }
      }
      val inputFilePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
          val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
            ?: uri.lastPathSegment
            ?: "input.bin"
          contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
          viewModel.setInputFile(uri, name)
        }
      }
      val outputFilePicker = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        if (uri != null) {
          val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
            ?: uri.lastPathSegment
            ?: "output.bin"
          contentResolver.takePersistableUriPermission(
            uri,
            android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
          )
          viewModel.setOutputFile(uri, name)
        }
      }
      LaunchedEffect(uiState.errorMessage) {
        uiState.errorMessage?.let {
          Toast.makeText(this@MainActivity, it, Toast.LENGTH_LONG).show()
          viewModel.errorMessageShown()
        }
      }
      ApplicationTheme {
        ModelRunnerScreen(
          uiState = uiState,
          onChooseModel = { modelPicker.launch(arrayOf("application/octet-stream", "application/x-tflite", "*/*")) },
          onSelectModel = viewModel::selectModel,
          onChooseInputFile = { inputFilePicker.launch(arrayOf("*/*")) },
          onClearInputFile = { viewModel.setInputFile(null, null) },
          onChooseOutputFile = { outputFilePicker.launch("output.bin") },
          onClearOutputFile = { viewModel.setOutputFile(null, null) },
          onSelectAccelerator = viewModel::selectAccelerator,
          onSelectGpuPrecision = viewModel::selectGpuPrecision,
          onSelectGpuBackend = viewModel::selectGpuBackend,
          onSelectGpuPriority = viewModel::selectGpuPriority,
          onSelectGpuBufferStorageType = viewModel::selectGpuBufferStorageType,
          onSetGpuPreferTextureWeights = viewModel::setGpuPreferTextureWeights,
          onSetGpuConstantTensorSharing = viewModel::setGpuConstantTensorSharing,
          onSetGpuInfiniteFloatCapping = viewModel::setGpuInfiniteFloatCapping,
          onSelectRunMode = viewModel::selectRunMode,
          onRunModel = viewModel::toggleModelRun,
        )
      }
    }
  }
}

@Composable
private fun ModelRunnerScreen(
  uiState: UiState,
  onChooseModel: () -> Unit,
  onSelectModel: (String) -> Unit,
  onChooseInputFile: () -> Unit,
  onClearInputFile: () -> Unit,
  onChooseOutputFile: () -> Unit,
  onClearOutputFile: () -> Unit,
  onSelectAccelerator: (AcceleratorChoice) -> Unit,
  onSelectGpuPrecision: (CompiledModel.GpuOptions.Precision) -> Unit,
  onSelectGpuBackend: (CompiledModel.GpuOptions.Backend) -> Unit,
  onSelectGpuPriority: (CompiledModel.GpuOptions.Priority) -> Unit,
  onSelectGpuBufferStorageType: (CompiledModel.GpuOptions.BufferStorageType) -> Unit,
  onSetGpuPreferTextureWeights: (Boolean) -> Unit,
  onSetGpuConstantTensorSharing: (Boolean) -> Unit,
  onSetGpuInfiniteFloatCapping: (Boolean) -> Unit,
  onSelectRunMode: (RunMode) -> Unit,
  onRunModel: () -> Unit,
) {
  Column(modifier = Modifier.fillMaxSize()) {
    TopAppBar(title = { Text("victortest") }, backgroundColor = MaterialTheme.colors.secondary)
    Column(
      modifier = Modifier.padding(20.dp).verticalScroll(rememberScrollState()),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Text("LiteRT CompiledModel runner", style = MaterialTheme.typography.h6)
      Text("Select a .tflite file. Random input tensors are generated from its tensor metadata; outputs are discarded.")
      ModelSelector(uiState, onChooseModel, onSelectModel)
      FileSelector(
        label = "Input tensor file",
        fileName = uiState.inputFileName,
        placeholder = "Random (default)",
        chooseLabel = "Choose input file",
        onChoose = onChooseInputFile,
        onClear = onClearInputFile,
      )
      FileSelector(
        label = "Output tensor file",
        fileName = uiState.outputFileName,
        placeholder = "Discarded (default)",
        chooseLabel = "Choose output file",
        onChoose = onChooseOutputFile,
        onClear = onClearOutputFile,
      )
      RunModeSelector(uiState.runMode, onSelectRunMode)
      AcceleratorSelector(uiState.accelerator, onSelectAccelerator)
      if (uiState.accelerator != AcceleratorChoice.CPU) {
        GpuPrecisionSelector(uiState.gpuPrecision, onSelectGpuPrecision)
        GpuBackendSelector(uiState.gpuBackend, onSelectGpuBackend)
        GpuPrioritySelector(uiState.gpuPriority, onSelectGpuPriority)
        GpuBufferStorageSelector(uiState.gpuBufferStorageType, onSelectGpuBufferStorageType)
        GpuOptionSwitch("Prefer texture weights", uiState.gpuPreferTextureWeights, onSetGpuPreferTextureWeights)
        GpuOptionSwitch("Constant tensor sharing", uiState.gpuConstantTensorSharing, onSetGpuConstantTensorSharing)
        GpuOptionSwitch("Infinite float capping", uiState.gpuInfiniteFloatCapping, onSetGpuInfiniteFloatCapping)
      }
      Button(
        onClick = onRunModel,
        enabled = uiState.selectedModelId != null,
        modifier = Modifier.fillMaxWidth(),
      ) { Text(if (uiState.isRunning) "Stop running model" else "Run model") }
      LogPanel(uiState.logLines)
      Box(
        modifier = Modifier.fillMaxWidth().height(220.dp),
        contentAlignment = Alignment.Center,
      ) {
        when (uiState.runMode) {
          RunMode.SYNCHRONOUS ->
            uiState.inferenceTime?.let { Text("$it ms", fontSize = 64.sp, style = MaterialTheme.typography.h3) }
          RunMode.ASYNCHRONOUS ->
            uiState.inferencesPerSecond?.let { rate ->
              Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("%.1f inferences/s".format(rate), fontSize = 48.sp, style = MaterialTheme.typography.h4)
                Text("avg %.2f ms/inference".format(1000.0 / rate), fontSize = 24.sp)
              }
            }
        }
      }
      uiState.tensorDescriptions.forEach { Text(it) }
    }
  }
}

@Composable
private fun LogPanel(logLines: List<String>) {
  Column(
    modifier = Modifier
      .fillMaxWidth()
      .height(720.dp)
      .background(Color(0xFF101010))
      .padding(10.dp)
      .verticalScroll(rememberScrollState()),
  ) {
    if (logLines.isEmpty()) {
      Text("> Ready", color = Color(0xFF9CDCFE), fontFamily = FontFamily.Monospace, fontSize = 10.sp)
    } else {
      logLines.forEach { line ->
        Text("> $line", color = Color(0xFFB5CEA8), fontFamily = FontFamily.Monospace, fontSize = 10.sp)
      }
    }
  }
}

@Composable
private fun RunModeSelector(
  selected: RunMode,
  onSelect: (RunMode) -> Unit,
) {
  var expanded by remember { mutableStateOf(false) }
  Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
    Text("Mode")
    OutlinedButton(onClick = { expanded = true }) { Text(selected.name) }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      RunMode.entries.forEach { mode ->
        DropdownMenuItem(onClick = { onSelect(mode); expanded = false }) {
          Text(mode.name)
        }
      }
    }
  }
}

@Composable
private fun GpuPrecisionSelector(
  selected: CompiledModel.GpuOptions.Precision,
  onSelect: (CompiledModel.GpuOptions.Precision) -> Unit,
) {
  var expanded by remember { mutableStateOf(false) }
  Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
    Text("GPU precision")
    OutlinedButton(onClick = { expanded = true }) { Text(selected.name) }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      CompiledModel.GpuOptions.Precision.entries.forEach { precision ->
        DropdownMenuItem(onClick = { onSelect(precision); expanded = false }) {
          Text(precision.name)
        }
      }
    }
  }
}

@Composable
private fun GpuBackendSelector(
  selected: CompiledModel.GpuOptions.Backend,
  onSelect: (CompiledModel.GpuOptions.Backend) -> Unit,
) {
  var expanded by remember { mutableStateOf(false) }
  Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
    Text("GPU backend")
    OutlinedButton(onClick = { expanded = true }) { Text(selected.name) }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      CompiledModel.GpuOptions.Backend.entries.forEach { backend ->
        DropdownMenuItem(onClick = { onSelect(backend); expanded = false }) {
          Text(backend.name)
        }
      }
    }
  }
}

@Composable
private fun GpuPrioritySelector(
  selected: CompiledModel.GpuOptions.Priority,
  onSelect: (CompiledModel.GpuOptions.Priority) -> Unit,
) {
  var expanded by remember { mutableStateOf(false) }
  Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
    Text("GPU priority")
    OutlinedButton(onClick = { expanded = true }) { Text(selected.name) }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      CompiledModel.GpuOptions.Priority.entries.forEach { priority ->
        DropdownMenuItem(onClick = { onSelect(priority); expanded = false }) {
          Text(priority.name)
        }
      }
    }
  }
}

@Composable
private fun GpuBufferStorageSelector(
  selected: CompiledModel.GpuOptions.BufferStorageType,
  onSelect: (CompiledModel.GpuOptions.BufferStorageType) -> Unit,
) {
  var expanded by remember { mutableStateOf(false) }
  Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
    Text("GPU buffer storage")
    OutlinedButton(onClick = { expanded = true }) { Text(selected.name) }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      CompiledModel.GpuOptions.BufferStorageType.entries.forEach { storageType ->
        DropdownMenuItem(onClick = { onSelect(storageType); expanded = false }) {
          Text(storageType.name)
        }
      }
    }
  }
}

@Composable
private fun GpuOptionSwitch(label: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
  Row(
    modifier = Modifier.fillMaxWidth(),
    horizontalArrangement = Arrangement.SpaceBetween,
    verticalAlignment = Alignment.CenterVertically,
  ) {
    Text(label)
    Switch(checked = checked, onCheckedChange = onCheckedChange)
  }
}

@Composable
private fun AcceleratorSelector(
  selected: AcceleratorChoice,
  onSelect: (AcceleratorChoice) -> Unit,
) {
  var expanded by remember { mutableStateOf(false) }
  Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
    Text("Accelerator")
    OutlinedButton(onClick = { expanded = true }) { Text(selected.displayName) }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      AcceleratorChoice.entries.forEach { accelerator ->
        DropdownMenuItem(onClick = { onSelect(accelerator); expanded = false }) {
          Text(accelerator.displayName)
        }
      }
    }
  }
}

@Composable
private fun FileSelector(
  label: String,
  fileName: String?,
  placeholder: String,
  chooseLabel: String,
  onChoose: () -> Unit,
  onClear: () -> Unit,
) {
  Column(modifier = Modifier.fillMaxWidth()) {
    Text(label)
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Text(fileName ?: placeholder, modifier = Modifier.weight(1f).align(Alignment.CenterVertically))
      OutlinedButton(onClick = onChoose) { Text(chooseLabel) }
      if (fileName != null) {
        OutlinedButton(onClick = onClear) { Text("Clear") }
      }
    }
  }
}

@Composable
private fun ModelSelector(
  uiState: UiState,
  onChooseModel: () -> Unit,
  onSelectModel: (String) -> Unit,
) {
  var expanded by remember { mutableStateOf(false) }
  val selected = uiState.models.firstOrNull { it.id == uiState.selectedModelId }
  Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    OutlinedButton(onClick = { expanded = true }, enabled = uiState.models.isNotEmpty(), modifier = Modifier.weight(1f)) {
      Text(selected?.displayName ?: "No model selected")
    }
    OutlinedButton(onClick = onChooseModel) { Text("Choose .tflite") }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      uiState.models.forEach { model ->
        DropdownMenuItem(onClick = { onSelectModel(model.id); expanded = false }) {
          Text(model.displayName)
        }
      }
    }
  }
  Spacer(modifier = Modifier.height(4.dp))
}
