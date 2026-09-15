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

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.Switch
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ai.edge.litert.CompiledModel

/** Full screen for both victortest and victortestcpp: model/file pickers, run controls, results, logs. */
@Composable
fun ModelRunnerScreen(
  appName: String,
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
    TopAppBar(title = { Text(appName) }, backgroundColor = MaterialTheme.colors.secondary)
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
