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

package com.google.ai.edge.examples.victortest

import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.ai.edge.examples.modelrunner.common.MainViewModel
import com.google.ai.edge.examples.modelrunner.common.ModelRunnerScreen
import com.google.ai.edge.examples.modelrunner.common.view.ApplicationTheme

// The Compose UI, UiState, and MainViewModel logic are shared with the victortestcpp sample app
// in the model_runner_common module. This Activity only wires up file pickers and this app's own
// pure-Kotlin InferenceRunner (ModelRunner).
class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) { context -> ModelRunner(context) } }
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
          appName = "victortest",
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
