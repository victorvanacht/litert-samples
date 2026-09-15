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

import android.net.Uri
import android.graphics.Color
import androidx.camera.core.CameraSelector
import androidx.compose.runtime.Immutable
import com.google.ai.edge.litert.CompiledModel

@Immutable
data class UiState(
  val mediaUri: Uri = Uri.EMPTY,
  val overlayInfo: OverlayInfo? = null,
  val lensFacing: Int = CameraSelector.LENS_FACING_BACK,
  val models: List<ModelOption> = emptyList(),
  val selectedModelId: String? = null,
  val inputFileUri: Uri? = null,
  val inputFileName: String? = null,
  val outputFileUri: Uri? = null,
  val outputFileName: String? = null,
  val accelerator: AcceleratorChoice = AcceleratorChoice.GPU,
  val gpuPrecision: CompiledModel.GpuOptions.Precision = CompiledModel.GpuOptions.Precision.FP16,
  val gpuBackend: CompiledModel.GpuOptions.Backend = CompiledModel.GpuOptions.Backend.OPENCL,
  val gpuPriority: CompiledModel.GpuOptions.Priority = CompiledModel.GpuOptions.Priority.HIGH,
  val gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType = CompiledModel.GpuOptions.BufferStorageType.BUFFER,
  val gpuPreferTextureWeights: Boolean = false,
  val gpuConstantTensorSharing: Boolean = false,
  val gpuInfiniteFloatCapping: Boolean = true,
  val runMode: RunMode = RunMode.SYNCHRONOUS,
  val inferenceTime: Long? = null,
  val inferencesPerSecond: Double? = null,
  val tensorDescriptions: List<String> = emptyList(),
  val logLines: List<String> = emptyList(),
  val isRunning: Boolean = false,
  val errorMessage: String? = null,
)

enum class AcceleratorChoice(val displayName: String) {
  CPU("CPU"),
  GPU("GPU"),
}

enum class RunMode {
  SYNCHRONOUS,
  ASYNCHRONOUS,
}

@Immutable
data class ModelOption(val id: String, val displayName: String, val uri: Uri)

data class ModelRunResult(
  val displayName: String,
  val inferenceTimeMillis: Long,
  val tensorDescriptions: List<String>,
)

data class ThroughputResult(val inferencesPerSecond: Double, val tensorDescriptions: List<String>)

@Immutable class OverlayInfo(val pixels: IntArray, val width: Int, val height: Int)

@Immutable
data class ColorLabel(val id: Int, val label: String, val rgbColor: Int) {
  fun getColor(): Int {
    return if (id == 0) Color.TRANSPARENT
    else Color.argb(128, Color.red(rgbColor), Color.green(rgbColor), Color.blue(rgbColor))
  }
}
