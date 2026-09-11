package com.google.ai.edge.examples.victortest

import android.net.Uri
import android.graphics.Color
import androidx.camera.core.CameraSelector
import androidx.compose.runtime.Immutable

@Immutable
data class UiState(
  val mediaUri: Uri = Uri.EMPTY,
  val overlayInfo: OverlayInfo? = null,
  val lensFacing: Int = CameraSelector.LENS_FACING_BACK,
  val models: List<ModelOption> = emptyList(),
  val selectedModelId: String? = null,
  val accelerator: AcceleratorChoice = AcceleratorChoice.CPU,
  val inferenceTime: Long? = null,
  val tensorDescriptions: List<String> = emptyList(),
  val logLines: List<String> = emptyList(),
  val isRunning: Boolean = false,
  val errorMessage: String? = null,
)

@Immutable
data class ModelOption(val id: String, val displayName: String, val uri: Uri)

@Immutable class OverlayInfo(val pixels: IntArray, val width: Int, val height: Int)

@Immutable
data class ColorLabel(val id: Int, val label: String, val rgbColor: Int) {
  fun getColor(): Int {
    return if (id == 0) Color.TRANSPARENT
    else Color.argb(128, Color.red(rgbColor), Color.green(rgbColor), Color.blue(rgbColor))
  }
}
