package com.google.ai.edge.examples.victortest

import android.content.Context
import com.google.ai.edge.examples.modelrunner.common.InferenceRunner

class ShellBenchmarkInstrumentation : com.google.ai.edge.examples.modelrunner.common.ShellBenchmarkInstrumentation() {
  override fun createModelRunner(context: Context): InferenceRunner = ModelRunner(context)
}