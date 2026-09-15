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

// Shared UI/state/view-model code for the victortest and victortestcpp sample apps. Each app
// includes this module by relative path (see their settings.gradle.kts) and supplies its own
// InferenceRunner implementation (JNI-backed vs pure-Kotlin CompiledModel).
plugins {
  alias(libs.plugins.android.library)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.compose.compiler)
}

android {
  namespace = "com.google.ai.edge.examples.modelrunner.common"
  compileSdk = 36

  defaultConfig { minSdk = 23 }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  kotlinOptions { jvmTarget = "1.8" }
  buildFeatures { compose = true }
}

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.material2)
  implementation(libs.androidx.camera.core)
  implementation(libs.litert) {
    exclude(group = "com.google.ai.edge.litert", module = "litert-support")
    exclude(group = "com.google.ai.edge.litert", module = "litert-support-api")
  }
  implementation(platform("org.jetbrains.kotlinx:kotlinx-coroutines-bom:1.10.2"))
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android")
}
