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
import com.google.ai.edge.litert.CompiledModel

/**
 * Common contract implemented by each sample app's model runner: victortest drives inference
 * through the pure-Kotlin `CompiledModel` API, victortestcpp through a JNI bridge to the LiteRT
 * C++ SDK. Both expose the same tunable knobs so [MainViewModel] and the shared Compose UI never
 * need to know which backend is in use.
 */
interface InferenceRunner {
  /** Runs one inference at a time, waiting for each result before starting the next. */
  suspend fun runSynchronous(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    gpuPrecision: CompiledModel.GpuOptions.Precision,
    gpuBackend: CompiledModel.GpuOptions.Backend,
    gpuPriority: CompiledModel.GpuOptions.Priority,
    gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
    gpuPreferTextureWeights: Boolean,
    gpuConstantTensorSharing: Boolean,
    gpuInfiniteFloatCapping: Boolean,
    inputFileUri: Uri?,
    outputFileUri: Uri?,
    onLog: suspend (String) -> Unit,
    onResult: suspend (ModelRunResult) -> Unit,
  )

  /** Keeps [concurrency] inferences in flight, reporting aggregate throughput instead of latency. */
  suspend fun runAsynchronous(
    uri: Uri,
    displayName: String,
    accelerator: AcceleratorChoice,
    gpuPrecision: CompiledModel.GpuOptions.Precision,
    gpuBackend: CompiledModel.GpuOptions.Backend,
    gpuPriority: CompiledModel.GpuOptions.Priority,
    gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
    gpuPreferTextureWeights: Boolean,
    gpuConstantTensorSharing: Boolean,
    gpuInfiniteFloatCapping: Boolean,
    inputFileUri: Uri?,
    outputFileUri: Uri?,
    concurrency: Int,
    onLog: suspend (String) -> Unit,
    onThroughput: suspend (ThroughputResult) -> Unit,
  )
}
