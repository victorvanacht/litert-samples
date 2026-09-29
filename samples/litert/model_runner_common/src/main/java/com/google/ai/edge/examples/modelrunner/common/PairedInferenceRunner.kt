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
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.coroutineScope

data class PairedInferenceConfig(
  val uri: Uri,
  val displayName: String,
  val inputFileUri: Uri?,
  val cpuOutputFileUri: Uri?,
  val gpuOutputFileUri: Uri?,
  val cpuThreadCount: Int,
  val cpuKernelMode: CpuKernelMode,
  val xnnpackFlags: Int,
  val gpuPrecision: CompiledModel.GpuOptions.Precision,
  val gpuBackend: CompiledModel.GpuOptions.Backend,
  val gpuPriority: CompiledModel.GpuOptions.Priority,
  val gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
  val gpuPreferTextureWeights: Boolean,
  val gpuConstantTensorSharing: Boolean,
  val gpuInfiniteFloatCapping: Boolean,
)

data class PairedInferenceProgress(
  val completedRuns: Int,
  val cpuCompletedRuns: Int,
  val gpuCompletedRuns: Int,
  val elapsedMillis: Long,
  val inferencesPerSecond: Double,
  val tensorDescriptions: List<String>,
)

typealias PairedInferenceResult = PairedInferenceProgress

/** Runs one independently compiled synchronous model on each accelerator. */
class PairedInferenceRunner(private val modelRunner: InferenceRunner) {
  suspend fun run(
    config: PairedInferenceConfig,
    warmupRuns: Int = 0,
    maxMeasuredRuns: Int? = null,
    onLog: suspend (String) -> Unit,
    onProgress: suspend (PairedInferenceProgress) -> Unit,
  ): PairedInferenceResult {
    require(warmupRuns >= 0) { "warmupRuns must be non-negative" }
    require(maxMeasuredRuns == null || maxMeasuredRuns > 0) { "maxMeasuredRuns must be positive" }
    require((config.cpuOutputFileUri == null) == (config.gpuOutputFileUri == null)) {
      "CPU and GPU output files must both be selected or both be omitted"
    }
    require(config.cpuOutputFileUri == null || config.cpuOutputFileUri != config.gpuOutputFileUri) {
      "CPU and GPU output files must be different"
    }

    val readyWorkers = AtomicInteger(0)
    val measurementStartNanos = CompletableDeferred<Long>()
    val measurementEndNanos = AtomicLong(0)
    val completedRuns = AtomicInteger(0)
    val cpuCompletedRuns = AtomicInteger(0)
    val gpuCompletedRuns = AtomicInteger(0)
    var latestTensorDescriptions = emptyList<String>()
    lateinit var workerJobs: List<Job>

    suspend fun runWorker(accelerator: AcceleratorChoice, outputUri: Uri?) {
      val workerName = accelerator.displayName
      modelRunner.runSynchronous(
        uri = config.uri,
        displayName = config.displayName,
        accelerator = accelerator,
        cpuThreadCount = config.cpuThreadCount,
        cpuKernelMode = config.cpuKernelMode,
        xnnpackFlags = config.xnnpackFlags,
        gpuPrecision = config.gpuPrecision,
        gpuBackend = config.gpuBackend,
        gpuPriority = config.gpuPriority,
        gpuBufferStorageType = config.gpuBufferStorageType,
        gpuPreferTextureWeights = config.gpuPreferTextureWeights,
        gpuConstantTensorSharing = config.gpuConstantTensorSharing,
        gpuInfiniteFloatCapping = config.gpuInfiniteFloatCapping,
        inputFileUri = config.inputFileUri,
        outputFileUri = outputUri,
        warmupRuns = warmupRuns,
        deferOutputWrites = true,
        onLog = { message ->
          if (!message.startsWith("Inference #")) onLog("[$workerName] $message")
        },
        onReady = {
          onLog("[$workerName] Ready after $warmupRuns warmup run(s)")
          if (readyWorkers.incrementAndGet() == 2) {
            measurementStartNanos.complete(System.nanoTime())
          }
          measurementStartNanos.await()
          onLog("[$workerName] Measurement started")
        },
      ) { result ->
        currentCoroutineContext().ensureActive()
        val nextCount = incrementWithinLimit(completedRuns, maxMeasuredRuns) ?: return@runSynchronous
        val workerCount = when (accelerator) {
          AcceleratorChoice.CPU -> cpuCompletedRuns.incrementAndGet()
          AcceleratorChoice.GPU -> gpuCompletedRuns.incrementAndGet()
          AcceleratorChoice.CPU_GPU -> error("Paired workers must use a single accelerator")
        }
        latestTensorDescriptions = result.tensorDescriptions
        val nowNanos = System.nanoTime()
        val startNanos = measurementStartNanos.await()
        val elapsedMillis = ((nowNanos - startNanos) / 1_000_000).coerceAtLeast(1)
        val progress = PairedInferenceProgress(
          completedRuns = nextCount,
          cpuCompletedRuns = cpuCompletedRuns.get(),
          gpuCompletedRuns = gpuCompletedRuns.get(),
          elapsedMillis = elapsedMillis,
          inferencesPerSecond = nextCount * 1_000_000_000.0 / (nowNanos - startNanos).coerceAtLeast(1),
          tensorDescriptions = result.tensorDescriptions,
        )
        onProgress(progress)
        if (maxMeasuredRuns != null && nextCount == maxMeasuredRuns) {
          measurementEndNanos.compareAndSet(0, System.nanoTime())
          workerJobs.forEach { it.cancel(PairedRunLimitReached()) }
        }
        if (workerCount == 1 || workerCount % 100 == 0) {
          onLog("[$workerName] Completed $workerCount measured inference(s)")
        }
      }
    }

    coroutineScope {
      val cpuJob = launch(start = CoroutineStart.LAZY) {
        runWorker(AcceleratorChoice.CPU, config.cpuOutputFileUri)
      }
      val gpuJob = launch(start = CoroutineStart.LAZY) {
        runWorker(AcceleratorChoice.GPU, config.gpuOutputFileUri)
      }
      workerJobs = listOf(cpuJob, gpuJob)
      workerJobs.forEach(Job::start)
      workerJobs.joinAll()
      currentCoroutineContext().ensureActive()
    }

    val startNanos = measurementStartNanos.await()
    val endNanos = measurementEndNanos.get().takeIf { it != 0L } ?: System.nanoTime()
    val elapsedNanos = (endNanos - startNanos).coerceAtLeast(1)
    val count = completedRuns.get()
    return PairedInferenceProgress(
      completedRuns = count,
      cpuCompletedRuns = cpuCompletedRuns.get(),
      gpuCompletedRuns = gpuCompletedRuns.get(),
      elapsedMillis = (elapsedNanos / 1_000_000).coerceAtLeast(1),
      inferencesPerSecond = count * 1_000_000_000.0 / elapsedNanos,
      tensorDescriptions = latestTensorDescriptions,
    )
  }

  private fun incrementWithinLimit(counter: AtomicInteger, limit: Int?): Int? {
    while (true) {
      val current = counter.get()
      if (limit != null && current >= limit) return null
      if (counter.compareAndSet(current, current + 1)) return current + 1
    }
  }

  private class PairedRunLimitReached : kotlinx.coroutines.CancellationException("Paired run limit reached")
}