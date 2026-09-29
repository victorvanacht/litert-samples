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
import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.ai.edge.litert.CompiledModel
import java.util.Collections
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PairedInferenceRunnerTest {
  @Test
  fun runsBothAcceleratorsAfterSharedWarmupAndStopsAtAggregateLimit() = runBlocking {
    val fakeRunner = FakeInferenceRunner()
    val progress = mutableListOf<PairedInferenceProgress>()
    val logLines = Collections.synchronizedList(mutableListOf<String>())

    val result = PairedInferenceRunner(fakeRunner).run(
      config = testConfig(),
      warmupRuns = 2,
      maxMeasuredRuns = 5,
      onLog = { logLines += it },
    ) { update ->
      assertTrue(logLines.contains("[CPU] Measurement started"))
      assertTrue(logLines.contains("[GPU] Measurement started"))
      progress += update
    }

    assertEquals(setOf(AcceleratorChoice.CPU, AcceleratorChoice.GPU), fakeRunner.accelerators.toSet())
    assertEquals(2, fakeRunner.receivedWarmupRuns.size)
    assertTrue(fakeRunner.receivedWarmupRuns.all { it == 2 })
    assertEquals(Uri.parse("file:///cpu.bin"), fakeRunner.outputUris[AcceleratorChoice.CPU])
    assertEquals(Uri.parse("file:///gpu.bin"), fakeRunner.outputUris[AcceleratorChoice.GPU])
    assertEquals(5, result.completedRuns)
    assertEquals(5, result.cpuCompletedRuns + result.gpuCompletedRuns)
    assertEquals(5, progress.last().completedRuns)
    assertTrue(result.elapsedMillis > 0)
    assertTrue(result.inferencesPerSecond.isFinite() && result.inferencesPerSecond > 0.0)
    assertEquals(setOf(AcceleratorChoice.CPU, AcceleratorChoice.GPU), fakeRunner.closedAccelerators.toSet())
  }

  @Test
  fun workerFailureCancelsAndJoinsItsSibling() {
    val fakeRunner = FakeInferenceRunner(failAccelerator = AcceleratorChoice.GPU)

    val thrown = runCatching {
      runBlocking {
        PairedInferenceRunner(fakeRunner).run(
          config = testConfig(),
          onLog = {},
          onProgress = {},
        )
      }
    }.exceptionOrNull()

    assertTrue(thrown is IllegalStateException)
    assertEquals(setOf(AcceleratorChoice.CPU, AcceleratorChoice.GPU), fakeRunner.closedAccelerators.toSet())
  }

  @Test
  fun rejectsOneOutputUriForBothWorkers() {
    val config = testConfig().copy(gpuOutputFileUri = Uri.parse("file:///cpu.bin"))
    val thrown = runCatching {
      runBlocking {
        PairedInferenceRunner(FakeInferenceRunner()).run(
          config = config,
          onLog = {},
          onProgress = {},
        )
      }
    }.exceptionOrNull()

    assertTrue(thrown is IllegalArgumentException)
  }

  @Test
  fun shellConfigAcceptsEitherRunModeForCpuGpu() {
    RunMode.entries.forEach { mode ->
      val config = ShellBenchmarkConfig.from(
        Bundle().apply {
          putString("accelerator", "CPU_GPU")
          putString("run_mode", mode.name)
          putString("runs", "5")
          putString("cpu_output", "file:///cpu.bin")
          putString("gpu_output", "file:///gpu.bin")
        },
      )

      assertEquals(AcceleratorChoice.CPU_GPU, config.accelerator)
      assertEquals(mode, config.runMode)
      assertEquals(5, config.runs)
      assertEquals(Uri.parse("file:///cpu.bin"), config.cpuOutputFileUri)
      assertEquals(Uri.parse("file:///gpu.bin"), config.gpuOutputFileUri)
    }
  }

  @Test
  fun shellConfigRejectsMissingPairedOutputDestination() {
    val thrown = runCatching {
      ShellBenchmarkConfig.from(
        Bundle().apply {
          putString("accelerator", "CPU_GPU")
          putString("cpu_output", "file:///cpu.bin")
        },
      )
    }.exceptionOrNull()

    assertTrue(thrown is IllegalArgumentException)
  }

  private fun testConfig() = PairedInferenceConfig(
    uri = Uri.parse("asset://model.tflite"),
    displayName = "model.tflite",
    inputFileUri = null,
    cpuOutputFileUri = Uri.parse("file:///cpu.bin"),
    gpuOutputFileUri = Uri.parse("file:///gpu.bin"),
    cpuThreadCount = 2,
    cpuKernelMode = CpuKernelMode.XNNPACK,
    xnnpackFlags = 0,
    gpuPrecision = CompiledModel.GpuOptions.Precision.FP16,
    gpuBackend = CompiledModel.GpuOptions.Backend.OPENCL,
    gpuPriority = CompiledModel.GpuOptions.Priority.HIGH,
    gpuBufferStorageType = CompiledModel.GpuOptions.BufferStorageType.BUFFER,
    gpuPreferTextureWeights = false,
    gpuConstantTensorSharing = false,
    gpuInfiniteFloatCapping = true,
  )

  private class FakeInferenceRunner(
    private val failAccelerator: AcceleratorChoice? = null,
  ) : InferenceRunner {
    val accelerators = Collections.synchronizedList(mutableListOf<AcceleratorChoice>())
    val receivedWarmupRuns = Collections.synchronizedList(mutableListOf<Int>())
    val outputUris = Collections.synchronizedMap(mutableMapOf<AcceleratorChoice, Uri?>())
    val closedAccelerators = Collections.synchronizedList(mutableListOf<AcceleratorChoice>())

    override suspend fun runSynchronous(
      uri: Uri,
      displayName: String,
      accelerator: AcceleratorChoice,
      cpuThreadCount: Int,
      cpuKernelMode: CpuKernelMode,
      xnnpackFlags: Int,
      gpuPrecision: CompiledModel.GpuOptions.Precision,
      gpuBackend: CompiledModel.GpuOptions.Backend,
      gpuPriority: CompiledModel.GpuOptions.Priority,
      gpuBufferStorageType: CompiledModel.GpuOptions.BufferStorageType,
      gpuPreferTextureWeights: Boolean,
      gpuConstantTensorSharing: Boolean,
      gpuInfiniteFloatCapping: Boolean,
      inputFileUri: Uri?,
      outputFileUri: Uri?,
      warmupRuns: Int,
      deferOutputWrites: Boolean,
      onLog: suspend (String) -> Unit,
      onReady: suspend () -> Unit,
      onResult: suspend (ModelRunResult) -> Unit,
    ) {
      accelerators += accelerator
      receivedWarmupRuns += warmupRuns
      outputUris[accelerator] = outputFileUri
      try {
        delay(if (accelerator == AcceleratorChoice.CPU) 10 else 1)
        onReady()
        if (accelerator == failAccelerator) error("${accelerator.displayName} failed")
        while (true) {
          delay(1)
          onResult(ModelRunResult(displayName, 1, listOf("output: test")))
        }
      } catch (cancel: CancellationException) {
        throw cancel
      } finally {
        closedAccelerators += accelerator
      }
    }

    override suspend fun runAsynchronous(
      uri: Uri,
      displayName: String,
      accelerator: AcceleratorChoice,
      cpuThreadCount: Int,
      cpuKernelMode: CpuKernelMode,
      xnnpackFlags: Int,
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
    ) = error("Paired inference must use independent synchronous workers")
  }
}