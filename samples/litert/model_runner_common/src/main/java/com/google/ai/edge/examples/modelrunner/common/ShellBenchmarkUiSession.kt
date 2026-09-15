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

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

object ShellBenchmarkUiSession {
  private val activeRequest = AtomicReference<Request?>()
  private val _request = MutableStateFlow<Request?>(null)
  val request: StateFlow<Request?> = _request

  fun start(config: ShellBenchmarkConfig): Request {
    val request = Request(config)
    check(activeRequest.compareAndSet(null, request)) { "A shell benchmark is already running" }
    _request.value = request
    return request
  }

  fun finish(request: Request, result: Result<ShellBenchmarkResult>) {
    if (activeRequest.compareAndSet(request, null)) {
      _request.value = null
      request.result.complete(result)
    }
  }

  class Request(val config: ShellBenchmarkConfig) {
    val result = CompletableDeferred<Result<ShellBenchmarkResult>>()
  }
}

@Composable
fun ShellBenchmarkUiDriver(viewModel: MainViewModel) {
  val request by ShellBenchmarkUiSession.request.collectAsState()
  LaunchedEffect(request) {
    val activeRequest = request ?: return@LaunchedEffect
    val result = runCatching { viewModel.runShellBenchmarkFromUi(activeRequest.config) }
    ShellBenchmarkUiSession.finish(activeRequest, result)
  }
}