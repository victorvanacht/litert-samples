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

import android.content.Context
import java.io.File

/** Mirrors the in-app log window to a file retrievable with `adb pull` (no root needed). */
class LogFileWriter(context: Context) {
  // Named after the app's own package segment so logs from different sample apps don't collide
  // if pulled into the same folder on a developer's machine.
  val file: File = File(context.getExternalFilesDir(null), "${context.packageName.substringAfterLast('.')}.log")

  /** Last line of the previous session, read before it gets cleared; null if no prior log exists. */
  fun readLastLine(): String? = if (file.exists()) file.readLines().lastOrNull { it.isNotBlank() } else null

  fun clear() {
    file.writeText("")
  }

  fun appendLine(line: String) {
    file.appendText("$line\n")
  }
}
