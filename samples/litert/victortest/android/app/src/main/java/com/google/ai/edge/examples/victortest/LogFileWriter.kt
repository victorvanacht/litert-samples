package com.google.ai.edge.examples.victortest

import android.content.Context
import java.io.File

/** Mirrors the in-app log window to a file retrievable with `adb pull` (no root needed). */
class LogFileWriter(context: Context) {
  val file: File = File(context.getExternalFilesDir(null), "victortest.log")

  /** Last line of the previous session, read before it gets cleared; null if no prior log exists. */
  fun readLastLine(): String? = if (file.exists()) file.readLines().lastOrNull { it.isNotBlank() } else null

  fun clear() {
    file.writeText("")
  }

  fun appendLine(line: String) {
    file.appendText("$line\n")
  }
}
