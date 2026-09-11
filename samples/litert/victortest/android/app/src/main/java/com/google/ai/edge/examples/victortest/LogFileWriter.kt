package com.google.ai.edge.examples.victortest

import android.content.Context
import java.io.File

/** Mirrors the in-app log window to a file retrievable with `adb pull` (no root needed). */
class LogFileWriter(context: Context) {
  val file: File = File(context.getExternalFilesDir(null), "victortest.log")

  fun clear() {
    file.writeText("")
  }

  fun appendLine(line: String) {
    file.appendText("$line\n")
  }
}
