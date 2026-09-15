# victortest shell benchmarks

The `victortest` and `victortestcpp` Android apps expose self-targeting instrumentation entry points for automated latency testing from `adb shell`. The command-line interface mirrors the UI-configurable model runner options and returns an ordered, parseable result block.

Each instrumentation run launches the app's normal foreground activity, scrolls down to the bottom of the screen content, and drives the same shared `MainViewModel` callbacks that back the UI controls. This keeps command-line benchmarks close to the manual UI path while still allowing scripts to set every option and read structured timing results.

Use this interface for scripts that need to:

- Install one or both sample apps.
- Push a model and optional input data to the device.
- Run a fixed number of synchronous inference runs.
- Ignore warmup runs.
- Read back the average inference time and the individual measured run times.
- Verify exactly which options were used for the run.

## Packages and instrumentation components

| App | Package | Instrumentation component |
| --- | --- | --- |
| `victortest` | `com.google.ai.edge.examples.victortest` | `com.google.ai.edge.examples.victortest/.ShellBenchmarkInstrumentation` |
| `victortestcpp` | `com.google.ai.edge.examples.victortestcpp` | `com.google.ai.edge.examples.victortestcpp/.ShellBenchmarkInstrumentation` |

## Recommended file locations

For automation, push models and input files into the app-specific external files directory:

```text
/sdcard/Android/data/<package>/files/models/
/sdcard/Android/data/<package>/files/inputs/
/sdcard/Android/data/<package>/files/outputs/
```

Do not use shared folders such as `/sdcard/Download` as raw file paths for `model`, `input`, or `output`. On modern Android, the app process running the instrumentation cannot reliably open those paths because of scoped storage restrictions.

## Minimal commands

Build and install the APKs from this repository, then push a model into each app's storage directory:

```sh
adb shell mkdir -p /sdcard/Android/data/com.google.ai.edge.examples.victortest/files/models
adb push yourmodel.tflite \
  /sdcard/Android/data/com.google.ai.edge.examples.victortest/files/models/yourmodel.tflite

adb shell am instrument -w \
  -e model /sdcard/Android/data/com.google.ai.edge.examples.victortest/files/models/yourmodel.tflite \
  -e model_display_name yourmodel.tflite \
  -e runs 20 \
  -e warmup_runs 5 \
  com.google.ai.edge.examples.victortest/.ShellBenchmarkInstrumentation
```

For `victortestcpp`, use the `victortestcpp` package and component:

```sh
adb shell mkdir -p /sdcard/Android/data/com.google.ai.edge.examples.victortestcpp/files/models
adb push yourmodel.tflite \
  /sdcard/Android/data/com.google.ai.edge.examples.victortestcpp/files/models/yourmodel.tflite

adb shell am instrument -w \
  -e model /sdcard/Android/data/com.google.ai.edge.examples.victortestcpp/files/models/yourmodel.tflite \
  -e model_display_name yourmodel.tflite \
  -e runs 20 \
  -e warmup_runs 5 \
  com.google.ai.edge.examples.victortestcpp/.ShellBenchmarkInstrumentation
```

## Full command example

This example specifies every supported command-line option:

```sh
adb shell am instrument -w \
  -e model /sdcard/Android/data/com.google.ai.edge.examples.victortest/files/models/yourmodel.tflite \
  -e model_display_name yourmodel.tflite \
  -e input /sdcard/Android/data/com.google.ai.edge.examples.victortest/files/inputs/input.bin \
  -e output /sdcard/Android/data/com.google.ai.edge.examples.victortest/files/outputs/output.bin \
  -e runs 20 \
  -e warmup_runs 5 \
  -e run_mode SYNCHRONOUS \
  -e accelerator GPU \
  -e gpu_precision FP16 \
  -e gpu_backend OPENCL \
  -e gpu_priority HIGH \
  -e gpu_buffer_storage_type BUFFER \
  -e gpu_prefer_texture_weights false \
  -e gpu_constant_tensor_sharing false \
  -e gpu_infinite_float_capping true \
  com.google.ai.edge.examples.victortest/.ShellBenchmarkInstrumentation
```

To run the same configuration against `victortestcpp`, replace the package paths and component with `com.google.ai.edge.examples.victortestcpp`.

## Command-line options

All options are passed with `adb shell am instrument -w -e <name> <value> ...`. Enum values are case-insensitive. Boolean values accept `true`, `false`, `1`, `0`, `yes`, `no`, `y`, `n`, `on`, and `off`.

| Option | Default | Valid values | Description |
| --- | --- | --- | --- |
| `model` | `asset://selfie_multiclass.tflite` | `asset://...`, `file://...`, or a raw file path | Model to benchmark. Values without `://` are treated as file paths and are reported back as `file:///...`. For custom files, push into the app-specific external files directory first. |
| `model_display_name` | Derived from `model` | Any string | Human-readable model name passed to the runner and echoed in the result. Usually the `.tflite` file name. |
| `input` | Empty | `asset://...`, `file://...`, or a raw file path | Optional raw input data file. If omitted, random input tensors are generated. If provided, bytes are consumed sequentially across input tensors and short files are zero-padded. |
| `output` | Empty | `file://...` or a raw file path | Optional output path. If omitted, outputs are read back for synchronization but discarded. If provided, output tensors are written as concatenated raw bytes. Prefer the app-specific external files directory. |
| `runs` | `1` | Integer greater than `0` | Number of measured inference runs. These runs are included in `total_inference_time_ms`, `average_inference_time_ms`, and `inference_times_ms`. |
| `warmup_runs` | `0` | Integer greater than or equal to `0` | Number of initial inference runs to execute before measuring. Warmup runs are not included in the reported timing metrics. |
| `run_mode` | `SYNCHRONOUS` | `SYNCHRONOUS` | Latency benchmarking currently supports synchronous mode only. `ASYNCHRONOUS` is a UI throughput mode and is rejected by this command-line benchmark. |
| `accelerator` | `GPU` | `CPU`, `GPU` | Accelerator used to compile and run the model. GPU-specific options are only applied when `accelerator=GPU`. |
| `gpu_precision` | `FP16` | `DEFAULT`, `FP16`, `FP32` | GPU precision preference. |
| `gpu_backend` | `OPENCL` | `AUTOMATIC`, `OPENCL`, `WEBGPU`, `OPENGL` | GPU backend preference. Availability depends on the device and LiteRT runtime. |
| `gpu_priority` | `HIGH` | `DEFAULT`, `LOW`, `NORMAL`, `HIGH` | GPU execution priority preference. |
| `gpu_buffer_storage_type` | `BUFFER` | `DEFAULT`, `BUFFER`, `TEXTURE_2D` | GPU tensor buffer storage preference. |
| `gpu_prefer_texture_weights` | `false` | Boolean | Prefer texture-backed GPU weights when supported. |
| `gpu_constant_tensor_sharing` | `false` | Boolean | Enable GPU constant tensor sharing when supported. |
| `gpu_infinite_float_capping` | `true` | Boolean | Enable GPU infinite float capping. |

## Instrumentation output

Successful runs print ordered `INSTRUMENTATION_RESULT` lines. The first block echoes the resolved files and run counts, the second block echoes execution options, and the third block reports status and timing metrics.

Example:

```text
INSTRUMENTATION_RESULT: model=file:///sdcard/Android/data/com.google.ai.edge.examples.victortest/files/models/yourmodel.tflite
INSTRUMENTATION_RESULT: model_display_name=yourmodel.tflite
INSTRUMENTATION_RESULT: input=
INSTRUMENTATION_RESULT: output=
INSTRUMENTATION_RESULT: runs=20
INSTRUMENTATION_RESULT: warmup_runs=5

INSTRUMENTATION_RESULT: run_mode=SYNCHRONOUS
INSTRUMENTATION_RESULT: accelerator=GPU
INSTRUMENTATION_RESULT: gpu_precision=FP16
INSTRUMENTATION_RESULT: gpu_backend=OPENCL
INSTRUMENTATION_RESULT: gpu_priority=HIGH
INSTRUMENTATION_RESULT: gpu_buffer_storage_type=BUFFER
INSTRUMENTATION_RESULT: gpu_prefer_texture_weights=false
INSTRUMENTATION_RESULT: gpu_constant_tensor_sharing=false
INSTRUMENTATION_RESULT: gpu_infinite_float_capping=true

INSTRUMENTATION_RESULT: status=ok
INSTRUMENTATION_RESULT: tensor_descriptions=input 0: FLOAT32, [1, 480, 480, 3]; output 0: FLOAT32, [1, 42525, 17]
INSTRUMENTATION_RESULT: total_inference_time_ms=993
INSTRUMENTATION_RESULT: average_inference_time_ms=49.65
INSTRUMENTATION_RESULT: inference_times_ms=[50, 50, 51, 51, 50, 50, 51, 50, 50, 50, 50, 51, 51, 50, 49, 49, 50, 47, 46, 47]
```

Failure output uses the same prefix and includes `status=error` plus an `error` message:

```text
INSTRUMENTATION_RESULT: status=error
INSTRUMENTATION_RESULT: error=runs must be greater than 0
```

Use `status=ok` as the primary success signal in scripts. Do not depend on `INSTRUMENTATION_CODE`; the ordered stream output intentionally focuses on key/value result lines.

## Python automation example

Use `subprocess.run(..., shell=False)` and pass command arguments as a list. This avoids quoting differences between Windows, macOS, and Linux shells.

```python
from __future__ import annotations

import re
import subprocess
from pathlib import Path


APPS = {
	"victortest": {
		"package": "com.google.ai.edge.examples.victortest",
		"apk": Path("samples/litert/victortest/android/app/build/outputs/apk/debug/victortest.apk"),
	},
	"victortestcpp": {
		"package": "com.google.ai.edge.examples.victortestcpp",
		"apk": Path("samples/litert/victortestcpp/android/app/build/outputs/apk/debug/victortestcpp.apk"),
	},
}


def adb(*args: str) -> str:
	completed = subprocess.run(
		["adb", *args],
		check=True,
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
	)
	return completed.stdout


def parse_instrumentation_results(output: str) -> dict[str, str]:
	results: dict[str, str] = {}
	pattern = re.compile(r"^INSTRUMENTATION_RESULT: ([^=]+)=(.*)$")
	for line in output.splitlines():
		match = pattern.match(line)
		if match:
			results[match.group(1)] = match.group(2)
	return results


def install_app(app_name: str) -> None:
	adb("install", "-r", str(APPS[app_name]["apk"]))


def push_model(app_name: str, local_model: Path) -> str:
	package = APPS[app_name]["package"]
	remote_dir = f"/sdcard/Android/data/{package}/files/models"
	remote_model = f"{remote_dir}/{local_model.name}"
	adb("shell", "mkdir", "-p", remote_dir)
	adb("push", str(local_model), remote_model)
	return remote_model


def run_benchmark(
	app_name: str,
	remote_model: str,
	runs: int = 20,
	warmup_runs: int = 5,
	accelerator: str = "GPU",
) -> dict[str, str]:
	package = APPS[app_name]["package"]
	component = f"{package}/.ShellBenchmarkInstrumentation"
	output = adb(
		"shell",
		"am",
		"instrument",
		"-w",
		"-e", "model", remote_model,
		"-e", "model_display_name", Path(remote_model).name,
		"-e", "runs", str(runs),
		"-e", "warmup_runs", str(warmup_runs),
		"-e", "run_mode", "SYNCHRONOUS",
		"-e", "accelerator", accelerator,
		"-e", "gpu_precision", "FP16",
		"-e", "gpu_backend", "OPENCL",
		"-e", "gpu_priority", "HIGH",
		"-e", "gpu_buffer_storage_type", "BUFFER",
		"-e", "gpu_prefer_texture_weights", "false",
		"-e", "gpu_constant_tensor_sharing", "false",
		"-e", "gpu_infinite_float_capping", "true",
		component,
	)
	results = parse_instrumentation_results(output)
	if results.get("status") != "ok":
		raise RuntimeError(f"Benchmark failed for {app_name}: {results.get('error', output)}")
	return results


def main() -> None:
	model = Path("yourmodel.tflite")
	for app_name in ("victortest", "victortestcpp"):
		install_app(app_name)
		remote_model = push_model(app_name, model)
		results = run_benchmark(app_name, remote_model)
		print(
			app_name,
			"average_ms=", results["average_inference_time_ms"],
			"runs=", results["runs"],
			"warmup_runs=", results["warmup_runs"],
		)


if __name__ == "__main__":
	main()
```

## Notes for test harnesses

- Run `adb devices` before starting and ensure exactly one expected device is connected, or pass `-s <serial>` to every `adb` command.
- Use the same model path layout for both apps, but keep files under each app's own package directory.
- Parse `average_inference_time_ms` as a floating-point number.
- Parse `total_inference_time_ms` as an integer.
- Parse `inference_times_ms` as a list of integer milliseconds if per-run analysis is needed.
- Compare `runs` and `warmup_runs` in the output with the requested values before accepting benchmark results.
- For CPU/GPU comparisons, run separate commands with `accelerator=CPU` and `accelerator=GPU`; GPU-only options are still echoed, but only applied when GPU is selected.