#include <jni.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"

namespace {

using litert::CompiledModel;
using litert::ElementType;
using litert::Environment;
using litert::Expected;
using litert::GpuOptions;
using litert::HwAccelerators;
using litert::Options;
using litert::TensorBuffer;

struct BufferSet {
  std::vector<TensorBuffer> inputs;
  std::vector<TensorBuffer> outputs;
};

struct Session {
  Environment environment;
  CompiledModel model;
  std::vector<std::string> tensor_descriptions;
  std::vector<ElementType> input_types;
  std::vector<size_t> input_elements;
  BufferSet synchronous_buffers;
};

void Throw(JNIEnv* env, const std::string& message) {
  jclass exception = env->FindClass("java/lang/RuntimeException");
  if (exception != nullptr) {
    env->ThrowNew(exception, message.c_str());
  }
}

size_t ElementCount(const litert::RankedTensorType& type) {
  size_t count = 1;
  for (int dimension : type.Layout().Dimensions()) {
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

void FillInput(TensorBuffer& buffer, ElementType type, size_t elements) {
  static thread_local std::mt19937 generator(std::random_device{}());
  auto memory = buffer.Lock(TensorBuffer::LockMode::kWrite);
  LITERT_ABORT_IF_ERROR(memory);

  switch (type) {
    case ElementType::Float32:
      for (size_t i = 0; i < elements; ++i) {
        static_cast<float*>(*memory)[i] =
            std::generate_canonical<float, 24>(generator);
      }
      break;
    case ElementType::Int32:
      for (size_t i = 0; i < elements; ++i) {
        static_cast<int32_t*>(*memory)[i] = static_cast<int32_t>(generator());
      }
      break;
    case ElementType::UInt8:
    case ElementType::Int8:
      for (size_t i = 0; i < elements; ++i) {
        static_cast<uint8_t*>(*memory)[i] = static_cast<uint8_t>(generator());
      }
      break;
    case ElementType::Int64:
      for (size_t i = 0; i < elements; ++i) {
        static_cast<int64_t*>(*memory)[i] = static_cast<int64_t>(generator());
      }
      break;
    case ElementType::Bool:
      std::memset(*memory, 1, elements);
      break;
    default: {
      auto packed_size = buffer.PackedSize();
      LITERT_ABORT_IF_ERROR(packed_size);
      std::memset(*memory, 0, *packed_size);
      break;
    }
  }
  LITERT_ABORT_IF_ERROR(buffer.Unlock());
}

Expected<BufferSet> CreateBuffers(const CompiledModel& model) {
  LITERT_ASSIGN_OR_RETURN(auto inputs, model.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto outputs, model.CreateOutputBuffers());
  return BufferSet{std::move(inputs), std::move(outputs)};
}

void InitializeInputs(Session& session, BufferSet& buffers) {
  for (size_t i = 0; i < buffers.inputs.size(); ++i) {
    FillInput(buffers.inputs[i], session.input_types[i], session.input_elements[i]);
  }
}

void RunBufferSet(Session& session, BufferSet& buffers) {
  LITERT_ABORT_IF_ERROR(session.model.Run(buffers.inputs, buffers.outputs));

  for (TensorBuffer& output : buffers.outputs) {
    auto memory = output.Lock(TensorBuffer::LockMode::kRead);
    LITERT_ABORT_IF_ERROR(memory);
    LITERT_ABORT_IF_ERROR(output.Unlock());
  }
}

Expected<std::unique_ptr<Session>> CreateSession(
    const std::string& model_path, int accelerator, int precision, int backend,
    int priority, int storage_type, bool prefer_texture_weights,
    bool constant_tensor_sharing, bool infinite_float_capping) {
  LITERT_ASSIGN_OR_RETURN(auto environment, Environment::Create({}));

  Options options;
  const auto hardware = accelerator == 1 ? HwAccelerators::kGpu
                                         : HwAccelerators::kCpu;
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(hardware));

  if (accelerator == 1) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu, options.GetGpuOptions());
    LITERT_RETURN_IF_ERROR(gpu.SetPrecision(
        static_cast<GpuOptions::Precision>(precision)));
    LITERT_RETURN_IF_ERROR(gpu.SetBackend(
        static_cast<GpuOptions::Backend>(backend)));
    LITERT_RETURN_IF_ERROR(gpu.SetPriority(
        static_cast<GpuOptions::Priority>(priority)));
    LITERT_RETURN_IF_ERROR(gpu.SetBufferStorageType(
        static_cast<GpuOptions::BufferStorageType>(storage_type)));
    LITERT_RETURN_IF_ERROR(
        gpu.SetPreferTextureWeights(prefer_texture_weights));
    LITERT_RETURN_IF_ERROR(
        gpu.EnableConstantTensorSharing(constant_tensor_sharing));
    LITERT_RETURN_IF_ERROR(
        gpu.EnableInfiniteFloatCapping(infinite_float_capping));
  }

  LITERT_ASSIGN_OR_RETURN(auto model,
                          CompiledModel::Create(environment, model_path, options));
  auto session = std::make_unique<Session>(
      Session{std::move(environment), std::move(model), {}, {}, {}, {}});

  LITERT_ASSIGN_OR_RETURN(auto input_names, session->model.GetSignatureInputNames());
  LITERT_ASSIGN_OR_RETURN(auto output_names, session->model.GetSignatureOutputNames());
  for (size_t i = 0; i < input_names.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto type, session->model.GetInputTensorType(0, i));
    session->input_types.push_back(type.ElementType());
    session->input_elements.push_back(ElementCount(type));
    session->tensor_descriptions.push_back(
        "input " + std::to_string(i) + ": " + std::string(input_names[i]));
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    session->tensor_descriptions.push_back(
        "output " + std::to_string(i) + ": " + std::string(output_names[i]));
  }
  LITERT_ASSIGN_OR_RETURN(session->synchronous_buffers,
                          CreateBuffers(session->model));
  InitializeInputs(*session, session->synchronous_buffers);
  return session;
}

Session* GetSession(jlong handle) {
  return reinterpret_cast<Session*>(handle);
}

extern "C" JNIEXPORT jlong JNICALL NativePrepare(
    JNIEnv* env, jobject, jstring model_path, jint accelerator, jint precision,
    jint backend, jint priority, jint storage_type, jboolean prefer_texture_weights,
    jboolean constant_tensor_sharing, jboolean infinite_float_capping) {
  const char* path = env->GetStringUTFChars(model_path, nullptr);
  auto session = CreateSession(path, accelerator, precision, backend, priority,
                               storage_type, prefer_texture_weights,
                               constant_tensor_sharing, infinite_float_capping);
  env->ReleaseStringUTFChars(model_path, path);
  if (!session) {
    Throw(env, session.Error().Message());
    return 0;
  }
  return reinterpret_cast<jlong>(std::move(*session).release());
}

extern "C" JNIEXPORT jobjectArray JNICALL NativeTensorDescriptions(
    JNIEnv* env, jobject, jlong handle) {
  const Session* session = GetSession(handle);
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(
      session->tensor_descriptions.size(), string_class, nullptr);
  for (size_t i = 0; i < session->tensor_descriptions.size(); ++i) {
    jstring value = env->NewStringUTF(session->tensor_descriptions[i].c_str());
    env->SetObjectArrayElement(result, i, value);
    env->DeleteLocalRef(value);
  }
  return result;
}

extern "C" JNIEXPORT jlong JNICALL NativeRun(JNIEnv*, jobject, jlong handle) {
  Session* session = GetSession(handle);
  const auto start = std::chrono::steady_clock::now();
  RunBufferSet(*session, session->synchronous_buffers);
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

extern "C" JNIEXPORT jdouble JNICALL NativeRunConcurrent(
    JNIEnv*, jobject, jlong handle, jint concurrency) {
  Session* session = GetSession(handle);
  std::vector<BufferSet> slots;
  slots.reserve(concurrency);
  for (int i = 0; i < concurrency; ++i) {
    auto buffers = CreateBuffers(session->model);
    if (!buffers.HasValue()) std::abort();
    InitializeInputs(*session, buffers.Value());
    slots.push_back(std::move(buffers.Value()));
  }
  const auto start = std::chrono::steady_clock::now();
  const auto deadline = start + std::chrono::milliseconds(200);
  std::vector<std::thread> workers;
  std::atomic<long> completed = 0;
  for (BufferSet& slot : slots) {
    workers.emplace_back([&session, &slot, &deadline, &completed] {
      while (std::chrono::steady_clock::now() < deadline) {
        RunBufferSet(*session, slot);
        ++completed;
      }
    });
  }
  for (auto& worker : workers) worker.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return completed / seconds;
}

extern "C" JNIEXPORT void JNICALL NativeClose(JNIEnv*, jobject, jlong handle) {
  delete GetSession(handle);
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }
  jclass runner = env->FindClass(
      "com/google/ai/edge/examples/victortestcpp/NativeModelRunner");
  if (runner == nullptr) return JNI_ERR;
  const JNINativeMethod methods[] = {
      {"nativePrepare", "(Ljava/lang/String;IIIIIZZZ)J", reinterpret_cast<void*>(NativePrepare)},
      {"nativeTensorDescriptions", "(J)[Ljava/lang/String;", reinterpret_cast<void*>(NativeTensorDescriptions)},
      {"nativeRun", "(J)J", reinterpret_cast<void*>(NativeRun)},
      {"nativeRunConcurrent", "(JI)D", reinterpret_cast<void*>(NativeRunConcurrent)},
      {"nativeClose", "(J)V", reinterpret_cast<void*>(NativeClose)},
  };
  if (env->RegisterNatives(runner, methods, sizeof(methods) / sizeof(methods[0])) != JNI_OK) {
    return JNI_ERR;
  }
  return JNI_VERSION_1_6;
}
