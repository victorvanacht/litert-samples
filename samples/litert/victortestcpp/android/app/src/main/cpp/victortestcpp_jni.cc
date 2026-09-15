#include <dlfcn.h>
#include <jni.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Status = int;
using Environment = void*;
using Model = void*;
using CompiledModel = void*;
using Options = void*;
using Tensor = void*;
using Signature = void*;
using TensorBuffer = void*;
using BufferRequirements = void*;

struct Layout {
  unsigned int rank : 7;
  bool has_strides : 1;
  int32_t dimensions[8];
  uint32_t strides[8];
};

struct RankedTensorType {
  int32_t element_type;
  Layout layout;
};

struct EnvOption;

constexpr Status kOk = 0;
constexpr int kCpu = 1;
constexpr int kGpu = 2;
constexpr int kRead = 0;
constexpr int kWrite = 1;

using CreateEnvironment = Status (*)(int, const EnvOption*, Environment*);
using DestroyEnvironment = void (*)(Environment);
using CreateModelFromFile = Status (*)(Environment, const char*, Model*);
using DestroyModel = void (*)(Model);
using CreateOptions = Status (*)(Options*);
using DestroyOptions = void (*)(Options);
using SetHardwareAccelerators = Status (*)(Options, int);
using CreateCompiledModel = Status (*)(Environment, Model, Options, CompiledModel*);
using DestroyCompiledModel = void (*)(CompiledModel);
using GetModelSignature = Status (*)(Model, size_t, Signature*);
using GetNumSignatureInputs = Status (*)(Signature, size_t*);
using GetNumSignatureOutputs = Status (*)(Signature, size_t*);
using GetSignatureInputTensor = Status (*)(Signature, size_t, Tensor*);
using GetSignatureOutputTensor = Status (*)(Signature, size_t, Tensor*);
using GetRankedTensorType = Status (*)(Tensor, RankedTensorType*);
using GetSignatureInputName = Status (*)(Signature, size_t, const char**);
using GetSignatureOutputName = Status (*)(Signature, size_t, const char**);
using GetInputRequirements = Status (*)(CompiledModel, size_t, size_t, BufferRequirements*);
using GetOutputRequirements = Status (*)(CompiledModel, size_t, size_t, BufferRequirements*);
using CreateManagedBuffer = Status (*)(Environment, int, const RankedTensorType*, size_t, TensorBuffer*);
using CreateManagedBufferFromRequirements = Status (*)(Environment, const RankedTensorType*, BufferRequirements, TensorBuffer*);
using DestroyBuffer = void (*)(TensorBuffer);
using LockBuffer = Status (*)(TensorBuffer, void**, int);
using UnlockBuffer = Status (*)(TensorBuffer);
using RunModel = Status (*)(CompiledModel, size_t, size_t, TensorBuffer*, size_t, TensorBuffer*);
using GetStatusString = const char* (*)(Status);

struct Api {
  void* library = nullptr;
  CreateEnvironment create_environment = nullptr;
  DestroyEnvironment destroy_environment = nullptr;
  CreateModelFromFile create_model_from_file = nullptr;
  DestroyModel destroy_model = nullptr;
  CreateOptions create_options = nullptr;
  DestroyOptions destroy_options = nullptr;
  SetHardwareAccelerators set_hardware_accelerators = nullptr;
  CreateCompiledModel create_compiled_model = nullptr;
  DestroyCompiledModel destroy_compiled_model = nullptr;
  GetModelSignature get_model_signature = nullptr;
  GetNumSignatureInputs get_num_signature_inputs = nullptr;
  GetNumSignatureOutputs get_num_signature_outputs = nullptr;
  GetSignatureInputTensor get_input_tensor = nullptr;
  GetSignatureOutputTensor get_output_tensor = nullptr;
  GetRankedTensorType get_ranked_tensor_type = nullptr;
  GetSignatureInputName get_input_name = nullptr;
  GetSignatureOutputName get_output_name = nullptr;
  GetInputRequirements get_input_requirements = nullptr;
  GetOutputRequirements get_output_requirements = nullptr;
  CreateManagedBuffer create_managed_buffer = nullptr;
  CreateManagedBufferFromRequirements create_managed_buffer_from_requirements = nullptr;
  DestroyBuffer destroy_buffer = nullptr;
  LockBuffer lock_buffer = nullptr;
  UnlockBuffer unlock_buffer = nullptr;
  RunModel run_model = nullptr;
  GetStatusString get_status_string = nullptr;
};

Api api;

template <typename Function>
Function Resolve(const char* name) {
  return reinterpret_cast<Function>(dlsym(api.library, name));
}

bool ResolveApi() {
  if (api.library != nullptr) {
    return true;
  }
  api.library = dlopen("libLiteRt.so", RTLD_NOW);
  if (api.library == nullptr) {
    return false;
  }
#define RESOLVE(field, name) api.field = Resolve<decltype(api.field)>("LiteRt" #name)
  RESOLVE(create_environment, CreateEnvironment);
  RESOLVE(destroy_environment, DestroyEnvironment);
  RESOLVE(create_model_from_file, CreateModelFromFile);
  RESOLVE(destroy_model, DestroyModel);
  RESOLVE(create_options, CreateOptions);
  RESOLVE(destroy_options, DestroyOptions);
  RESOLVE(set_hardware_accelerators, SetOptionsHardwareAccelerators);
  RESOLVE(create_compiled_model, CreateCompiledModel);
  RESOLVE(destroy_compiled_model, DestroyCompiledModel);
  RESOLVE(get_model_signature, GetModelSignature);
  RESOLVE(get_num_signature_inputs, GetNumSignatureInputs);
  RESOLVE(get_num_signature_outputs, GetNumSignatureOutputs);
  RESOLVE(get_input_tensor, GetSignatureInputTensorByIndex);
  RESOLVE(get_output_tensor, GetSignatureOutputTensorByIndex);
  RESOLVE(get_ranked_tensor_type, GetRankedTensorType);
  RESOLVE(get_input_name, GetSignatureInputName);
  RESOLVE(get_output_name, GetSignatureOutputName);
  RESOLVE(get_input_requirements, GetCompiledModelInputBufferRequirements);
  RESOLVE(get_output_requirements, GetCompiledModelOutputBufferRequirements);
  RESOLVE(create_managed_buffer, CreateManagedTensorBuffer);
  RESOLVE(create_managed_buffer_from_requirements, CreateManagedTensorBufferFromRequirements);
  RESOLVE(destroy_buffer, DestroyTensorBuffer);
  RESOLVE(lock_buffer, LockTensorBuffer);
  RESOLVE(unlock_buffer, UnlockTensorBuffer);
  RESOLVE(run_model, RunCompiledModel);
  RESOLVE(get_status_string, GetStatusString);
#undef RESOLVE
  return api.create_environment != nullptr && api.destroy_environment != nullptr &&
         api.create_model_from_file != nullptr && api.destroy_model != nullptr &&
         api.create_options != nullptr && api.destroy_options != nullptr &&
         api.set_hardware_accelerators != nullptr && api.create_compiled_model != nullptr &&
         api.destroy_compiled_model != nullptr && api.get_model_signature != nullptr &&
         api.get_num_signature_inputs != nullptr && api.get_num_signature_outputs != nullptr &&
         api.get_input_tensor != nullptr && api.get_output_tensor != nullptr &&
         api.get_ranked_tensor_type != nullptr && api.get_input_name != nullptr &&
         api.get_output_name != nullptr && api.get_input_requirements != nullptr &&
         api.get_output_requirements != nullptr && api.create_managed_buffer_from_requirements != nullptr &&
         api.destroy_buffer != nullptr && api.lock_buffer != nullptr && api.unlock_buffer != nullptr &&
         api.run_model != nullptr && api.get_status_string != nullptr;
}

std::string StatusText(const char* operation, Status status) {
  return std::string(operation) + " failed: " + api.get_status_string(status);
}

void Throw(JNIEnv* env, const std::string& message) {
  jclass exception = env->FindClass("java/lang/RuntimeException");
  if (exception != nullptr) {
    env->ThrowNew(exception, message.c_str());
  }
}

struct TensorInfo {
  RankedTensorType type{};
  BufferRequirements requirements = nullptr;
  std::string name;
  size_t elements = 0;
};

struct Slot {
  std::vector<TensorBuffer> inputs;
  std::vector<TensorBuffer> outputs;

  ~Slot() {
    for (TensorBuffer buffer : inputs) {
      api.destroy_buffer(buffer);
    }
    for (TensorBuffer buffer : outputs) {
      api.destroy_buffer(buffer);
    }
  }
};

struct Session {
  Environment environment = nullptr;
  Model model = nullptr;
  CompiledModel compiled_model = nullptr;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  std::unique_ptr<Slot> synchronous_slot;
  std::mutex error_mutex;
  Status last_status = kOk;

  ~Session() {
    if (synchronous_slot) {
      DestroySlot(*synchronous_slot);
    }
    if (compiled_model != nullptr) {
      api.destroy_compiled_model(compiled_model);
    }
    if (model != nullptr) {
      api.destroy_model(model);
    }
    if (environment != nullptr) {
      api.destroy_environment(environment);
    }
  }

  void DestroySlot(Slot& slot) {
    for (TensorBuffer buffer : slot.inputs) {
      api.destroy_buffer(buffer);
    }
    for (TensorBuffer buffer : slot.outputs) {
      api.destroy_buffer(buffer);
    }
    slot.inputs.clear();
    slot.outputs.clear();
  }

  Status CreateSlot(Slot& slot) {
    for (const TensorInfo& info : inputs) {
      TensorBuffer buffer = nullptr;
      Status status = api.create_managed_buffer_from_requirements(
          environment, &info.type, info.requirements, &buffer);
      if (status != kOk) {
        DestroySlot(slot);
        return status;
      }
      slot.inputs.push_back(buffer);
    }
    for (const TensorInfo& info : outputs) {
      TensorBuffer buffer = nullptr;
      Status status = api.create_managed_buffer_from_requirements(
          environment, &info.type, info.requirements, &buffer);
      if (status != kOk) {
        DestroySlot(slot);
        return status;
      }
      slot.outputs.push_back(buffer);
    }
    return kOk;
  }

  Status RunSlot(Slot& slot) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      void* memory = nullptr;
      Status status = api.lock_buffer(slot.inputs[i], &memory, kWrite);
      if (status != kOk) return status;
      FillInput(memory, inputs[i]);
      status = api.unlock_buffer(slot.inputs[i]);
      if (status != kOk) return status;
    }
    Status status = api.run_model(compiled_model, 0, slot.inputs.size(), slot.inputs.data(),
                                  slot.outputs.size(), slot.outputs.data());
    if (status != kOk) return status;
    for (TensorBuffer buffer : slot.outputs) {
      void* memory = nullptr;
      status = api.lock_buffer(buffer, &memory, kRead);
      if (status != kOk) return status;
      status = api.unlock_buffer(buffer);
      if (status != kOk) return status;
    }
    return kOk;
  }

  void FillInput(void* memory, const TensorInfo& info) {
    static thread_local std::mt19937 generator(std::random_device{}());
    switch (info.type.element_type) {
      case 1:
        for (size_t i = 0; i < info.elements; ++i) static_cast<float*>(memory)[i] = std::generate_canonical<float, 24>(generator);
        break;
      case 2:
        for (size_t i = 0; i < info.elements; ++i) static_cast<int32_t*>(memory)[i] = static_cast<int32_t>(generator());
        break;
      case 3:
      case 9:
        for (size_t i = 0; i < info.elements; ++i) static_cast<uint8_t*>(memory)[i] = static_cast<uint8_t>(generator());
        break;
      case 4:
        for (size_t i = 0; i < info.elements; ++i) static_cast<int64_t*>(memory)[i] = static_cast<int64_t>(generator());
        break;
      case 6:
        std::memset(memory, 1, info.elements);
        break;
      default:
        std::memset(memory, 0, info.elements * 4);
        break;
    }
  }
};

size_t ElementCount(const Layout& layout) {
  size_t count = 1;
  for (unsigned int i = 0; i < layout.rank; ++i) {
    count *= static_cast<size_t>(layout.dimensions[i]);
  }
  return count;
}

Session* GetSession(jlong handle) {
  return reinterpret_cast<Session*>(handle);
}

extern "C" JNIEXPORT jlong JNICALL NativePrepare(JNIEnv* env, jobject, jstring model_path,
                                                   jint accelerator) {
  if (!ResolveApi()) {
    Throw(env, "Unable to resolve LiteRT C API from libLiteRt.so");
    return 0;
  }
  const char* path = env->GetStringUTFChars(model_path, nullptr);
  auto session = std::make_unique<Session>();
  Status status = api.create_environment(0, nullptr, &session->environment);
  if (status == kOk) status = api.create_model_from_file(session->environment, path, &session->model);
  Options options = nullptr;
  if (status == kOk) status = api.create_options(&options);
  if (status == kOk) status = api.set_hardware_accelerators(options, accelerator == 1 ? kGpu : kCpu);
  if (status == kOk) status = api.create_compiled_model(session->environment, session->model, options, &session->compiled_model);
  if (options != nullptr) api.destroy_options(options);
  env->ReleaseStringUTFChars(model_path, path);
  if (status != kOk) {
    Throw(env, StatusText("LiteRT model creation", status));
    return 0;
  }
  Signature signature = nullptr;
  status = api.get_model_signature(session->model, 0, &signature);
  if (status != kOk) {
    Throw(env, StatusText("LiteRT signature query", status));
    return 0;
  }
  size_t input_count = 0;
  size_t output_count = 0;
  status = api.get_num_signature_inputs(signature, &input_count);
  if (status == kOk) status = api.get_num_signature_outputs(signature, &output_count);
  for (size_t i = 0; status == kOk && i < input_count; ++i) {
    TensorInfo info;
    Tensor tensor = nullptr;
    const char* name = nullptr;
    status = api.get_input_tensor(signature, i, &tensor);
    if (status == kOk) status = api.get_ranked_tensor_type(tensor, &info.type);
    if (status == kOk) status = api.get_input_name(signature, i, &name);
    if (status == kOk) info.name = name;
    if (status == kOk) status = api.get_input_requirements(session->compiled_model, 0, i, &info.requirements);
    if (status == kOk) info.elements = ElementCount(info.type.layout);
    if (status == kOk) session->inputs.push_back(std::move(info));
  }
  for (size_t i = 0; status == kOk && i < output_count; ++i) {
    TensorInfo info;
    Tensor tensor = nullptr;
    const char* name = nullptr;
    status = api.get_output_tensor(signature, i, &tensor);
    if (status == kOk) status = api.get_ranked_tensor_type(tensor, &info.type);
    if (status == kOk) status = api.get_output_name(signature, i, &name);
    if (status == kOk) info.name = name;
    if (status == kOk) status = api.get_output_requirements(session->compiled_model, 0, i, &info.requirements);
    if (status == kOk) info.elements = ElementCount(info.type.layout);
    if (status == kOk) session->outputs.push_back(std::move(info));
  }
  if (status == kOk) {
    session->synchronous_slot = std::make_unique<Slot>();
    status = session->CreateSlot(*session->synchronous_slot);
  }
  if (status != kOk) {
    Throw(env, StatusText("LiteRT tensor setup", status));
    return 0;
  }
  return reinterpret_cast<jlong>(session.release());
}

extern "C" JNIEXPORT jobjectArray JNICALL NativeTensorDescriptions(JNIEnv* env, jobject, jlong handle) {
  Session* session = GetSession(handle);
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(session->inputs.size() + session->outputs.size(), string_class, nullptr);
  size_t index = 0;
  auto add = [&](const char* kind, const std::vector<TensorInfo>& tensors) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      std::string description = std::string(kind) + " " + std::to_string(i) + ": " + tensors[i].name;
      jstring value = env->NewStringUTF(description.c_str());
      env->SetObjectArrayElement(result, index++, value);
      env->DeleteLocalRef(value);
    }
  };
  add("input", session->inputs);
  add("output", session->outputs);
  return result;
}

extern "C" JNIEXPORT jlong JNICALL NativeRun(JNIEnv* env, jobject, jlong handle) {
  Session* session = GetSession(handle);
  auto start = std::chrono::steady_clock::now();
  Status status = session->RunSlot(*session->synchronous_slot);
  if (status != kOk) {
    Throw(env, StatusText("LiteRT inference", status));
    return 0;
  }
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

extern "C" JNIEXPORT jdouble JNICALL NativeRunConcurrent(JNIEnv* env, jobject, jlong handle, jint concurrency) {
  if (concurrency < 1) {
    Throw(env, "Concurrency must be positive");
    return 0;
  }
  Session* session = GetSession(handle);
  std::vector<std::unique_ptr<Slot>> slots(concurrency);
  for (auto& slot : slots) {
    slot = std::make_unique<Slot>();
    Status status = session->CreateSlot(*slot);
    if (status != kOk) {
      Throw(env, StatusText("LiteRT buffer setup", status));
      return 0;
    }
  }
  std::atomic<long> completed = 0;
  std::atomic<Status> failure = kOk;
  const auto start = std::chrono::steady_clock::now();
  const auto deadline = start + std::chrono::milliseconds(200);
  std::vector<std::thread> workers;
  for (auto& slot : slots) {
    Slot* slot_ptr = slot.get();
    workers.emplace_back([&session, &completed, &failure, &deadline, slot_ptr]() {
      while (std::chrono::steady_clock::now() < deadline && failure.load() == kOk) {
        Status status = session->RunSlot(*slot_ptr);
        if (status != kOk) {
          failure.store(status);
          return;
        }
        completed.fetch_add(1);
      }
    });
  }
  for (auto& worker : workers) worker.join();
  if (failure.load() != kOk) {
    Throw(env, StatusText("LiteRT asynchronous inference", failure.load()));
    return 0;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return completed.load() / seconds;
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
  if (runner == nullptr) {
    return JNI_ERR;
  }
  const JNINativeMethod methods[] = {
      {"nativePrepare", "(Ljava/lang/String;I)J", reinterpret_cast<void*>(NativePrepare)},
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