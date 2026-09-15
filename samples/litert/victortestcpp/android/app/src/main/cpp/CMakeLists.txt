cmake_minimum_required(VERSION 3.22.1)
project(victortestcpp_jni CXX)

set(LITERT_SDK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/litert_cc_sdk")
set(LITERT_SDK_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_cc_sdk")

# The official SDK is header/source-only; its matching runtime is supplied per
# ABI from the LiteRT Android artifact.
file(COPY "${LITERT_SDK_SOURCE_DIR}/"
	DESTINATION "${LITERT_SDK_BUILD_DIR}"
	PATTERN "prebuilt" EXCLUDE)
file(COPY "${LITERT_SDK_SOURCE_DIR}/prebuilt/${ANDROID_ABI}/libLiteRt.so"
	DESTINATION "${LITERT_SDK_BUILD_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LITERT_SDK_BUILD_DIR}")
include("${LITERT_SDK_BUILD_DIR}/abseil-cpp.cmake")
include("${LITERT_SDK_BUILD_DIR}/flatbuffers.cmake")
if(TARGET flatbuffers AND NOT TARGET flatbuffers::flatbuffers)
	add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
endif()

# FlatBuffers needs a host-side flatc while the rest of the SDK is cross-built
# for Android. Build that small tool with the host CMake toolchain first.
set(LITERT_HOST_FLATC_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/host_flatc")
execute_process(
	COMMAND ${CMAKE_COMMAND} -S "${CMAKE_BINARY_DIR}/flatbuffers"
					-B "${LITERT_HOST_FLATC_BUILD_DIR}"
					-DFLATBUFFERS_BUILD_TESTS=OFF
					-DFLATBUFFERS_BUILD_FLATLIB=OFF
					-DFLATBUFFERS_BUILD_FLATC=ON
					  -DFLATBUFFERS_BUILD_GRPCTEST=OFF
	RESULT_VARIABLE LITERT_FLATC_CONFIGURE_RESULT)
if(NOT LITERT_FLATC_CONFIGURE_RESULT EQUAL 0)
	message(FATAL_ERROR "Failed to configure host flatc")
endif()
execute_process(
	COMMAND ${CMAKE_COMMAND} --build "${LITERT_HOST_FLATC_BUILD_DIR}"
					--target flatc --parallel
	RESULT_VARIABLE LITERT_FLATC_BUILD_RESULT)
if(NOT LITERT_FLATC_BUILD_RESULT EQUAL 0)
	message(FATAL_ERROR "Failed to build host flatc")
endif()
set(FLATBUFFERS_FLATC_EXECUTABLE
		"${LITERT_HOST_FLATC_BUILD_DIR}/flatc" CACHE FILEPATH "Host flatc")
set(FLATBUFFERS_INCLUDE_DIR
	"${CMAKE_BINARY_DIR}/flatbuffers/include" CACHE PATH "FlatBuffers headers")

add_subdirectory("${LITERT_SDK_BUILD_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/litert_sdk")

add_library(victortestcpp_jni SHARED victortestcpp_jni.cc)
target_compile_features(victortestcpp_jni PRIVATE cxx_std_20)
target_compile_options(victortestcpp_jni PRIVATE -Wall -Werror -Wno-deprecated-declarations)
target_link_libraries(victortestcpp_jni PRIVATE litert_cc_api android log)
target_link_options(victortestcpp_jni PRIVATE "-Wl,-z,max-page-size=16384")