cmake_minimum_required(VERSION 3.22.1)
project(victortestcpp_jni CXX)

add_library(victortestcpp_jni SHARED victortestcpp_jni.cc)
target_compile_features(victortestcpp_jni PRIVATE cxx_std_20)
target_compile_options(victortestcpp_jni PRIVATE -Wall -Werror)
target_link_libraries(victortestcpp_jni PRIVATE android log)
target_link_options(victortestcpp_jni PRIVATE "-Wl,-z,max-page-size=16384")