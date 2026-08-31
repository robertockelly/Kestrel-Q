cmake_minimum_required(VERSION 3.24)
project(kq_llama_multi_token_oracle LANGUAGES CXX)

if(NOT DEFINED LLAMA_SOURCE_ROOT OR NOT DEFINED LLAMA_BUILD_ROOT)
    message(FATAL_ERROR "LLAMA_SOURCE_ROOT and LLAMA_BUILD_ROOT are required")
endif()

foreach(component IN ITEMS llama ggml ggml-base ggml-cpu)
    string(REPLACE "-" "_" target_component "${component}")
    add_library(kq_oracle_${target_component} SHARED IMPORTED)
endforeach()
set_target_properties(kq_oracle_llama PROPERTIES
    IMPORTED_IMPLIB "${LLAMA_BUILD_ROOT}/src/Release/llama.lib"
    IMPORTED_LOCATION "${LLAMA_BUILD_ROOT}/bin/Release/llama.dll")
set_target_properties(kq_oracle_ggml PROPERTIES
    IMPORTED_IMPLIB "${LLAMA_BUILD_ROOT}/ggml/src/Release/ggml.lib"
    IMPORTED_LOCATION "${LLAMA_BUILD_ROOT}/bin/Release/ggml.dll")
set_target_properties(kq_oracle_ggml_base PROPERTIES
    IMPORTED_IMPLIB "${LLAMA_BUILD_ROOT}/ggml/src/Release/ggml-base.lib"
    IMPORTED_LOCATION "${LLAMA_BUILD_ROOT}/bin/Release/ggml-base.dll")
set_target_properties(kq_oracle_ggml_cpu PROPERTIES
    IMPORTED_IMPLIB "${LLAMA_BUILD_ROOT}/ggml/src/Release/ggml-cpu.lib"
    IMPORTED_LOCATION "${LLAMA_BUILD_ROOT}/bin/Release/ggml-cpu.dll")

add_executable(kq_llama_multi_token_oracle
    ../llama-multi-token-oracle.cpp)
target_compile_features(kq_llama_multi_token_oracle PRIVATE cxx_std_17)
target_include_directories(kq_llama_multi_token_oracle PRIVATE
    "${LLAMA_SOURCE_ROOT}/include"
    "${LLAMA_SOURCE_ROOT}/ggml/include")
target_link_libraries(kq_llama_multi_token_oracle PRIVATE
    kq_oracle_llama kq_oracle_ggml kq_oracle_ggml_base
    kq_oracle_ggml_cpu psapi)

if(MSVC)
    target_compile_options(kq_llama_multi_token_oracle PRIVATE /W4 /EHsc)
else()
    target_compile_options(kq_llama_multi_token_oracle PRIVATE
        -Wall -Wextra -Wpedantic)
endif()
