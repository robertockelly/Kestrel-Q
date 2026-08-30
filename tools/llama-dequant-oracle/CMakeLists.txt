cmake_minimum_required(VERSION 3.24)
project(kq_llama_dequant_oracle LANGUAGES CXX)

if(NOT DEFINED LLAMA_CPP_SOURCE_DIR OR NOT DEFINED LLAMA_CPP_BUILD_DIR)
    message(FATAL_ERROR
        "Set LLAMA_CPP_SOURCE_DIR and LLAMA_CPP_BUILD_DIR to the pinned ignored checkout/build")
endif()

add_executable(kq_llama_dequant_oracle
    ../llama-dequant-oracle.cpp
)
target_compile_features(kq_llama_dequant_oracle PRIVATE cxx_std_17)
target_include_directories(kq_llama_dequant_oracle PRIVATE
    "${LLAMA_CPP_SOURCE_DIR}/ggml/include"
)
target_link_libraries(kq_llama_dequant_oracle PRIVATE
    "${LLAMA_CPP_BUILD_DIR}/ggml/src/Release/ggml-base.lib"
)
if(MSVC)
    target_compile_options(kq_llama_dequant_oracle PRIVATE /W4 /fp:strict)
endif()

add_custom_command(TARGET kq_llama_dequant_oracle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMA_CPP_BUILD_DIR}/bin/Release/ggml-base.dll"
        "$<TARGET_FILE_DIR:kq_llama_dequant_oracle>"
)
