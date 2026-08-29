cmake_minimum_required(VERSION 3.24)

project(kestrel_q
    VERSION 0.0.1
    LANGUAGES C
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(KQ_ENABLE_CUDA "Enable the CUDA smoke backend" OFF)
option(KQ_BUILD_TESTS "Build tests" ON)

function(kq_enable_c_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

function(kq_configure_cuda_target target)
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )

    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
        )
    endif()
endfunction()

add_library(kq_core
    src/kq_file_win32.c
    src/kq_gguf.c
    src/kq_model.c
    src/kq_runtime.c
    src/kq_status.c
)

target_include_directories(kq_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

kq_enable_c_warnings(kq_core)

add_executable(kq
    src/main.c
)

target_link_libraries(kq PRIVATE kq_core)
kq_enable_c_warnings(kq)

add_executable(kq_inspect
    src/inspect_main.c
)
set_target_properties(kq_inspect PROPERTIES OUTPUT_NAME "kq-inspect")
target_link_libraries(kq_inspect PRIVATE kq_core)
kq_enable_c_warnings(kq_inspect)

if(KQ_ENABLE_CUDA)
    enable_language(CUDA)

    add_library(kq_cuda_backend STATIC
        src/cuda/kq_cuda_smoke.cu
    )

    target_include_directories(kq_cuda_backend
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    kq_configure_cuda_target(kq_cuda_backend)

    set(KQ_GIT_COMMIT "unknown")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE KQ_GIT_REVISION_RESULT
            OUTPUT_VARIABLE KQ_GIT_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(NOT KQ_GIT_REVISION_RESULT EQUAL 0)
            set(KQ_GIT_COMMIT "unknown")
        else()
            execute_process(
                COMMAND ${GIT_EXECUTABLE} status --porcelain
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE KQ_GIT_STATUS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(NOT KQ_GIT_STATUS STREQUAL "")
                string(APPEND KQ_GIT_COMMIT "-dirty")
            endif()
        endif()
    endif()

    add_executable(kq_cuda_bandwidth
        bench/cuda_bandwidth.cu
    )
    kq_configure_cuda_target(kq_cuda_bandwidth)
    target_compile_definitions(kq_cuda_bandwidth PRIVATE
        KQ_GIT_COMMIT="${KQ_GIT_COMMIT}"
    )
endif()

if(KQ_BUILD_TESTS)
    enable_testing()
    add_executable(kq_smoke_test tests/smoke.c)
    target_link_libraries(kq_smoke_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_smoke_test)
    add_test(NAME kq_smoke COMMAND kq_smoke_test)

    add_executable(kq_gguf_test tests/gguf_test.c)
    target_link_libraries(kq_gguf_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_gguf_test)
    add_test(NAME kq_gguf_synthetic COMMAND kq_gguf_test)

    add_executable(kq_model_test tests/model_test.c)
    target_link_libraries(kq_model_test PRIVATE kq_core)
    target_include_directories(kq_model_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_model_test)
    add_test(NAME kq_model_semantic_synthetic COMMAND kq_model_test)

    add_executable(kq_gguf_integration_test tests/gguf_integration.c)
    target_link_libraries(kq_gguf_integration_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_gguf_integration_test)
    add_test(NAME kq_gguf_real_artifact COMMAND kq_gguf_integration_test)
    set_tests_properties(kq_gguf_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME kq_model_semantic_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-semantic-registry.py
                --inspect $<TARGET_FILE:kq_inspect>
                --mapping
                ${CMAKE_CURRENT_SOURCE_DIR}/research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/canonical-gguf-mapping.csv
        )
        set_tests_properties(kq_model_semantic_oracle PROPERTIES
            SKIP_RETURN_CODE 77
        )
    endif()

    if(KQ_ENABLE_CUDA)
        add_executable(kq_cuda_smoke_test tests/cuda_smoke.c)
        target_link_libraries(kq_cuda_smoke_test PRIVATE kq_cuda_backend)
        kq_enable_c_warnings(kq_cuda_smoke_test)
        add_test(NAME kq_cuda_smoke COMMAND kq_cuda_smoke_test)

        add_test(
            NAME kq_cuda_bandwidth_self_test
            COMMAND kq_cuda_bandwidth --self-test --machine-id KQ-01
        )
    endif()
endif()
