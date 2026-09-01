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

add_library(kq_utf8proc STATIC
    third_party/utf8proc/utf8proc.c
)
target_include_directories(kq_utf8proc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/utf8proc
)
target_compile_definitions(kq_utf8proc PUBLIC UTF8PROC_STATIC)
if(MSVC)
    target_compile_options(kq_utf8proc PRIVATE /W0)
else()
    target_compile_options(kq_utf8proc PRIVATE -w)
endif()

add_library(kq_core
    src/kq_chat.c
    src/kq_file_win32.c
    src/kq_gdn.c
    src/kq_gdn_qwen38.c
    src/kq_gguf.c
    src/kq_layer.c
    src/kq_layer_qwen38.c
    src/kq_model.c
    src/kq_model_exec.c
    src/kq_moe.c
    src/kq_moe_qwen38.c
    src/kq_numeric.c
    src/kq_ple.c
    src/kq_ple_value.c
    src/kq_ple_value_qwen38.c
    src/kq_qsa.c
    src/kq_qsa_qwen38.c
    src/kq_runtime.c
    src/kq_sampling.c
    src/kq_sha256.c
    src/kq_status.c
    src/kq_tensor_view.c
    src/kq_tokenizer.c
    src/kq_weight_provider.c
)

target_include_directories(kq_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kq_core PRIVATE kq_utf8proc)

kq_enable_c_warnings(kq_core)

if(MSVC)
    set_property(SOURCE src/kq_numeric.c APPEND PROPERTY
        COMPILE_OPTIONS /fp:strict)
    set_property(SOURCE src/kq_gdn.c src/kq_gdn_qwen38.c
        src/kq_qsa.c src/kq_qsa_qwen38.c
        src/kq_moe.c src/kq_moe_qwen38.c
        src/kq_ple_value.c src/kq_ple_value_qwen38.c APPEND PROPERTY
        COMPILE_OPTIONS /fp:strict)
    set_property(SOURCE src/kq_layer.c src/kq_layer_qwen38.c
        src/kq_model_exec.c src/kq_weight_provider.c APPEND PROPERTY
        COMPILE_OPTIONS /fp:strict)
    set_property(SOURCE src/kq_sampling.c APPEND PROPERTY
        COMPILE_OPTIONS /fp:strict)
else()
    set_property(SOURCE src/kq_numeric.c APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
    set_property(SOURCE src/kq_gdn.c src/kq_gdn_qwen38.c
        src/kq_qsa.c src/kq_qsa_qwen38.c
        src/kq_moe.c src/kq_moe_qwen38.c
        src/kq_ple_value.c src/kq_ple_value_qwen38.c APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
    set_property(SOURCE src/kq_layer.c src/kq_layer_qwen38.c
        src/kq_model_exec.c src/kq_weight_provider.c APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
    set_property(SOURCE src/kq_sampling.c APPEND PROPERTY
        COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
endif()

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

add_executable(kq_run
    src/run_main.c
)
set_target_properties(kq_run PROPERTIES OUTPUT_NAME "kq-run")
target_link_libraries(kq_run PRIVATE kq_core)
kq_enable_c_warnings(kq_run)

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

    add_executable(kq_sha256_test tests/sha256_test.c)
    target_link_libraries(kq_sha256_test PRIVATE kq_core)
    target_include_directories(kq_sha256_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_sha256_test)
    add_test(NAME kq_sha256_known_vectors COMMAND kq_sha256_test)

    add_executable(kq_chat_test tests/chat_test.c)
    target_link_libraries(kq_chat_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_chat_test)
    add_test(NAME kq_chat_synthetic COMMAND kq_chat_test)

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

    add_executable(kq_model_exec_test tests/model_exec_test.c)
    target_link_libraries(kq_model_exec_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_model_exec_test)
    add_test(NAME kq_model_exec_synthetic COMMAND kq_model_exec_test)

    add_executable(kq_first_token_integration_test
        tests/first_token_integration.c)
    target_link_libraries(kq_first_token_integration_test PRIVATE kq_core)
    target_include_directories(kq_first_token_integration_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_first_token_integration_test)
    add_test(NAME kq_first_token_real_artifact
        COMMAND kq_first_token_integration_test)
    set_tests_properties(kq_first_token_real_artifact PROPERTIES
        SKIP_RETURN_CODE 77)
    add_executable(kq_multi_token_integration_test
        tests/multi_token_integration.c)
    target_link_libraries(kq_multi_token_integration_test PRIVATE kq_core)
    target_include_directories(kq_multi_token_integration_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_multi_token_integration_test)
    add_test(NAME kq_multi_token_real_artifact
        COMMAND kq_multi_token_integration_test)
    set_tests_properties(kq_multi_token_real_artifact PROPERTIES
        SKIP_RETURN_CODE 77)
    add_executable(kq_sampled_generation_integration_test
        tests/sampled_generation_integration.c)
    target_link_libraries(kq_sampled_generation_integration_test PRIVATE
        kq_core)
    target_include_directories(kq_sampled_generation_integration_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_sampled_generation_integration_test)
    add_test(NAME kq_sampled_generation_real_artifact
        COMMAND kq_sampled_generation_integration_test)
    set_tests_properties(kq_sampled_generation_real_artifact PROPERTIES
        SKIP_RETURN_CODE 77)
    add_test(NAME kq_run_rejects_vision
        COMMAND kq_run placeholder.gguf --prompt x --max-new-tokens 1
                --greedy --vision)
    add_test(NAME kq_run_rejects_mtp
        COMMAND kq_run placeholder.gguf --prompt x --max-new-tokens 1
                --greedy --mtp)
    add_test(NAME kq_run_rejects_zero_generation
        COMMAND kq_run placeholder.gguf --prompt x --max-new-tokens 0
                --greedy)
    add_test(NAME kq_run_rejects_sample_without_seed
        COMMAND kq_run placeholder.gguf --prompt x --max-new-tokens 1
                --sample)
    add_test(NAME kq_run_rejects_sample_invalid_stream
        COMMAND kq_run placeholder.gguf --prompt x --max-new-tokens 1
                --sample --seed 0 --stream 9223372036854775808)
    add_test(NAME kq_run_rejects_mode_conflict
        COMMAND kq_run placeholder.gguf --prompt x --max-new-tokens 1
                --greedy --seed 0)
    set_tests_properties(
        kq_run_rejects_vision
        kq_run_rejects_mtp
        kq_run_rejects_zero_generation
        kq_run_rejects_sample_without_seed
        kq_run_rejects_sample_invalid_stream
        kq_run_rejects_mode_conflict
        PROPERTIES
        WILL_FAIL TRUE)

    add_executable(kq_tensor_view_test tests/tensor_view_test.c)
    target_link_libraries(kq_tensor_view_test PRIVATE kq_core)
    target_include_directories(kq_tensor_view_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_tensor_view_test)
    add_test(NAME kq_tensor_view_synthetic COMMAND kq_tensor_view_test)

    add_executable(kq_tokenizer_integration_test tests/tokenizer_integration.c)
    target_link_libraries(kq_tokenizer_integration_test PRIVATE kq_core)
    target_include_directories(kq_tokenizer_integration_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_tokenizer_integration_test)
    add_test(NAME kq_tokenizer_real_artifact COMMAND kq_tokenizer_integration_test)
    set_tests_properties(kq_tokenizer_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(kq_tokenizer_probe tests/tokenizer_probe.c)
    target_link_libraries(kq_tokenizer_probe PRIVATE kq_core)
    kq_enable_c_warnings(kq_tokenizer_probe)

    add_executable(kq_ple_test tests/ple_test.c)
    target_link_libraries(kq_ple_test PRIVATE kq_core)
    target_include_directories(kq_ple_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_ple_test)
    add_test(NAME kq_ple_synthetic COMMAND kq_ple_test)

    add_executable(kq_ple_integration_test tests/ple_integration.c)
    target_link_libraries(kq_ple_integration_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_ple_integration_test)
    add_test(NAME kq_ple_real_artifact COMMAND kq_ple_integration_test)
    set_tests_properties(kq_ple_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(kq_ple_probe tests/ple_probe.c)
    target_link_libraries(kq_ple_probe PRIVATE kq_core)
    kq_enable_c_warnings(kq_ple_probe)

    add_executable(kq_numeric_test tests/numeric_test.c)
    target_link_libraries(kq_numeric_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_numeric_test)
    add_test(NAME kq_numeric_synthetic COMMAND kq_numeric_test)

    add_executable(kq_sampling_test tests/sampling_test.c)
    target_link_libraries(kq_sampling_test PRIVATE kq_core)
    target_include_directories(kq_sampling_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_sampling_test)
    add_test(NAME kq_sampling_synthetic COMMAND kq_sampling_test)

    add_executable(kq_sampling_probe tests/sampling_probe.c)
    target_link_libraries(kq_sampling_probe PRIVATE kq_core)
    target_include_directories(kq_sampling_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_sampling_probe)

    add_executable(kq_numeric_probe tests/numeric_probe.c)
    target_link_libraries(kq_numeric_probe PRIVATE kq_core)
    kq_enable_c_warnings(kq_numeric_probe)

    add_executable(kq_gdn_probe tests/gdn_probe.c)
    target_link_libraries(kq_gdn_probe PRIVATE kq_core)
    target_include_directories(kq_gdn_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_gdn_probe)

    add_executable(kq_gdn_test tests/gdn_test.c)
    target_link_libraries(kq_gdn_test PRIVATE kq_core)
    target_include_directories(kq_gdn_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_gdn_test)
    add_test(NAME kq_gdn_synthetic COMMAND kq_gdn_test)

    add_executable(kq_gdn_integration_test tests/gdn_integration.c)
    target_link_libraries(kq_gdn_integration_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_gdn_integration_test)
    add_test(NAME kq_gdn_real_artifact COMMAND kq_gdn_integration_test)
    set_tests_properties(kq_gdn_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(kq_qsa_test tests/qsa_test.c)
    target_link_libraries(kq_qsa_test PRIVATE kq_core)
    target_include_directories(kq_qsa_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_qsa_test)
    add_test(NAME kq_qsa_synthetic COMMAND kq_qsa_test)

    add_executable(kq_qsa_integration_test tests/qsa_integration.c)
    target_link_libraries(kq_qsa_integration_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_qsa_integration_test)
    add_test(NAME kq_qsa_real_artifact COMMAND kq_qsa_integration_test)
    set_tests_properties(kq_qsa_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(kq_qsa_probe tests/qsa_probe.c)
    target_link_libraries(kq_qsa_probe PRIVATE kq_core)
    target_include_directories(kq_qsa_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_qsa_probe)

    add_executable(kq_moe_test tests/moe_test.c)
    target_link_libraries(kq_moe_test PRIVATE kq_core)
    target_include_directories(kq_moe_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_moe_test)
    add_test(NAME kq_moe_synthetic COMMAND kq_moe_test)

    add_executable(kq_moe_integration_test tests/moe_integration.c)
    target_link_libraries(kq_moe_integration_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_moe_integration_test)
    add_test(NAME kq_moe_real_artifact COMMAND kq_moe_integration_test)
    set_tests_properties(kq_moe_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(kq_moe_probe tests/moe_probe.c)
    target_link_libraries(kq_moe_probe PRIVATE kq_core)
    target_include_directories(kq_moe_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_moe_probe)

    add_executable(kq_ple_value_test tests/ple_value_test.c)
    target_link_libraries(kq_ple_value_test PRIVATE kq_core)
    target_include_directories(kq_ple_value_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_ple_value_test)
    add_test(NAME kq_ple_value_synthetic COMMAND kq_ple_value_test)

    add_executable(kq_ple_value_integration_test tests/ple_value_integration.c)
    target_link_libraries(kq_ple_value_integration_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_ple_value_integration_test)
    add_test(NAME kq_ple_value_real_artifact COMMAND kq_ple_value_integration_test)
    set_tests_properties(kq_ple_value_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(kq_ple_value_probe tests/ple_value_probe.c)
    target_link_libraries(kq_ple_value_probe PRIVATE kq_core)
    target_include_directories(kq_ple_value_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_ple_value_probe)

    add_executable(kq_layer_test tests/layer_test.c)
    target_link_libraries(kq_layer_test PRIVATE kq_core)
    target_include_directories(kq_layer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_layer_test)
    add_test(NAME kq_layer_synthetic COMMAND kq_layer_test)

    add_executable(kq_layer_integration_test tests/layer_integration.c)
    target_link_libraries(kq_layer_integration_test PRIVATE kq_core)
    kq_enable_c_warnings(kq_layer_integration_test)
    add_test(NAME kq_layer_real_artifact COMMAND kq_layer_integration_test)
    set_tests_properties(kq_layer_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(kq_target_layer_integration_test
        tests/target_layer_integration.c)
    target_link_libraries(kq_target_layer_integration_test PRIVATE kq_core)
    target_include_directories(kq_target_layer_integration_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    kq_enable_c_warnings(kq_target_layer_integration_test)
    add_test(NAME kq_target_layer_real_artifact
        COMMAND kq_target_layer_integration_test)
    set_tests_properties(kq_target_layer_real_artifact PROPERTIES
        SKIP_RETURN_CODE 77)

    add_executable(kq_layer_probe tests/layer_probe.c)
    target_link_libraries(kq_layer_probe PRIVATE kq_core)
    target_include_directories(kq_layer_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    kq_enable_c_warnings(kq_layer_probe)

    add_executable(kq_numeric_integration_test tests/numeric_integration.c)
    target_link_libraries(kq_numeric_integration_test PRIVATE kq_core)
    target_include_directories(kq_numeric_integration_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    kq_enable_c_warnings(kq_numeric_integration_test)
    add_test(NAME kq_numeric_real_artifact COMMAND kq_numeric_integration_test)
    set_tests_properties(kq_numeric_real_artifact PROPERTIES SKIP_RETURN_CODE 77)

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
        add_test(
            NAME kq_tensor_view_geometry_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-tensor-view-geometry.py
                --inspect $<TARGET_FILE:kq_inspect>
                --inventory
                ${CMAKE_CURRENT_SOURCE_DIR}/research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/gguf-tensor-inventory.csv
        )
        set_tests_properties(kq_tensor_view_geometry_oracle PROPERTIES
            SKIP_RETURN_CODE 77
        )
        add_test(
            NAME kq_tokenizer_canonical_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-tokenizer.py
                --probe $<TARGET_FILE:kq_tokenizer_probe>
                --golden-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/goldens/Qwen3.8-Flash-Next
                --differential
                ${CMAKE_CURRENT_SOURCE_DIR}/research/tokenizer/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json
        )
        set_tests_properties(kq_tokenizer_canonical_oracle PROPERTIES
            SKIP_RETURN_CODE 77
        )
        add_test(
            NAME kq_ple_canonical_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-ple.py
                --probe $<TARGET_FILE:kq_ple_probe>
                --golden
                ${CMAKE_CURRENT_SOURCE_DIR}/research/goldens/Qwen3.8-Flash-Next/canonical/ple-address-vectors.json
                --differential
                ${CMAKE_CURRENT_SOURCE_DIR}/research/ple/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json
        )
        set_tests_properties(kq_ple_canonical_oracle PROPERTIES
            SKIP_RETURN_CODE 77
        )
        add_test(
            NAME kq_numeric_independent_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-numerics.py
                --probe $<TARGET_FILE:kq_numeric_probe>
                --evidence-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/numerics/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
        )
        add_test(
            NAME kq_gdn_independent_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-gdn.py
                --probe $<TARGET_FILE:kq_gdn_probe>
                --evidence-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
        )
        add_test(
            NAME kq_qsa_independent_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-qsa.py
                --probe $<TARGET_FILE:kq_qsa_probe>
                --evidence-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
                --verify
        )
        add_test(
            NAME kq_moe_independent_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-moe.py
                --probe $<TARGET_FILE:kq_moe_probe>
                --evidence-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
                --verify
        )
        add_test(
            NAME kq_ple_value_independent_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-ple-value.py
                --probe $<TARGET_FILE:kq_ple_value_probe>
                --evidence-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
                --verify
        )
        add_test(
            NAME kq_layer_independent_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-layer.py
                --probe $<TARGET_FILE:kq_layer_probe>
                --evidence-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
                --verify
        )
        add_test(
            NAME kq_target_layer_independent_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-target-layer.py
                --probe $<TARGET_FILE:kq_target_layer_integration_test>
                --evidence-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
                --verify
        )
        set_tests_properties(kq_target_layer_independent_oracle PROPERTIES
            SKIP_RETURN_CODE 77)
        add_test(
            NAME kq_sampling_independent_oracle
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate-native-sampling.py
                --probe $<TARGET_FILE:kq_sampling_probe>
                --evidence-dir
                ${CMAKE_CURRENT_SOURCE_DIR}/research/sampling/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
                --verify
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
