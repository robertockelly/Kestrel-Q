cmake_minimum_required(VERSION 3.24)

project(kestrel_q
    VERSION 0.0.1
    LANGUAGES C
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(KQ_ENABLE_CUDA "Enable the CUDA backend when implemented" OFF)
option(KQ_BUILD_TESTS "Build tests" ON)

add_library(kq_core
    src/kq_runtime.c
)

target_include_directories(kq_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(kq_core PRIVATE /W4)
else()
    target_compile_options(kq_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(kq
    src/main.c
)

target_link_libraries(kq PRIVATE kq_core)

if(KQ_BUILD_TESTS)
    enable_testing()
    add_executable(kq_smoke_test tests/smoke.c)
    target_link_libraries(kq_smoke_test PRIVATE kq_core)
    add_test(NAME kq_smoke COMMAND kq_smoke_test)
endif()
