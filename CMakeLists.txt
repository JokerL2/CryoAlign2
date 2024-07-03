cmake_minimum_required(VERSION 3.10)

project(test_project)
# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
# 包含子目录
add_subdirectory(alignment)
add_subdirectory(sample_cluster)

# 包含头文件目录
include_directories(
    alignment
    sample_cluster
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# 源文件
set(TEST_SOURCES main.cpp)

# 添加可执行文件
add_executable(CryoAlign ${TEST_SOURCES})

# 链接到必要的库
target_link_libraries(CryoAlign 
    alignment_lib 
    sample_cluster_lib
)

# 添加编译器优化和特定标志
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(CryoAlign PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(CryoAlign PRIVATE /O2 /arch:AVX)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(CryoAlign PRIVATE -g)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()
