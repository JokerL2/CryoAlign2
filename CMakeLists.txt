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
set(TEST1_SOURCES main_cryoalign.cpp)

# 添加可执行文件
add_executable(CryoAlign ${TEST1_SOURCES})

# 链接到必要的库
target_link_libraries(CryoAlign
    alignment_lib 
    sample_cluster_lib
)


# 源文件
set(TEST2_SOURCES main_extract_keypoints.cpp)

# 添加可执行文件
add_executable(CryoAlign_extract_keypoints ${TEST2_SOURCES})

# 链接到必要的库
target_link_libraries(CryoAlign_extract_keypoints
    sample_cluster_lib
)

# 源文件
set(TEST3_SOURCES main_alignment.cpp)

# 添加可执行文件
add_executable(CryoAlign_alignment ${TEST3_SOURCES})

# 链接到必要的库
target_link_libraries(CryoAlign_alignment
    alignment_lib 
)

# 添加编译器优化和特定标志
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(CryoAlign PRIVATE -O3 -march=native)
	target_compile_options(CryoAlign_extract_keypoints PRIVATE -O3 -march=native)
	target_compile_options(CryoAlign_alignment PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(CryoAlign PRIVATE /O2 /arch:AVX)
	target_compile_options(CryoAlign_extract_keypoints PRIVATE /O2 /arch:AVX)
	target_compile_options(CryoAlign_alignment PRIVATE /O2 /arch:AVX)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(CryoAlign PRIVATE -g)
	target_compile_options(CryoAlign_extract_keypoints PRIVATE -g)
	target_compile_options(CryoAlign_alignment PRIVATE -g)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()
