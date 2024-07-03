cmake_minimum_required(VERSION 3.0)
project(sample_cluster)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework serialization)
find_package(CUDA REQUIRED)
find_package(Armadillo REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.2")

# 指定CUDA编译器的路径
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc")

list(APPEND CMAKE_PREFIX_PATH "~/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

# 手动设置 mlpack 的包含目录和库路径
find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack)

# 包含头文件目录
include_directories(
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${MLPACK_INCLUDE_DIRS}
)

# 源文件
set(SOURCES  VoxEM.cpp)

# 添加库
add_library(sample_cluster_lib ${SOURCES})

# 链接到必要的库
target_link_libraries(sample_cluster_lib 
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    ${ARMADILLO_LIBRARIES}
    ${MLPACK_LIBRARIES}
    stdc++fs
)

# 添加编译器优化和特定标志
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sample_cluster_lib PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(sample_cluster_lib PRIVATE /O2 /arch:AVX)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sample_cluster_lib PRIVATE -g)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()
