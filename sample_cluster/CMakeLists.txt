cmake_minimum_required(VERSION 3.0)
project(sample_cluster)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework serialization)
find_package(CUDA REQUIRED)
find_package(Armadillo REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.2")

# 指定CUDA编译器的路径
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc")

if(LIBTORCH_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake/Torch")
elseif(DEFINED ENV{LIBTORCH_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}/share/cmake/Torch")
endif()
find_package(Torch REQUIRED)
foreach(_torch_target torch torch_cpu torch_cuda c10 c10_cuda)
    if(TARGET ${_torch_target})
        get_target_property(_is_imported ${_torch_target} IMPORTED)
        if(_is_imported)
            set_target_properties(${_torch_target} PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endif()
endforeach()

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
    ${CMAKE_CURRENT_SOURCE_DIR}/vectorize
)

# 源文件
set(SOURCES
    VoxEM.cpp
    vectorize/mrc.c
    vectorize/sample_vectorize.c
)

# 添加库
add_library(sample_cluster_lib ${SOURCES})

# 链接到必要的库
target_link_libraries(sample_cluster_lib 
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    ${ARMADILLO_LIBRARIES}
    ${MLPACK_LIBRARIES}
    m
    stdc++fs
)
if(TARGET OpenMP::OpenMP_C)
    target_link_libraries(sample_cluster_lib OpenMP::OpenMP_C)
endif()

add_executable(Sample vectorize/main.c vectorize/mrc.c)
set_target_properties(Sample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(Sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vectorize)
target_link_libraries(Sample m)
if(TARGET OpenMP::OpenMP_C)
    target_link_libraries(Sample OpenMP::OpenMP_C)
endif()

# 添加编译器优化和特定标志
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sample_cluster_lib PRIVATE -O3 -march=native)
    target_compile_options(Sample PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(sample_cluster_lib PRIVATE /O2 /arch:AVX)
    target_compile_options(Sample PRIVATE /O2 /arch:AVX)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sample_cluster_lib PRIVATE -g)
    target_compile_options(Sample PRIVATE -g)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()
