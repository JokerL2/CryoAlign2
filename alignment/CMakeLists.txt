cmake_minimum_required(VERSION 3.10)
project(alignment)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# 查找 OpenMP
find_package(OpenMP REQUIRED)

# 查找 Boost
find_package(Boost REQUIRED COMPONENTS date_time program_options unit_test_framework serialization)

# 查找 Open3D
find_package(Open3D REQUIRED)

# 查找 Eigen
find_package(Eigen3 REQUIRED)

# 查找 teaser++
set(TEASERPP_INCLUDE_DIRS /usr/local/include/teaser)


# 设置 CNPY 的包含目录和库文件路径
set(CNPY_INCLUDE_DIRS /usr/local/include)
set(CNPY_LIBRARIES /usr/local/lib/libcnpy.so)


# 设置 FLANN 的包含目录和库文件路径
set(FLANN_INCLUDE_DIRS /usr/include/flann)
set(FLANN_LIBRARIES /usr/lib/x86_64-linux-gnu/libflann.so)

find_package(PCL REQUIRED)
# 包含头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${Open3D_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${TEASERPP_INCLUDE_DIRS}
    ${CNPY_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

# 源文件
set(SRCS Registration.cpp)

# 添加库
add_library(alignment_lib ${SRCS})

# 链接库
target_link_libraries(alignment_lib 
    OpenMP::OpenMP_CXX
    ${Boost_LIBRARIES}
    ${Open3D_LIBRARIES}
    Eigen3::Eigen
    teaser_io 
    teaser_registration
    ${CNPY_LIBRARIES}
    ${FLANN_LIBRARIES}
    ${PCL_LIBRARIES}
    stdc++fs
)
