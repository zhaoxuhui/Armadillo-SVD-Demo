cmake_minimum_required(VERSION 3.11)
project(armadillo_demo)

set(CMAKE_CXX_STANDARD 11)

find_package(Armadillo)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(svd_demo svd_demo.cpp)
target_link_libraries(svd_demo ${ARMADILLO_LIB_DIRS} armadillo)

add_executable(mat_convert mat_convert.cpp)
target_link_libraries(mat_convert ${OpenCV_LIBS} ${ARMADILLO_LIB_DIRS} armadillo)

add_executable(comare_performance compare_performance.cpp)
target_link_libraries(comare_performance ${ARMADILLO_LIB_DIRS} armadillo)