cmake_minimum_required(VERSION 3.5)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(Tensorrt_LIBS nvinfer nvinfer_plugin cudart)

# libyoloplugin.so
cuda_add_library(yoloplugin SHARED ${PROJECT_SOURCE_DIR}/src/yololayer.cu)
target_link_libraries(yoloplugin ${Tensorrt_LIBS})


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
file(GLOB_RECURSE library_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.h src/*.cpp)

# libyolov5_trt.so
add_library(yolov5_trt SHARED ${library_src})
target_link_libraries(yolov5_trt ${OpenCV_LIBS} ${Tensorrt_LIBS})

# libyolov5_trt.a
add_library(yolov5_trt_static STATIC ${library_src})
SET_TARGET_PROPERTIES(yolov5_trt_static PROPERTIES OUTPUT_NAME "yolov5_trt")
target_link_libraries(yolov5_trt_static ${OpenCV_LIBS} ${Tensorrt_LIBS})

# tttt
add_executable(tttt ${PROJECT_SOURCE_DIR}/test.cpp)
target_link_libraries(tttt yoloplugin yolov5_trt ${Tensorrt_LIBS} )

# add_definitions(-O2 -pthread)

