cmake_minimum_required(VERSION 3.5)
project(Y5Detect_trt)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


# # CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "    compiler path: ${CMAKE_CUDA_COMPILER}")
enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# # opencv
find_package(OpenCV REQUIRED)
message(STATUS "Opencv library status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "> libraries: ${OpenCV_LIBS}")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS} ")
include_directories(${OpenCV_INCLUDE_DIRS})


# # TensorRT
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
set(Tensorrt_LIBS nvonnxparser nvinfer cudart)


include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE library_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.h ./*.hpp ./*.cpp)

# share
add_library(yolov5_trt SHARED ${library_src})
target_link_libraries(yolov5_trt ${OpenCV_LIBS} ${Tensorrt_LIBS})

# static
add_library(yolov5_trt_static STATIC ${library_src})
SET_TARGET_PROPERTIES(yolov5_trt_static PROPERTIES OUTPUT_NAME "yolov5_trt")
target_link_libraries(yolov5_trt_static ${OpenCV_LIBS} ${Tensorrt_LIBS})

# bin
file(GLOB_RECURSE sample_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test.cpp)
add_executable(Y5Detect_trt.bin ${sample_src})
target_link_libraries(Y5Detect_trt.bin yolov5_trt ${OpenCV_LIBS} ${Tensorrt_LIBS} )