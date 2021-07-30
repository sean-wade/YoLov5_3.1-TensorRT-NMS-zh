cmake_minimum_required(VERSION 3.10.2)
project(testv5)

find_package(OpenCV 4.1.1 REQUIRED)
message(STATUS "Opencv library status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "> libraries: ${OpenCV_LIBS}")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS} ")
message(STATUS "> libs: ${OpenCV_LIBRARY_DIRS} ")


find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64/)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(/usr/local/include)
link_directories(/usr/local/lib)


set(Boost_LIBRARIES 
    -lboost_thread
    -lboost_filesystem
    -lboost_system
)


#link_directories(../)


add_executable(tttt test.cpp)
target_link_libraries(tttt
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/libyolov5_trt.so
    #${PROJECT_SOURCE_DIR}/libyoloplugin.so
    )
