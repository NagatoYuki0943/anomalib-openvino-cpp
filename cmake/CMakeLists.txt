cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

# opencv
set(OpenCV_DIR D:/ai/openvino/openvino_2022.1.0.643/extras/opencv/cmake)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBRARIES: " ${OpenCV_LIBRARIES})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# openvino
set(OpenVINO_DIR D:/ai/openvino/openvino_2022.1.0.643/runtime/cmake)
find_package(OpenVINO REQUIRED)
include_directories(D:/ai/openvino/openvino_2022.1.0.643/runtime/include)

# rapidjson 为相对目录,可以更改为绝对目录
include_directories(../include)

# 链接所有库，不指定cpp文件
link_libraries(${OpenCV_LIBRARIES}
                openvino::runtime)

add_executable(cpp main.cpp utils.cpp opencv_utils.cpp)
