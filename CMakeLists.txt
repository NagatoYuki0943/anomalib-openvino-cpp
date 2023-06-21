cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR D:/ml/opencv/build)
set(OpenVINO_DIR D:/ml/w_openvino_toolkit_windows/runtime/cmake)

# rapidjson 为相对目录,可以更改为绝对目录
include_directories(include)

# opencv
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBS: " ${OpenCV_LIBS})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# openvino
find_package(OpenVINO REQUIRED)
include_directories(openvino::include_directories)

# 链接所有库，不指定cpp文件
link_libraries(${OpenCV_LIBS}
                openvino::runtime)

add_executable(main main.cpp utils.cpp opencv_utils.cpp inference.hpp)
