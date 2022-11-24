# 说明

> 适用于anomalib导出的openvino格式的模型



# 安装openvino

## 下载安装

> https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html
>
> 下载2022.1版本，最新版本没提供对应的opencv
>
> 勾选C++
>
> 下载完成后安装
>
> [openvino文档](https://docs.openvino.ai/latest/)

## 下载对应的opencv

> 在`openvino_2022.1.0.643\extras\scripts\`文件夹下有`download_opencv.ps1`
>
> 运行会将opencv下载到`openvino_2022.1.0.643\extras\`目录下
>
> 可以运行opencv目录下的`ffmpeg-download.ps1`下载ffmpeg

## 配置环境变量

```yaml
#opencv
D:\ai\openvino\openvino_2022.1.0.643\extras\opencv\bin

#openvino
D:\ai\openvino\openvino_2022.1.0.643\runtime\bin\intel64\Debug
D:\ai\openvino\openvino_2022.1.0.643\runtime\bin\intel64\Release
D:\ai\openvino\openvino_2022.1.0.643\runtime\3rdparty\tbb\bin
```

# 错误

## 找不到`opencv_core_parallel_onetbb455_64d.dll`

> 没有问题，参考https://github.com/opencv/opencv/issues/20113
>
> debug模式下会显示，release不会显示



# 关于include文件夹

> include文件夹是rapidjson的文件，用来解析json



# Cmake

> cmake和vs的代码一致，指示引入方式有差别
>
> cmake版本要设置 `CMakeLists.txt` 中 opencv，openvino的路径为自己的路径

```cmake
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
```

# VS

> 使用vs2019

属性

- C/C++

  - 附加包含目录 release debug 都包含

    ```python
    D:\ai\openvino\openvino_2022.1.0.643\runtime\include
    D:\ai\openvino\openvino_2022.1.0.643\runtime\include\ie
    D:\ai\openvino\openvino_2022.1.0.643\runtime\include\ngraph
    D:\ai\openvino\openvino_2022.1.0.643\runtime\include\openvino
    D:\ai\openvino\openvino_2022.1.0.643\extras\opencv\include
    ..\include	# rapidjson 为相对目录,可以更改为绝对目录
    ```

- 链接器

  - 附加库目录 release debug 分开包含

    ```python
    # debug
    D:\ai\openvino\openvino_2022.1.0.643\runtime\lib\intel64\Debug
    D:\ai\openvino\openvino_2022.1.0.643\extras\opencv\lib

    # release
    D:\ai\openvino\openvino_2022.1.0.643\runtime\lib\intel64\Release
    D:\ai\openvino\openvino_2022.1.0.643\extras\opencv\lib
    ```

  - 输入

    - 附加依赖项 release debug 分开包含 (PS: opencv全包含了大多数,但是不知道哪些是用不到的)

      ```python
      # debug
      openvinod.lib
      openvino_cd.lib
      openvino_ir_frontendd.lib
      opencv_core455d.lib
      opencv_dnn455d.lib
      opencv_highgui455d.lib
      opencv_imgcodecs455d.lib
      opencv_imgproc455d.lib
      
      # release
      openvino.lib
      openvino_c.lib
      openvino_ir_frontend.lib
      opencv_core455.lib
      opencv_dnn455.lib
      opencv_highgui455.lib
      opencv_imgcodecs455.lib
      opencv_imgproc455.lib
      ```

# 错误

### c2760 意外标记 “）”

属性

- C/C++
  - 语言 符合模式改为否



# 待解决

## 1.自己的图片预处理结果不对

##  ~~2.vs使用cv::imshow会卡住~~

> 自己莫名其妙好了。。。



# 第三方库

处理json使用了rapidjson https://rapidjson.org/zh-cn/

opencv方式参考了mmdeploy https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
