# 说明

> 适用于anomalib导出的openvino格式的模型
>
> 测试了 patchcore和fastflow模型

```yaml
# 模型配置文件中设置为openvino,导出openvino会导出onnx
optimization:
  export_mode: openvino # options: torch, onnx, openvino
```

# 安装openvino

## 下载安装

> https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html
>
> 下载最新版
>
> 勾选C++
>
> 下载完成后安装
>
> [openvino文档](https://docs.openvino.ai/latest/)

## 下载opencv

> https://opencv.org

## 配置环境变量

```yaml
#opencv
$opencv_path\build\x64\vc16\bin

#openvino
$openvino_path\runtime\bin\intel64\Debug
$openvino_path\runtime\bin\intel64\Release
$openvino_path\runtime\3rdparty\tbb\bin
```

# 关于include文件夹

> include文件夹是rapidjson的文件，用来解析json

# Cmake

> cmake和vs的代码一致，指示引入方式有差别
>
> cmake版本要设置 `CMakeLists.txt` 中 opencv，openvino的路径为自己的路径

# 查看是否缺失dll

> https://github.com/lucasg/Dependencies 这个工具可以查看exe工具是否缺失dll

# 第三方库

处理json使用了rapidjson https://rapidjson.org/zh-cn/

opencv方式参考了mmdeploy https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
