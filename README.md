# 说明

> 适用于anomalib导出的openvino格式的模型
>
> 测试了patchcore,fastflow和efficient_ad模型

```yaml
# 模型配置文件中设置为openvino,导出openvino会导出onnx
optimization:
  export_mode: openvino # options: torch, onnx, openvino
```

# 注意事项

## patchcore
> patchcore模型训练配置文件需要调整center_crop为 `center_crop: null`
> 只缩放图片，不再剪裁图片
>

## efficient_ad

> 使用efficient_ad模型需要给 `Inference` 添加 `bool efficient_ad = true`参数
> 原因是efficient_ad模型的标准化在模型中做了，不需要在外部再做
> 

# 其他推理方式

> [anomalib-onnxruntime-cpp](https://github.com/NagatoYuki0943/anomalib-onnxruntime-cpp)
>
> [anomalib-openvino-cpp](https://github.com/NagatoYuki0943/anomalib-openvino-cpp)
>
> [anomalib-tensorrt-cpp](https://github.com/NagatoYuki0943/anomalib-tensorrt-cpp)

# example

```C++
#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // patchcore模型训练配置文件调整center_crop为 `center_crop: null`
    string model_path = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/model.xml";
    string meta_path  = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-openvino-cpp/result"; // 注意目录不会自动创建,要手动创建才会保存
    string device     = "CPU";
    bool openvino_preprocess = true;    // 是否使用openvino图片预处理,使用dynamic shape必须要用openvino_preprocess
    bool efficient_ad = true;           // 是否使用efficient_ad模型 vs2022 debug模式使用efficient_ad模型载入会失败,release模式成功

    // 创建推理器
    auto inference = Inference(model_path, meta_path, device, openvino_preprocess, efficient_ad);

    // 单张图片推理
    cv::Mat image = readImage(image_path);
    Result result = inference.single(image);
    saveScoreAndImages(result.score, result.anomaly_map, image_path, save_dir);
    cv::resize(result.anomaly_map, result.anomaly_map, { 2000, 500 });
    cv::imshow("result", result.anomaly_map);
    cv::waitKey(0);

    // 多张图片推理
    inference.multi(image_dir, save_dir);
    return 0;
}
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

> 设置 `CMakeLists.txt` 中 opencv，openvino的路径为自己的路径

# 查看是否缺失dll

> https://github.com/lucasg/Dependencies 这个工具可以查看exe工具是否缺失dll

# 第三方库

处理json使用了rapidjson https://rapidjson.org/zh-cn/

opencv方式参考了mmdeploy https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
