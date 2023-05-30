#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // patchcore模型训练配置文件删除了center_crop
    string model_path = "D:/ml/code/anomalib/results/patchcore/mvtec/bottle/run/weights/openvino/model.xml";
    string meta_path  = "D:/ml/code/anomalib/results/patchcore/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-openvino-cpp/result"; // 注意目录不会自动创建,要手动创建才会保存
    string device     = "CPU";
    bool openvino_preprocess = true;    // 是否使用openvino图片预处理

    // 创建推理器
    auto inference = Inference(model_path, meta_path, device, openvino_preprocess);

    // 单张图片推理
    cv::Mat result = inference.single(image_path, save_dir);
    cv::imshow("result", result);
    cv::waitKey(0);

    // 多张图片推理
    // inference.multi(image_dir, save_dir);
    return 0;
}
