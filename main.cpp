#include <string>
#include <numeric>
#include <vector>
#include "inference.hpp"

using namespace std;

/**
 * 单张图片推理
 * @param model_path    模型路径
 * @param meta_path     超参数路径
 * @param image_path    图片路径
 * @param save_dir      保存路径
 * @param device        CPU or GPU 推理
 * @param openvino_preprocess   是否使用openvino图片预处理
 */
void single(string& model_path, string& meta_path, string& image_path, string& save_dir,
    string& device, bool openvino_preprocess = true) {
    // 1.创建推理器
    Inference inference = Inference(model_path, meta_path, device, openvino_preprocess);

    // 2.读取图片
    cv::Mat image = readImage(image_path);

    // time
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // 3.推理单张图片
    Result result = inference.infer(image);
    cout << "score: " << result.score << endl;

    // 4.生成其他图片(mask,mask边缘,热力图和原图的叠加)
    vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
    // time
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    cout << "infer time: " << end - start << " ms" << endl;

    // 5.保存显示图片
    // 将mask转化为3通道,不然没法拼接图片
    cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);

    saveScoreAndImages(result.score, images, image_path, save_dir);

    cv::imshow("result", images[2]);
    cv::waitKey(0);
}


/**
 * 多张图片推理
 * @param model_path    模型路径
 * @param meta_path     超参数路径
 * @param image_dir     图片文件夹路径
 * @param save_dir      保存路径
 * @param device        CPU or GPU 推理
 * @param openvino_preprocess   是否使用openvino图片预处理
 */
void multi(string& model_path, string& meta_path, string& image_dir, string& save_dir,
    string& device, bool openvino_preprocess = true) {
    // 1.创建推理器
    Inference inference = Inference(model_path, meta_path, device, openvino_preprocess);

    // 2.读取全部图片路径
    vector<cv::String> paths = getImagePaths(image_dir);

    vector<float> times;
    for (auto& image_path : paths) {
        // 3.读取单张图片
        cv::Mat image = readImage(image_path);

        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 4.推理单张图片
        Result result = inference.infer(image);
        cout << "score: " << result.score << endl;

        // 5.图片生成其他图片(mask,mask边缘,热力图和原图的叠加)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time: " << end - start << " ms" << endl;
        times.push_back(end - start);

        // 6.保存图片
        // 将mask转化为3通道,不然没法拼接图片
        cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
        saveScoreAndImages(result.score, images, image_path, save_dir);
    }

    // 6.统计数据
    double sumValue = accumulate(begin(times), end(times), 0.0); // accumulate函数就是求vector和的函数；
    double avgValue = sumValue / times.size();                   // 求均值
    cout << "avg infer time: " << avgValue << " ms" << endl;
}


int main() {
    // 注意使用非patchcore模型时报错可以查看utils.cpp中infer_height和infer_width中的[1] 都改为 [0]，具体查看注释和metadata.json文件
    string model_path = "D:/ml/code/anomalib/results/patchcore/mvtec/bottle/run/weights/openvino/model.xml";
    string param_path = "D:/ml/code/anomalib/results/patchcore/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-openvino-cpp/result"; // 注意目录不会自动创建,要手动创建才会保存
    // 是否使用openvino图片预处理
    bool openvino_preprocess = true;
    string device = "CPU";

    single(model_path, param_path, image_path, save_dir, device, openvino_preprocess);
    // multi(model_path, param_path, image_dir, save_dir, device, openvino_preprocess);
    return 0;
}
