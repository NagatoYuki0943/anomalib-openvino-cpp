#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <numeric>
#include <vector>
#include <Windows.h>
#include"utils.h"


using namespace std;


class Inference{
private:
    MetaData meta{};                    // 超参数
    ov::CompiledModel compiled_model;   // 编译好的模型
    ov::InferRequest infer_request;     // 推理请求
    string device;                      // 使用的设备
    bool openvino_preprocess;           // 是否使用openvino图片预处理

public:
    /**
     * @param model_path    模型路径
     * @param meta_path     超参数路径
     * @param device        CPU or GPU 推理
     * @param openvino_preprocess   是否使用openvino图片预处理
     */
    Inference(string& model_path, string& meta_path, string& device, bool openvino_preprocess){
        // 1.读取meta
        this->meta   = getJson(meta_path);
        this->device = device;
        this->openvino_preprocess = openvino_preprocess;
        // 2.创建模型
        this->get_openvino_model(model_path);
        // 3.创建推理请求
        this->infer_request = this->compiled_model.create_infer_request();
        // 4.模型预热
        this->warm_up();
    }

    /**
     * get openvino model
     * @param model_path
     */
    void get_openvino_model(string& model_path){
        vector<float> mean = {0.485 * 255, 0.456 * 255, 0.406 * 255};
        vector<float> std  = {0.229 * 255, 0.224 * 255, 0.225 * 255};

        // Step 1. Initialize OpenVINO Runtime core
        ov::Core core;
        // Step 2. Read a model
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        if(this->openvino_preprocess){
            // Step 4. Inizialize Preprocessing for the model
            // https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
            // https://blog.csdn.net/sandmangu/article/details/107181289
            // https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
            ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
            // Specify input image format   input(0) refers to the 0th input.
            ppp.input(0).tensor()
                    .set_element_type(ov::element::f32)                    // u8 -> f32
                    .set_layout(ov::Layout("NCHW"))         // NHWC -> NCHW
                    .set_color_format(ov::preprocess::ColorFormat::RGB);
            // Specify preprocess pipeline to input image without resizing
            ppp.input(0).preprocess()
                    .convert_element_type(ov::element::f32)
                    .mean(mean)
                    .scale(std);
            // Specify model's input layout
            ppp.input(0).model().set_layout(ov::Layout("NCHW"));
            // Specify output results format
            ppp.output(0).tensor().set_element_type(ov::element::f32);
            ppp.output(1).tensor().set_element_type(ov::element::f32);
            // Embed above steps in the graph
            model = ppp.build();
        }
        // 模型
        this->compiled_model = core.compile_model(model, this->device);
    }


    /**
     * 模型预热
     */
    void warm_up(){
        // 输入数据
        cv::Size size = cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::Mat input = cv::Mat(size, CV_8UC3, color);
        this->infer(input);
    }


    /**
     * 图片预处理
     * @param image 预处理图片
     * @return      经过预处理的图片
     */
    cv::Mat preProcess(cv::Mat& image) {
        vector<float> mean = {0.485, 0.456, 0.406};
        vector<float> std  = {0.229, 0.224, 0.225};

        // 缩放 w h
        cv::Mat resized_image = Resize(image, this->meta.infer_size[0], this->meta.infer_size[1], "bilinear");

        // 归一化
        // convertTo直接将所有值除以255,normalize的NORM_MINMAX是将原始数据范围变换到0~1之间,convertTo更符合深度学习的做法
        resized_image.convertTo(resized_image, CV_32FC3, 1.0/255, 0);
        //cv::normalize(resized_image, resized_image, 0, 1, cv::NormTypes::NORM_MINMAX, CV_32FC3);

        // 标准化
        resized_image = Normalize(resized_image, mean, std);
        return resized_image;
    }


    /**
     * 后处理部分,标准化热力图和得分,还原热力图到原图尺寸
     *
     * @param anomaly_map   未经过标准化的热力图
     * @param pred_score    未经过标准化的得分
     * @return result		热力图和得分vector
     */
    vector<cv::Mat> postProcess(cv::Mat& anomaly_map, cv::Mat& pred_score) {
        // 标准化热力图和得分
        anomaly_map = cvNormalizeMinMax(anomaly_map, this->meta.pixel_threshold, this->meta.min, this->meta.max);
        pred_score  = cvNormalizeMinMax(pred_score, this->meta.image_threshold, this->meta.min, this->meta.max);

        // 还原到原图尺寸
        anomaly_map = Resize(anomaly_map, this->meta.image_size[0], this->meta.image_size[1], "bilinear");

        // 返回热力图和得分
        return vector<cv::Mat>{anomaly_map, pred_score};
    }


    /**
     * 推理单张图片
     * @param image 原始图片
     * @return      叠加热力图的原图和得分
     */
    Result infer(cv::Mat& image){
        // 1.保存图片原始高宽
        this->meta.image_size[0] = image.size[0];
        this->meta.image_size[1] = image.size[1];

        // 2.图片预处理
        cv::Mat resized_image;
        if (this->openvino_preprocess){
            // 不需要resize,blobFromImage会resize
            resized_image = image;
        }else{
            resized_image = this->preProcess(image);
        }
        // [H, W, C] -> [N, C, H, W]
        // 这里只转换维度,其他预处理都做了,python版本是否使用openvino图片预处理都需要这一步,C++只是自己的预处理需要这一步
        // openvino如果使用这一步的话需要将输入的类型由 u8 转换为 f32, Layout 由 NHWC 改为 NCHW  (38, 39行)
        resized_image = cv::dnn::blobFromImage(resized_image, 1.0,
                                               {this->meta.infer_size[1], this->meta.infer_size[0]},
                                               {0, 0, 0},
                                               false, false, CV_32F);

        // 输入全为1测试
        // cv::Size size = cv::Size(224, 224);
        // cv::Scalar color = cv::Scalar(1, 1, 1);
        // resized_image = cv::Mat(size, CV_32FC3, color);

        // 3.从图像创建tensor
        auto *input_data = (float *) resized_image.data;
        ov::Tensor input_tensor = ov::Tensor(this->compiled_model.input().get_element_type(),
                                             this->compiled_model.input().get_shape(), input_data);

        // 4.推理
        this->infer_request.set_input_tensor(input_tensor);
        this->infer_request.infer();

        // 5.获取输出
        const ov::Tensor &result1 = this->infer_request.get_output_tensor(0);
        const ov::Tensor &result2 = this->infer_request.get_output_tensor(1);
        // cout << result1.get_shape() << endl;    //{1, 1, 224, 224}
        // cout << result2.get_shape() << endl;    //{1}

        // 6.将输出转换为Mat
        // result1.data<float>() 返回指针 放入Mat中不能解印用
        cv::Mat anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]),
                                      CV_32FC1, result1.data<float>());
        cv::Mat pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, result2.data<float>());
        cout << "pred_score: " << pred_score << endl;   // 4.0252275

        // 7.后处理:标准化,缩放到原图
        vector<cv::Mat> result = this->postProcess(anomaly_map, pred_score);
        anomaly_map = result[0];
        float score = result[1].at<float>(0, 0);

        // 8.叠加原图和热力图,从这里开始就是为了显示做准备了
        anomaly_map = superimposeAnomalyMap(anomaly_map, image);

        // 9.给图片添加分数
        anomaly_map = addLabel(anomaly_map, score);

        // 10.返回结果
        return Result {anomaly_map, score};
    }
};


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
            string& device, bool openvino_preprocess = true){
    // 1.创建推理器
    Inference inference = Inference(model_path, meta_path, device, openvino_preprocess);

    // 2.读取图片
    cv::Mat image = readImage(image_path);

    // 3.推理单张图片
    Result result = inference.infer(image);

    // 4.保存显示图片
    cout << "score: " << result.score << endl;
    saveScoreAndImage(result.score, result.anomaly_map, image_path, save_dir);
    cv::imshow("result", result.anomaly_map);
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
           string& device, bool openvino_preprocess = true){
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
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time:" << end - start << "ms" << endl;
        times.push_back(end - start);
        // 5.保存图片
        saveScoreAndImage(result.score, result.anomaly_map, image_path, save_dir);
    }

    // 6.统计数据
    double sumValue = accumulate(begin(times), end(times), 0.0);  // accumulate函数就是求vector和的函数；
    double meanValue = sumValue / times.size();                   // 求均值
    cout << "mean infer time: " << meanValue << endl;
}


int main(){
    string model_path = "D:/ai/code/abnormal/anomalib/results/patchcore/mvtec/bottle-cls/optimization/openvino/model.xml";
    string meta_path  = "D:/ai/code/abnormal/anomalib/results/patchcore/mvtec/bottle-cls/optimization/meta_data.json";
    string image_path = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ai/code/abnormal/anomalib-patchcore-openvino/cmake/result";
    // 是否使用openvino图片预处理
    bool openvino_preprocess = true;
    string device = "CPU";
    single(model_path, meta_path, image_path, save_dir,device, openvino_preprocess);
    // multi(model_path, meta_path, image_dir, save_dir, device, openvino_preprocess);
    return 0;
}
