#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <numeric>
#include <vector>
#include <Windows.h>
#include"opencv_utils.h"
#include"utils.h"


using namespace std;


/**
 * 推理单张图片
 * @param compiled_model    编译后的模型
 * @param infer_request     推理请求
 * @param image             原始图片
 * @param meta              超参数
 * @param openvino_preprocess 是否使用openvino图片预处理,目前只能使用它的预处理,自己写的有问题
 * @return                  叠加热力图的原图和得分
 */
Result infer(ov::CompiledModel& compiled_model, ov::InferRequest& infer_request,
             cv::Mat& image, MetaData& meta, bool openvino_preprocess){
    // 1.保存图片原始高宽
    meta.image_size[0] = image.size[0];
    meta.image_size[1] = image.size[1];

    // 2.图片预处理
    cv::Mat resized_image;
    if (openvino_preprocess){
        // 不需要resize,blobFromImage会resize
        resized_image = image;
    }else{
        resized_image = preProcess(image, meta);
    }
    // [H, W, C] -> [N, C, H, W]
    // 这里只转换维度,其他预处理都做了,python版本是否使用openvino图片预处理都需要这一步,C++只是自己的预处理需要这一步
    // openvino如果使用这一步的话需要将输入的类型由 u8 转换为 f32, Layout 由 NHWC 改为 NCHW  (38, 39行)
    resized_image = cv::dnn::blobFromImage(resized_image, 1.0,
                                           {meta.infer_size[1], meta.infer_size[0]},
                                           {0, 0, 0},
                                           false, false, CV_32F);

    // 输入全为1测试
    // cv::Size size = cv::Size(224, 224);
    // cv::Scalar color = cv::Scalar(1, 1, 1);
    // resized_image = cv::Mat(size, CV_32FC3, color);

    // 3.从图像创建tensor
    auto *input_data = (float *) resized_image.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);

    // 4.推理
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // 5.获取输出
    const ov::Tensor &result1 = infer_request.get_output_tensor(0);
    const ov::Tensor &result2 = infer_request.get_output_tensor(1);
    // cout << result1.get_shape() << endl;    //{1, 1, 224, 224}
    // cout << result2.get_shape() << endl;    //{1}

    // 6.将输出转换为Mat
    // result1.data<float>() 返回指针 放入Mat中不能解印用
    cv::Mat anomaly_map = cv::Mat(cv::Size(meta.infer_size[1], meta.infer_size[0]), CV_32FC1, result1.data<float>());
    cv::Mat pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, result2.data<float>());
    cout << "pred_score: " << pred_score << endl;   // 4.0252275

    // 7.后处理:标准化,缩放到原图
    vector<cv::Mat> result = postProcess(anomaly_map, pred_score, meta);
    anomaly_map = result[0];
    float score = result[1].at<float>(0, 0);

    // 8.叠加原图和热力图,从这里开始就是为了显示做准备了
    anomaly_map = superimposeAnomalyMap(anomaly_map, image);

    // 9.给图片添加分数
    anomaly_map = addLabel(anomaly_map, score);

    // 10.返回结果
    return Result {anomaly_map, score};
}


/**
 * get      openvino model
 * @param   超参数
 * @param   model_path
 * @return  CompiledModel
 */
ov::CompiledModel get_openvino_model(string& model_path, MetaData& meta,  string& device, bool openvino_preprocess=false){
    vector<float> mean = {0.485 * 255, 0.456 * 255, 0.406 * 255};
    vector<float> std  = {0.229 * 255, 0.224 * 255, 0.225 * 255};

    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    if(openvino_preprocess){
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

    ov::CompiledModel compiled_model = core.compile_model(model, device);

    // **模型预热**
    // 输入数据
    cv::Size size = cv::Size(meta.infer_size[1], meta.infer_size[0]);
    cv::Scalar color = cv::Scalar(0, 0, 0);
    cv::Mat input = cv::Mat(size, CV_8UC3, color);
    // 推理请求
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer(compiled_model, infer_request, input, meta, openvino_preprocess);

    return compiled_model;
}


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
    // 1.读取meta
    MetaData meta = getJson(meta_path);

    // 2.获取模型
    ov::CompiledModel compiled_model = get_openvino_model(model_path, meta, device, openvino_preprocess);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 3.读取图片
    cv::Mat image = readImage(image_path);

    // 4.推理单张图片
    Result result = infer(compiled_model, infer_request, image, meta, openvino_preprocess);

    // 5.保存显示图片
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
void multi(string& model_path, string& meta_path, string& image_dir, string& save_dir, string& device, bool openvino_preprocess = true){
    // 1.读取meta
    MetaData meta = getJson(meta_path);

    // 2.获取模型
    ov::CompiledModel compiled_model = get_openvino_model(model_path, meta, device, openvino_preprocess);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 3.读取全部图片路径
    vector<cv::String> paths = getImagePaths(image_dir);

    vector<float> times;
    for (auto& image_path : paths) {
        // 4.读取单张图片
        cv::Mat image = readImage(image_path);

        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 5.推理单张图片
        Result result = infer(compiled_model, infer_request, image, meta, openvino_preprocess);
        cout << "score: " << result.score << endl;
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time:" << end - start << "ms" << endl;
        times.push_back(end - start);
        // 6.保存图片
        saveScoreAndImage(result.score, result.anomaly_map, image_path, save_dir);
    }

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
    //multi(model_path, meta_path, image_dir, save_dir, device, openvino_preprocess);
    return 0;
}
