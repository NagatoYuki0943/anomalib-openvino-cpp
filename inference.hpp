#pragma once

#include <string>
#include <numeric>
#include <vector>
#include <Windows.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "utils.h"

using namespace std;

/**
 * input(0)/output(0) 按照id找指定的输入输出，不指定找全部的输入输出
 *
 *  input().tensor()       有7个方法
 *  ppp.input().tensor().set_color_format().set_element_type().set_layout()
 *                      .set_memory_type().set_shape().set_spatial_dynamic_shape().set_spatial_static_shape();
 *
 *  output().tensor()      有2个方法
 *  ppp.output().tensor().set_layout().set_element_type();
 *
 *  input().preprocess()   有8个方法
 *  ppp.input().preprocess().convert_color().convert_element_type().mean().scale()
 *                          .convert_layout().reverse_channels().resize().custom();
 *
 *  output().postprocess() 有3个方法
 *  ppp.output().postprocess().convert_element_type().convert_layout().custom();
 *
 *  input().model()  只有1个方法
 *  ppp.input().model().set_layout();
 *
 *  output().model() 只有1个方法
 *  ppp.output().model().set_layout();
 **/

class Inference {
private:
    bool openvino_preprocess;           // 是否使用openvino图片预处理
    MetaData meta{};                    // 超参数
    ov::CompiledModel compiled_model;   // 编译好的模型
    ov::InferRequest infer_request;     // 推理请求
    vector<ov::Output<const ov::Node>> inputs;  // 模型的输入列表名称
    vector<ov::Output<const ov::Node>> outputs; // 模型的输出列表名称

public:
    /**
     * @param model_path    模型路径
     * @param meta_path     超参数路径
     * @param device        CPU or GPU 推理
     * @param openvino_preprocess   是否使用openvino图片预处理
     */
    Inference(string& model_path, string& meta_path, string& device, bool openvino_preprocess) {
        this->openvino_preprocess = openvino_preprocess;
        // 1.读取meta
        this->meta = getJson(meta_path);
        // 2.创建模型
        this->compiled_model = this->get_model(model_path, device);
        // 3.获取模型的输入输出
        this->inputs = this->compiled_model.inputs();
        this->outputs = this->compiled_model.outputs();
        // 4.创建推理请求
        this->infer_request = this->compiled_model.create_infer_request();
        // 5.模型预热
        this->warm_up();
    }

    /**
     * get openvino model
     * @param model_path 模型路径
     * @param device     使用的设备
     */
    ov::CompiledModel get_model(string& model_path, string& device) const {
        vector<float> mean = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
        vector<float> std = { 0.229 * 255, 0.224 * 255, 0.225 * 255 };

        // Step 1. Initialize OpenVINO Runtime core
        ov::Core core;
        // Step 2. Read a Model from a Drive
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        if (this->openvino_preprocess) {
            // Step 3. Inizialize Preprocessing for the model
            // https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
            // https://blog.csdn.net/sandmangu/article/details/107181289
            // https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
            ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

            // Specify input image format
            ppp.input(0).tensor()
                .set_color_format(ov::preprocess::ColorFormat::RGB)     // BGR -> RGB
                .set_element_type(ov::element::f32)                     // u8 -> f32
                .set_layout(ov::Layout("NCHW"));                        // NHWC -> NCHW

        // Specify preprocess pipeline to input image without resizing
            ppp.input(0).preprocess()
                //  .convert_color(ov::preprocess::ColorFormat::RGB)
                //  .convert_element_type(ov::element::f32)
                .mean(mean)
                .scale(std);

            // Specify model's input layout
            ppp.input(0).model().set_layout(ov::Layout("NCHW"));

            // Specify output results format
            ppp.output(0).tensor().set_element_type(ov::element::f32);
            if (model->outputs().size() == 2) {
                ppp.output(1).tensor().set_element_type(ov::element::f32);
            }

            // Embed above steps in the graph
            model = ppp.build();

        }
        // Step 4. Load the Model to the Device
        return core.compile_model(model, device);
    }

    /**
     * 模型预热
     */
    void warm_up() {
        // 输入数据
        cv::Size size = cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::Mat input = cv::Mat(size, CV_8UC3, color);
        this->infer(input);
    }

    /**
     * 推理单张图片
     * @param image 原始图片
     * @return      标准化的并所放到原图热力图和得分
     */
    Result infer(cv::Mat& image) {
        // 1.保存图片原始高宽
        this->meta.image_size[0] = image.size[0];
        this->meta.image_size[1] = image.size[1];

        // 2.图片预处理
        cv::Mat resized_image;
        if (this->openvino_preprocess) {
            // 不需要resize,blobFromImage会resize
            resized_image = image;
        }
        else {
            resized_image = pre_process(image, meta);
        }
        // [H, W, C] -> [N, C, H, W]
        // 这里只转换维度,其他预处理都做了,python版本是否使用openvino图片预处理都需要这一步,C++只是自己的预处理需要这一步
        // openvino如果使用这一步的话需要将输入的类型由 u8 转换为 f32, Layout 由 NHWC 改为 NCHW  (38, 39行)
        resized_image = cv::dnn::blobFromImage(resized_image, 1.0,
            { this->meta.infer_size[1], this->meta.infer_size[0] },
            { 0, 0, 0 },
            false, false, CV_32F);

        // 输入全为1测试
        // cv::Size size = cv::Size(224, 224);
        // cv::Scalar color = cv::Scalar(1, 1, 1);
        // resized_image = cv::Mat(size, CV_32FC3, color);

        // 3.从图像创建tensor
        auto* input_data = (float*)resized_image.data;
        ov::Tensor input_tensor = ov::Tensor(this->compiled_model.input(0).get_element_type(),
            this->compiled_model.input(0).get_shape(), input_data);

        // 4.推理
        this->infer_request.set_input_tensor(input_tensor);
        this->infer_request.infer();

        // 5.获取热力图
        ov::Tensor result1;
        result1 = this->infer_request.get_output_tensor(0);
        // cout << result1.get_shape() << endl;    //{1, 1, 224, 224}

        // 6.将热力图转换为Mat
        // result1.data<float>() 返回指针 放入Mat中不能解引用
        cv::Mat anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]),
            CV_32FC1, result1.data<float>());
        cv::Mat pred_score;

        // 7.针对不同输出数量获取得分
        if (this->outputs.size() == 2) {
            ov::Tensor result2 = this->infer_request.get_output_tensor(1);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, result2.data<float>());  // {1}
        }
        else {
            double _, maxValue;    // 最大值，最小值
            cv::minMaxLoc(anomaly_map, &_, &maxValue);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue);
        }
        cout << "pred_score: " << pred_score << endl;   // 4.0252275

        // 8.后处理:标准化,缩放到原图
        vector<cv::Mat> result = post_process(anomaly_map, pred_score, meta);
        anomaly_map = result[0];
        float score = result[1].at<float>(0, 0);

        // 9.返回结果
        return Result{ anomaly_map, score };
    }

    /**
     * 单张图片推理
     * @param image_path    图片路径
     * @param save_dir      保存路径
     */
    cv::Mat single(string& image_path, string& save_dir) {
        // 1.读取图片
        cv::Mat image = readImage(image_path);

        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 2.推理单张图片
        Result result = this->infer(image);
        cout << "score: " << result.score << endl;

        // 3.生成其他图片(mask,mask边缘,热力图和原图的叠加)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time: " << end - start << " ms" << endl;

        // 4.保存显示图片
        // 将mask转化为3通道,不然没法拼接图片
        cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);

        saveScoreAndImages(result.score, images, image_path, save_dir);

        return images[2];
    }


    /**
     * 多张图片推理
     * @param image_dir 图片文件夹路径
     * @param save_dir  保存路径
     */
    void multi(string& image_dir, string& save_dir) {
        // 1.读取全部图片路径
        vector<cv::String> paths = getImagePaths(image_dir);

        vector<float> times;
        for (auto& image_path : paths) {
            // 2.读取单张图片
            cv::Mat image = readImage(image_path);

            // time
            auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            // 3.推理单张图片
            Result result = this->infer(image);
            cout << "score: " << result.score << endl;

            // 4.图片生成其他图片(mask,mask边缘,热力图和原图的叠加)
            vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
            // time
            auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            cout << "infer time: " << end - start << " ms" << endl;
            times.push_back(end - start);

            // 5.保存图片
            // 将mask转化为3通道,不然没法拼接图片
            cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
            saveScoreAndImages(result.score, images, image_path, save_dir);
        }

        // 6.统计数据
        double sumValue = accumulate(begin(times), end(times), 0.0); // accumulate函数就是求vector和的函数；
        double avgValue = sumValue / times.size();                   // 求均值
        cout << "avg infer time: " << avgValue << " ms" << endl;
    }
};