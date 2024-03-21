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
    bool openvino_preprocess;                   // 是否使用openvino图片预处理
    bool efficient_ad;                          // 是否使用efficient_ad模型
    MetaData meta{};                            // 超参数
    ov::CompiledModel compiled_model;           // 编译好的模型
    ov::InferRequest infer_request;             // 推理请求
    vector<ov::Output<const ov::Node>> inputs;  // 模型的输入列表名称
    vector<ov::Output<const ov::Node>> outputs; // 模型的输出列表名称

public:
    /**
     * @param model_path    模型路径
     * @param meta_path     超参数路径
     * @param device        CPU or GPU 推理
     * @param openvino_preprocess   是否使用openvino图片预处理
     * @param efficient_ad          是否使用efficient_ad模型
     */
    Inference(string& model_path, string& meta_path, string& device, bool openvino_preprocess, bool efficient_ad = false) {
        this->efficient_ad = efficient_ad;
        this->openvino_preprocess = openvino_preprocess;
        // 1.读取meta
        this->meta = getJson(meta_path);
        // 2.创建模型
        this->get_model(model_path, device);
        // 3.获取模型的输入输出
        this->inputs = this->compiled_model.inputs();
        this->outputs = this->compiled_model.outputs();

        // 打印输入输出形状
        //dynamic shape model without openvino_preprocess coundn't print input and output shape
        for (auto input : this->inputs) {
            cout << "Input: " << input.get_any_name() << ": [ ";
            for (auto j : input.get_shape()) {
                cout << j << " ";
            }
            cout << "] ";
            cout << "dtype: " << input.get_element_type() << endl;
        }

        for (auto output : this->outputs) {
            cout << "Output: " << output.get_any_name() << ": [ ";
            for (auto j : output.get_shape()) {
                cout << j << " ";
            }
            cout << "] ";
            cout << "dtype: " << output.get_element_type() << endl;
        }

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
    void get_model(string& model_path, string& device) {
        // Step 1. Initialize OpenVINO Runtime core
        ov::Core core;
        // Step 2. Read a Model from a Drive
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        if (this->openvino_preprocess) {
            vector<float> mean;
            vector<float> std;
            if (!this->efficient_ad) {
                mean = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
                std = { 0.229 * 255, 0.224 * 255, 0.225 * 255 };
            }
            else {
                mean = { 0., 0., 0. };
                std = { 255., 255., 255. };
            }

            // Step 3. Inizialize Preprocessing for the model
            // https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
            // https://blog.csdn.net/sandmangu/article/details/107181289
            // https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
            ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

            // Specify input image format
            ppp.input(0).tensor()
                .set_color_format(ov::preprocess::ColorFormat::RGB)     // BGR -> RGB
                .set_element_type(ov::element::u8)
                .set_layout(ov::Layout("HWC"));                         // HWC NHWC NCHW

        // Specify preprocess pipeline to input image without resizing
            ppp.input(0).preprocess()
                //  .convert_color(ov::preprocess::ColorFormat::RGB)
                .convert_element_type(ov::element::f32)
                .mean(mean)
                .scale(std);

            // Specify model's input layout
            ppp.input(0).model().set_layout(ov::Layout("NCHW"));

            // Specify output results format
            for (size_t i = 0; i < model->outputs().size(); i++) {
                ppp.output(i).tensor().set_element_type(ov::element::f32);
            }

            // Embed above steps in the graph
            model = ppp.build();
        }
        // Step 4. Load the Model to the Device
        this->compiled_model = core.compile_model(model, device);
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
        this->meta.image_size[0] = image.size().height;
        this->meta.image_size[1] = image.size().width;

        // 2.图片预处理
        cv::Mat blob;
        if (this->openvino_preprocess) {
            cv::resize(image, blob, { this->meta.infer_size[0], this->meta.infer_size[1] });
        }
        else {
            blob = pre_process(image, this->meta, this->efficient_ad);
            // [H, W, C] -> [N, C, H, W]
            blob = cv::dnn::blobFromImage(blob);
        }

        // 3.从图像创建tensor
        ov::Tensor input_tensor = ov::Tensor(this->compiled_model.input(0).get_element_type(),
            this->compiled_model.input(0).get_shape(), (float*)blob.data);

        // 4.推理
        this->infer_request.set_input_tensor(input_tensor);
        this->infer_request.infer();

        // 5.获取热力图
        ov::Tensor result1;
        result1 = this->infer_request.get_output_tensor(0);
        // cout << result1.get_shape() << endl;    //{1, 1, 224, 224}

        // 6.将热力图转换为Mat
        // result1.data<float>() 返回指针 放入Mat中不能解引用
        cv::Mat anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]), CV_32FC1, result1.data<float>());
        cv::Mat pred_score;

        // 7.针对不同输出数量获取得分
        // efficient_ad模型有3个输出,不过只有第1个是anomaly_map,其余不用处理
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
        vector<cv::Mat> post_mat = post_process(anomaly_map, pred_score, this->meta);
        anomaly_map = post_mat[0];
        float score = post_mat[1].at<float>(0, 0);

        // 9.返回结果
        return Result{ anomaly_map, score };
    }

    /**
     * 单张图片推理
     * @param image    RGB图片
     * @return      标准化的并所放到原图热力图和得分
     */
    Result single(cv::Mat& image) {
        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 1.推理单张图片
        Result result = this->infer(image);
        cout << "score: " << result.score << endl;

        // 2.生成其他图片(mask,mask抠图,mask边缘,热力图和原图的叠加)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time: " << end - start << " ms" << endl;

        // 3.保存显示图片
        // 拼接图片
        cv::Mat res;
        cv::hconcat(images, res);

        return Result{ res, result.score };
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

            // 4.生成其他图片(mask,mask抠图,mask边缘,热力图和原图的叠加)
            vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
            // time
            auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            cout << "infer time: " << end - start << " ms" << endl;
            times.push_back(end - start);

            // 5.保存图片
            // 拼接图片
            cv::Mat res;
            cv::hconcat(images, res);
            saveScoreAndImages(result.score, res, image_path, save_dir);
        }

        // 6.统计数据
        double sumValue = accumulate(begin(times), end(times), 0.0); // accumulate函数就是求vector和的函数；
        double avgValue = sumValue / times.size();                   // 求均值
        cout << "avg infer time: " << avgValue << " ms" << endl;
    }
};
