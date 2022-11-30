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
 * input(0)/output(0) ����id��ָ���������������ָ����ȫ�����������
 *
 *  input().tensor()       ��7������
 *  ppp.input().tensor().set_color_format().set_element_type().set_layout()
 *                      .set_memory_type().set_shape().set_spatial_dynamic_shape().set_spatial_static_shape();
 *
 *  output().tensor()      ��2������
 *  ppp.output().tensor().set_layout().set_element_type();
 *
 *  input().preprocess()   ��8������
 *  ppp.input().preprocess().convert_color().convert_element_type().mean().scale()
 *                          .convert_layout().reverse_channels().resize().custom();
 *
 *  output().postprocess() ��3������
 *  ppp.output().postprocess().convert_element_type().convert_layout().custom();
 *
 *  input().model()  ֻ��1������
 *  ppp.input().model().set_layout();
 *
 *  output().model() ֻ��1������
 *  ppp.output().model().set_layout();
 **/

class Inference {
private:
    bool openvino_preprocess;           // �Ƿ�ʹ��openvinoͼƬԤ����
    MetaData meta{};                    // ������
    ov::CompiledModel compiled_model;   // ����õ�ģ��
    ov::InferRequest infer_request;     // ��������
    vector<ov::Output<const ov::Node>> inputs;  // ģ�͵������б�����
    vector<ov::Output<const ov::Node>> outputs; // ģ�͵�����б�����

public:
    /**
     * @param model_path    ģ��·��
     * @param meta_path     ������·��
     * @param device        CPU or GPU ����
     * @param openvino_preprocess   �Ƿ�ʹ��openvinoͼƬԤ����
     */
    Inference(string& model_path, string& meta_path, string& device, bool openvino_preprocess) {
        this->openvino_preprocess = openvino_preprocess;
        // 1.��ȡmeta
        this->meta = getJson(meta_path);
        // 2.����ģ��
        this->compiled_model = this->get_model(model_path, device);
        // 3.��ȡģ�͵��������
        this->inputs = this->compiled_model.inputs();
        this->outputs = this->compiled_model.outputs();
        // 4.������������
        this->infer_request = this->compiled_model.create_infer_request();
        // 5.ģ��Ԥ��
        this->warm_up();
    }

    /**
     * get openvino model
     * @param model_path ģ��·��
     * @param device     ʹ�õ��豸
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
     * ģ��Ԥ��
     */
    void warm_up() {
        // ��������
        cv::Size size = cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::Mat input = cv::Mat(size, CV_8UC3, color);
        this->infer(input);
    }


    /**
     * ������ͼƬ
     * @param image ԭʼͼƬ
     * @return      ��׼���Ĳ����ŵ�ԭͼ����ͼ�͵÷�
     */
    Result infer(cv::Mat& image) {
        // 1.����ͼƬԭʼ�߿�
        this->meta.image_size[0] = image.size[0];
        this->meta.image_size[1] = image.size[1];

        // 2.ͼƬԤ����
        cv::Mat resized_image;
        if (this->openvino_preprocess) {
            // ����Ҫresize,blobFromImage��resize
            resized_image = image;
        }
        else {
            resized_image = pre_process(image, meta);
        }
        // [H, W, C] -> [N, C, H, W]
        // ����ֻת��ά��,����Ԥ��������,python�汾�Ƿ�ʹ��openvinoͼƬԤ������Ҫ��һ��,C++ֻ���Լ���Ԥ������Ҫ��һ��
        // openvino���ʹ����һ���Ļ���Ҫ������������� u8 ת��Ϊ f32, Layout �� NHWC ��Ϊ NCHW  (38, 39��)
        resized_image = cv::dnn::blobFromImage(resized_image, 1.0,
            { this->meta.infer_size[1], this->meta.infer_size[0] },
            { 0, 0, 0 },
            false, false, CV_32F);

        // ����ȫΪ1����
        // cv::Size size = cv::Size(224, 224);
        // cv::Scalar color = cv::Scalar(1, 1, 1);
        // resized_image = cv::Mat(size, CV_32FC3, color);

        // 3.��ͼ�񴴽�tensor
        auto* input_data = (float*)resized_image.data;
        ov::Tensor input_tensor = ov::Tensor(this->compiled_model.input(0).get_element_type(),
            this->compiled_model.input(0).get_shape(), input_data);

        // 4.����
        this->infer_request.set_input_tensor(input_tensor);
        this->infer_request.infer();

        // 5.��ȡ����ͼ
        ov::Tensor result1;
        result1 = this->infer_request.get_output_tensor(0);
        // cout << result1.get_shape() << endl;    //{1, 1, 224, 224}

        // 6.������ͼת��ΪMat
        // result1.data<float>() ����ָ�� ����Mat�в��ܽ�����
        cv::Mat anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]),
            CV_32FC1, result1.data<float>());
        cv::Mat pred_score;

        // 7.��Բ�ͬ���������ȡ�÷�
        if (this->outputs.size() == 2) {
            ov::Tensor result2 = this->infer_request.get_output_tensor(1);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, result2.data<float>());  // {1}
        }
        else {
            double _, maxValue;    // ���ֵ����Сֵ
            cv::minMaxLoc(anomaly_map, &_, &maxValue);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue);
        }
        cout << "pred_score: " << pred_score << endl;   // 4.0252275

        // 8.����:��׼��,���ŵ�ԭͼ
        vector<cv::Mat> result = post_process(anomaly_map, pred_score, meta);
        anomaly_map = result[0];
        float score = result[1].at<float>(0, 0);

        // 9.���ؽ��
        return Result{ anomaly_map, score };
    }
};