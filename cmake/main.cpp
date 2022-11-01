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
    MetaData meta{};                    // ������
    ov::CompiledModel compiled_model;   // ����õ�ģ��
    ov::InferRequest infer_request;     // ��������
    string device;                      // ʹ�õ��豸
    bool openvino_preprocess;           // �Ƿ�ʹ��openvinoͼƬԤ����

public:
    /**
     * @param model_path    ģ��·��
     * @param meta_path     ������·��
     * @param device        CPU or GPU ����
     * @param openvino_preprocess   �Ƿ�ʹ��openvinoͼƬԤ����
     */
    Inference(string& model_path, string& meta_path, string& device, bool openvino_preprocess){
        // 1.��ȡmeta
        this->meta   = getJson(meta_path);
        this->device = device;
        this->openvino_preprocess = openvino_preprocess;
        // 2.����ģ��
        this->get_openvino_model(model_path);
        // 3.������������
        this->infer_request = this->compiled_model.create_infer_request();
        // 4.ģ��Ԥ��
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
        // ģ��
        this->compiled_model = core.compile_model(model, this->device);
    }


    /**
     * ģ��Ԥ��
     */
    void warm_up(){
        // ��������
        cv::Size size = cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::Mat input = cv::Mat(size, CV_8UC3, color);
        this->infer(input);
    }


    /**
     * ͼƬԤ����
     * @param image Ԥ����ͼƬ
     * @return      ����Ԥ�����ͼƬ
     */
    cv::Mat preProcess(cv::Mat& image) {
        vector<float> mean = {0.485, 0.456, 0.406};
        vector<float> std  = {0.229, 0.224, 0.225};

        // ���� w h
        cv::Mat resized_image = Resize(image, this->meta.infer_size[0], this->meta.infer_size[1], "bilinear");

        // ��һ��
        // convertToֱ�ӽ�����ֵ����255,normalize��NORM_MINMAX�ǽ�ԭʼ���ݷ�Χ�任��0~1֮��,convertTo���������ѧϰ������
        resized_image.convertTo(resized_image, CV_32FC3, 1.0/255, 0);
        //cv::normalize(resized_image, resized_image, 0, 1, cv::NormTypes::NORM_MINMAX, CV_32FC3);

        // ��׼��
        resized_image = Normalize(resized_image, mean, std);
        return resized_image;
    }


    /**
     * ������,��׼������ͼ�͵÷�,��ԭ����ͼ��ԭͼ�ߴ�
     *
     * @param anomaly_map   δ������׼��������ͼ
     * @param pred_score    δ������׼���ĵ÷�
     * @return result		����ͼ�͵÷�vector
     */
    vector<cv::Mat> postProcess(cv::Mat& anomaly_map, cv::Mat& pred_score) {
        // ��׼������ͼ�͵÷�
        anomaly_map = cvNormalizeMinMax(anomaly_map, this->meta.pixel_threshold, this->meta.min, this->meta.max);
        pred_score  = cvNormalizeMinMax(pred_score, this->meta.image_threshold, this->meta.min, this->meta.max);

        // ��ԭ��ԭͼ�ߴ�
        anomaly_map = Resize(anomaly_map, this->meta.image_size[0], this->meta.image_size[1], "bilinear");

        // ��������ͼ�͵÷�
        return vector<cv::Mat>{anomaly_map, pred_score};
    }


    /**
     * ������ͼƬ
     * @param image ԭʼͼƬ
     * @return      ��������ͼ��ԭͼ�͵÷�
     */
    Result infer(cv::Mat& image){
        // 1.����ͼƬԭʼ�߿�
        this->meta.image_size[0] = image.size[0];
        this->meta.image_size[1] = image.size[1];

        // 2.ͼƬԤ����
        cv::Mat resized_image;
        if (this->openvino_preprocess){
            // ����Ҫresize,blobFromImage��resize
            resized_image = image;
        }else{
            resized_image = this->preProcess(image);
        }
        // [H, W, C] -> [N, C, H, W]
        // ����ֻת��ά��,����Ԥ��������,python�汾�Ƿ�ʹ��openvinoͼƬԤ������Ҫ��һ��,C++ֻ���Լ���Ԥ������Ҫ��һ��
        // openvino���ʹ����һ���Ļ���Ҫ������������� u8 ת��Ϊ f32, Layout �� NHWC ��Ϊ NCHW  (38, 39��)
        resized_image = cv::dnn::blobFromImage(resized_image, 1.0,
                                               {this->meta.infer_size[1], this->meta.infer_size[0]},
                                               {0, 0, 0},
                                               false, false, CV_32F);

        // ����ȫΪ1����
        // cv::Size size = cv::Size(224, 224);
        // cv::Scalar color = cv::Scalar(1, 1, 1);
        // resized_image = cv::Mat(size, CV_32FC3, color);

        // 3.��ͼ�񴴽�tensor
        auto *input_data = (float *) resized_image.data;
        ov::Tensor input_tensor = ov::Tensor(this->compiled_model.input().get_element_type(),
                                             this->compiled_model.input().get_shape(), input_data);

        // 4.����
        this->infer_request.set_input_tensor(input_tensor);
        this->infer_request.infer();

        // 5.��ȡ���
        const ov::Tensor &result1 = this->infer_request.get_output_tensor(0);
        const ov::Tensor &result2 = this->infer_request.get_output_tensor(1);
        // cout << result1.get_shape() << endl;    //{1, 1, 224, 224}
        // cout << result2.get_shape() << endl;    //{1}

        // 6.�����ת��ΪMat
        // result1.data<float>() ����ָ�� ����Mat�в��ܽ�ӡ��
        cv::Mat anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]),
                                      CV_32FC1, result1.data<float>());
        cv::Mat pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, result2.data<float>());
        cout << "pred_score: " << pred_score << endl;   // 4.0252275

        // 7.����:��׼��,���ŵ�ԭͼ
        vector<cv::Mat> result = this->postProcess(anomaly_map, pred_score);
        anomaly_map = result[0];
        float score = result[1].at<float>(0, 0);

        // 8.����ԭͼ������ͼ,�����￪ʼ����Ϊ����ʾ��׼����
        anomaly_map = superimposeAnomalyMap(anomaly_map, image);

        // 9.��ͼƬ��ӷ���
        anomaly_map = addLabel(anomaly_map, score);

        // 10.���ؽ��
        return Result {anomaly_map, score};
    }
};


/**
 * ����ͼƬ����
 * @param model_path    ģ��·��
 * @param meta_path     ������·��
 * @param image_path    ͼƬ·��
 * @param save_dir      ����·��
 * @param device        CPU or GPU ����
 * @param openvino_preprocess   �Ƿ�ʹ��openvinoͼƬԤ����
 */
void single(string& model_path, string& meta_path, string& image_path, string& save_dir,
            string& device, bool openvino_preprocess = true){
    // 1.����������
    Inference inference = Inference(model_path, meta_path, device, openvino_preprocess);

    // 2.��ȡͼƬ
    cv::Mat image = readImage(image_path);

    // 3.������ͼƬ
    Result result = inference.infer(image);

    // 4.������ʾͼƬ
    cout << "score: " << result.score << endl;
    saveScoreAndImage(result.score, result.anomaly_map, image_path, save_dir);
    cv::imshow("result", result.anomaly_map);
    cv::waitKey(0);
}


/**
 * ����ͼƬ����
 * @param model_path    ģ��·��
 * @param meta_path     ������·��
 * @param image_dir     ͼƬ�ļ���·��
 * @param save_dir      ����·��
 * @param device        CPU or GPU ����
 * @param openvino_preprocess   �Ƿ�ʹ��openvinoͼƬԤ����
 */
void multi(string& model_path, string& meta_path, string& image_dir, string& save_dir,
           string& device, bool openvino_preprocess = true){
    // 1.����������
    Inference inference = Inference(model_path, meta_path, device, openvino_preprocess);

    // 2.��ȡȫ��ͼƬ·��
    vector<cv::String> paths = getImagePaths(image_dir);

    vector<float> times;
    for (auto& image_path : paths) {
        // 3.��ȡ����ͼƬ
        cv::Mat image = readImage(image_path);

        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 4.������ͼƬ
        Result result = inference.infer(image);
        cout << "score: " << result.score << endl;
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time:" << end - start << "ms" << endl;
        times.push_back(end - start);
        // 5.����ͼƬ
        saveScoreAndImage(result.score, result.anomaly_map, image_path, save_dir);
    }

    // 6.ͳ������
    double sumValue = accumulate(begin(times), end(times), 0.0);  // accumulate����������vector�͵ĺ�����
    double meanValue = sumValue / times.size();                   // ���ֵ
    cout << "mean infer time: " << meanValue << endl;
}


int main(){
    string model_path = "D:/ai/code/abnormal/anomalib/results/patchcore/mvtec/bottle-cls/optimization/openvino/model.xml";
    string meta_path  = "D:/ai/code/abnormal/anomalib/results/patchcore/mvtec/bottle-cls/optimization/meta_data.json";
    string image_path = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ai/code/abnormal/anomalib-patchcore-openvino/cmake/result";
    // �Ƿ�ʹ��openvinoͼƬԤ����
    bool openvino_preprocess = true;
    string device = "CPU";
    single(model_path, meta_path, image_path, save_dir,device, openvino_preprocess);
    // multi(model_path, meta_path, image_dir, save_dir, device, openvino_preprocess);
    return 0;
}
