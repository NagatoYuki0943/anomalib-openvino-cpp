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
 * get      openvino model
 * @param   model_path
 * @return  CompiledModel
 */
ov::CompiledModel get_openvino_model(const string& model_path, const string& device="CPU", bool openvino_preprocess=false){
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
        ppp.input(0).tensor().
                set_element_type(ov::element::u8).
                set_layout(ov::Layout("NHWC"))
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
    return compiled_model;
}


/**
 * ͼƬԤ����
 * TODO ������,openvino����������
 * @param path	ͼƬ·��
 * @param meta  ������,������ԭͼ�Ŀ��
 * @return x	tensor���͵�ͼƬ
 */
cv::Mat preProcess(cv::Mat& image, MetaData& meta) {
    vector<float> mean = {0.485, 0.456, 0.406};
    vector<float> std  = {0.229, 0.224, 0.225};

    // ���� w h
    cv::Mat resized_image = Resize(image, meta.infer_size[0], meta.infer_size[1], "bilinear");
    //cout << resized_image << endl;
//    double *maxVal;
//    double *minVal;
//    cv::minMaxLoc(resized_image, minVal, maxVal);
//    cout << *minVal << " " << *maxVal << endl;
//    cout << cv::mean(resized_image) << endl;    // [138.361, 137.557, 142.034, 0]
//    cout << resized_image.channels() << endl;     // [138.361, 137.557, 142.034, 0]

    // ��һ��
    // convertToֱ�ӽ�����ֵ����255,normalize��NORM_MINMAX�ǽ�ԭʼ���ݷ�Χ�任��0~1֮��,convertTo���������ѧϰ������
    resized_image.convertTo(resized_image, CV_32FC3, 1.0/255, 0);
    //cv::normalize(resized_image, resized_image, 0, 1, cv::NormTypes::NORM_MINMAX, CV_32FC3);
    // ��׼��
    resized_image = Normalize(resized_image, mean, std);
    return resized_image;
}


/**
 * ������ͼƬ
 * @param compiled_model    ������ģ��
 * @param infer_request     ��������
 * @param image             ԭʼͼƬ
 * @param meta              ������
 * @param openvino_preprocess �Ƿ�ʹ��openvinoͼƬԤ����,Ŀǰֻ��ʹ������Ԥ����,�Լ�д��������
 * @return                  ��������ͼ��ԭͼ�͵÷�
 */
Result infer(ov::CompiledModel& compiled_model, ov::InferRequest& infer_request,
             cv::Mat& image, MetaData& meta, bool openvino_preprocess){
    // 1.����ͼƬԭʼ�߿�
    meta.image_size[0] = image.size[0];
    meta.image_size[1] = image.size[1];

    // 2.ͼƬԤ����
    cv::Mat resized_image;
    if (openvino_preprocess){
        resized_image = Resize(image, meta.infer_size[0], meta.infer_size[1], "bilinear");
    }else{
        resized_image = preProcess(image, meta);
    }

    // ����ȫΪ1����
    // cv::Scalar color = cv::Scalar(1, 1, 1);
    // cv::Size size = cv::Size(224, 224);
    // image = cv::Mat(size, CV_32FC3, color);

    // 3.��ͼ�񴴽�tensor
    auto *input_data = (float *) resized_image.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);

    // 4.����
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // 5.��ȡ���
    const ov::Tensor &result1 = infer_request.get_output_tensor(0);
    const ov::Tensor &result2 = infer_request.get_output_tensor(1);
    // cout << result1.get_shape() << endl;    //{1, 1, 224, 224}
    // cout << result2.get_shape() << endl;    //{1}

    // 6.�����ת��ΪMat
    // result1.data<float>() ����ָ�� ����Mat�в��ܽ�ӡ��
    cv::Mat anomaly_map = cv::Mat(cv::Size(meta.infer_size[1], meta.infer_size[0]), CV_32FC1, result1.data<float>());
    cv::Mat pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, result2.data<float>());
    cout << "pred_score: " << pred_score << endl;   // 4.0252275

    // 7.����:��׼��,���ŵ�ԭͼ
    vector<cv::Mat> result = postProcess(anomaly_map, pred_score, meta);
    anomaly_map = result[0];
    float score = result[1].at<float>(0, 0);

    // 8.����ԭͼ������ͼ,�����￪ʼ����Ϊ����ʾ��׼����
    anomaly_map = superimposeAnomalyMap(anomaly_map, image);

    // 9.��ͼƬ��ӷ���
    anomaly_map = addLabel(anomaly_map, score);

    // 10.���ؽ��
    return Result {anomaly_map, score};
}


/**
 * ����ͼƬ����
 * @param model_path    ģ��·��
 * @param meta_path     ������·��
 * @param image_path    ͼƬ·��
 * @param save_dir      ����·��
 * @param openvino_preprocess   �Ƿ�ʹ��openvinoͼƬԤ����
 */
void single(string& model_path, string& meta_path, string& image_path, string& save_dir, bool openvino_preprocess = true){
    // 1.��ȡmeta
    MetaData meta = getJson(meta_path);

    // 2.��ȡģ��
    ov::CompiledModel compiled_model = get_openvino_model(model_path, "CPU", openvino_preprocess);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 3.��ȡͼƬ
    cv::Mat image = readImage(image_path);

    // 4.������ͼƬ
    Result result = infer(compiled_model, infer_request, image, meta, openvino_preprocess);

    // 5.������ʾͼƬ
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
 * @param openvino_preprocess   �Ƿ�ʹ��openvinoͼƬԤ����
 */
void multi(string& model_path, string& meta_path, string& image_dir, string& save_dir, bool openvino_preprocess = true){
    // 1.��ȡmeta
    MetaData meta = getJson(meta_path);

    // 2.��ȡģ��
    ov::CompiledModel compiled_model = get_openvino_model(model_path, "CPU", openvino_preprocess);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 3.��ȡȫ��ͼƬ·��
    vector<cv::String> paths = getImagePaths(image_dir);

    vector<float> times;
    for (auto& image_path : paths) {
        // 4.��ȡ����ͼƬ
        cv::Mat image = readImage(image_path);

        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 5.������ͼƬ
        Result result = infer(compiled_model, infer_request, image, meta, openvino_preprocess);
        cout << "score: " << result.score << endl;
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time:" << end - start << "ms" << endl;
        times.push_back(end - start);
        // 6.����ͼƬ
        saveScoreAndImage(result.score, result.anomaly_map, image_path, save_dir);
    }

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
    // �Ƿ�ʹ��openvinoͼƬԤ����, Ŀǰ����ʹ����, �Լ�д��Ԥ����������
    bool openvino_preprocess = true;
    single(model_path, meta_path, image_path, save_dir, openvino_preprocess);
    //multi(model_path, meta_path, image_dir, save_dir, openvino_preprocess);
    return 0;
}

