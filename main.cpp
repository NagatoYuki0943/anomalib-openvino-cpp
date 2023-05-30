#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // ע��ʹ�÷�patchcoreģ��ʱ�������Բ鿴utils.cpp��infer_height��infer_width�е�[1] ����Ϊ [0]������鿴ע�ͺ�metadata.json�ļ�
    string model_path = "D:/ml/code/anomalib/results/patchcore/mvtec/bottle/run/weights/openvino/model.xml";
    string meta_path  = "D:/ml/code/anomalib/results/patchcore/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-openvino-cpp/result"; // ע��Ŀ¼�����Զ�����,Ҫ�ֶ������Żᱣ��
    string device     = "CPU";
    bool openvino_preprocess = true;    // �Ƿ�ʹ��openvinoͼƬԤ����

    // ����������
    auto inference = Inference(model_path, meta_path, device, openvino_preprocess);

    // ����ͼƬ����
    cv::Mat result = inference.single(image_path, save_dir);
    cv::imshow("result", result);
    cv::waitKey(0);

    // ����ͼƬ����
    inference.multi(image_dir, save_dir);
    return 0;
}
