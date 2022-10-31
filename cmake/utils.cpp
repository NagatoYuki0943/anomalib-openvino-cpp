#include "utils.h"


MetaData getJson(const string& json_path) {
    FILE *fp;
    fopen_s(&fp, json_path.c_str(), "r");

    char readBuffer[1000];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document doc;
    doc.ParseStream(is);
    fclose(fp);

    float image_threshold = doc["image_threshold"].GetFloat();
    float pixel_threshold = doc["pixel_threshold"].GetFloat();
    float min             = doc["min"].GetFloat();
    float max             = doc["max"].GetFloat();
    // 列表分别取出
    auto infer_size       = doc["infer_size"].GetArray();
    int infer_height      = infer_size[0].GetInt();
    int infer_width       = infer_size[1].GetInt();

    // cout << image_threshold << endl;
    // cout << pixel_threshold << endl;
    // cout << min << endl;
    // cout << max << endl;
    // cout << infer_height << endl;
    // cout << infer_width << endl;

    return MetaData {image_threshold, pixel_threshold, min, max, {infer_height, infer_width}};
}


vector<cv::String> getImagePaths(string& path) {
    vector<cv::String> paths;
    // for (auto& path : paths) {
    //     //cout << path << endl;
    //     // D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large\000.png
    // }
    cv::glob(path, paths, false);
    return paths;
}


cv::Mat readImage(string& path) {
    auto image = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);    // BGR2RGB
    return image;
}


void saveScoreAndImage(float score, cv::Mat& mixed_image_with_label, cv::String& image_path, string& save_dir) {
    // 获取图片文件名
    // 这样基本确保无论使用 \ / 作为分隔符都能找到文件名字
    auto start = image_path.rfind('\\');
    if (start < 0 || start > image_path.length()){
        start = image_path.rfind('/');
    }
    auto end = image_path.substr(start + 1).rfind('.');
    auto image_name = image_path.substr(start + 1).substr(0, end);  // 000

    // 写入得分
    ofstream ofs;
    ofs.open(save_dir + "/" + image_name + ".txt", ios::out);
    ofs << score;
    ofs.close();

    // 写入图片
    cv::imwrite(save_dir + "/" + image_name + ".jpg", mixed_image_with_label);
}


cv::Mat preProcess(cv::Mat& image, MetaData& meta) {
    vector<float> mean = {0.485, 0.456, 0.406};
    vector<float> std  = {0.229, 0.224, 0.225};

    // 缩放 w h
    cv::Mat resized_image = Resize(image, meta.infer_size[0], meta.infer_size[1], "bilinear");

    // 归一化
    // convertTo直接将所有值除以255,normalize的NORM_MINMAX是将原始数据范围变换到0~1之间,convertTo更符合深度学习的做法
    resized_image.convertTo(resized_image, CV_32FC3, 1.0/255, 0);
    //cv::normalize(resized_image, resized_image, 0, 1, cv::NormTypes::NORM_MINMAX, CV_32FC3);

    // 标准化
    resized_image = Normalize(resized_image, mean, std);
    return resized_image;
}


cv::Mat cvNormalizeMinMax(cv::Mat& targets, float threshold, float min_val, float max_val) {
    auto normalized = ((targets - threshold) / (max_val - min_val)) + 0.5;
    cv::Mat normalized1;
    // normalized = np.clip(normalized, 0, 1) 去除小于0和大于1的
    // 设置上下限: https://blog.csdn.net/simonyucsdy/article/details/106525717
    // 设置上限为1
    cv::threshold(normalized, normalized1, 1, 1, cv::ThresholdTypes::THRESH_TRUNC);
    // 设置下限为0
    cv::threshold(normalized1, normalized1, 0, 0, cv::ThresholdTypes::THRESH_TOZERO);
    return normalized1;
}


vector<cv::Mat> postProcess(cv::Mat& anomaly_map, cv::Mat& pred_score, MetaData& meta) {
    // 标准化热力图和得分
    anomaly_map = cvNormalizeMinMax(anomaly_map, meta.pixel_threshold, meta.min, meta.max);
    pred_score  = cvNormalizeMinMax(pred_score, meta.image_threshold, meta.min, meta.max);

    // 还原到原图尺寸
    anomaly_map = Resize(anomaly_map, meta.image_size[0], meta.image_size[1], "bilinear");

    // 返回热力图和得分
    return vector<cv::Mat>{anomaly_map, pred_score};
}


cv::Mat superimposeAnomalyMap(const cv::Mat& anomaly_map, cv::Mat& origin_image) {
    cv::cvtColor(origin_image, origin_image, cv::ColorConversionCodes::COLOR_RGB2BGR);    // RGB2BGR

    auto anomaly = anomaly_map.clone();
    // 归一化，图片效果更明显
    //python代码： anomaly_map = (anomaly - anomaly.min()) / np.ptp(anomaly) np.ptp()函数实现的功能等同于np.max(array) - np.min(array)
    double minValue, maxValue;    // 最大值，最小值
    cv::minMaxLoc(anomaly, &minValue, &maxValue);
    anomaly = (anomaly - minValue) / (maxValue - minValue);

    //转换为整形
    anomaly.convertTo(anomaly, CV_8UC1, 255, 0);
    //单通道转化为3通道
    cv::applyColorMap(anomaly, anomaly, cv::ColormapTypes::COLORMAP_JET);
    //合并原图和热力图
    cv::Mat combine;
    cv::addWeighted(anomaly, 0.4, origin_image, 0.6, 0, combine);

    return combine;
}


cv::Mat addLabel(cv::Mat& mixed_image, float score, int font) {
    string text = "Confidence Score " + to_string(score);
    int font_size = mixed_image.cols / 1024 + 1;
    int baseline = 0;
    int thickness = font_size / 2;
    cv::Size textsize = cv::getTextSize(text, font, font_size, thickness, &baseline);
    //cout << textsize << endl;	//[1627 x 65]

    //背景
    cv::rectangle(mixed_image, cv::Point(0, 0), cv::Point(textsize.width + 10, textsize.height + 10),
                  cv::Scalar(225, 252, 134), cv::FILLED);

    //添加文字
    cv::putText(mixed_image, text, cv::Point(0, textsize.height + 10), font, font_size,
                cv::Scalar(0, 0, 0), thickness);

    return mixed_image;
}

