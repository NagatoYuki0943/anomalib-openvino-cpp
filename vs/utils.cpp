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
    // ÁÐ±í·Ö±ðÈ¡³ö
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
    // »ñÈ¡Í¼Æ¬ÎÄ¼þÃû
    // ÕâÑù»ù±¾È·±£ÎÞÂÛÊ¹ÓÃ \ / ×÷Îª·Ö¸ô·û¶¼ÄÜÕÒµ½ÎÄ¼þÃû×Ö
    auto start = image_path.rfind('\\');
    if (start < 0 || start > image_path.length()){
        start = image_path.rfind('/');
    }
    auto end = image_path.substr(start + 1).rfind('.');
    auto image_name = image_path.substr(start + 1).substr(0, end);  // 000

    // Ð´ÈëµÃ·Ö
    ofstream ofs;
    ofs.open(save_dir + "/" + image_name + ".txt", ios::out);
    ofs << score;
    ofs.close();

    // Ð´ÈëÍ¼Æ¬
    cv::imwrite(save_dir + "/" + image_name + ".jpg", mixed_image_with_label);
}


cv::Mat cvNormalizeMinMax(cv::Mat& targets, float threshold, float min_val, float max_val) {
    auto normalized = ((targets - threshold) / (max_val - min_val)) + 0.5;
    cv::Mat normalized1;
    // normalized = np.clip(normalized, 0, 1) È¥³ýÐ¡ÓÚ0ºÍ´óÓÚ1µÄ
    // ÉèÖÃÉÏÏÂÏÞ: https://blog.csdn.net/simonyucsdy/article/details/106525717
    // ÉèÖÃÉÏÏÞÎª1
    cv::threshold(normalized, normalized1, 1, 1, cv::ThresholdTypes::THRESH_TRUNC);
    // ÉèÖÃÏÂÏÞÎª0
    cv::threshold(normalized1, normalized1, 0, 0, cv::ThresholdTypes::THRESH_TOZERO);
    return normalized1;
}


cv::Mat superimposeAnomalyMap(const cv::Mat& anomaly_map, cv::Mat& origin_image) {
    cv::cvtColor(origin_image, origin_image, cv::ColorConversionCodes::COLOR_RGB2BGR);    // RGB2BGR

    auto anomaly = anomaly_map.clone();
    // ¹éÒ»»¯£¬Í¼Æ¬Ð§¹û¸üÃ÷ÏÔ
    //python´úÂë£º anomaly_map = (anomaly - anomaly.min()) / np.ptp(anomaly) np.ptp()º¯ÊýÊµÏÖµÄ¹¦ÄÜµÈÍ¬ÓÚnp.max(array) - np.min(array)
    double minValue, maxValue;    // ×î´óÖµ£¬×îÐ¡Öµ
    cv::minMaxLoc(anomaly, &minValue, &maxValue);
    anomaly = (anomaly - minValue) / (maxValue - minValue);

    //×ª»»ÎªÕûÐÎ
    anomaly.convertTo(anomaly, CV_8UC1, 255, 0);
    //µ¥Í¨µÀ×ª»¯Îª3Í¨µÀ
    cv::applyColorMap(anomaly, anomaly, cv::ColormapTypes::COLORMAP_JET);
    //ºÏ²¢Ô­Í¼ºÍÈÈÁ¦Í¼
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
    //cout << textsize << endl; //[1627 x 65]

    //±³¾°
    cv::rectangle(mixed_image, cv::Point(0, 0), cv::Point(textsize.width + 10, textsize.height + 10),
                  cv::Scalar(225, 252, 134), cv::FILLED);

    //Ìí¼ÓÎÄ×Ö
    cv::putText(mixed_image, text, cv::Point(0, textsize.height + 10), font, font_size,
                cv::Scalar(0, 0, 0), thickness);

    return mixed_image;
}

