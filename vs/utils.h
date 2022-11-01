#pragma once
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include "rapidjson/document.h"         //https://github.com/Tencent/rapidjson
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"
#include "opencv_utils.h"


using namespace std;

struct MetaData {
public:
    float image_threshold;
    float pixel_threshold;
    float min;
    float max;
    int infer_size[2];  // h w
    int image_size[2];  // h w
};


/**
 * 获取json配置文件
 * @param json_path 配置文件路径
 * @return
 */
MetaData getJson(const string& json_path);


/**
 * 获取文件夹下全部图片的绝对路径
 *
 * @param path          图片文件夹路径
 * @return result       全部图片绝对路径列表
 */
vector<cv::String> getImagePaths(string& path);



/**
 * 读取图像 BGR2RGB
 *
 * @param path  图片路径
 * @return      图片
 */
cv::Mat readImage(string& path);


/**
 * 保存图片和分数
 *
 * @param score      得分
 * @param mixed_image_with_label 混合后的图片
 * @param image_path 输入图片的路径
 * @param save_dir   保存的路径
 */
void saveScoreAndImage(float score, cv::Mat& mixed_image_with_label, cv::String& image_path, string& save_dir);


/**
 * opencv标准化热力图
 *
 * @param targets       热力图
 * @param threshold     阈值,meta中的参数
 * @param min_val       最小值,meta中的参数
 * @param max_val       最大值,meta中的参数
 * @return normalized   经过标准化后的结果
 */
cv::Mat cvNormalizeMinMax(cv::Mat& targets, float threshold, float min_val, float max_val);


/**
 * 叠加图片
 *
 * @param anomaly_map   混合后的图片
 * @param origin_image  原始图片
 * @return result       叠加后的图像
 */
cv::Mat superimposeAnomalyMap(const cv::Mat& anomaly_map, cv::Mat& origin_image);


/**
 * 给图片添加标签
 *
 * @param mixed_image   混合后的图片
 * @param score         得分
 * @param font          字体
 * @return mixed_image  添加标签的图像
 */
cv::Mat addLabel(cv::Mat& mixed_image, float score, int font = cv::FONT_HERSHEY_PLAIN);


/**
 * 推理结果：热力图 + 得分
 */
struct Result{
public:
    cv::Mat anomaly_map;
    float score;
};
