#ifndef YOLOV5_H_
#define YOLOV5_H_

#include "common.hpp"
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
// #include "logging.h"


// #define NMS_THRESH 0.4
// #define CONF_THRESH 0.5

#define BATCH_SIZE 1
#define KEEP_TOPK 20
#define INPUT_H 640
#define INPUT_W 640


typedef struct _RecBox 
{
    // int x1;
    // int y1;
    // int x2;
    // int y2;

    cv::Rect rect;
    float conf;
    int class_id;
} RecBox;


// typedef struct _YoloV5DetectionResult 
// {
//     std::vector<RecBox> recboxes;
// } YoloV5DetectionResult;



class YoloV5Detector 
{

public:
    YoloV5Detector();
    ~YoloV5Detector();

    bool InitModel(std::string engine_name, int gpu_id=0);    // 
    // void Forward(cv::Mat& img, YoloV5DetectionResult *det_result);
    int Process(cv::Mat &img, std::vector<RecBox> &result);

private:
    void doInference();


private:
    IRuntime *m_runtime_;
    ICudaEngine *m_engine_;
    IExecutionContext *m_context_;

    
    // prepare input data ---------------------------
    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    int counts[BATCH_SIZE];
    float boxes[BATCH_SIZE * KEEP_TOPK * 4];
    float scores[BATCH_SIZE * KEEP_TOPK];
    float classes[BATCH_SIZE * KEEP_TOPK];


    const char *INPUT_NAME = "data";
    const char *OUTPUT_COUNTS = "count";
    const char *OUTPUT_BOXES = "box";
    const char *OUTPUT_SCORES = "score";
    const char *OUTPUT_CLASSES = "class";

};
#endif