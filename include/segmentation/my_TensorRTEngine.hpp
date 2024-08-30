#pragma once

#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <filesystem> // C++17에서 std::filesystem 사용을 위한 헤더 
#include <fstream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>


class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};
class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();
    // 엔진을 생성하는 멤버함수
    bool buildEngine(const std::string& onnxFile, const std::string& engineFileName);
    // 생성된 엔진으로 추론을 수행하는 멤버함수
    //bool runInference(const std::string& imageFile);
    //동영상으로 추론을 진행하는 멤버함수 원본
    //bool runInference(cv::Mat& frame,cv::Mat& result);
    //비동기 추론 모드함수
    // 추가된 멤버 함수
    
    bool runInference(cv::Mat& frame, cv::Mat& result, cudaStream_t stream);
    bool processVideo(const std::string& videoFile);//동영상 불러오기
    int inputH, inputW, numClasses;
private:
    Logger logger;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::vector<void*> buffers;
    std::vector<int64_t> bufferSizes;
    std::vector<cv::Vec3b> colors;  // 컬러맵을 멤버 변수로 선언
    void readFile(const std::string& fileName, std::vector<char>& buffer);
    void cleanup();
};
