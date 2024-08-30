#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
// 전역 컬러맵 초기화 함수 선언
void initializeColors(int numClasses);

class ImageProcessor {
public:
    // 이미지 파일 경로를 인자로 받는 기존 함수
    //static bool preprocessImage(const std::string& imageFile, float* buffer, int inputH, int inputW);
    //static void createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses);
    // cv::Mat 객체를 인자로 받는 새 함수
    static bool preprocessImage(const cv::Mat& image, float* buffer, int inputH, int inputW);
    // cv::Mat 타입의 마스크를 반환하도록 수정된 함수
    
    //static cv::Mat createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses);
    
    // segmentation 마스크 컬러를 사용자 정의한함수
    static void colorizeSegmentation(const cv::Mat &score, cv::Mat &segm,int numClasses);
    //static cv::Mat createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses, bool applySoftmax = false);
    //.static cv::Mat createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses);
    static cv::Mat createSegmentationMask(float* outputBuffer, int inputH, int inputW);
private:
    // 후처리 함수에서 사용할 소프트맥스 함수
    static void softmax(float* input, int size);
};
