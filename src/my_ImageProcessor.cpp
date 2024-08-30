#include "segmentation/my_ImageProcessor.hpp"  
#include "segmentation/my_TensorRTEngine.hpp"
#include <iostream>               
#include <algorithm>              
#include <cmath>                  
#include <vector>                 



// 전역 변수로 colors 선언
std::vector<cv::Vec3b> colors; 
extern std::vector<int> neededClasses;
// 전역 컬러맵 초기화 함수 정의
void initializeColors(int numClasses) {
    if (colors.empty()) {
        // Generate colors.
        colors.push_back(cv::Vec3b(0, 0, 0)); // Background color
        for (int i = 1; i < numClasses; ++i) {
            cv::Vec3b color;
            for (int j = 0; j < 3; ++j)
                color[j] = (colors[i - 1][j] + rand() % 256) / 2;
            colors.push_back(color);
        }
    } else if (numClasses != static_cast<int>(colors.size())) {
        CV_Error(cv::Error::StsError, cv::format("Number of output classes does not match number of colors (%d != %zu)", numClasses, colors.size()));
    }
}


// 이미지 전처리 함수: 이미지 파일을 읽고 모델의 입력에 맞게 전처리
/*bool ImageProcessor::preprocessImage(const std::string& imageFile, float* buffer, int inputH, int inputW) {
    cv::Mat img = cv::imread(imageFile);  // 이미지 파일읽기
    if (img.empty()) {  // 이미지 파일을 제대로 읽지 못했을 경우
        std::cerr << "이미지 로드 실패" << imageFile << std::endl;  // 오류 메시지 출력
        return false;  
    }

    cv::Mat resized;  // 리사이즈된 이미지를 저장할 행렬 선언
    cv::resize(img, resized, cv::Size(inputW, inputH));  // 이미지를 지정된 크기로 리사이즈
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);  // 이미지의 픽셀 값을 0-1 범위로 정규화 (float 형식)

    cv::imshow("src", resized);  // 리사이즈된 이미지를 화면에 표시 (디버깅 용도)
    cv::waitKey(1);  

    std::vector<cv::Mat> channels(3);  // BGR 채널을 저장할 벡터 생성
    cv::split(resized, channels);  // 리사이즈된 이미지를 B, G, R 세 개의 채널로 분리

    for (int i = 0; i < 3; ++i) {  // 각 채널에 대해 반복
        std::memcpy(buffer + i * inputH * inputW, channels[i].data, inputH * inputW * sizeof(float));  
        // 각 채널의 데이터를 모델 입력 버퍼에 복사
    }
    return true;  // 성공을 나타내는 true 반환
}*/


// 소프트맥스 함수: 입력 배열을 확률로 변환하는 함수
void ImageProcessor::softmax(float* input, int size) {
    float max = *std::max_element(input, input + size);  // 입력 배열에서 최대값을 찾아서 변수 max에 저장
    float sum = 0;  // 소프트맥스 계산을 위한 합 변수 초기화
    for (int i = 0; i < size; i++) {  // 입력 배열의 각 원소에 대해 반복
        input[i] = std::exp(input[i] - max);  // 입력 값에서 최대값을 뺀 후 지수 함수 적용
        sum += input[i];  // 지수 함수 결과를 sum에 추가하여 합 계산
    }
    for (int i = 0; i < size; i++) {  // 다시 입력 배열의 각 원소에 대해 반복
        input[i] /= sum;  // 각 원소를 합으로 나누어 확률로 변환
    }
}

// 세그멘테이션 마스크 생성 함수: 모델의 출력을 기반으로 세그멘테이션 마스크를 생성
/*void ImageProcessor::createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses) {
    int outputH = inputH;  // 출력 이미지의 높이를 입력 이미지의 높이로 설정
    int outputW = inputW;  // 출력 이미지의 너비를 입력 이미지의 너비로 설정
    // 세그멘테이션 마스크를 저장할 그레이스케일 이미지 (8비트, 단일 채널) 생성
    cv::Mat segmentationMask(outputH, outputW, CV_8UC1);  
    std::vector<int> classCounts(numClasses, 0);  // 각 클래스의 픽셀 개수를 저장할 벡터 초기화
    for (int y = 0; y < outputH; ++y) {  // 이미지의 각 픽셀 행에 대해 반복
        for (int x = 0; x < outputW; ++x) {  // 이미지의 각 픽셀 열에 대해 반복
            std::vector<float> pixelProbs(numClasses);  // 각 픽셀에 대해 클래스 확률을 저장할 벡터 생성
            for (int c = 0; c < numClasses; ++c) {  // 각 클래스에 대해 반복
            // 모델의 출력 버퍼에서 해당 픽셀의 클래스 확률을 가져와 저장
                pixelProbs[c] = outputBuffer[(c * outputH * outputW) + (y * outputW) + x];  
            }
            
            softmax(pixelProbs.data(), numClasses);  // 픽셀별 클래스 확률에 소프트맥스 적용하여 확률 값으로 변환
            // 가장 높은 확률을 가진 클래스 인덱스 찾기
            int maxClass = std::max_element(pixelProbs.begin(), pixelProbs.end()) - pixelProbs.begin();
            // 클래스 인덱스를 0-255 사이로 변환하여 마스크에 저장  
            segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass * 255 / (numClasses - 1));  
            classCounts[maxClass]++;  // 해당 클래스의 픽셀 개수 증가
        }
    }

    cv::Mat resizedMask;  // 리사이즈된 마스크 이미지를 저장할 행렬 선언
    cv::resize(segmentationMask, resizedMask, cv::Size(inputW, inputH), 0, 0, cv::INTER_NEAREST);  // 세그멘테이션 마스크를 원본 크기로 리사이즈
    cv::Mat colorMask;  // 컬러맵이 적용된 마스크 이미지를 저장할 행렬 선언
    cv::applyColorMap(resizedMask, colorMask, cv::COLORMAP_JET);  // 컬러맵을 적용하여 시각화하기 쉽게 변환
    cv::imshow("colorMask", colorMask);  // 컬러 마스크 이미지를 화면에 표시
    cv::waitKey(0);  // 키 입력 대기 (무한 대기)
    //cv::imwrite("segmentation_mask.png", colorMask);  // 컬러 마스크 이미지를 파일로 저장
    //std::cout << "Segmentation mask saved as 'segmentation_mask.png'" << std::endl;
    //cv::imwrite("segmentation_mask_gray.png", resizedMask);  // 그레이스케일 마스크 이미지를 파일로 저장
    //std::cout << "Grayscale segmentation mask saved as 'segmentation_mask_gray.png'" << std::endl;
}*/
// 새로 추가된 cv::Mat 객체를 인자로 받는 함수 정의
bool ImageProcessor::preprocessImage(const cv::Mat& image, float* buffer, int inputH, int inputW) {
    if (image.empty()) {
        std::cerr << "Image is empty." << std::endl;
        return false;
    }
    //cv::imshow("src",image);
    //cv::waitKey(0);
    // 입력 크기가 0이 아닌지 확인
    if (inputW <= 0 || inputH <= 0) {
        std::cerr << "Invalid input dimensions: " << inputW << "x" << inputH << std::endl;
        return false;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputW, inputH)); // 이미지 크기 조정
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f); // 0-255 범위를 0-1로 정규화
    //cv::imshow("src",resized);
    //cv::waitKey(0);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int i = 0; i < 3; ++i) {
        std::memcpy(buffer + i * inputH * inputW, channels[i].data, inputH * inputW * sizeof(float));
    }
    //cv::imshow("src",resized);
    //cv::waitKey(0);
    return true;
}

//사용자 정의 컬러맵
//void  ImageProcessor::colorizeSegmentation(const cv::Mat &score, cv::Mat &segm, int numClasses) {
    

    



//    const int rows = score.size[1];
//    const int cols = score.size[0];
//    const int chns = numClasses;
    
    

    /*cv::Mat maxCl = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1, score.data);
    for (int ch = 1; ch < chns; ch++) {
        for (int row = 0; row < rows; row++) {
            //const float *ptrScore = score.ptr<float>(0, ch, row);
            const uchar *ptrScore = score.ptr<uchar>(0, ch, row);
            uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
            //float *ptrMaxVal = maxVal.ptr<float>(row);
            uchar *ptrMaxVal = maxVal.ptr<uchar>(row);
            for (int col = 0; col < cols; col++) {
                if (ptrScore[col] > ptrMaxVal[col]) {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++) {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);  // 수정된 구문
        for (int col = 0; col < cols; col++) {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }*/
//    segm.create(rows, cols, CV_8UC3);
//    for (int row = 0; row < rows; row++) {
//        const uchar *ptrMaxCl = score.ptr<uchar>(row);
//        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);  // 수정된 구문
//        for (int col = 0; col < cols; col++) {
//           ptrSegm[col] = colors[ptrMaxCl[col]];
//        }
//    }
   
//}



//기존 createSegmentation마스크 함수 (Mat)객체 반환
/*cv::Mat ImageProcessor::createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses) {
    auto start4 = std::chrono::high_resolution_clock::now();//디버깅 용도 softMax 시작
    cv::Mat segmentationMask(inputH, inputW, CV_8UC1);
    for (int y = 0; y < inputH; ++y) {
        uchar *p = segmentationMask.ptr<uchar>(y);
        for (int x = 0; x < inputW; ++x) {
            int maxClass = 0;
            float maxProb = outputBuffer[y * inputW + x];
            for (int c = 1; c < numClasses; ++c) {
                float prob = outputBuffer[(c * inputH * inputW) + (y * inputW) + x];
                if (prob > maxProb) {
                    maxClass = c;
                    maxProb = prob;
                }
            }
            //segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass * 255 / (numClasses - 1));
            p[x] = static_cast<uchar>(maxClass * 255 / (numClasses - 1));
        }
    }
    cv::Mat colorMask;
    cv::applyColorMap(segmentationMask, colorMask,  cv::COLORMAP_RAINBOW);
    //colorizeSegmentation(segmentationMask, colorMask,numClasses);
     auto end4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime4 = end4 - start4;
    std::cout << "softMAX!!시간: " << inferenceTime4.count() << " seconds." << std::endl;//후처리 시간 측정끝
    return colorMask;
}*/


 // maxClass에 따른 마스크 값 설정
            /*if (validClasses.find(maxClass) != validClasses.end()) {
                // 유효한 클래스는 해당 클래스 ID로 설정 (장애물로 인식)
                segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass);
            } else if (maxClass == floorClassIndex) {
                // 바닥으로 인식할 클래스 (ID: 3)
                segmentationMask.at<uchar>(y, x) = static_cast<uchar>(floorClassIndex);  // 바닥은 자체 ID 유지
            } else {
                // 나머지 클래스는 모두 배경으로 처리 (0으로 설정)
                segmentationMask.at<uchar>(y, x) = 0;
            }*/

// 필요한 클래스 인덱스만을 사용하여 세그멘테이션 마스크 생성  최종 1
/*cv::Mat ImageProcessor::createSegmentationMask(float* outputBuffer, int inputH, int inputW) {
    cv::Mat segmentationMask(inputH, inputW, CV_8UC1);

    // 필요한 클래스 인덱스를 정의합니다.
    //std::vector<int> neededClassIndices = {12, 33, 15, 19, 5, 0, 3};  // 3번은 바닥 (floor) 인덱스
    int numNeededClasses = neededClasses.size();  // 필요로 하는 클래스 수
    std::cout<<"클래수 개수:"<<numNeededClasses<<std::endl;
    auto start4 = std::chrono::high_resolution_clock::now();  // 디버깅 용도 시작 시간
    for (int y = 0; y < inputH; ++y) {
        uchar *p = segmentationMask.ptr<uchar>(y);
        for (int x = 0; x < inputW; ++x) {
            int maxClass = 0;
            float maxProb = outputBuffer[y * inputW + x];
            for (int c = 1; c < numNeededClasses; ++c) {
                float prob = outputBuffer[(c * inputH * inputW) + (y * inputW) + x];
                if (prob > maxProb) {
                    maxClass = c;
                    maxProb = prob;
                }
            }
            //segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass * 255 / (numClasses - 1));
            p[x] = static_cast<uchar>(maxClass * 255 / (numNeededClasses - 1));
        }
    
           
    }

    // 컬러맵을 적용하여 시각화 가능하도록 변환
    cv::Mat colorMask;
    cv::applyColorMap(segmentationMask, colorMask, cv::COLORMAP_RAINBOW);

    auto end4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime4 = end4 - start4;
    std::cout << "소프트맥스 계산 시간: " << inferenceTime4.count() << " seconds." << std::endl;  // 처리 시간 출력

    return colorMask;
}*/










// softMAx 옛날꺼
/*cv::Mat ImageProcessor::createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses, bool applySoftmax) {
    cv::Mat segmentationMask(inputH, inputW, CV_8UC1);
    // 확률 계산을 위해 소프트맥스를 적용해야 하는 경우
    auto start4 = std::chrono::high_resolution_clock::now();//디버깅 용도 softMax 시작
    if (applySoftmax) {
        for (int y = 0; y < inputH; ++y) {
            for (int x = 0; x < inputW; ++x) {
                std::vector<float> pixelProbs(numClasses);
                for (int c = 0; c < numClasses; ++c) {
                    pixelProbs[c] = outputBuffer[(c * inputH * inputW) + (y * inputW) + x];
                }
                // 소프트맥스 적용
                softmax(pixelProbs.data(), numClasses);
                // 최대 확률을 가진 클래스 찾기
                int maxClass = std::max_element(pixelProbs.begin(), pixelProbs.end()) - pixelProbs.begin();
                segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass * 255 / (numClasses - 1));
            }
        }
    } else { 
        // 소프트맥스가 필요 없는 경우
        for (int y = 0; y < inputH; ++y) {
            for (int x = 0; x < inputW; ++x) {
                int maxClass = 0;
                float maxProb = outputBuffer[y * inputW + x];
                for (int c = 1; c < numClasses; ++c) {
                    float prob = outputBuffer[(c * inputH * inputW) + (y * inputW) + x];
                    if (prob > maxProb) {
                        maxClass = c;
                        maxProb = prob;
                    }
                }
                segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass * 255 / (numClasses - 1));
            }
        }
    }
    auto end4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime4 = end4 - start4;
    std::cout << "softMAX!!시간: " << inferenceTime4.count() << " seconds." << std::endl;//후처리 시간 측정끝
    // 컬러 맵 적용
    auto start5 = std::chrono::high_resolution_clock::now();//디버깅 용도 map 시작
    cv::Mat colorMask;
    colorizeSegmentation(segmentationMask, colorMask, numClasses);
    auto end5 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime5 = end5 - start5;
    std::cout << "map$$시간: " << inferenceTime5.count() << " seconds." << std::endl;//후처리 시간 측정끝
    return colorMask;
}*/

void ImageProcessor::colorizeSegmentation(const cv::Mat& score, cv::Mat& segm, int numClasses) {
    const int rows = score.rows;
    const int cols = score.cols;

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++) {
        const uchar *ptrMaxCl = score.ptr<uchar>(row);
        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
        for (int col = 0; col < cols; col++) {
            ptrSegm[col] = colors[ptrMaxCl[col]]; // colors는 전역 또는 클래스 외부에서 선언된 변수
        }
    }
}