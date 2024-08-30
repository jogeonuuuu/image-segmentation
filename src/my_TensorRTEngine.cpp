

#include "segmentation/my_TensorRTEngine.hpp"  // TensorRTEngine 클래스의 선언을 포함하는 헤더 파일
#include "segmentation/my_ImageProcessor.hpp"  // 이미지 전처리 및 후처리를 위한 유틸리티 함수들이 정의된 헤더 파일
#include <fstream>  // 파일 입출력을 위한 헤더 파일
#include <iostream> // 표준 입출력 사용을 위한 헤더 파일
#include "cuda_runtime_api.h"  // CUDA 런타임 API 사용을 위한 헤더 파일



//extern "C" void findMaxClass(const float* d_output, unsigned char* d_segmentationMask, int inputH, int inputW, int numNeededClasses);
void wrapperfindMaxClass(const float* d_output, unsigned char* d_segmentationMask, int inputH, int inputW, int numNeededClasses);


//std::vector<cv::Vec3b>& colors;
// Logger 클래스의 log 함수 구현: 주어진 심각도에 따라 메시지를 출력
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) // 경고 이상의 심각도인 경우 메시지를 출력
        std::cout << msg << std::endl;
}

std::vector<int> neededClasses = {12, 33, 15, 19, 5,0,3};

// TensorRTEngine 클래스 생성자: 멤버 변수 초기화
TensorRTEngine::TensorRTEngine() 
    : runtime(nullptr), engine(nullptr), context(nullptr), inputH(0), inputW(0), numClasses(0) {}

// TensorRTEngine 클래스 소멸자: 자원을 해제하는 cleanup 함수 호출
TensorRTEngine::~TensorRTEngine() {
    cleanup();
}

// 파일을 읽어 벡터에 데이터를 저장하는 함수
void TensorRTEngine::readFile(const std::string& fileName, std::vector<char>& buffer) {
    std::ifstream file(fileName, std::ios::binary | std::ios::ate); // 파일을 바이너리 모드로 열고 끝으로 이동
    if (!file.is_open()) { // 파일 열기에 실패한 경우 에러 메시지 출력 후 프로그램 종료
        std::cerr << "파일 열기 실패" << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    std::streamsize size = file.tellg(); // 파일 크기 가져오기
    file.seekg(0, std::ios::beg); // 파일의 시작으로 이동
    buffer.resize(size); // 버퍼의 크기를 파일 크기로 설정
    if (!file.read(buffer.data(), size)) { // 파일 내용을 버퍼에 읽어오기
        std::cerr << "파일 읽기 실패 " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ONNX 파일을 사용하여 TensorRT 엔진을 생성하고 파일로 저장하는 함수 원래 함수
/*bool TensorRTEngine::buildEngine(const std::string& onnxFile, const std::string& engineFileName) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger); // TensorRT 빌더 생성
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0); // 빈 네트워크 정의 생성
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger); // ONNX 파서 생성
    
    std::vector<char> modelData;
    readFile(onnxFile, modelData); // ONNX 모델 파일 읽어오기
    if (!parser->parse(modelData.data(), modelData.size())) { // 모델 파싱
        std::cerr << "onnx모델 파서 실패" << std::endl;
        return false; // 파싱 실패 시 false 반환
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig(); // 빌더 설정 생성
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 작업 공간 메모리 풀 제한 설정 (1GB)

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config); // 직렬화된 엔진 생성
    if (!serializedEngine) { // 엔진 생성 실패 시
        std::cerr << "엔진 빌드 실패" << std::endl;
        return false;
    }
    
     // 직렬화된 엔진 데이터를 멤버 변수에 저장
    //serializedEngineData.resize(serializedEngine->size());
    //memcpy(serializedEngineData.data(), serializedEngine->data(), serializedEngine->size());
    std::ofstream engineFile(engineFileName, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();
    //엔진 파일 생성끝
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << "Failed to create CUDA engine" << std::endl;
        return false;
    }
    context = engine->createExecutionContext();
    nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0)); // 입력 텐서 차원 가져오기
    inputH = inputDims.d[2]; // 입력 텐서의 높이
    inputW = inputDims.d[3]; // 입력 텐서의 너비
    nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1)); // 출력 텐서 차원 가져오기
    numClasses = outputDims.d[1]; // 출력 텐서의 클래스 수
    // 할당된 자원 정리
    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;
    return true; // 엔진 빌드 성공 시 true 반환
}*/

// ONNX 파일을 사용하여 TensorRT 엔진을 생성하고 파일로 저장하거나 기존 엔진을 로드하는 함수  엔진있는지 확인후 없으면생성 
/*bool TensorRTEngine::buildEngine(const std::string& onnxFile, const std::string& engineFileName) {
    // Step 1: 엔진 파일이 이미 존재하는지 확인
    if (std::filesystem::exists(engineFileName)) {
        // 엔진 파일이 존재하면, 빌드하지 않고 로드
        std::ifstream engineFile(engineFileName, std::ios::binary);
        if (!engineFile.is_open()) {
            std::cerr << "엔진 파일 열기 실패: " << engineFileName << std::endl;
            return false;
        }
        
        // 엔진 파일의 크기를 얻고 데이터를 읽음
        engineFile.seekg(0, std::ios::end);
        std::streamsize engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        
        // 엔진 파일을 메모리에 읽어옴
        std::vector<char> engineData(engineSize);
        if (!engineFile.read(engineData.data(), engineSize)) {
            std::cerr << "엔진 파일 읽기 실패: " << engineFileName << std::endl;
            return false;
        }
        engineFile.close();

        // 엔진 데이터를 이용해 CUDA 엔진을 디시리얼라이즈
        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);
        if (!engine) {
            std::cerr << "엔진 파일에서 CUDA 엔진 생성 실패" << std::endl;
            return false;
        }

        // 컨텍스트와 입력, 출력 차원 정보 설정
        context = engine->createExecutionContext();
        nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
        inputH = inputDims.d[2];
        inputW = inputDims.d[3];
        nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
        numClasses = outputDims.d[1];

        return true; // 엔진 파일에서 성공적으로 로드
    }

    // Step 2: 엔진 파일이 없을 경우, ONNX 파일로부터 새 엔진을 빌드
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    
    std::vector<char> modelData;
    readFile(onnxFile, modelData);
    if (!parser->parse(modelData.data(), modelData.size())) {
        std::cerr << "ONNX 모델 파서 실패" << std::endl;
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "엔진 빌드 실패" << std::endl;
        return false;
    }
    
    std::ofstream engineFile(engineFileName, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << "CUDA 엔진 생성 실패" << std::endl;
        return false;
    }

    // 컨텍스트와 입력, 출력 차원 정보 설정
    context = engine->createExecutionContext();
    nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
    numClasses = outputDims.d[1];

    // 자원 해제
    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true; // 엔진 빌드 성공 시 true 반환
}*/



//엔진 생성하여 저장, 기존 엔진이 있는경우 로드 (FP16으로 최적화)
bool TensorRTEngine::buildEngine(const std::string& onnxFile, const std::string& engineFileName) {
    // Step 1: 엔진 파일이 이미 존재하는지 확인
    if (std::filesystem::exists(engineFileName)) {
        // 엔진 파일이 존재하면, 빌드하지 않고 로드
        std::ifstream engineFile(engineFileName, std::ios::binary);
        if (!engineFile.is_open()) {
            std::cerr << "엔진 파일 열기 실패: " << engineFileName << std::endl;
            return false;
        }

        // 엔진 파일의 크기를 얻고 데이터를 읽음
        engineFile.seekg(0, std::ios::end);
        std::streamsize engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);

        // 엔진 파일을 메모리에 읽어옴
        std::vector<char> engineData(engineSize);
        if (!engineFile.read(engineData.data(), engineSize)) {
            std::cerr << "엔진 파일 읽기 실패: " << engineFileName << std::endl;
            return false;
        }
        engineFile.close();

        // 엔진 데이터를 이용해 CUDA 엔진을 디시리얼라이즈
        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);
        if (!engine) {
            std::cerr << "엔진 파일에서 CUDA 엔진 생성 실패" << std::endl;
            return false;
        }

        // 컨텍스트와 입력, 출력 차원 정보 설정
        context = engine->createExecutionContext();
        nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
        inputH = inputDims.d[2];
        inputW = inputDims.d[3];
        nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
        numClasses = outputDims.d[1];

        return true; // 엔진 파일에서 성공적으로 로드
    }

    // Step 2: 엔진 파일이 없을 경우, ONNX 파일로부터 새 엔진을 빌드
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    std::vector<char> modelData;
    readFile(onnxFile, modelData);
    if (!parser->parse(modelData.data(), modelData.size())) {
        std::cerr << "ONNX 모델 파서 실패" << std::endl;
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // FP16 모드 활성화
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1,3,512,512));
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1,3,512,512));
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1,3,512,512));

    config->addOptimizationProfile(profile);
    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "엔진 빌드 실패" << std::endl;
        return false;
    }

    std::ofstream engineFile(engineFileName, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << "CUDA 엔진 생성 실패" << std::endl;
        return false;
    }

    // 컨텍스트와 입력, 출력 차원 정보 설정
    context = engine->createExecutionContext();
    nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
    numClasses = outputDims.d[1];

    // 자원 해제
    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true; // 엔진 빌드 성공 시 true 반환
}


// 생성된 TensorRT 엔진을 사용하여 이미지를 입력으로 받아 추론을 수행하는 함수
/*bool TensorRTEngine::runInference(const std::string& imageFile) {
    if (serializedEngineData.empty()) {
        std::cerr << "Serialized engine data is empty. Build the engine first." << std::endl;
        return false;
    }
    // 런타임과 엔진을 생성하는 코드
    
    //auto engine_start=std::chrono::high_resolution_clock::now();
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedEngineData.data(), serializedEngineData.size());
    if (!engine) {
        std::cerr << "엔진 생성 실패" << std::endl;
        return false;
    }
    if(!context){
    context = engine->createExecutionContext(); // 추론 실행 컨텍스트 생성
    // 입력 텐서와 출력 텐서의 모양을 가져옵니다.
    nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0)); // 입력 텐서 차원 가져오기
    inputH = inputDims.d[2]; // 입력 텐서의 높이
    inputW = inputDims.d[3]; // 입력 텐서의 너비
    nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1)); // 출력 텐서 차원 가져오기
    numClasses = outputDims.d[1]; // 출력 텐서의 클래스 수
    }
    int32_t nIO = engine->getNbIOTensors(); // 엔진의 입력과 출력 텐서 개수 가져오기
    std::vector<std::string> vTensorName(nIO); // 텐서 이름을 저장할 벡터
    buffers.resize(nIO); // CUDA 버퍼 포인터를 저장할 벡터 크기 조정
    bufferSizes.resize(nIO); // 각 버퍼의 크기를 저장할 벡터 크기 조정

    for (int i = 0; i < nIO; ++i) { // 각 텐서에 대해 반복
        vTensorName[i] = std::string(engine->getIOTensorName(i)); // 텐서 이름 가져오기
        nvinfer1::Dims dims = context->getTensorShape(vTensorName[i].c_str()); // 텐서의 차원 가져오기
        int64_t size = 1; // 텐서의 크기 계산을 위한 초기값
        for (int j = 0; j < dims.nbDims; ++j)
            size *= dims.d[j]; // 모든 차원을 곱하여 텐서의 총 요소 수 계산
        bufferSizes[i] = size * sizeof(float); // 텐서의 버퍼 크기 계산

        cudaMalloc(&buffers[i], bufferSizes[i]); // CUDA 메모리에 버퍼 할당
    }

    float* inputBuffer = new float[bufferSizes[0] / sizeof(float)]; // 입력 버퍼 할당
    if (!ImageProcessor::preprocessImage(imageFile, inputBuffer, inputH, inputW)) { // 이미지 전처리 수행
        std::cerr << "이미지 전처리 실패" << std::endl;
        return false;
    }
    cudaMemcpy(buffers[0], inputBuffer, bufferSizes[0], cudaMemcpyHostToDevice); // 전처리된 이미지를 GPU 메모리로 복사

    for (int i = 0; i < nIO; ++i) { // 텐서 주소 설정
        context->setTensorAddress(vTensorName[i].c_str(), buffers[i]);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream); // CUDA 스트림 생성
    auto start = std::chrono::high_resolution_clock::now();//추론시작 
    bool status = context->enqueueV3(stream); // 추론 수행
    if (!status) { // 추론 실패 시
        std::cerr << "추론실패!" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream); // CUDA 스트림이 완료될 때까지 대기
    
    auto end = std::chrono::high_resolution_clock::now();// 추론 종료 시간 기록
    std::chrono::duration<double> inferenceTime = end - start;
    // 추론 시간 출력
    std::cout << "추론하는데 걸린시간:" << inferenceTime.count() << "초." << std::endl;//추론하는데 걸리는시간 출력
    float* outputBuffer = new float[bufferSizes[1] / sizeof(float)]; // 출력 버퍼 할당
    cudaMemcpy(outputBuffer, buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost); // 추론 결과를 GPU에서 CPU로 복사

    ImageProcessor::createSegmentationMask(outputBuffer, inputH, inputW, numClasses); // 추론 결과를 사용하여 분할 마스크 생성

    delete[] inputBuffer; // 입력 버퍼 메모리 해제
    delete[] outputBuffer; // 출력 버퍼 메모리 해제
    for (void* buffer : buffers) { // 각 CUDA 버퍼 메모리 해제
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream); // CUDA 스트림 제거

    return true; // 추론 성공 시 true 반환
}*/

//추론함수 원본
/*bool TensorRTEngine::runInference(cv::Mat& frame,cv::Mat& result) {
   
    // 컨텍스트가 초기화되지 않은 경우 초기화
    
    // CUDA 버퍼를 한 번만 설정
        int32_t nIO = engine->getNbIOTensors();
        //std::cout<<"nIO::"<<nIO<<std::endl;
        std::vector<std::string> vTensorName(nIO);
        //std::cout<<"vTensorName 개수:"<<vTensorName.size()<<std::endl;
        buffers.resize(nIO);
        bufferSizes.resize(nIO);
        for (int i = 0; i < nIO; ++i) {
            vTensorName[i] = std::string(engine->getIOTensorName(i));
            nvinfer1::Dims dims = context->getTensorShape(engine->getIOTensorName(i));
            int64_t size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                size *= dims.d[j];
            }
            bufferSizes[i] = size * sizeof(float);
            cudaMalloc(&buffers[i], bufferSizes[i]); // 메모리를 한 번만 할당
        }
    // 이미지 전처리 수행
    float* inputBuffer = new float[bufferSizes[0] / sizeof(float)];
    if (!ImageProcessor::preprocessImage(frame, inputBuffer, inputH, inputW)) {
        std::cerr << "Image preprocessing failed." << std::endl;
        delete[] inputBuffer;
        return false;
    }
    //auto start = std::chrono::high_resolution_clock::now();
    // 입력 데이터를 GPU 메모리로 복사
    cudaMemcpy(buffers[0], inputBuffer, bufferSizes[0], cudaMemcpyHostToDevice);
    //std::cout<<"이름나오기 직전:"<<std::endl;
    for (int i = 0; i < nIO; ++i) { // defining tensor adress 
        context->setTensorAddress(vTensorName[i].c_str(), buffers[i]);
    }
    //std::cout<<"오류 부분::"<<vTensorName[2]<<std::endl;
    // 추론 수행
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Inference failed!" << std::endl;
        delete[] inputBuffer;
        cudaStreamDestroy(stream);
        return false;
    }
    cudaStreamSynchronize(stream);
    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> inferenceTime = end - start;
    //std::cout << "추론 시간: " << inferenceTime.count() << " seconds." << std::endl;
    // 출력 데이터를 CPU 메모리로 복사
    float* outputBuffer = new float[bufferSizes[1] / sizeof(float)];
    cudaMemcpy(outputBuffer, buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost);
    

    // 세그멘테이션 마스크 생성
    result = ImageProcessor::createSegmentationMask(outputBuffer, inputH, inputW, numClasses);

    // 메모리 해제
    delete[] inputBuffer;
    delete[] outputBuffer;
    for (void* buffer : buffers) {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream);

    return true; // 결과 마스크 반환
}*/



//수정된 추론함수(비동기 모드)GPU CPU 따로 작업
bool TensorRTEngine::runInference(cv::Mat& frame, cv::Mat& result, cudaStream_t stream){
 int32_t nIO = engine->getNbIOTensors();
    std::vector<std::string> vTensorName(nIO);
    buffers.resize(nIO);
    bufferSizes.resize(nIO);

    for (int i = 0; i < nIO; ++i) {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nvinfer1::Dims dims = context->getTensorShape(engine->getIOTensorName(i));
        int64_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        bufferSizes[i] = size * sizeof(float);
        cudaMalloc(&buffers[i], bufferSizes[i]);
    }
    // 고정된 메모리 할당
    auto start2 = std::chrono::high_resolution_clock::now(); //전처리 시간측정 시작
    float* inputBuffer;
    cudaMallocHost((void**)&inputBuffer, bufferSizes[0]); // 고정된 메모리 할당
    // 이미지 전처리 수행 원본
    //float* inputBuffer = new float[bufferSizes[0] / sizeof(float)];
    if (!ImageProcessor::preprocessImage(frame, inputBuffer, inputH, inputW)) {
        std::cerr << "Image preprocessing failed." << std::endl;
        //delete[] inputBuffer;
        return false;
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime2 = end2 - start2;
    std::cout << "전처리##시간: " << inferenceTime2.count() << " seconds." << std::endl;// 전처리시간측정 끝
    auto start1 = std::chrono::high_resolution_clock::now();
    // 입력 데이터를 GPU 메모리로 비동기 복사
    auto start7 = std::chrono::high_resolution_clock::now();//inputBuffer 시간 시작
    cudaMemcpyAsync(buffers[0], inputBuffer, bufferSizes[0], cudaMemcpyHostToDevice, stream);
    cudaError_t err = cudaGetLastError();
    auto end7 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime7 = end7 - start7;
    std::cout << "inputBuffer__cudaMemcpyAsync 시간: " << inferenceTime7.count() << " seconds." << std::endl;// 전처리시간측정 끝
    for (int i = 0; i < nIO; ++i) {
        context->setTensorAddress(vTensorName[i].c_str(), buffers[i]);
    }
    //delete[] inputBuffer;
    // 비동기 추론 수행
    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Inference failed!" << std::endl;
        //delete[] inputBuffer;
        return false;
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime1 = end1 - start1;
    std::cout << "진짜++추론 시간: " << inferenceTime1.count() << " seconds." << std::endl;
    //출력 데이터를 비동기 복사 고정된 메모리 할당
    auto start3 = std::chrono::high_resolution_clock::now();//후처리 시간 시작

   
    //cv::cuda::GpuMat d_segmentationMask(inputH, inputW, CV_8UC3);
    
    // 출력 데이터를 비동기로 CPU 메모리로 복사 원본
    auto start6 = std::chrono::high_resolution_clock::now();//메모리 할당 시간 시작
    //float* outputBuffer = new float[bufferSizes[1] / sizeof(float)];
    
    /*float* outputBuffer = new float[6 * inputH * inputW];  // 필요한 클래스(0~4)만큼의 버퍼 크기 할당

    size_t numElementsPerClass = inputH * inputW;
    size_t offset = 0;
    for (int i = 0; i < 6; ++i) {  // 필요한 클래스(0~5)만큼 반복
        cudaMemcpyAsync(outputBuffer + offset, reinterpret_cast<float*>(buffers[1]) + i * numElementsPerClass, numElementsPerClass * sizeof(float), cudaMemcpyDeviceToHost, stream);
        offset += numElementsPerClass;
    }*/

    // 필요한 클래스만큼의 메모리 할당
    //float* outputBuffer = new float[neededClasses.size() * inputH * inputW];
    //size_t numElementsPerClass = inputH * inputW;
    //size_t offset = 0;
    // 필요한 클래스 인덱스만 저장하는 벡터
    
    // 필요한 클래스 인덱스만 복사
    /*for (int i = 0; i < neededClassIndices.size(); ++i) {


        int idx = neededClassIndices[i];
        cudaMemcpyAsync(outputBuffer + offset, reinterpret_cast<float*>(buffers[1]) + idx * numElementsPerClass, numElementsPerClass * sizeof(float), cudaMemcpyDeviceToHost, stream);
        offset += numElementsPerClass;
    }*/
    
    //막은부분
    /*for (int i = 0; i < neededClasses.size(); ++i) {
        int idx = neededClasses[i];
        cudaMemcpyAsync(outputBuffer + offset, reinterpret_cast<float*>(buffers[1]) + idx * numElementsPerClass, numElementsPerClass * sizeof(float), cudaMemcpyDeviceToHost, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Memcpy Error: " << cudaGetErrorString(err) << std::endl;
            delete[] outputBuffer;
            return false;
        }
        offset += numElementsPerClass;
    }*/
    

    //cudaMemcpyAsync(outputBuffer, buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost, stream);//범인 0.027 잡아먹음
    auto end6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime6 = end6 - start6;
    std::cout << "outputBuffer^^__cufaMemcpyAsync시간: " << inferenceTime6.count() << " seconds." << std::endl;//메모리 시간 측정끝~~
    // GPU 작업이 완료될 때까지 기다리도록 동기화 (메인 코드에서)
    
    
    // 세그멘테이션 마스크 생성
    // 동기화: GPU 작업이 완료될 때까지 대기
    
    
    //result = ImageProcessor::createSegmentationMask(outputBuffer, inputH, inputW, numClasses);
    //result = ImageProcessor::createSegmentationMask(outputBuffer, inputH, inputW, 5);//5개의 클래스 원본
   
    unsigned char* outputBuffer;
    //cudaMallocHost((void**)&outputBuffer, inputH*inputW); // 고정된 메모리 할당
    cudaMallocHost((void**)&outputBuffer, inputH * inputW * sizeof(unsigned char)); // 메모리 크기 수정: unsigned char
    unsigned char* d_segmentationMask;
    cudaMalloc(&d_segmentationMask, inputH * inputW * sizeof(unsigned char));
    
    wrapperfindMaxClass ((const float*)buffers[1], d_segmentationMask, inputH, inputW, numClasses);
    cudaMemcpyAsync(outputBuffer, d_segmentationMask, inputH*inputW* sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    
    // CUDA 작업이 완료될 때까지 대기
    cudaStreamSynchronize(stream);
    
    cv::Mat segmentationMask(inputH, inputW, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < inputH; ++y) {
        for (int x = 0; x < inputW; ++x) {
           segmentationMask.at<uchar>(y, x) = outputBuffer[y*inputW + x];                       
        }
    }        
    

    //cv::Mat colorMask(inputW,inputH,CV_8UC3,cv::Scalar(0,0,0));
    //cv::applyColorMap(segmentationMask, colorMask,  cv::COLORMAP_JET );
    std::vector<cv::Vec3b> colors(150);
    colors[12]=cv::Vec3b(255,0,255);
    colors[33]=colors[15]=colors[56]=colors[64]=cv::Vec3b(255,255,0);
    colors[19]=colors[30]=colors[31]=colors[75]=colors[101]=cv::Vec3b(0,255,255); // chair..
    colors[5]=cv::Vec3b(128,128,128);
    colors[0]=cv::Vec3b(255,0,0);
    colors[3]=cv::Vec3b(255,255,255);
    colors[10]=cv::Vec3b(0,255,0);
    colors[14]=cv::Vec3b(128,128,0);
    colors[41]=cv::Vec3b(0,0,255);
    colors[50]=cv::Vec3b(0,128,128);
    colors[74]=colors[143]=cv::Vec3b(128,0,128);
    colors[139]=cv::Vec3b(0,200,0); // fan

    cv::Mat colorMask(inputH, inputW, CV_8UC3, cv::Scalar(0,0,0));

    for (int row = 0; row < segmentationMask.rows; row++) {
        for (int col = 0; col < segmentationMask.cols; col++) {
            colorMask.at<cv::Vec3b>(row, col) = colors[segmentationMask.at<uchar>(row, col)];
        }
    }

    result = colorMask;
    
    // 메모리 해제
    for (void* buffer : buffers) {
        cudaFree(buffer);
    }    
    cudaFreeHost(inputBuffer);
    cudaFreeHost(outputBuffer);
    cudaFree(d_segmentationMask);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime3 = end3 - start3;
    std::cout << "후처리@@시간: " << inferenceTime3.count() << " seconds." << std::endl;//후처리 시간 측정끝
    
    return true;
}


// 할당된 자원을 해제하는 함수
void TensorRTEngine::cleanup() {
    if (context) {
        delete context; // 컨텍스트 해제
        context = nullptr;
    }
    if (engine) {
        delete engine; // 엔진 해제
        engine = nullptr;
    }
    if (runtime) {
        delete runtime; // 런타임 해제
        runtime = nullptr;
    }
}
