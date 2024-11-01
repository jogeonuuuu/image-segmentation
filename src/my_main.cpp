#include "segmentation/my_TensorRTEngine.hpp"
#include"segmentation/my_ImageProcessor.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <sys/time.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <memory>
#include <functional>
using std::placeholders::_1;


void mysub_callback(rclcpp::Node::SharedPtr node, 
    const sensor_msgs::msg::CompressedImage::SharedPtr msg,
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr mypub, 
    TensorRTEngine& engine)
{
    auto start = std::chrono::high_resolution_clock::now(); //count start(seconds)

    // Step 3: 영상 받기
    cv::Mat frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (frame.empty()) {
            std::cerr << "End of video or failed to grab sub_frame" << std::endl;
            return;
    }

    // Step 4: 각 프레임에 대해 추론 수행
    cv::Mat sgmt_img;
    cudaStream_t stream;
    cudaStreamCreate(&stream); // CUDA 스트림 생성
    engine.runInference(frame, sgmt_img, stream); // 비동기 추론 수행
    if (sgmt_img.empty()) {
        std::cerr << "segmentation mask error" << std::endl;
        return;
    }
    
    // Step 5: GPU 작업이 완료될 때까지 동기화 (추론이 끝났을 때), (비동기 모드일 때)
    cudaStreamSynchronize(stream);

    // Step 6: "gray_segmentation_image/compressed" TOPIC으로 Publishing
    std_msgs::msg::Header hdr;
    cv::resize(sgmt_img, sgmt_img, cv::Size(640, 360));
    auto img = cv_bridge::CvImage(hdr, "mono8", sgmt_img).toCompressedImageMsg();
    mypub->publish(*img); //sensor_msgs::msg::CompressedImage::SharedPtr img;

    // Step 7: 원본 프레임 및 추론된 마스크를 화면에 표시
    cv::imshow("Input Frame", frame);
    cv::imshow("Segmentation Image", sgmt_img);
    cv::waitKey(1);

    auto end = std::chrono::high_resolution_clock::now(); //count stop
    std::chrono::duration<double> inferenceTime = end - start;
    std::cout << "추론 시간: " << inferenceTime.count() << " seconds." << std::endl;
}

int main(int argc, char** argv)
{
    //const std::string onnxFile = "/home/linux/ros2_ws/src/segmentation/models/segmenter_low_model.onnx";
    //const std::string engineFileName = "/home/linux/ros2_ws/src/segmentation/models/segmenter_low.engine";
    const std::string onnxFile = "/home/linux/ros2_ws/src/segmentation/models/segmenter_high_model.onnx";
    const std::string engineFileName = "/home/linux/ros2_ws/src/segmentation/models/segmenter_high.engine";

    // Step 1: TensorRT 엔진 빌드
    TensorRTEngine engine;
    if (!engine.buildEngine(onnxFile, engineFileName)) {
        std::cerr << "Failed to build engine" << std::endl;
        return EXIT_FAILURE;
    }

    // Step 2: Opencv -> 직접 초기화 (한 번만 할 수 있도록?)
    initializeColors(engine.numClasses);

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("img_sgmt_Node");

    auto mypub = node->create_publisher<sensor_msgs::msg::CompressedImage>("gray_segmentation_image/compressed", 
        rclcpp::QoS(rclcpp::KeepLast(10)).best_effort());

    std::function<void(const sensor_msgs::msg::CompressedImage::SharedPtr msg)> fn;
    fn = std::bind(mysub_callback, node, _1, mypub, std::ref(engine));
    //engine -> std::ref(engine) : 객체의 복사본이 아니라 참조를 사용하도록 
    auto mysub = node->create_subscription<sensor_msgs::msg::CompressedImage>("image/compressed",
        rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(), fn);
    
    rclcpp::spin(node);
    rclcpp::shutdown();

    return EXIT_SUCCESS;
}
