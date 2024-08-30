cmake_minimum_required(VERSION 3.10)
project(segmentation LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# CUDA 표준 설정 (필요한 경우)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link directories
link_directories(${CUDA_LIBRARIES})
link_directories(${TENSORRT_LIB_DIR})

# Add source files
set(SOURCES
    src/my_main.cpp
    src/my_ImageProcessor.cpp
    src/my_TensorRTEngine.cpp
    src/my_kernel.cu
)

# CUDA 옵션 설정
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_89,code=sm_89")

#build
add_executable(${PROJECT_NAME} ${SOURCES})
ament_target_dependencies(${PROJECT_NAME} rclcpp sensor_msgs cv_bridge OpenCV)

# 타겟 설정에서 CUDA 아키텍처를 설정합니다. (필요한 경우)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"  # 원하는 CUDA 아키텍처를 지정합니다.
)

# Include directories for the target
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)

# Install
install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

ament_package()
