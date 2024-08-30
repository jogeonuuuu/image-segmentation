// CUDA 커널 함수 정의
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
__global__ void findMaxClass(const float* d_output, unsigned char* d_segmentationMask, int inputH, int inputW, int numNeededClasses) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < inputW && y < inputH) {
        int maxClass = 0;
        float maxProb = d_output[y * inputW + x];

        for (int c = 1; c < numNeededClasses; ++c) {
            float prob = d_output[(c * inputH * inputW) + (y * inputW) + x];
            if (prob > maxProb) {
                maxClass = c;
                maxProb = prob;
            }
        }

        d_segmentationMask[y * inputW + x] = static_cast<unsigned char>(maxClass);
    }
}
void wrapperfindMaxClass(const float* d_output, unsigned char* d_segmentationMask, int inputH, int inputW, int numNeededClasses)
{
    dim3 blockSize(16,16);
    dim3 gridSize((inputW + blockSize.x - 1) / blockSize.x, (inputH + blockSize.y - 1) / blockSize.y);
    //findMaxClass <<<gridSize, blockSize >>> (outputBuffer, d_segmentationMask, inputH, inputW, numClasses);
    findMaxClass<<<gridSize, blockSize >>>(d_output,d_segmentationMask,inputH,inputW,numNeededClasses);
}
