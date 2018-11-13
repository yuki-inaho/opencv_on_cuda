#include "myKernel.cuh"
#include <opencv2/core.hpp>
#include <opencv2/cudev.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void myKernel(cv::cuda::PtrStepSz<uchar> src, cv::cuda::PtrStepSz<uchar> dst)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if((x < dst.cols) && (y < dst.rows)){
        dst.ptr(y)[x] = UCHAR_MAX - src.ptr(y)[x];
    }
}

void launchMyKernel(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    cv::cuda::PtrStepSz<uchar> pSrc = 
        cv::cuda::PtrStepSz<uchar>(src.rows, src.cols * src.channels(), src.ptr<uchar>(), src.step);
    cv::cuda::PtrStepSz<uchar> pDst = 
        cv::cuda::PtrStepSz<uchar>(dst.rows, dst.cols * dst.channels(), dst.ptr<uchar>(), dst.step);

    const dim3 block(64, 8);
    const dim3 grid(cv::cudev::divUp(src.cols, block.x), cv::cudev::divUp(src.rows, block.y));

    myKernel<<<grid, block>>>(pSrc, pDst);
}

