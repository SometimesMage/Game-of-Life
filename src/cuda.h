#ifndef __CUDA_H__
#define __CUDA_H__

__global__ void computeFrame(char *in, char *out, int width, int height);

__global__ void computeFrame2(char *in, char *out, int width, int height);

#endif //__CUDA_H__