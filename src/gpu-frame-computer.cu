extern "C" {
    #include "gpu-frame-computer.h"
}

#include <stdio.h>
#include "cuda.h"

void checkCudaError(cudaError_t error, const char *file, int line);

extern "C"
double gpuComputeFrame(Game *game)
{
    char *d_in, *d_out;

    checkCudaError(cudaMalloc(&d_in, (game->width + 2) * (game->height + 2) * sizeof(char)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&d_out, (game->width + 2) * (game->height + 2) * sizeof(char)), __FILE__, __LINE__);

    checkCudaError(cudaMemset(d_out, 0, (game->width + 2) * (game->height + 2) * sizeof(char)), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(d_in, game->data->data, (game->width + 2) * (game->height + 2) * sizeof(char), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid(ceil(game->width/(float)BLOCK_SIZE), ceil(game->height/(float)BLOCK_SIZE), 1);
    int sharedMemorySize = (BLOCK_SIZE + 2) * (BLOCK_SIZE * 2) * sizeof(char);

    double startTime = currentTime();
    computeFrame2<<<grid, block, sharedMemorySize>>>(d_in, d_out, game->width, game->height);
    cudaDeviceSynchronize();
    double endTime = currentTime();

    checkCudaError(cudaGetLastError(), __FILE__, __LINE__);

    checkCudaError(cudaMemcpy(game->data->data, d_out, (game->width + 2) * (game->height + 2) * sizeof(char), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    checkCudaError(cudaFree(d_in), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_out), __FILE__, __LINE__);

    return endTime - startTime;
}

void checkCudaError(cudaError_t error, const char *file, int line)
{
    if(error != 0) {
        printf("Cuda Error: %s (%s:%d)\n", cudaGetErrorName(error), file, line);
    }
}