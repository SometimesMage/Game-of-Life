extern "C" {
    #include "gpu-frame-computer.h"
}

#include <stdio.h>
#include "cuda.h"

void checkCudaError(cudaError_t error);

extern "C"
double gpuComputeFrame(Game *game)
{
    char *d_in, *d_out;

    checkCudaError(cudaMalloc(&d_in, (game->width + 2) * (game->height + 2) * sizeof(char)));
    checkCudaError(cudaMalloc(&d_out, (game->width + 2) * (game->height + 2) * sizeof(char)));

    checkCudaError(cudaMemset(d_out, 0, (game->width + 2) * (game->height + 2) * sizeof(char)));
    checkCudaError(cudaMemcpy(d_in, game->data->data, (game->width + 2) * (game->height + 2) * sizeof(char), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid(ceil(game->width/(float)BLOCK_SIZE), ceil(game->height/(float)BLOCK_SIZE), 1);
    int sharedMemorySize = (BLOCK_SIZE + 2) * (BLOCK_SIZE * 2) * sizeof(char);

    double startTime = currentTime();

    computeFrame2<<<grid, block, sharedMemorySize>>>(d_in, d_out, game->width, game->height);

    double endTime = currentTime();

    checkCudaError(cudaGetLastError());

    checkCudaError(cudaMemcpy(game->data->data, d_out, (game->width + 2) * (game->height + 2) * sizeof(char), cudaMemcpyDeviceToHost));

    checkCudaError(cudaFree(d_in));
    checkCudaError(cudaFree(d_out));

    return endTime - startTime;
}

void checkCudaError(cudaError_t error)
{
    if(error != 0) {
        printf("Cuda Error: %s\n", cudaGetErrorName(error));
    }
}