extern "C" {
    #include "gpu-frame-computer.h"
}

#include <stdio.h>
#include "cuda.h"

void checkCudaError(cudaError_t error);

extern "C"
void gpuComputeFrame(Game *game)
{
    char *d_in, *d_out;

    checkCudaError(cudaMalloc(&d_in, (game->width + 2) * (game->height + 2) * sizeof(char)));
    checkCudaError(cudaMalloc(&d_out, (game->width + 2) * (game->height + 2) * sizeof(char)));

    checkCudaError(cudaMemset(d_out, 0, (game->width + 2) * (game->height + 2) * sizeof(char)));
    checkCudaError(cudaMemcpy(d_in, game->data->data, (game->width + 2) * (game->height + 2) * sizeof(char), cudaMemcpyHostToDevice));

    dim3 block(128, 1, 1);
    dim3 grid(ceil(game->width/(float)128), game->height, 1);

    computeFrame<<<grid, block>>>(d_in, d_out, game->width, game->height);
    checkCudaError(cudaGetLastError());

    checkCudaError(cudaMemcpy(game->data->data, d_out, (game->width + 2) * (game->height + 2) * sizeof(char), cudaMemcpyDeviceToHost));

    checkCudaError(cudaFree(d_in));
    checkCudaError(cudaFree(d_out));
}

void checkCudaError(cudaError_t error)
{
    if(error != 0) {
        printf("Cuda Error: %s\n", cudaGetErrorName(error));
    }
}