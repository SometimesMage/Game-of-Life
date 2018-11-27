extern "C" {
    #include "gpu-frame-computer.h"
}

#include "cuda.h"

extern "C"
void gpuComputeFrame(Game *game)
{
    char *d_in, *d_out;

    cudaMalloc(&d_in, (game->width + 2) * (game->height + 2) * sizeof(char));
    cudaMalloc(&d_out, (game->width + 2) * (game->height + 2) * sizeof(char));

    cudaMemset(d_out, 0, (game->width + 2) * (game->height + 2) * sizeof(char));
    cudaMemcpy(d_in, game->data->data, (game->width + 2) * (game->height + 2) * sizeof(char), cudaMemcpyHostToDevice);

    dim3 block(128, 1, 1);
    dim3 grid(ceil(game->width/(float)128), game->height, 1);

    computeFrame<<<grid, block>>>(d_in, d_out, game->width, game->height);

    cudaMemcpy(game->data->data, d_out, (game->width + 2) * (game->height + 2) * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}