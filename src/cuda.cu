#include "cuda.h"

__global__ void computeFrame(char *in, char *out, int width, int height)
{
    int g_x = blockDim.x * blockIdx.x + threadIdx.x;
    int g_y = blockDim.y * blockIdx.y + threadIdx.y;

    if(g_x < width && g_y < height) {
        g_x += 1; g_y += 1; //Offset by one since there is a border buffer

        int sum = 0;
        sum += in[(g_x + 1) + (g_y + 1) * (width + 2)];
        sum += in[(g_x + 0) + (g_y + 1) * (width + 2)];
        sum += in[(g_x - 1) + (g_y + 1) * (width + 2)];
        sum += in[(g_x + 1) + (g_y + 0) * (width + 2)];
        sum += in[(g_x - 1) + (g_y + 0) * (width + 2)];
        sum += in[(g_x + 1) + (g_y - 1) * (width + 2)];
        sum += in[(g_x + 0) + (g_y - 1) * (width + 2)];
        sum += in[(g_x - 1) + (g_y - 1) * (width + 2)];

        if(in[g_x + g_y * (width + 2)]) {
            out[g_x + g_y * (width + 2)] = sum == 2 || sum == 3;
        } else {
            out[g_x + g_y * (width + 2)] = sum == 3;
        }
    }
}

__global__ void computeFrame2(char *in, char *out, int width, int height)
{
    extern __shared__ char s_in[];
    int g_x = blockDim.x * blockIdx.x + threadIdx.x;
    int g_y = blockDim.y * blockIdx.y + threadIdx.y;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    if(g_x < width && g_y < height) {
        g_x += 1; g_y += 1; t_x += 1; t_y += 1; //Offset by one since there is a border buffer

        //Base Cell
        s_in[t_x + t_y * (blockDim.x + 2)] = in[g_x + g_y * (width * 2)];

        //Upper Cell
        if(t_y == 1) {
            s_in[t_x + (t_y - 1) * (blockDim.x + 2)] = in[g_x + (g_y - 1) * (width * 2)];
        }

        //Lower Cell
        if(t_y == blockDim.y || g_y == height - 1) {
            s_in[t_x + (t_y + 1) * (blockDim.x + 2)] = in[g_x + (g_y + 1) * (width * 2)];
        }

        //Left Cell
        if(t_x == 1) {
            s_in[(t_x - 1) + t_y * (blockDim.x + 2)] = in[(g_x - 1) + g_y * (width * 2)];
        }

        //Rigth Cell
        if(t_x == blockDim.x || g_x == width - 1) {
            s_in[(t_x + 1) + t_y * (blockDim.x + 2)] = in[(g_x + 1) + g_y * (width * 2)];
        }

        //Upper-Left Corner
        if(t_x == 1 && t_y == 1) {
            s_in[(t_x - 1) + (t_y - 1) * (blockDim.x + 2)] = in[(g_x - 1) + (g_y - 1) * (width * 2)];
        }

        //Lower-Left Corner
        if(t_x == 1 && ((t_y == blockDim.y || g_y == height - 1))) {
            s_in[(t_x - 1) + (t_y + 1) * (blockDim.x + 2)] = in[(g_x - 1) + (g_y + 1) * (width * 2)];
        }

        //Upper-Right Corner
        if((t_x == blockDim.x || g_x == width - 1) && t_y == 1) {
            s_in[(t_x + 1) + (t_y - 1) * (blockDim.x + 2)] = in[(g_x + 1) + (g_y - 1) * (width * 2)];
        }

        //Lower-Right Corner
        if((t_x == blockDim.x || g_x == width - 1) && (t_y == blockDim.y || g_y == height - 1)) {
            s_in[(t_x + 1) + (t_y + 1) * (blockDim.x + 2)] = in[(g_x + 1) + (g_y + 1) * (width * 2)];
        }

        __syncthreads();

        int sum = 0;
        sum += s_in[(t_x + 1) + (t_y + 1) * (blockDim.x + 2)];
        sum += s_in[(t_x + 0) + (t_y + 1) * (blockDim.x + 2)];
        sum += s_in[(t_x - 1) + (t_y + 1) * (blockDim.x + 2)];
        sum += s_in[(t_x + 1) + (t_y + 0) * (blockDim.x + 2)];
        sum += s_in[(t_x - 1) + (t_y + 0) * (blockDim.x + 2)];
        sum += s_in[(t_x + 1) + (t_y - 1) * (blockDim.x + 2)];
        sum += s_in[(t_x + 0) + (t_y - 1) * (blockDim.x + 2)];
        sum += s_in[(t_x - 1) + (t_y - 1) * (blockDim.x + 2)];

        if(s_in[t_x + t_y * (blockDim.x + 2)]) {
            out[g_x + g_y * (width + 2)] = sum == 2 || sum == 3;
        } else {
            out[g_x + g_y * (width + 2)] = sum == 3;
        }
    }
}