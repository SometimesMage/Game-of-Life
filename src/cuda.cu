#include "cuda.h"

__device__ char getCell(char *data, int x, int y, int width);
__device__ void setCell(char *data, int x, int y, int widht, char value);

__global__ void computeFrame(char *in, char *out, int width, int height)
{
    int g_x = blockDim.x * blockIdx.x + threadIdx.x;
    int g_y = blockDim.y * blockIdx.y + threadIdx.y;

    if(g_x < width && g_y < height) {
        int sum = 0;
        sum += getCell(in, g_x - 1, g_y - 1, width);
        sum += getCell(in, g_x + 0, g_y - 1, width);
        sum += getCell(in, g_x + 1, g_y - 1, width);
        sum += getCell(in, g_x - 1, g_y + 0, width);
        sum += getCell(in, g_x + 1, g_y + 0, width);
        sum += getCell(in, g_x - 1, g_y + 1, width);
        sum += getCell(in, g_x + 0, g_y + 1, width);
        sum += getCell(in, g_x + 1, g_y + 1, width);

        if(getCell(in, g_x, g_y, width)) {
            setCell(out, g_x, g_y, width, sum == 2 || sum == 3);
        } else {
            setCell(out, g_x, g_y, width, sum == 3);
        }
    }
}

__device__ char getCell(char *data, int x, int y, int width)
{
    return data[(x + 1) * (y + 1) * (width + 2)];
}

__device__ void setCell(char *data, int x, int y, int width, char value)
{
    data[(x + 1) * (y + 1) * (width + 2)] = value;
}