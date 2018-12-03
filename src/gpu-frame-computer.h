#ifndef __GPU_FRAME_COMPUTER_H__
#define __GPU_FRAME_COMPUTER_H__

#include "game.h"
#include "timing.h"

int BLOCK_SIZE;

double gpuComputeFrame(Game *game);

#endif //__GPU_FRAME_COMPUTER_H__