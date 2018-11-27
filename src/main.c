#include <stdio.h>
#include <stdlib.h>

#include "game.h"
#include "cpu-frame-computer.h"
#include "gpu-frame-computer.h"

void printUsage();

int main(int argc, char **args)
{
    /*if(argc < 3) {
        printUsage();
        return 0;
    }*/

    Game *game = Game_Init(80, 60, 8, 5, cpuComputeFrame);

    if(game) {
        Game_Start(game);
        Game_Clean(game);
    }

    return 0;
}

void printUsage() 
{
    printf("Usage: ./gol [tile_size] [width] [height] (fps)\n");
}