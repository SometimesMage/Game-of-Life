#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "game.h"
#include "cpu-frame-computer.h"
#include "gpu-frame-computer.h"

void printUsage();
void setupCE(int argc, char **args);
void setupGE(int argc, char **args);
void setupCF(int argc, char **args);
void setupGF(int argc, char **args);
void setupCT(int argc, char **args);
void setupGT(int argc, char **args);

int main(int argc, char **args)
{
    if(argc < 2) {
        printUsage();
        exit(0);
    }

    if(strcmp("-h", args[1]) == 0) {
        printUsage();
    } else if(strcmp("-ce", args[1]) == 0) {
        setupCE(argc, args);
    } else if(strcmp("-ge", args[1]) == 0) {
        setupGE(argc, args);
    } else if(strcmp("-cf", args[1]) == 0) {
        setupCF(argc, args);
    } else if(strcmp("-gf", args[1]) == 0) {
        setupGF(argc, args);
    } else if(strcmp("-ct", args[1]) == 0) {
        setupCT(argc, args);
    } else if(strcmp("-gt", args[1]) == 0) {
        setupGT(argc, args);
    }

    return 0;
}

void setupCE(int argc, char **args)
{
    if(argc < 5) {
        printUsage();
        exit(0);
    }

    int width = atoi(args[2]);
    int height = atoi(args[3]);
    int cell_size = atoi(args[4]);

    if(!width || !height || !cell_size) {
        printf("Error: Invalid parameter!\n");
        printUsage();
        exit(0);
    }

    Game *game = Game_Init(width, height, cell_size, 5, cpuComputeFrame);

    if(!game) {
        printf("Error: Coulnd't initialize game!\n");
        exit(0);
    }

    Game_Start(game);
    Game_Clean(game);
}

void setupGE(int argc, char **args) 
{
    if(argc < 6) {
        printUsage();
        exit(0);
    }

    int width = atoi(args[2]);
    int height = atoi(args[3]);
    int cell_size = atoi(args[4]);
    BLOCK_SIZE = atoi(args[5]);

    if(!width || !height || !cell_size || !BLOCK_SIZE) {
        printf("Error: Invalid parameter!\n");
        printUsage();
        exit(0);
    }

    Game *game = Game_Init(width, height, cell_size, 5, gpuComputeFrame);

    if(!game) {
        printf("Error: Coulnd't initialize game!\n");
        exit(0);
    }

    Game_Start(game);
    Game_Clean(game);
}

void setupCF(int argc, char **args) 
{
    if(argc < 4) {
        printUsage();
        exit(0);
    }

    int cell_size = atoi(args[3]);

    if(!cell_size) {
        printf("Error: Invalid parameter!\n");
        printUsage();
        exit(0);
    }

    Game *game = Game_InitWithFile(args[2], cell_size, 5, cpuComputeFrame);

    if(!game) {
        printf("Error: Coulnd't initialize game!\n");
        exit(0);
    }

    Game_Start(game);
    Game_Clean(game);
}

void setupGF(int argc, char **args) 
{
    if(argc < 5) {
        printUsage();
        exit(0);
    }

    int cell_size = atoi(args[3]);
    BLOCK_SIZE = atoi(args[4]);

    if(!cell_size) {
        printf("Error: Invalid parameter!\n");
        printUsage();
        exit(0);
    }

    Game *game = Game_InitWithFile(args[2], cell_size, 5, gpuComputeFrame);

    if(!game) {
        printf("Error: Coulnd't initialize game!\n");
        exit(0);
    }

    Game_Start(game);
    Game_Clean(game);
}

void setupCT(int argc, char **args) 
{
    if(argc < 5) {
        printUsage();
        exit(0);
    }

    int cell_size = atoi(args[3]);
    int frames = atoi(args[4]);

    if(!cell_size || !frames) {
        printf("Error: Invalid parameter!\n");
        printUsage();
        exit(0);
    }

    Game *game = Game_InitWithFile(args[2], cell_size, 5, cpuComputeFrame);

    if(!game) {
        printf("Error: Coulnd't initialize game!\n");
        exit(0);
    }

    Game_StartFrames(game, frames);
    Game_Clean(game);
}

void setupGT(int argc, char **args) 
{
    if(argc < 6) {
        printUsage();
        exit(0);
    }

    int cell_size = atoi(args[3]);
    int frames = atoi(args[4]);
    BLOCK_SIZE = atoi(args[5]);

    if(!cell_size) {
        printf("Error: Invalid parameter!\n");
        printUsage();
        exit(0);
    }

    Game *game = Game_InitWithFile(args[2], cell_size, 5, gpuComputeFrame);

    if(!game) {
        printf("Error: Coulnd't initialize game!\n");
        exit(0);
    }

    Game_StartFrames(game, frames);
    Game_Clean(game);
}

void printUsage() 
{
    printf("Usage:\n\t./project -ce [width] [height] [cell_size]\n");
    printf("\t./project -ge [width] [height] [cell_size] [block_size]\n");
    printf("\t./project -cf [file_name] [cell_size]\n");
    printf("\t./project -gf [file_name] [cell_size] [block_size]\n");
    printf("\t./project -ct [file_name] [cell_size] [frames]\n");
    printf("\t./project -gt [file_name] [cell_size] [frames] [block_size]\n");
    printf("\t./project -h\n");
    printf("\nFlags:\n\t-ce : Edit, compute via CPU\n");
    printf("\t-ge : Edit, compute via GPU\n");
    printf("\t-cf : Open and edit, compute via CPU\n");
    printf("\t-gf : Open and edit, compute via GPU\n");
    printf("\t-ct : Open and time, compute via CPU\n");
    printf("\t-gt : Open and time, compute via GPU\n");
    printf("\t-h  : Prints out this messages\n");
    printf("\nParameters:\n\twidth : The amount of cells in the x direction\n");
    printf("\theight : The amount of cells in the y direction\n");
    printf("\tcell_size : The amount of pixels a cell takes up on the screen\n");
    printf("\tblock_size : The amount of threads a block will be on the gpu\n");
    printf("\tframes : The number of frames to run the timing\n");
    printf("\tfile_name : The name of the file you want to open\n");
    printf("\nEditor:\n\t'l' : Advance one frame\n");
    printf("\t'k' : Play/pause game\n");
    printf("\t'e' : Export out game board to file name 'export.gol'\n");
    printf("\t'c' : Clear game board\n");
    printf("\t'-' : Decreases the FPS of computing a generation\n");
    printf("\t'=' : Increases the FPS of computing a generation\n");
}