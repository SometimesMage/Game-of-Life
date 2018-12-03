#ifndef __GAME_H__
#define __GAME_H__

#include <SDL2/SDL.h>
#include "cell-data.h"

typedef struct game {
    int width;
    int height;
    int cellSize;
    int fps;
    int lastFrameTime;
    int currentDelta;
    int currentPlayDelta;
    int totalComputedFrames;
    double totalComputeTime;
    SDL_bool quit;
    SDL_bool play;
    CellData *data;
    SDL_Window *window;
    SDL_Surface *surface;
    double (*computeFrame)(struct game *game);
} Game;

Game* Game_Init(int width, int height, int cellSize, int fps, double (*computeFrame)(Game *game));

Game* Game_InitWithFile(char *fileName, int cellSize, int fps, double (*computeFrame)(Game *game));

void Game_Start(Game *game);

void Game_StartFrames(Game *game, int frames);

void Game_HandleEvents(Game *game);

void Game_Render(Game *game);

void Game_OnMouseDown(Game *game);

void Game_Clean(Game *game);
#endif //__GAME_H__