#ifndef __GAME_H__
#define __GAME_H__

#include <SDL2/SDL.h>
#include "cell-data.h"

typedef struct game {
    int width;
    int height;
    int tileSize;
    int fps;
    int lastFrameTime;
    int currentDelta;
    int currentPlayDelta;
    SDL_bool quit;
    SDL_bool play;
    CellData *data;
    SDL_Window *window;
    SDL_Surface *surface;
    void (*computeFrame)(struct game *game);
} Game;

Game* Game_Init(int width, int height, int tileSize, int fps, void (*computeFrame)(Game *game));

void Game_Start(Game *game);

void Game_HandleEvents(Game *game);

void Game_Render(Game *game, int delta);

void Game_OnMouseDown(Game *game);

void Game_Clean(Game *game);
#endif //__GAME_H__