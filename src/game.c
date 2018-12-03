#include <stdlib.h>
#include <stdio.h>

#include "game.h"

#define RENDER_FPS 60
#define MILLS_PER_FRAME(fps) (1000 / fps)

void printSDLError(const char *msg);

Game* Game_Init(int width, int height, int cellSize, int fps, double (*computeFrame)(Game *game))
{
    Game *game = (Game*) calloc(1, sizeof(Game));
    game->width = width;
    game->height = height;
    game->cellSize = cellSize;
    game->fps = fps;
    game->computeFrame = computeFrame;
    game->lastFrameTime = 0;
    game->currentDelta = 0;
    game->currentPlayDelta = 0;
    game->totalComputedFrames = 0;
    game->totalComputeTime = 0.0f;
    game->quit = SDL_FALSE;
    game->play = SDL_FALSE;

    game->window = NULL;
    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
        printSDLError("SDL could not initialize!");
        free(game);
        return NULL;
    }

    game->window = SDL_CreateWindow("Game of Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width * cellSize, height * cellSize, SDL_WINDOW_SHOWN);
    if(game->window == NULL) {
        printSDLError("Window could not be created!");
        free(game);
        SDL_Quit();
        return NULL;
    }

    game->surface = SDL_GetWindowSurface(game->window);
    game->data = Cell_Init(width, height);

    return game;
}

Game* Game_InitWithFile(char *fileName, int cellSize, int fps, double (*computeFrame)(Game *game))
{
    Game *game = (Game*) calloc(1, sizeof(Game));
    game->data = Cell_InitWithFile(fileName);
    game->width = game->data->width;
    game->height = game->data->height;
    game->cellSize = cellSize;
    game->fps = fps;
    game->computeFrame = computeFrame;
    game->lastFrameTime = 0;
    game->currentDelta = 0;
    game->currentPlayDelta = 0;
    game->totalComputedFrames = 0;
    game->totalComputeTime = 0.0f;
    game->quit = SDL_FALSE;
    game->play = SDL_FALSE;

    game->window = NULL;
    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
        printSDLError("SDL could not initialize!");
        free(game);
        return NULL;
    }

    game->window = SDL_CreateWindow("Game of Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, game->width * cellSize, game->height * cellSize, SDL_WINDOW_SHOWN);
    if(game->window == NULL) {
        printSDLError("Window could not be created!");
        free(game);
        SDL_Quit();
        return NULL;
    }

    game->surface = SDL_GetWindowSurface(game->window);

    return game;
}

void Game_Start(Game *game)
{
    game->lastFrameTime = SDL_GetTicks();

    while(!game->quit) {
        int currentFrameTime = SDL_GetTicks();
        game->currentDelta += currentFrameTime - game->lastFrameTime;

        Game_HandleEvents(game);

        if(game->play) {
            game->currentPlayDelta += currentFrameTime - game->lastFrameTime;
            if(game->currentPlayDelta >= MILLS_PER_FRAME(game->fps)) {
                game->totalComputedFrames++;
                game->totalComputeTime += game->computeFrame(game);
                game->currentPlayDelta = 0;
            }
        }

        if(game->currentDelta >= MILLS_PER_FRAME(RENDER_FPS)) {
            Game_Render(game);
            game->currentDelta = 0;
        }

        game->lastFrameTime = currentFrameTime;
    }

    printf("Total Compute Time %f\n", game->totalComputeTime);
    printf("Total Frames Played: %d\n", game->totalComputedFrames);
    printf("Avg. Compute Time: %f\n", game->totalComputeTime / game->totalComputedFrames);
}

void Game_StartFrames(Game *game, int frames)
{
    int frame = 0;
    game->play = SDL_TRUE;
    game->lastFrameTime = SDL_GetTicks();

    while(frame < frames) {
        int currentFrameTime = SDL_GetTicks();
        game->currentDelta += currentFrameTime - game->lastFrameTime;

        if(game->currentDelta >= MILLS_PER_FRAME(game->fps)) { 
            game->totalComputedFrames++;
            game->totalComputeTime += game->computeFrame(game);
            Game_Render(game);
            game->currentDelta = 0;
            frame++;
        }

        game->lastFrameTime = currentFrameTime;
    }

    printf("Total Compute Time %f\n", game->totalComputeTime);
    printf("Total Frames Played: %d\n", game->totalComputedFrames);
    printf("Avg. Compute Time: %f\n", game->totalComputeTime / game->totalComputedFrames);

    Cell_Export(game->data);
}

void Game_HandleEvents(Game *game)
{
    SDL_Event e;
    while(SDL_PollEvent(&e) != 0) {
        if(e.type == SDL_QUIT) {
            game->quit = SDL_TRUE;
        } else if (e.type == SDL_MOUSEBUTTONDOWN) {
            if(!game->play) {
                Game_OnMouseDown(game);
            }
        } else if(e.type == SDL_KEYDOWN) {
            switch(e.key.keysym.sym) {
                case SDLK_l:
                    if(!game->play) {
                        game->totalComputeTime += game->computeFrame(game);
                        game->totalComputedFrames++;
                    }
                    break;
                case SDLK_k:
                    game->play = game->play ? SDL_FALSE : SDL_TRUE;
                    break;
                case SDLK_c:
                    if(!game->play) {
                        Cell_Clear(game->data);
                    }
                    break;
                case SDLK_e:
                    Cell_Export(game->data);
                    printf("Game data exported!\n");
                    break;
                case SDLK_MINUS:
                    game->fps = game->fps - 1;
                    break;
                case SDLK_EQUALS:
                    game->fps = game->fps + 1;
                    break;
            }
        }
    }
}

void Game_Render(Game *game) 
{
    //Clear surface
    int32_t color;

    if(game->play) {
        color = SDL_MapRGB(game->surface->format, 0x33, 0x33, 0x33);
    } else {
        color = SDL_MapRGB(game->surface->format, 0x55, 0x55, 0x55);
    }

    SDL_FillRect(game->surface, NULL, color);

    for(int y = 0; y < game->height; y++) {
        for(int x = 0; x < game->width; x++) {
            if(Cell_GetAt(game->data, x, y) == 1) {
                SDL_Rect rect = {x * game->cellSize, y * game->cellSize, game->cellSize, game->cellSize};
                SDL_FillRect(game->surface, &rect, SDL_MapRGB(game->surface->format, 0xFF, 0xFF, 0xFF));
            }
        }
    }

    SDL_UpdateWindowSurface(game->window);
}

void Game_OnMouseDown(Game *game)
{
    int x, y;
    SDL_GetMouseState(&x, &y);
    Cell_FlipAt(game->data, x/game->cellSize, y/game->cellSize);
}

void Game_Clean(Game *game)
{
    Cell_Clean(game->data);
    SDL_DestroyWindow(game->window);
    SDL_Quit();
    free(game);
}

void printSDLError(const char *msg)
{
    printf("%s SDL_Error: %s\n", msg, SDL_GetError());
}