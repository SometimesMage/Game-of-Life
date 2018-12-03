#include "cpu-frame-computer.h"
#include "timing.h"

double cpuComputeFrame(Game *game)
{
    CellData *afterData = Cell_Init(game->width, game->height);

    double startTime = currentTime();
    
    for(int y = 0; y < game->height; y++) {
        for(int x = 0; x < game->width; x++) {
            int sum = 0;
            sum += Cell_GetAt(game->data, x - 1, y - 1); //NE
            sum += Cell_GetAt(game->data, x + 0, y - 1); //N
            sum += Cell_GetAt(game->data, x + 1, y - 1); //NW
            sum += Cell_GetAt(game->data, x - 1, y + 0); //E
            sum += Cell_GetAt(game->data, x + 1, y + 0); //W
            sum += Cell_GetAt(game->data, x - 1, y + 1); //SE
            sum += Cell_GetAt(game->data, x + 0, y + 1); //S
            sum += Cell_GetAt(game->data, x + 1, y + 1); //SW

            if(Cell_GetAt(game->data, x, y) == 1) {
                Cell_SetAt(afterData, x, y, sum == 2 || sum == 3);
            } else {
                Cell_SetAt(afterData, x, y, sum == 3);
            }
        }
    }

    double endTime = currentTime();

    Cell_Copy(game->data, afterData);
    Cell_Clean(afterData);

    return endTime - startTime;
}