#include <stdlib.h>
#include <string.h>

#include "cell-data.h"

CellData* Cell_Init(int width, int height)
{
    CellData *cellData = (CellData*) calloc(1, sizeof(CellData));
    cellData->width = width;
    cellData->height = height;
    cellData->data = (char*) calloc((width + 2) * (height + 2), sizeof(char));
    return cellData;
}

void Cell_Copy(CellData *dest, CellData *src)
{
    if(dest->width <= src->width && dest->height <= src->height) {
        memcpy(dest->data, src->data, (dest->width + 2) * (dest->height + 2) * sizeof(char));
    }
}

char Cell_GetAt(CellData *cellData, int x, int y)
{
    return cellData->data[(x + 1) + (y + 1) * (cellData->width + 2)];
}

void Cell_FlipAt(CellData *cellData, int x, int y)
{
    if(cellData->data[(x + 1) + (y + 1) * (cellData->width + 2)] == 1) {
        cellData->data[(x + 1) + (y + 1) * (cellData->width + 2)] = 0;
    } else {
        cellData->data[(x + 1) + (y + 1) * (cellData->width + 2)] = 1;
    }
}

void Cell_SetAt(CellData *cellData, int x, int y, char alive)
{
    cellData->data[(x + 1) + (y + 1) * (cellData->width + 2)] = alive;
}

void Cell_Clean(CellData *cellData)
{
    free(cellData->data);
    free(cellData);
}