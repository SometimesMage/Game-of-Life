#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "cell-data.h"

CellData* Cell_Init(int width, int height)
{
    CellData *cellData = (CellData*) calloc(1, sizeof(CellData));
    cellData->width = width;
    cellData->height = height;
    cellData->data = (char*) calloc((width + 2) * (height + 2), sizeof(char));
    return cellData;
}

CellData* Cell_InitWithFile(char *fileName)
{
    FILE *file = fopen(fileName, "r");

    int width, height;

    fscanf(file, "%d", &width);
    fscanf(file, "%d", &height);

    CellData *cellData = (CellData*) calloc(1, sizeof(CellData));
    cellData->width = width;
    cellData->height = height;
    cellData->data = (char*) calloc((width + 2) * (height + 2), sizeof(char));

    int x, y;
    int data;

    for(y = 0; y < height; y++) {
        for(x = 0; x < width; x++) {
            fscanf(file, "%d", &data);
            Cell_SetAt(cellData, x, y, data);
        }
    }

    fclose(file);
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

void Cell_Export(CellData *cellData)
{
    FILE *file = fopen("export.gol", "w");

    fprintf(file, "%d %d\n", cellData->width, cellData->height);

    int x, y;
    for(y = 0; y < cellData->height; y++) {
        for(x = 0; x < cellData->width; x++) {
            fprintf(file, "%d ", Cell_GetAt(cellData, x, y));
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void Cell_Clear(CellData *cellData)
{
    memset(cellData->data, 0, (cellData->width + 2) * (cellData->height +2) * sizeof(char));
}

void Cell_Clean(CellData *cellData)
{
    free(cellData->data);
    free(cellData);
}