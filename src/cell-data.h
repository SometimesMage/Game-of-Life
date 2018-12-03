#ifndef __CELL_DATA_H__
#define __CELL_DATA_H__

typedef struct cell_data {
    char *data;
    int width;
    int height;
} CellData;

CellData* Cell_Init(int width, int height);

CellData* Cell_InitWithFile(char *fileName);

void Cell_Copy(CellData *dest, CellData *src);

char Cell_GetAt(CellData *cellData, int x, int y);

void Cell_FlipAt(CellData *cellData, int x, int y);

void Cell_SetAt(CellData *cellData, int x, int y, char alive);

void Cell_Export(CellData *cellData);

void Cell_Clear(CellData *cellData);

void Cell_Clean(CellData *cellData);

#endif //__CELL_DATA_H__