#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void printUsage();

int main(int argc, char **args) 
{
    if(argc < 5) {
        printUsage();
        exit(0);
    }

    srand(time(0));

    int width = atoi(args[2]);
    int height = atoi(args[3]);
    int freq = atoi(args[4]);

    if(!width || !height || !freq) {
        printUsage();
        exit(0);
    }

    FILE *file = fopen(args[1], "w");
    fprintf(file, "%d %d\n", width, height);

    int x, y;
    for(y = 0; y < height; y++) {
        for(x = 0; x < width; x++) {
            if(rand() % freq == 0) {
                fprintf(file, "1 ");
            } else {
                fprintf(file, "0 ");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void printUsage()
{
    printf("./generate [fileName] [width] [height] [frequency]\n");
}