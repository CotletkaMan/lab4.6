#ifndef PARALEL_H_
#define PARALEL_H_

#include <stdio.h>

FILE* openFile(char*);
void closeFile(FILE*);
void writeToFile(FILE* , double* , int* , int x , int y);
void createScript(int X , int Y , double max);

int** createBlockOrder(int* order , int x , int y , int countBlocks);

#endif