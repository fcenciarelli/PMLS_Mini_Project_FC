


#ifndef __BACKPROPAGATION_H__
#define __BACKPROPAGATION_H__



#include "cnn.h"

#define UPDATE 1
#define FIND 0



#define ITERATIONS 1000
#define NUM_EPOCHS 100
#define BATCH_SIZE 20
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001


void output_layer3(const int32_t out, uint8_t *final);

int mod2(int val);

void get_bias(int8_t biases[10]);
void set_bias(int8_t biases[10]);

void get_weights(uint8_t  weights[10][192]);
void set_weights(uint8_t  weights[10][192]);

float cross_entropy_loss(float* softmax_output, int* lables, int output_size, int batch_size);

#endif
