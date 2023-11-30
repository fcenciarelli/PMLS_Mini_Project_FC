
// Inspired by https://github.com/Gio200023/Continual-Learning-on-Max78000/tree/main/python_scripts

#ifndef __CNN_H__
#define __CNN_H__


#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

#define CNN_FAIL 0
#define CNN_OK 1


#define CNN_NUM_OUTPUTS 10
#define CNN_NUM_OUTPUTS_LAYER_0 48
#define CNN_NUM_OUTPUTS_LAYER_1 10
#define CNN_NUM_OUTPUTS_FROZEN_LAYER 192