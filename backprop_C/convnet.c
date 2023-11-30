// convnet.c

#include "convnet.h"
#include <stdio.h>
#include <stdlib.h>




void ConvNet_init(ConvNet *net, int C, int H, int W, int hidden_dim, int num_classes, double weight_scale, double reg )
{
    int conv_out_H = H / 4;
    int conv_out_W = W / 4;

    net->W1 = calloc(64 * C * 3 * 3, sizeof(double));
    net->b1 = calloc(64, sizeof(double));
    net->W2 = calloc(64 * 64 * 3 * 3, sizeof(double));
    net->b2 = calloc(64, sizeof(double));
    net->W3 = calloc(64 * conv_out_H * conv_out_W * hidden_dim, sizeof(double));
    net->b3 = calloc(hidden_dim, sizeof(double));
    net->W4 = calloc(hidden_dim * num_classes, sizeof(double));
    net->b4 = calloc(num_classes, sizeof(double));

    for (int i = 0; i < 64 * C * 3 * 3; ++i) net->W1[i] = ((double)rand() / RAND_MAX) * weight_scale;
    for (int i = 0; i < 64 * 64 * 3 * 3; ++i) net->W2[i] = ((double)rand() / RAND_MAX) * weight_scale;
    for (int i = 0; i < 64 * conv_out_H * conv_out_W * hidden_dim; ++i) net->W3[i] = ((double)rand() / RAND_MAX) * weight_scale;
    for (int i = 0; i < hidden_dim * num_classes; ++i) net->W4[i] = ((double)rand() / RAND_MAX) * weight_scale;

    net->reg = reg;
}