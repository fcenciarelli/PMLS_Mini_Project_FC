// convnet.h

#ifndef CONVNET_H
#define CONVNET_H

typedef struct {
    double *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
    int input_C, input_H, input_W;
    int hidden_dim, num_classes;
    double weight_scale, reg;

} ConvNet;


void ConvNet_init(ConvNet *net, int C, int H, int W, int hidden_dim, int num_classes, double weight_scale, double reg );
double* ConvNet_forward(ConvNet *net, double *input);
void ConvNet_backward(ConvNet *net, double *dout);
void ConvNet_train(ConvNet *net, double *x, double *y, double lr, int batch_size, int epochs);
double ConvNet_eval(ConvNet *net, double *x, double *y);








#endif