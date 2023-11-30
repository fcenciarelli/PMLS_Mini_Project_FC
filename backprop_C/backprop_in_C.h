#ifndef BACKPROP_IN_C_H
#define BACKPROP_IN_C_H




int add(int a, int b);

void conv_forward(double *x, double *w, double *b, double *out,int N, int C, int H, int W, int F, int FH, int FW );

void FC_forward(double *x, double *w, double *b, double *out, int n, int m, int p );
void FC_backward(double *dout, double *x, double *w, double *dx, double *dw, double *db, int n, int m, int p);


typedef struct {
    int pool_height; 
    int pool_width;
    int stride;
} MaxPoolCache;


double* max_pool_forward(double* x, int N, int C, int H, int W, int pool_height, int pool_width, int stride, int* output_shape, MaxPoolCache* cache);


void relu_forward(double *x, double *out, int size);

void relu_backward(double *dout, double *cache, double *dx, int size);


#endif
