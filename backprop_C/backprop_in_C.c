#include "backprop_in_C.h"
#include <stdlib.h>

int add(int a, int b)
{
    return a + b;
}



void conv_forward(double *x, double *w, double *b, double *out,int N, int C, int H, int W, int F, int FH, int FW )
{
    int stride =1;
    int padding = (FH - 1) / 2;
    int out_H  = H;
    int out_W = W;



    int padded_H  = H + 2 * padding;
    int padded_W = W + 2 * padding;
    double *padded_x = (double *)calloc(N * C * padded_H * padded_W, sizeof(double));


    for (int n= 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int h = 0; h < H; ++h)
            {
                for (int w =0; w < W; ++w)
                {
                    int padded_index = ((n * C + c) * padded_H + (h + padding)) * padded_W + (w + padding);
                    int input_index = ((n * C + c) * H + h) * W + w;
                    padded_x[padded_index] = x[input_index];
                }
            }
        }


        for (int f = 0; f < F; ++f)
        {
            for (int h = 0; h < out_H; ++h) 
            {
                for (int w_2= 0; w_2 < out_W; ++w_2)
                {
                    double sum  = 0.0;
                    for (int fh = 0; fh < FH; ++fh)
                    {
                        for (int fw = 0; fw < FW; ++fw)
                        {
                            for (int c = 0; c < C; ++c)
                            {
                                int padded_index = ((n * C + c) * padded_H + (h * stride + fh)) * padded_W + (w_2 * stride + fw);
                                int weight_index = ((f * C + c) * FH + fh) * FW + fw;
                                sum += padded_x[padded_index] * w[weight_index];
                            }
                        }
                    }
                    int output_index = ((n * F + f) * out_H + h) * out_W + w_2;
                    out[output_index] = sum + b[f];
                }
            }
        }
    }

    free(padded_x);

}

// Forward for fully connected
// x: input matrix, size (n x m)
// w: weight matrix, size (m, p)
// b: bias vector, size (p)
// out: output matrix, size (n x p)
// n: samples number,  m: input features, p: output features
void FC_forward(double *x, double *w, double *b, double *out, int n, int m, int p) 
{
    // flattening can be added before

    for (int i = 0; i < n; ++i)
    {
        for (int j=0; j < p; ++j)
        {
            out[i * p + j] = b[j];
            for (int k = 0; k < m; ++k)
            {
                out[i * p + j] += x[i * m + k] * w[k * p + j];
            }
        }
    } 
}


void FC_backward(double *dout, double *x, double *w , double *dx, double *dw, double *db, int n, int m, int p )
{

    for (int i = 0; i < n * m; ++i) dx[i] = 0.0;
    for (int i = 0; i < m * p; ++i) dw[i] = 0.0;
    for (int i = 0; i < p; ++i) db[i] = 0.0;

    for (int i =0; i < n; ++i)
    {
        for (int j = 0; j < p; ++j) 
        {
            db[j] += dout[i * p +j];
            for (int k = 0; k < m; ++k)
            {
                dx[i * m + k] += w[k * p + j] * dout[i * p + j];
                dw[k * p + j] += x[i * m + k] * dout[i * p + j];
            }
        }
    }

}

double* max_pool_forward(double* x, int N, int C, int H, int W, int pool_height, int pool_width, int stride, int* output_shape, MaxPoolCache* cache)
{
    int out_H = 1 + (H - pool_height) / stride;
    int out_W = 1 + (W - pool_width) / stride;

    double* out = (double*)malloc(N * C * out_H * out_W * sizeof(double));

    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int h = 0; h < out_H; ++h)
            {
                for (int w = 0; w < out_W; ++w)
                {
                    double max_val = -1.0e30;
                    for (int ph = 0; ph < pool_height; ++ph) 
                    {
                        for (int pw = 0; pw < pool_width; ++pw) 
                        {
                            int h_index = h * stride + ph;
                            int w_index = w * stride + pw;
                            int idx = ((n * C + c) * H + h_index) * W + w_index;
                            if (x[idx] > max_val) 
                            {
                                max_val = x[idx];
                            }
                        }
                    }
                    out[((n * C + c) * out_H + h) * out_W + w] = max_val;
                }
            }
        }

    }

    output_shape[0] = N;
    output_shape[1] = C;
    output_shape[2] = out_H;
    output_shape[3] = out_W;
    cache ->pool_height = pool_height;
    cache -> pool_width = pool_width;
    cache ->stride = stride;

    return out;

} 



void relu_forward(double *x, double *out, int size)
{
    for (int i= 0; i < size; ++i)
    {
        out[i] = x[i] > 0 ? x[i] : 0;
    }

}

void relu_backward(double *dout, double *cache, double *dx, int size)
{
    for (int i= 0; i < size; ++i)
    {
        dx[i] = dout[i] * (cache[i] > 0);
    }

}
