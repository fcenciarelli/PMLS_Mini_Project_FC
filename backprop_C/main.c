#include <stdio.h>
#include <stdlib.h>
#include "backprop_in_C.h"




void testing_FC_forward() 
{

    int n = 10, m = 5, p = 3;

    double x[n *m];
    double w[m * p];
    double b[p];
    double out[n * p];

    for (int i = 0; i < n *m; ++i) x[i] = i;
    for (int i = 0; i < m *p; ++i) w[i] = i;
    for (int i = 0; i < p; ++i) b[i] = i;

    FC_forward(x, w, b, out, n, m , p);

    printf("Output matrix:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            printf("%f ", out[i * p + j]);
        }
        printf("\n");
    }

}


void testing_FC_backward()
{
    int n = 2, m = 3, p = 2;

    double *x = (double *)malloc(n * m * sizeof(double));
    double *w = (double *)malloc(m * p * sizeof(double));
    double *dout = (double *)malloc(n * p * sizeof(double));
    double *dx = (double *)malloc(n * m * sizeof(double));
    double *dw = (double *)malloc(m * p * sizeof(double));
    double *db = (double *)malloc(p * sizeof(double));

    double value = 1.0;
    for (int i = 0; i < n * m; ++i) x[i] = value++;
    for (int i = 0; i < m * p; ++i) w[i] = value++;
    for (int i = 0; i < n * p; ++i) dout[i] = value++;


    FC_backward(dout,x, w, dx, dw, db, n , m ,p);

    printf("dx:\n");
    for (int i = 0; i < n * m; ++i) {
        printf("%f ", dx[i]);
        if ((i + 1) % m == 0) printf("\n");
    }

    printf("\ndw:\n");
    for (int i = 0; i < m * p; ++i) {
        printf("%f ", dw[i]);
        if ((i + 1) % p == 0) printf("\n");
    }

    printf("\ndb:\n");
    for (int i = 0; i < p; ++i) {
        printf("%f ", db[i]);
    }
    printf("\n");

    free(x); free(w); free(dout); free(dx); free(dw); free(db);

}



void testing_max_pool()
{
    int N = 1, C = 1, H = 4, W = 4;
    int pool_height = 2, pool_width = 2, stride = 2;

    double x[N * C * H * W];

    for (int i=0; i < N * C * H * W; ++i)
    {
        x[i] = i + 1;
    }

    printf("Input of Max Pooling:\n");
    for (int i = 0; i < N * C * H * W; ++i) {
        printf("%f ", x[i]);
        if ((i + 1) % W == 0) printf("\n");
        if ((i + 1) % (H * W) == 0) printf("\n");
    }

    int output_shape[4];
    MaxPoolCache cache;

    double* out = max_pool_forward(x, N, C, H, W, pool_height, pool_width, stride, output_shape, &cache);

    printf("Max Pooling Output:\n");
    for (int i = 0; i < output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]; ++i) {
        printf("%f ", out[i]);
        if ((i + 1) % output_shape[3] == 0) printf("\n");
    }
    free(out);

}



int main() {
    int three = 1 + 2;

    testing_max_pool();
    printf("The sum of 1 and 2 is: ");
    printf("%d", three);
    printf("\n");
    return 0;
}