#include <stdlib.h>
#include <cassert>

#include <iostream>
#include <fstream>

#include <cblas.h>

#include <vector>
#include <algorithm>
#include <random>

void hadamard_product(double *a, double *b, double *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

inline double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

inline double d_sigmoid(double x)
{
    return x * (1 - x);
}

typedef struct mlp
{
    int input_size;
    int hidden_size;
    int output_size;

    double *hidden_weights;
    double *output_weights;

    double *hidden_bias;
    double *output_bias;

    double *input_layer;
    double *hidden_layer;
    double *output_layer;
} mlp_t;

double *rand_vector(int n)
{
    double *v = (double *)malloc(n * sizeof(double));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0, 1);

    for (int i = 0; i < n; i++)
    {
        v[i] = dist(gen);
    }

    return v;
}

mlp_t *create_mlp(int input_size, int hidden_size, int output_size)
{
    mlp_t *mlp = (mlp_t *)malloc(sizeof(mlp_t));

    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    mlp->hidden_weights = rand_vector(input_size * hidden_size);
    mlp->output_weights = rand_vector(hidden_size * output_size);

    mlp->hidden_bias = rand_vector(hidden_size);
    mlp->output_bias = rand_vector(output_size);

    mlp->hidden_layer = (double *)malloc(hidden_size * sizeof(double));
    mlp->output_layer = (double *)malloc(output_size * sizeof(double));

    return mlp;
}

void mlp_forward(mlp_t *mlp, double *x)
{
    mlp->input_layer = x;

    // h = W_h x
    cblas_dgemv(CblasRowMajor, CblasNoTrans, mlp->hidden_size, mlp->input_size, 1.0, mlp->hidden_weights, mlp->input_size, mlp->input_layer, 1, 0.0, mlp->hidden_layer, 1);
    // h += b_h
    cblas_daxpy(mlp->hidden_size, 1.0, mlp->hidden_bias, 1, mlp->hidden_layer, 1);
    // h = sigmoid(h)
    std::transform(mlp->hidden_layer, mlp->hidden_layer + mlp->hidden_size, mlp->hidden_layer, sigmoid);

    // y_hat = W_o h
    cblas_dgemv(CblasRowMajor, CblasNoTrans, mlp->output_size, mlp->hidden_size, 1.0, mlp->output_weights, mlp->hidden_size, mlp->hidden_layer, 1, 0.0, mlp->output_layer, 1);
    // y_hat += b_o
    cblas_daxpy(mlp->output_size, 1.0, mlp->output_bias, 1, mlp->output_layer, 1);
    // y_hat = sigmoid(y_hat)
    std::transform(mlp->output_layer, mlp->output_layer + mlp->output_size, mlp->output_layer, sigmoid);
}

void mlp_backprop(mlp_t *mlp, double *y, double alpha)
{
    double *delta_o = (double *)malloc(mlp->output_size * sizeof(double));
    double *delta_h = (double *)malloc(mlp->hidden_size * sizeof(double));

    // delta_o = y - y_hat
    cblas_dcopy(mlp->output_size, y, 1, delta_o, 1);
    cblas_daxpy(mlp->output_size, -1.0, mlp->output_layer, 1, delta_o, 1);
    // We overwrite our previous output with the derivative of sigmoid applied to it
    std::transform(mlp->output_layer, mlp->output_layer + mlp->output_size, mlp->output_layer, d_sigmoid);
    // delta_o = delta_o * d_sigmoid(output)
    hadamard_product(delta_o, mlp->output_layer, delta_o, mlp->output_size);

    // delta_h = W_o^T delta_o
    cblas_dgemv(CblasRowMajor, CblasTrans, mlp->hidden_size, mlp->output_size, 1.0, mlp->output_weights, mlp->hidden_size, delta_o, 1, 0.0, delta_h, 1);
    // We overwrite the hidden output with the derivative of sigmoid applied to it
    std::transform(mlp->hidden_layer, mlp->hidden_layer + mlp->hidden_size, mlp->hidden_layer, d_sigmoid);
    // delta_h *= d_sigmoid(hidden)
    hadamard_product(delta_h, mlp->hidden_layer, delta_h, mlp->hidden_size);

    // W_o += alpha * delta_o h^T
    cblas_dger(CblasRowMajor, mlp->output_size, mlp->hidden_size, alpha, delta_o, 1, mlp->hidden_layer, 1, mlp->output_weights, mlp->hidden_size);
    // b_o += alpha * delta_o
    cblas_daxpy(mlp->output_size, alpha, delta_o, 1, mlp->output_bias, 1);

    // W_h += alpha * delta_h x^T
    cblas_dger(CblasRowMajor, mlp->hidden_size, mlp->input_size, alpha, delta_h, 1, mlp->input_layer, 1, mlp->hidden_weights, mlp->input_size);
    // b_h += alpha * delta_h
    cblas_daxpy(mlp->hidden_size, alpha, delta_h, 1, mlp->hidden_bias, 1);

    fprintf(stderr, "Completed backprop\n");

    free(delta_h);
    free(delta_o);
}

void delete_mlp(mlp_t *mlp)
{
    free(mlp->hidden_weights);
    free(mlp->output_weights);

    free(mlp->hidden_bias);
    free(mlp->output_bias);

    free(mlp->hidden_layer);
    free(mlp->output_layer);

    free(mlp);
}

void print_vector(double *v, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n");
}

int main()
{
    int input_size = 1;
    int hidden_size = 8;
    int output_size = 1;

    mlp_t *mlp = create_mlp(input_size, hidden_size, output_size);

    print_vector(mlp->hidden_weights, input_size * hidden_size);
    print_vector(mlp->output_weights, hidden_size * output_size);

    print_vector(mlp->hidden_bias, hidden_size);
    print_vector(mlp->output_bias, output_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0, 1);

    for (int i = 0; i < 100000; i++)
    {
        double x[1] = {dist(gen)};
        double y[1] = {sin(x[0])};

        fprintf(stderr, "x: %f, y: %f\n", x[0], y[0]);

        fprintf(stderr, "Forward %d\n", i);
        mlp_forward(mlp, x);
        fprintf(stderr, "Output: %f\n", mlp->output_layer[0]);
        fprintf(stderr, "Backprop %d\n", i);
        mlp_backprop(mlp, y, 0.1);
    }

    double x[1] = {0.5};
    mlp_forward(mlp, x);

    fprintf(stderr, "Final output: %f\n", mlp->output_layer[0]);

    delete_mlp(mlp);
}