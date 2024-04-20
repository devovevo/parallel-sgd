#include <stdlib.h>
#include <cassert>

#include <iostream>
#include <fstream>

#include <cblas.h>

#include <vector>
#include <algorithm>
#include <random>

#include "activations.h"

typedef double (*fun_t)(double);

void hadamard_product(double *a, double *b, double *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

typedef struct mlp
{
    int num_layers;
    int *layer_sizes;

    double **weights;
    double **biases;

    fun_t *activations;
    fun_t *d_activations;

    double **layers;
    double **deltas;
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

void print_vector(double *v, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n");
}

mlp_t *create_mlp(int num_layers, int *layers, fun_t *activations, fun_t *d_activations)
{
    mlp_t *mlp = (mlp_t *)malloc(sizeof(mlp_t));

    mlp->num_layers = num_layers;
    mlp->layer_sizes = (int *)malloc(num_layers * sizeof(int));

    mlp->weights = (double **)malloc((num_layers - 1) * sizeof(double *));
    mlp->biases = (double **)malloc((num_layers - 1) * sizeof(double *));

    mlp->activations = activations;
    mlp->d_activations = d_activations;

    mlp->layers = (double **)malloc(num_layers * sizeof(double *));
    mlp->deltas = (double **)malloc(num_layers * sizeof(double *));

    for (int i = 0; i < num_layers; i++)
    {
        mlp->layer_sizes[i] = layers[i];

        if (i < num_layers - 1)
        {
            mlp->weights[i] = rand_vector(layers[i + 1] * layers[i]);
            mlp->biases[i] = rand_vector(layers[i + 1]);
        }

        mlp->layers[i] = (double *)malloc(layers[i] * sizeof(double));
        mlp->deltas[i] = (double *)malloc(layers[i] * sizeof(double));
    }

    return mlp;
}

void mlp_forward(mlp_t *mlp, double *x)
{
    cblas_dcopy(mlp->layer_sizes[0], x, 1, mlp->layers[0], 1);

    for (int i = 0; i < mlp->num_layers - 1; i++)
    {
        // h_{i + 1} = W_{h_i} h_{i}
        cblas_dgemv(CblasRowMajor, CblasNoTrans, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], 1.0, mlp->weights[i], mlp->layer_sizes[i], mlp->layers[i], 1, 0.0, mlp->layers[i + 1], 1);
        // h_{i + 1} += b_{i}
        cblas_daxpy(mlp->layer_sizes[i + 1], 1.0, mlp->biases[i], 1, mlp->layers[i + 1], 1);
        // h_{i + 1} = sigmoid(h_{i + 1})
        std::transform(mlp->layers[i + 1], mlp->layers[i + 1] + mlp->layer_sizes[i + 1], mlp->layers[i + 1], mlp->activations[i]);
    }
}

void mlp_backprop(mlp_t *mlp, double *y, double alpha)
{
    int output_index = mlp->num_layers - 1;

    // delta_o = y_hat
    cblas_dcopy(mlp->layer_sizes[output_index], mlp->layers[output_index], 1, mlp->deltas[output_index], 1);
    // delta_o -= y
    cblas_daxpy(mlp->layer_sizes[output_index], -1.0, y, 1, mlp->deltas[output_index], 1);

    for (int i = mlp->num_layers - 2; i >= 0; i--)
    {
        // We overwrite our previous output with the derivative of activation applied to it
        std::transform(mlp->layers[i + 1], mlp->layers[i + 1] + mlp->layer_sizes[i + 1], mlp->layers[i + 1], mlp->d_activations[i]);
        // delta_{i + 1} = delta_{i + 1} * d_activation(h_{i + 1})
        hadamard_product(mlp->deltas[i + 1], mlp->layers[i + 1], mlp->deltas[i + 1], mlp->layer_sizes[i + 1]);

        // W_i -= alpha * delta_{i + 1} h_i^T
        cblas_dger(CblasRowMajor, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], -1.0 * alpha, mlp->deltas[i + 1], 1, mlp->layers[i], 1, mlp->weights[i], mlp->layer_sizes[i]);

        // b_i -= alpha * delta_{i + 1}
        cblas_daxpy(mlp->layer_sizes[i + 1], -1.0 * alpha, mlp->deltas[i + 1], 1, mlp->biases[i], 1);

        // delta_{i - 1} = W_i^T delta_{i + 1}
        cblas_dgemv(CblasRowMajor, CblasTrans, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], 1.0, mlp->weights[i], mlp->layer_sizes[i], mlp->deltas[i + 1], 1, 0.0, mlp->deltas[i], 1);
    }
}

void delete_mlp(mlp_t *mlp)
{
    free(mlp->layer_sizes);

    for (int i = 0; i < mlp->num_layers; i++)
    {
        if (i < mlp->num_layers - 1)
        {
            free(mlp->weights[i]);
            free(mlp->biases[i]);
        }

        free(mlp->layers[i]);
        free(mlp->deltas[i]);
    }

    free(mlp->weights);
    free(mlp->biases);

    free(mlp->layers);
    free(mlp->deltas);

    free(mlp);
}

void print_mlp(mlp_t *mlp)
{
    fprintf(stderr, "Input:\n");
    print_vector(mlp->layers[0], mlp->layer_sizes[0]);

    for (int i = 1; i < mlp->num_layers - 1; i++)
    {
        fprintf(stderr, "Hidden %d:\n", i);
        print_vector(mlp->layers[i], mlp->layer_sizes[i]);
    }

    fprintf(stderr, "Output:\n");
    print_vector(mlp->layers[mlp->num_layers - 1], mlp->layer_sizes[mlp->num_layers - 1]);

    for (int i = 0; i < mlp->num_layers - 1; i++)
    {
        fprintf(stderr, "Weights %d:\n", i);
        print_vector(mlp->weights[i], mlp->layer_sizes[i] * mlp->layer_sizes[i + 1]);

        fprintf(stderr, "Biases %d:\n", i);
        print_vector(mlp->biases[i], mlp->layer_sizes[i + 1]);

        fprintf(stderr, "Deltas %d:\n", i);
        print_vector(mlp->deltas[i], mlp->layer_sizes[i]);
    }

    fprintf(stderr, "Final Delta:\n");
    print_vector(mlp->deltas[mlp->num_layers - 1], mlp->layer_sizes[mlp->num_layers - 1]);
}

int main()
{
    int num_layers = 5;
    int layers[] = {1, 8, 8, 8, 1};

    fun_t activations[] = {relu, relu, relu, identity};
    fun_t d_activations[] = {d_relu, d_relu, d_relu, d_identity};

    mlp_t *mlp = create_mlp(num_layers, layers, activations, d_activations);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0, 1);

    for (int i = 0; i < 5000; i++)
    {
        double x = dist(gen);
        double y = sin(x) * sin(x);

        mlp_forward(mlp, &x);
        mlp_backprop(mlp, &y, 0.0001);

        print_mlp(mlp);
    }

    std::ofstream file("data.csv");

    for (int i = 0; i < 1000; i++)
    {
        double x = dist(gen);
        double y = sin(x) * sin(x);

        mlp_forward(mlp, &x);
        file << x << "," << y << "," << mlp->layers[mlp->num_layers - 1][0] << std::endl;
    }

    file.close();

    delete_mlp(mlp);
}