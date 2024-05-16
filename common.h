#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

const int NUM_SAMPLES = 20000;
// batch size per machine. total batch size is BATCH_SIZE * (NUM_MACHINES)
// const int BATCH_SIZE = 1;

typedef double (*fun_t)(double);

void hadamard_product(double *a, double *b, double *c, int n);

double *rand_vector(int n);

void print_matrix(double *m, int rows, int cols);

void print_vector(double *v, int n);

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

mlp_t *create_mlp(int num_layers, int *layers, fun_t *activations, fun_t *d_activations);

void print_mlp(mlp_t *mlp);

void delete_mlp(mlp_t *mlp);

void print_mlp(mlp_t *mlp);