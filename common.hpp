#pragma once

#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

typedef double (*fun_t)(double);

void hadamard_product(double *a, double *b, double *c, int n);

double *rand_vector(int n);

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

void delete_mlp(mlp_t *mlp);

void print_mlp(mlp_t *mlp);
