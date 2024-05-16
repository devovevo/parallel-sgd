#include "common.hpp"

void hadamard_product(double *a, double *b, double *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

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
        printf("%f\n", v[i]);
    }
}

void print_matrix(double *m, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", m[i + j * rows]);
        }
        printf("\n");
    }
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

        mlp->layers[i] = (double *)aligned_alloc(64, layers[i] * sizeof(double));
        mlp->deltas[i] = (double *)aligned_alloc(64, layers[i] * sizeof(double));
    }

    return mlp;
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
        print_matrix(mlp->weights[i], mlp->layer_sizes[i + 1], mlp->layer_sizes[i]);

        fprintf(stderr, "Biases %d:\n", i);
        print_vector(mlp->biases[i], mlp->layer_sizes[i + 1]);

        fprintf(stderr, "Deltas %d:\n", i);
        print_vector(mlp->deltas[i], mlp->layer_sizes[i]);
    }

    fprintf(stderr, "Final Delta:\n");
    print_vector(mlp->deltas[mlp->num_layers - 1], mlp->layer_sizes[mlp->num_layers - 1]);
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