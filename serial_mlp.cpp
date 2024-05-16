#include <iostream>
#include <fstream>

#include <cblas.h>

#include <chrono>
using namespace std::chrono;

#include "activations.h"
#include "common.h"

void mlp_forward(mlp_t *mlp, double *x)
{
    cblas_dcopy(mlp->layer_sizes[0], x, 1, mlp->layers[0], 1);

    for (int i = 0; i < mlp->num_layers - 1; i++)
    {
        // h_{i + 1} = W_{h_i} h_{i}
        cblas_dgemv(CblasColMajor, CblasNoTrans, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], 1.0, mlp->weights[i], mlp->layer_sizes[i + 1], mlp->layers[i], 1, 0.0, mlp->layers[i + 1], 1);
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
        cblas_dger(CblasColMajor, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], -1.0 * alpha, mlp->deltas[i + 1], 1, mlp->layers[i], 1, mlp->weights[i], mlp->layer_sizes[i + 1]);

        // b_i -= alpha * delta_{i + 1}
        cblas_daxpy(mlp->layer_sizes[i + 1], -1.0 * alpha, mlp->deltas[i + 1], 1, mlp->biases[i], 1);

        // delta_{i} = W_i^T delta_{i + 1}
        cblas_dgemv(CblasColMajor, CblasTrans, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], 1.0, mlp->weights[i], mlp->layer_sizes[i + 1], mlp->deltas[i + 1], 1, 0.0, mlp->deltas[i], 1);
    }
}

int main()
{
    int num_layers = 5;
    int layers[] = {1, 8, 8, 8, 1};

    fun_t activations[] = {sigmoid, sigmoid, sigmoid, sigmoid};
    fun_t d_activations[] = {d_sigmoid, d_sigmoid, d_sigmoid, d_sigmoid};

    mlp_t *mlp = create_mlp(num_layers, layers, activations, d_activations);

    for (int k = 0; k < 10000; k++)
    {
        for (int i = 0; i < 100; i++)
        {
            double x = (double)i / 100;
            double y = sin(M_PI * x) * sin(M_PI * x);

            mlp_forward(mlp, &x);
            mlp_backprop(mlp, &y, 0.5);
        }
    }

    for (int i = 0; i < 100; i++)
    {
        double x = (double)i / 100;
        mlp_forward(mlp, &x);
        fprintf(stderr, "%f %f %f\n", x, sin(M_PI * x) * sin(M_PI * x), mlp->layers[num_layers - 1][0]);
    }

    delete_mlp(mlp);
}