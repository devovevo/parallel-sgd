
#include <cblas.h>
#include "common.hpp"
#include "activations.h"
#include "data.hpp"

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

int main()
{
    int num_layers = 5;
    int layers[] = {1, 8, 8, 8, 1};

    fun_t activations[] = {relu, relu, relu, identity};
    fun_t d_activations[] = {d_relu, d_relu, d_relu, d_identity};

    mlp_t *mlp = create_mlp(num_layers, layers, activations, d_activations);

    /*std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0, 1);*/
    double *x = new double[NUM_SAMPLES];
    double *y = new double[NUM_SAMPLES];
    read_data(x, y, NUM_SAMPLES);
    double seconds = 0.0;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        mlp_forward(mlp, &x[i]);

        auto start_time = std::chrono::steady_clock::now();
        mlp_backprop(mlp, &y[i], 0.0001);
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        seconds += diff.count();
    }
    std::cerr << "Serial training took " << seconds << " seconds." << std::endl;
    /*std::ofstream file("data.csv");
    read_data(x, y);
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        mlp_forward(mlp, &x[i]);
        file << x[i] << "," << y[i] << "," << mlp->layers[mlp->num_layers - 1][0] << std::endl;
    }

    file.close();*/

    delete_mlp(mlp);
}