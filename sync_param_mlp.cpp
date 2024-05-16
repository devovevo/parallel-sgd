#include <cblas.h>
#include <mpi.h>
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

/*
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

        // delta_{i} = W_i^T delta_{i + 1}
        cblas_dgemv(CblasRowMajor, CblasTrans, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], 1.0, mlp->weights[i], mlp->layer_sizes[i], mlp->deltas[i + 1], 1, 0.0, mlp->deltas[i], 1);
    }
}
*/

void mlp_gradient(mlp_t *mlp, double *y, double* w_update, double* b_update, int lyr) {
    // We overwrite our previous output with the derivative of activation applied to it
    std::transform(mlp->layers[lyr + 1], mlp->layers[lyr + 1] + mlp->layer_sizes[lyr + 1], mlp->layers[lyr + 1], mlp->d_activations[lyr]);
    // delta_{i + 1} = delta_{i + 1} * d_activation(h_{i + 1})
    hadamard_product(mlp->deltas[lyr + 1], mlp->layers[lyr + 1], mlp->deltas[lyr + 1], mlp->layer_sizes[lyr + 1]);
    //cblas_dger(CblasRowMajor, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], -1.0 * alpha, mlp->deltas[i + 1], 1, mlp->layers[i], 1, mlp->weights[i], mlp->layer_sizes[i]);
    cblas_dger(CblasRowMajor, mlp->layer_sizes[lyr + 1], mlp->layer_sizes[lyr], 1.0, mlp->deltas[lyr + 1], 1, mlp->layers[lyr], 1, w_update, mlp->layer_sizes[lyr]);
    cblas_dcopy(mlp->layer_sizes[lyr + 1], mlp->deltas[lyr + 1], 1, b_update, 1);

}
void mlp_update_weights(mlp_t *mlp, double alpha, double* w_update, double* b_update, int lyr) {
    // W_i -= alpha * delta_{i + 1} h_i^T
    //cblas_dger(CblasRowMajor, mlp->layer_sizes[lyr + 1], mlp->layer_sizes[lyr], -1.0 * alpha, mlp->deltas[lyr + 1], 1, mlp->layers[lyr], 1, mlp->weights[lyr], mlp->layer_sizes[lyr]);
    cblas_daxpy(mlp->layer_sizes[lyr + 1]* mlp->layer_sizes[lyr], -1.0 * alpha, w_update, 1, mlp->weights[lyr], 1);
    // b_i -= alpha * delta_{i + 1}
    //cblas_daxpy(mlp->layer_sizes[lyr + 1], -1.0 * alpha, mlp->deltas[lyr + 1], 1, mlp->biases[lyr], 1);
    cblas_daxpy(mlp->layer_sizes[lyr + 1], -1.0 * alpha, b_update, 1, mlp->biases[lyr], 1);
    // delta_{i} = W_i^T delta_{i + 1}
    cblas_dgemv(CblasRowMajor, CblasTrans, mlp->layer_sizes[lyr + 1], mlp->layer_sizes[lyr], 1.0, mlp->weights[lyr], mlp->layer_sizes[lyr], mlp->deltas[lyr + 1], 1, 0.0, mlp->deltas[lyr], 1);
    //cblas_dgemv(CblasRowMajor, CblasTrans, mlp->layer_sizes[lyr + 1], mlp->layer_sizes[lyr], 1.0, mlp->weights[lyr], mlp->layer_sizes[lyr], b_update, 1, 0.0, mlp->deltas[lyr], 1);

}

int main(int argc, char** argv)
{
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_layers = 5;
    int layers[] = {1, 8, 8, 8, 1};

    fun_t activations[] = {relu, relu, relu, identity};
    fun_t d_activations[] = {d_relu, d_relu, d_relu, d_identity};

    mlp_t *mlp = create_mlp(num_layers, layers, activations, d_activations);

    double* x = new double[NUM_SAMPLES];
    double* y = new double[NUM_SAMPLES];
    read_data(x, y);

    double seconds = 0.0;
    double** ws = (double**) malloc(sizeof(double*) * mlp->num_layers - 2);
    double** bs = (double**) malloc(sizeof(double*) * mlp->num_layers - 2);

    for (int lyr = mlp->num_layers - 2; lyr >= 0; lyr--) {
        int w_size = mlp->layer_sizes[lyr + 1] * mlp->layer_sizes[lyr];
        ws[lyr] = new double[w_size];//(double*) realloc(w_update, w_size * sizeof(double));
        int b_size = mlp->layer_sizes[lyr + 1];
        bs[lyr] = new double[b_size]; //(double*) realloc(b_update, b_size * sizeof(double));
    }

    for (int i = 0; i < NUM_SAMPLES / num_procs; i++)
    {
        mlp_forward(mlp, &x[rank + num_procs * i]);

        auto start_time = std::chrono::steady_clock::now();
        int output_index = mlp->num_layers - 1;
        // delta_o = y_hat
        cblas_dcopy(mlp->layer_sizes[output_index], mlp->layers[output_index], 1, mlp->deltas[output_index], 1);
        // delta_o -= y
        cblas_daxpy(mlp->layer_sizes[output_index], -1.0, &y[rank + num_procs * i], 1, mlp->deltas[output_index], 1);

        //double* b_update;
        //double* w_update;

        for (int lyr = mlp->num_layers - 2; lyr >= 0; lyr--) {
            int w_size = mlp->layer_sizes[lyr + 1] * mlp->layer_sizes[lyr];
            //w_update = new double[w_size];//(double*) realloc(w_update, w_size * sizeof(double));
            int b_size = mlp->layer_sizes[lyr + 1];
            //b_update = new double[b_size]; //(double*) realloc(b_update, b_size * sizeof(double));
            std::fill_n(ws[lyr], w_size, 0.0);
            std::fill_n(bs[lyr], b_size, 0.0);

            mlp_gradient(mlp, &y[rank + num_procs * i], ws[lyr], bs[lyr], lyr);
            MPI_Allreduce(MPI_IN_PLACE, bs[lyr], b_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, ws[lyr], w_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            //MPI_Allreduce(MPI_IN_PLACE, mlp->deltas[lyr+1], mlp->layer_sizes[lyr+1], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            //cblas_dscal(mlp->layer_sizes[lyr+1], (1.0/static_cast<double>(num_procs)), mlp->deltas[lyr+1], 1);
            cblas_dscal(w_size, (1.0/static_cast<double>(num_procs)), ws[lyr], 1);
            cblas_dscal(b_size, (1.0/static_cast<double>(num_procs)), bs[lyr], 1);
            mlp_update_weights(mlp, lyr, ws[lyr], bs[lyr], 0.0001);

            //delete w_update;
            //delete b_update;
        }

        //free(w_update);
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        seconds += diff.count();
        //print_mlp(mlp);
    }
    if (rank == 0) std::cout << "Training took " << seconds << " seconds for " << num_procs << "processes" << std::endl; 
    
    delete_mlp(mlp);
    MPI_Finalize();
}

