#include <cblas.h>
#include <mpi.h>
#include "common.h"
#include "activations.h"


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

void mlp_gradient(mlp_t *mlp, double *y, /*double* w_update, double* b_update,*/ int lyr) {
    // We overwrite our previous output with the derivative of activation applied to it
    std::transform(mlp->layers[lyr + 1], mlp->layers[lyr + 1] + mlp->layer_sizes[lyr + 1], mlp->layers[lyr + 1], mlp->d_activations[lyr]);
    // delta_{i + 1} = delta_{i + 1} * d_activation(h_{i + 1})
    hadamard_product(mlp->deltas[lyr + 1], mlp->layers[lyr + 1], mlp->deltas[lyr + 1], mlp->layer_sizes[lyr + 1]);
    //cblas_dger(CblasRowMajor, mlp->layer_sizes[i + 1], mlp->layer_sizes[i], -1.0 * alpha, mlp->deltas[i + 1], 1, mlp->layers[i], 1, mlp->weights[i], mlp->layer_sizes[i]);
    //cblas_dger(CblasRowMajor, mlp->layer_sizes[lyr + 1], mlp->layer_sizes[lyr], 1.0, mlp->deltas[lyr + 1], 1, mlp->layers[lyr], 1, w_update, mlp->layer_sizes[lyr]);
    //cblas_dcopy(mlp->layer_sizes[lyr + 1], mlp->deltas[lyr + 1], 1, b_update, 1);

}

void mlp_update_weights(mlp_t *mlp, double alpha, /*double* w_update, double* b_update,*/ int lyr) {
    // W_i -= alpha * delta_{i + 1} h_i^T
    cblas_dger(CblasRowMajor, mlp->layer_sizes[lyr + 1], mlp->layer_sizes[lyr], -1.0 * alpha, mlp->deltas[lyr + 1], 1, mlp->layers[lyr], 1, mlp->weights[lyr], mlp->layer_sizes[lyr]);
    //cblas_daxpy(mlp->layer_sizes[lyr + 1]* mlp->layer_sizes[lyr], -1.0 * alpha, w_update, 1, mlp->weights[lyr], 1);
    // b_i -= alpha * delta_{i + 1}
    cblas_daxpy(mlp->layer_sizes[lyr + 1], -1.0 * alpha, mlp->deltas[lyr + 1], 1, mlp->biases[lyr], 1);
    //cblas_daxpy(mlp->layer_sizes[lyr + 1], -1.0 * alpha, b_update, 1, mlp->biases[lyr], 1);
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

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0, 1);

    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_SAMPLES / num_procs; i++)
    {
        double x = dist(gen);
        double y = sin(x) * sin(x);

        mlp_forward(mlp, &x);

        int output_index = mlp->num_layers - 1;
        // delta_o = y_hat
        cblas_dcopy(mlp->layer_sizes[output_index], mlp->layers[output_index], 1, mlp->deltas[output_index], 1);
        // delta_o -= y
        cblas_daxpy(mlp->layer_sizes[output_index], -1.0, &y, 1, mlp->deltas[output_index], 1);

        //double* b_update = NULL;
        //double* w_update = NULL;
        for (int lyr = mlp->num_layers - 2; lyr >= 0; lyr--) {
            /*int w_size = mlp->layer_sizes[lyr + 1] * mlp->layer_sizes[lyr];
            w_update = (double*) realloc(w_update, w_size * sizeof(double));
            int b_size = mlp->layer_sizes[lyr + 1];
            b_update = (double*) realloc(b_update, b_size * sizeof(double));
            std::fill_n(w_update, w_size, 0.0);
            std::fill_n(b_update, b_size, 0.0);*/

            mlp_gradient(mlp, &y,/* w_update, b_update,*/ lyr);
            //MPI_Allreduce(MPI_IN_PLACE, b_update, b_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            //MPI_Allreduce(MPI_IN_PLACE, w_update, w_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, mlp->deltas[lyr+1], mlp->layer_sizes[lyr+1], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cblas_dscal(mlp->layer_sizes[lyr+1], (1.0/static_cast<double>(num_procs)), mlp->deltas[lyr+1], 1);
            //cblas_dscal(w_size, (1.0/static_cast<double>(num_procs)), w_update, 1);
            //cblas_dscal(b_size, (1.0/static_cast<double>(num_procs)), b_update, 1);
            mlp_update_weights(mlp, lyr, /*w_update, b_update,*/ 0.0001);
        }

        //free(w_update);

        //print_mlp(mlp);
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    if (rank == 0) std::cout << "Training took " << seconds << " seconds for " << num_procs << "processes" << std::endl; 
    
    if (rank == 0) {
        std::ofstream file("data.csv");

        for (int i = 0; i < 1000; i++)
        {
            double x = dist(gen);
            double y = sin(x) * sin(x);

            mlp_forward(mlp, &x);
            file << x << "," << y << "," << mlp->layers[mlp->num_layers - 1][0] << std::endl;
        }

        file.close();
    }

    delete_mlp(mlp);
    MPI_Finalize();
}

