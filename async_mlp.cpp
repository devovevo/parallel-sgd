#include <cblas.h>
#include <mpi.h>
#include "common.hpp"
#include "activations.h"
#include "data.hpp"
#include <upcxx/upcxx.hpp>
#include <algorithm>

/*
class async_mlp
{
public:
    int num_layers;
    int *layer_sizes;

    double** weights;
    double** biases;

    fun_t *activations;
    fun_t *d_activations;

    double **layers;
    double **deltas;

    async_mlp(int arg_num_layers, int *arg_layers, fun_t *activations, fun_t *d_activations)
    {

        num_layers = arg_num_layers;
        layer_sizes = (int *)malloc(num_layers * sizeof(int));

        weights = (double **)malloc((num_layers - 1) * sizeof(double *));
        biases = (double **)malloc((num_layers - 1) * sizeof(double *));

        activations = activations;
        d_activations = d_activations;

        layers = (double **)malloc(num_layers * sizeof(double *));
        deltas = (double **)malloc(num_layers * sizeof(double *));

        for (int i = 0; i < num_layers; i++)
        {
            layer_sizes[i] = arg_layers[i];

            if (i < num_layers - 1)
            {
                weights[i] = rand_vector(arg_layers[i + 1] * arg_layers[i]);
                biases[i] = rand_vector(arg_layers[i + 1]);
            }

            layers[i] = (double *)malloc(arg_layers[i] * sizeof(double));
            deltas[i] = (double *)malloc(arg_layers[i] * sizeof(double));
        }

    }


    void mlp_forward(double *x)
    {
        //// std::cerr << "copying" << std::endl;
        cblas_dcopy(layer_sizes[0], x, 1, layers[0], 1);

        for (int i = 0; i < num_layers - 1; i++)
        {   
            std::cerr << "dgemv" << std::endl;
            // h_{i + 1} = W_{h_i} h_{i}
            cblas_dgemv(CblasRowMajor, CblasNoTrans, layer_sizes[i + 1], layer_sizes[i], 1.0, weights[i], layer_sizes[i], layers[i], 1, 0.0, layers[i + 1], 1);
            // h_{i + 1} += b_{i}
            std::cerr << "daxpy" << std::endl;
            cblas_daxpy(layer_sizes[i + 1], 1.0, biases[i], 1, layers[i + 1], 1);
            // h_{i + 1} = sigmoid(h_{i + 1})
            std::cerr << "transform" << std::endl;
            //std::transform(mlp->layers[i + 1], mlp->layers[i + 1] + mlp->layer_sizes[i + 1], mlp->layers[i + 1], mlp->activations[i])
            std::transform(layers[i + 1], layers[i + 1] + layer_sizes[i + 1], layers[i + 1], activations[i]);
            std::cerr << "finish transform" << std::endl;
        }
    }

    void mlp_gradient(double* w_update, double* b_update, int lyr) {
        std::transform(layers[lyr + 1], layers[lyr + 1] + layer_sizes[lyr + 1], layers[lyr + 1], d_activations[lyr]);
        // delta_{i + 1} = delta_{i + 1} * d_activation(h_{i + 1})
        hadamard_product(deltas[lyr + 1], layers[lyr + 1], deltas[lyr + 1], layer_sizes[lyr + 1]);
        //cblas_dger(CblasRowMajor, layer_sizes[i + 1], layer_sizes[i], -1.0 * alpha, deltas[i + 1], 1, layers[i], 1, weights[i], layer_sizes[i]);
        cblas_dger(CblasRowMajor, layer_sizes[lyr + 1], layer_sizes[lyr], 1.0, deltas[lyr + 1], 1, layers[lyr], 1, w_update, layer_sizes[lyr]);
        cblas_dcopy(layer_sizes[lyr + 1], deltas[lyr + 1], 1, b_update, 1);
    }

    void mlp_update_weights(double alpha, const double* w_update, const double* b_update, int lyr) {
        // W_i -= alpha * delta_{i + 1} h_i^T
        cblas_daxpy(layer_sizes[lyr + 1] * layer_sizes[lyr], -1.0 * alpha, w_update, 1, weights[lyr], 1);
        // b_i -= alpha * delta_{i + 1}
        cblas_daxpy(layer_sizes[lyr + 1], -1.0 * alpha, b_update, 1, biases[lyr], 1);
        // delta_{i} = W_i^T delta_{i + 1}
        cblas_dgemv(CblasRowMajor, CblasTrans, layer_sizes[lyr + 1], layer_sizes[lyr], 1.0, weights[lyr], layer_sizes[lyr], deltas[lyr + 1], 1, 0.0, deltas[lyr], 1);
    }
};
*/
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

void mlp_gradient(mlp_t *mlp, double* w_update, double* b_update, int lyr) {
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

using grad_stack_t = upcxx::dist_object<upcxx::global_ptr<double>>;

int main() {
    upcxx::init();
    int num_layers = 5;
    int layers[] = {1, 8, 8, 8, 1};

    fun_t activations[] = {relu, relu, relu, identity};
    fun_t d_activations[] = {d_relu, d_relu, d_relu, d_identity};
    mlp_t *mlp = create_mlp(num_layers, layers, activations, d_activations);

    double *x = new double[NUM_SAMPLES / upcxx::rank_n()];
    double *y = new double[NUM_SAMPLES / upcxx::rank_n()];
    read_data(x, y, NUM_SAMPLES / upcxx::rank_n());
    double seconds = 0.0;

    std::vector<grad_stack_t> p_wgrads;
    p_wgrads.resize(mlp->num_layers - 1);
    std::vector<grad_stack_t> p_bgrads;
    p_bgrads.resize(mlp->num_layers - 1);
    std::vector<double *> all_ws;
    all_ws.resize(mlp->num_layers - 1);
    std::vector<double *> all_bs;
    all_bs.resize(mlp->num_layers - 1);

    // std::cerr << "init" << std::endl;
        /*for (int lyr = mlp->num_layers - 2; lyr >= 0; lyr--) {
            int w_size = mlp->layer_sizes[lyr + 1] * mlp->layer_sizes[lyr] * (std::rank_n() - 1);
            ws[lyr] = new double[w_size];//(double*) realloc(w_update, w_size * sizeof(double));
            int b_size = mlp->layer_sizes[lyr + 1];
            bs[lyr] = new double[b_size]; //(double*) realloc(b_update, b_size * sizeof(double));
        }*/
        
    for (int lyr = 0; lyr <= mlp->num_layers - 2; lyr++) {
        int w_size = mlp->layer_sizes[lyr + 1] * mlp->layer_sizes[lyr] * (upcxx::rank_n() - 1);
        p_wgrads[lyr] = grad_stack_t(upcxx::new_array<double>(w_size)); 
        all_ws[lyr] = new double[w_size];
        int b_size = mlp->layer_sizes[lyr + 1] * (upcxx::rank_n() - 1);
        all_bs[lyr] = new double[b_size];
        p_bgrads[lyr] = grad_stack_t(upcxx::new_array<double>(b_size));
    }
    // std::cerr << "finished creating pgrads" << std::endl;
    

    upcxx::dist_object<upcxx::global_ptr<uint64_t>> ptr(upcxx::new_<uint64_t>(0));
    upcxx::atomic_domain<uint64_t> ad = upcxx::atomic_domain<uint64_t>({upcxx::atomic_op::fetch_add});;
    // std::cerr << "finished that" << std::endl;
    for (int i = 0; i < NUM_SAMPLES / upcxx::rank_n(); i++)
    {   

        if (upcxx::rank_me() > 0) {
            mlp_forward(mlp, &x[i]);
        }
        // std::cerr << "passed forward" << std::endl;

        upcxx::barrier();
        auto start_time = std::chrono::steady_clock::now();
        if (upcxx::rank_me() > 0) {
            int output_index = mlp->num_layers - 1;
            // delta_o = y_hat
            cblas_dcopy(mlp->layer_sizes[output_index], mlp->layers[output_index], 1, mlp->deltas[output_index], 1);
            // delta_o -= y
            cblas_daxpy(mlp->layer_sizes[output_index], -1.0, &y[i], 1, mlp->deltas[output_index], 1);
        }
        //std::cerr << "passed initial setup" << std::endl;
        for (int lyr = mlp->num_layers - 2; lyr >= 0; lyr--) {
            int w_size = mlp->layer_sizes[lyr + 1] * mlp->layer_sizes[lyr];
            double* w_update = all_ws[lyr];
            int b_size = mlp->layer_sizes[lyr + 1];
            double* b_update = all_bs[lyr];
            // std::cerr << "initializating w and b" << std::endl;
            std::fill_n(w_update, w_size, 0.0);
            std::fill_n(b_update, b_size, 0.0);
            // std::cerr << "Starting Gradient" << std::endl;
            upcxx::promise<> prom;
            if (upcxx::rank_me() > 0) {
                mlp_gradient(mlp, w_update, b_update, lyr);
                auto idx = ad.fetch_add(ptr.fetch(0).wait(), 1, std::memory_order_relaxed).wait();
                // std::cerr << "Got Index " << idx << std::endl;
                auto w_addr = p_wgrads[lyr].fetch(0).wait();
                upcxx::rput(w_update, w_addr + idx * w_size, w_size, upcxx::operation_cx::as_promise(prom));
                // std::cerr << "Sent First Rput" << std::endl;
                auto b_addr = p_bgrads[lyr].fetch(0).wait();
                upcxx::rput(b_update, b_addr + idx * b_size, b_size, upcxx::operation_cx::as_promise(prom));
                // std::cerr << "Finished Rput" << std::endl;
            }
            prom.finalize().wait();
            upcxx::barrier();
            if (upcxx::rank_me() == 0) {
                // std::cerr << "Dereferencing p grads" << std::endl;
                double* w_grads = p_wgrads[lyr]->local();
                double* b_grads = p_bgrads[lyr]->local();
                for (uint64_t p = 1; p < upcxx::rank_n(); p++) {
                    cblas_daxpy(w_size, 1.0, w_grads + (p-1) * w_size, 1, w_update, 1);
                    cblas_daxpy(b_size, 1.0, b_grads + (p-1) * b_size, 1, b_update, 1);
                }
                cblas_dscal(w_size, 1.0/(upcxx::rank_n() - 1), w_update, 1);
                cblas_dscal(b_size, 1.0/(upcxx::rank_n() - 1), b_update, 1);
                std::copy(w_grads, w_grads + w_size, w_update);
                std::copy(b_grads, b_grads + b_size, b_update);
                // std::cerr << "setting pointer to 0" << std::endl;
                *(ptr->local()) = 0;
            }

            if (upcxx::rank_me() > 0) {
                // std::cerr << "doing rgets" << std::endl;
                auto fut_w = rget(p_wgrads[lyr].fetch(0).wait(), w_update, w_size);
                auto fut_b = rget(p_bgrads[lyr].fetch(0).wait(), b_update, b_size);
                // std::cerr << "finished rgets" << std::endl;
                fut_w.wait();
                fut_b.wait();
                mlp_update_weights(mlp, 0.0001, w_update, b_update, lyr);
            }
        }
       
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        seconds += diff.count();
    }

    if (upcxx::rank_me() == 0)
        std::cerr << "Async training took " << seconds << " seconds for " << upcxx::rank_n() << " processes." << std::endl;

    delete_mlp(mlp);
    upcxx::finalize();

    return 0;
}