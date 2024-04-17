#include <stdlib.h>
#include <cassert>
#include <vector>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;

namespace nn
{
    class MLP
    {
    public:
        int nlayers;
        std::vector<int> layer_sizes;

    private:
        std::vector<MatrixXd> weights;
        std::vector<MatrixXd> biases;
        std::vector<MatrixXd> activations;

    protected:
        explicit MLP(std::vector<int> &layer_sizes)
        {
            this->nlayers = layer_sizes.size();
            this->layer_sizes = layer_sizes;

            for (int i = 1; i < this->nlayers; i++)
            {
                        }
        }

        std::vector<double> forward(std::vector<double> &x)
        {
        }
    };
}