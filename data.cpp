#include "data.hpp"

double generateData(int x)
{
    return sin(x) * sin(x);
}

void read_data(double *x, double *y)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0, 1);

    // Read data from file into the array
    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        x[i] = dist(gen);
        y[i] = generateData(x[i]);
    }
}