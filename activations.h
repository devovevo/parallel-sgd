#include <math.h>

inline double identity(double x)
{
    return x;
}

inline double d_identity(double x)
{
    return 1;
}

inline double relu(double x)
{
    return x > 0 ? x : 0;
}

inline double d_relu(double x)
{
    return x > 0 ? 1 : 0;
}

inline double tanh(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

inline double d_tanh(double x)
{
    return 1 - x * x;
}

inline double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

inline double d_sigmoid(double x)
{
    return x * (1 - x);
}