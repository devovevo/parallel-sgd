#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <immintrin.h>

void hadamard(double *a, double *b, double *c, int n)
{
    int aligned_n = n / 4 * 4;
    int i = 0;

    for (; i < aligned_n; i += 4)
    {
        __m256d a_v = _mm256_loadu_pd(a + i);
        __m256d b_v = _mm256_loadu_pd(b + i);
        __m256d c_v = _mm256_mul_pd(a_v, b_v);
        _mm256_storeu_pd(c + i, c_v);
    }

    for (; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

double *rand_vector(int n)
{
    double *v = (double *)aligned_alloc(64, n * sizeof(double));
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{0, 1 / sqrt(n)};

    for (int i = 0; i < n; i++)
    {
        v[i] = d(gen);
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