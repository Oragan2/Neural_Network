#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

extern "C" {
    __global__ void forward_kernel(const double* inputs, const double* weights, const double* biases,
                                   double* outputs, int num_inputs, int num_outputs);

    __global__ void compute_output_delta(const double* outputs, const double* targets,
                                         const double* weighted_sums, double* deltas, int size);

    __global__ void compute_hidden_delta(const double* next_deltas, const double* next_weights,
                                         const double* weighted_sums, double* deltas,
                                         int num_current, int num_next);

    __global__ void update_weights_kernel(double* weights, double* biases,
                                          const double* deltas, const double* inputs,
                                          int num_inputs, int num_outputs, double lr);
}

#endif // KERNELS_H