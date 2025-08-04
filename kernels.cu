#include "kernels.h"

__global__ void forward_kernel(const double* inputs, const double* weights, const double* biases,
                               double* outputs, int num_inputs, int num_outputs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_outputs) {
        double sum = biases[i];
        for (int j = 0; j < num_inputs; ++j) {
            sum += inputs[j] * weights[i * num_inputs + j];
        }
        outputs[i] = sum > 0 ? sum : 0.0; // ReLU
    }
}

__global__ void compute_output_delta(const double* outputs, const double* targets,
                                     const double* weighted_sums, double* deltas, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double error = outputs[i] - targets[i];
        double grad = weighted_sums[i] > 0.0 ? 1.0 : 0.0;
        deltas[i] = error * grad;
    }
}

__global__ void compute_hidden_delta(const double* next_deltas, const double* next_weights,
                                     const double* weighted_sums, double* deltas,
                                     int num_current, int num_next) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_current) {
        double grad = weighted_sums[i] > 0.0 ? 1.0 : 0.0;
        double sum = 0.0;
        for (int j = 0; j < num_next; ++j) {
            sum += next_deltas[j] * next_weights[j * num_current + i];
        }
        deltas[i] = grad * sum;
    }
}

__global__ void update_weights_kernel(double* weights, double* biases,
                                      const double* deltas, const double* inputs,
                                      int num_inputs, int num_outputs, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_outputs) {
        for (int j = 0; j < num_inputs; ++j) {
            weights[i * num_inputs + j] -= lr * deltas[i] * inputs[j];
        }
        biases[i] -= lr * deltas[i];
    }
}