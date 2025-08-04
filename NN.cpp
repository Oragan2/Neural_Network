#include <vector>
#include <stdexcept>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <memory>
#include "kernels.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#endif

namespace py = pybind11;

std::mt19937 neuron_rng(std::random_device{}());
std::uniform_real_distribution<> weight_bias_dist(-0.1, 0.1);

double relu(double x) {
    return std::max(0.0, x);
}

double relu_prime(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// --- CPU-BASED CLASSES ---

class Neuron {
private:
    std::vector<double> weights;
    double bias;
    double weighted_sum;
    double activation;
    double delta;

public:
    Neuron(int num_inputs) {
        bias = weight_bias_dist(neuron_rng);
        weights.resize(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            weights[i] = weight_bias_dist(neuron_rng);
        }
    }

    double calc(const std::vector<double>& inputs) {
        weighted_sum = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            weighted_sum += inputs[i] * weights[i];
        }
        activation = relu(weighted_sum);
        return activation;
    }

    const std::vector<double>& get_weights() const { return weights; }
    double get_weighted_sum() const { return weighted_sum; }
    double get_activation() const { return activation; }
    double get_delta() const { return delta; }
    void set_delta(double d) { delta = d; }
    
    void update(const std::vector<double>& inputs, double learning_rate) {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate * delta * inputs[i];
        }
        bias -= learning_rate * delta;
    }
};

class Layer {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int num_neurons_in_layer, int num_inputs_from_prev_layer) {
        for (int i = 0; i < num_neurons_in_layer; ++i) {
            neurons.emplace_back(num_inputs_from_prev_layer);
        }
    }

    std::vector<double> forward_pass(const std::vector<double>& inputs) {
        std::vector<double> outputs(neurons.size());
        for(size_t i = 0; i < neurons.size(); ++i) {
            outputs[i] = neurons[i].calc(inputs);
        }
        return outputs;
    }

    std::vector<std::vector<double>> get_all_weights() const {
        std::vector<std::vector<double>> all_weights;
        all_weights.reserve(neurons.size());
        for (const auto& n : neurons) {
            all_weights.push_back(n.get_weights());
        }
        return all_weights;
    }

    std::vector<double> get_all_deltas() const {
        std::vector<double> all_deltas;
        all_deltas.reserve(neurons.size());
        for (const auto& n : neurons) {
            all_deltas.push_back(n.get_delta());
        }
        return all_deltas;
    }

    void backward_pass_output_layer(const std::vector<double>& inputs, const std::vector<double>& target_outputs, double learning_rate) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            double error = neurons[i].get_activation() - target_outputs[i];
            double delta = error * relu_prime(neurons[i].get_weighted_sum());
            neurons[i].set_delta(delta);
            neurons[i].update(inputs, learning_rate);
        }
    }

    void backward_pass_hidden_layer(const std::vector<double>& inputs, const std::vector<double>& next_deltas, 
                                    const std::vector<std::vector<double>>& next_weights, 
                                    double learning_rate) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            double error_gradient = 0.0;
            for (size_t j = 0; j < next_deltas.size(); ++j) {
                error_gradient += next_deltas[j] * next_weights[j][i];
            }
            double delta = relu_prime(neurons[i].get_weighted_sum()) * error_gradient;
            neurons[i].set_delta(delta);
            neurons[i].update(inputs, learning_rate);
        }
    }
};

class NeuralNetworkCPU {
private:
    std::vector<int> neurons_layout;
    std::vector<Layer> layers;
    std::vector<std::vector<double>> activations_by_layer;

public:
    NeuralNetworkCPU(const std::vector<int>& layout) : neurons_layout(layout) {
        if (layout.size() < 2) {
            throw std::runtime_error("Neural network layout must have at least input and output layers.");
        }
        for (size_t i = 1; i < layout.size(); ++i) {
            layers.emplace_back(layout[i], layout[i-1]);
        }
    }

    py::array_t<double> forward_pass(py::array_t<double> input_data_py) {
        auto inputs_buf = input_data_py.request();
        std::vector<double> current_output_vec(static_cast<const double*>(inputs_buf.ptr), static_cast<const double*>(inputs_buf.ptr) + inputs_buf.shape[0]);
        
        for (size_t i = 0; i < layers.size(); ++i) {
            current_output_vec = layers[i].forward_pass(current_output_vec);
        }
        return py::cast(current_output_vec);
    }
    
    std::vector<double> forward_pass_and_store(const std::vector<double>& input_data) {
        activations_by_layer.clear();
        activations_by_layer.push_back(input_data);
        
        std::vector<double> current_output = input_data;
        for (size_t i = 0; i < layers.size(); ++i) {
            current_output = layers[i].forward_pass(current_output);
            activations_by_layer.push_back(current_output);
        }
        return current_output;
    }

    void train(py::array_t<double> training_inputs_py, 
               py::array_t<double> training_labels_py, 
               double learning_rate, int epochs) {

        auto inputs_buf = training_inputs_py.request();
        auto labels_buf = training_labels_py.request();
        
        if (inputs_buf.ndim != 2 || labels_buf.ndim != 2) {
            throw std::runtime_error("Training data must be 2D NumPy arrays.");
        }
        size_t num_samples = inputs_buf.shape[0];
        size_t input_size = inputs_buf.shape[1];
        size_t output_size = labels_buf.shape[1];
        if (input_size != static_cast<size_t>(neurons_layout.front()) || 
            output_size != static_cast<size_t>(neurons_layout.back())) {
            throw std::runtime_error("Training data dimensions do not match network layout.");
        }
        const double* inputs_data = static_cast<const double*>(inputs_buf.ptr);
        const double* labels_data = static_cast<const double*>(labels_buf.ptr);
        std::vector<int> training_indices(num_samples);
        std::iota(training_indices.begin(), training_indices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(training_indices.begin(), training_indices.end(), neuron_rng);

            for (int index : training_indices) {
                std::vector<double> input_sample(inputs_data + index * input_size, inputs_data + (index + 1) * input_size);
                std::vector<double> label_sample(labels_data + index * output_size, labels_data + (index + 1) * output_size);

                forward_pass_and_store(input_sample);

                layers.back().backward_pass_output_layer(activations_by_layer[layers.size() - 1], label_sample, learning_rate);
                
                for (int i = layers.size() - 2; i >= 0; --i) {
                    const auto& next_layer_deltas = layers[i+1].get_all_deltas();
                    const auto& next_layer_weights = layers[i+1].get_all_weights();
                    const auto& current_layer_inputs = activations_by_layer[i];
                    layers[i].backward_pass_hidden_layer(current_layer_inputs, next_layer_deltas, next_layer_weights, learning_rate);
                }
            }

            if (epoch % 1000 == 0) {
                double total_loss = 0.0;
                for (size_t i = 0; i < num_samples; ++i) {
                    std::vector<double> input_sample(inputs_data + i * input_size, inputs_data + (i + 1) * input_size);
                    std::vector<double> label_sample(labels_data + i * output_size, labels_data + (i + 1) * output_size);
                    std::vector<double> output_vec = forward_pass_and_store(input_sample);
                    for(size_t j = 0; j < output_vec.size(); ++j) {
                        total_loss += std::pow(output_vec[j] - label_sample[j], 2);
                    }
                }
                double avg_loss = total_loss / (num_samples * output_size);
                std::cout << "Epoch " << epoch << ", Avg Loss: " << avg_loss << std::endl;
            }
        }
    }
};


// --- GPU-BASED CLASSES ---

class NeuralNetworkGPU {
private:
    std::vector<int> layout;
    std::vector<double*> d_weights, d_biases, d_weighted_sums, d_activations, d_deltas;
    double* d_input = nullptr;
    int num_layers;

    void allocate_layer(int layer_idx, int in_size, int out_size) {
        double* w; cudaMalloc(&w, in_size * out_size * sizeof(double));
        d_weights.push_back(w);

        double* b; cudaMalloc(&b, out_size * sizeof(double));
        d_biases.push_back(b);

        double* s; cudaMalloc(&s, out_size * sizeof(double));
        d_weighted_sums.push_back(s);

        double* a; cudaMalloc(&a, out_size * sizeof(double));
        d_activations.push_back(a);

        double* d; cudaMalloc(&d, out_size * sizeof(double));
        d_deltas.push_back(d);
    }

public:
    NeuralNetworkGPU(const std::vector<int>& net_layout) : layout(net_layout) {
        num_layers = layout.size() - 1;
        for (int i = 0; i < num_layers; ++i) {
            allocate_layer(i, layout[i], layout[i + 1]);
        }
    }

    py::array_t<double> forward_pass(py::array_t<double> input_data_py) {
        auto buf = input_data_py.request();
        int input_size = buf.shape[0];
        cudaMalloc(&d_input, input_size * sizeof(double));
        cudaMemcpy(d_input, buf.ptr, input_size * sizeof(double), cudaMemcpyHostToDevice);

        double* prev_output = d_input;

        for (int i = 0; i < num_layers; ++i) {
            int in_size = layout[i];
            int out_size = layout[i + 1];
            int threads = 256, blocks = (out_size + threads - 1) / threads;

            forward_kernel<<<blocks, threads>>>(
                prev_output, d_weights[i], d_biases[i],
                d_activations[i], in_size, out_size
            );
            prev_output = d_activations[i];
        }

        int final_size = layout.back();
        std::vector<double> result(final_size);
        cudaMemcpy(result.data(), d_activations.back(), final_size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_input);
        return py::cast(result);
    }

    void train(py::array_t<double> x_py, py::array_t<double> y_py, double lr, int epochs) {
        auto x_buf = x_py.request(), y_buf = y_py.request();
        int num_samples = x_buf.shape[0], input_size = x_buf.shape[1], output_size = y_buf.shape[1];

        const double* x_ptr = static_cast<const double*>(x_buf.ptr);
        const double* y_ptr = static_cast<const double*>(y_buf.ptr);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int s = 0; s < num_samples; ++s) {
                const double* input_sample = x_ptr + s * input_size;
                const double* label_sample = y_ptr + s * output_size;

                cudaMemcpy(d_input, input_sample, input_size * sizeof(double), cudaMemcpyHostToDevice);
                double* prev = d_input;

                // Forward
                for (int i = 0; i < num_layers; ++i) {
                    int in_size = layout[i], out_size = layout[i + 1];
                    int threads = 256, blocks = (out_size + threads - 1) / threads;
                    forward_kernel<<<blocks, threads>>>(prev, d_weights[i], d_biases[i], d_activations[i], in_size, out_size);
                    prev = d_activations[i];
                }

                // Output delta
                int last = num_layers - 1;
                int threads = 256, blocks = (layout.back() + threads - 1) / threads;
                cudaMemcpy(d_activations.back(), prev, output_size * sizeof(double), cudaMemcpyDeviceToDevice);
                compute_output_delta<<<blocks, threads>>>(
                    d_activations[last], label_sample, d_weighted_sums[last],
                    d_deltas[last], layout.back()
                );

                // Hidden deltas
                for (int i = num_layers - 2; i >= 0; --i) {
                    int cur = layout[i + 1], next = layout[i + 2];
                    int threads = 256, blocks = (cur + threads - 1) / threads;
                    compute_hidden_delta<<<blocks, threads>>>(
                        d_deltas[i + 1], d_weights[i + 1], d_weighted_sums[i],
                        d_deltas[i], cur, next
                    );
                }

                // Update weights
                prev = d_input;
                for (int i = 0; i < num_layers; ++i) {
                    int in_size = layout[i], out_size = layout[i + 1];
                    int threads = 256, blocks = (out_size + threads - 1) / threads;
                    update_weights_kernel<<<blocks, threads>>>(
                        d_weights[i], d_biases[i], d_deltas[i], prev,
                        in_size, out_size, lr
                    );
                    prev = d_activations[i];
                }
            }
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << " complete\n";
            }
        }
        cudaFree(d_input);
    }

    ~NeuralNetworkGPU() {
        for (auto& p : d_weights) cudaFree(p);
        for (auto& p : d_biases) cudaFree(p);
        for (auto& p : d_activations) cudaFree(p);
        for (auto& p : d_weighted_sums) cudaFree(p);
        for (auto& p : d_deltas) cudaFree(p);
    }
};

// --- WRAPPER CLASS FOR CONDITIONAL EXECUTION ---

class NeuralNetwork {
private:
    std::unique_ptr<NeuralNetworkCPU> cpu_net;
    std::unique_ptr<NeuralNetworkGPU> gpu_net;
    bool using_gpu = false;

public:
    NeuralNetwork(const std::vector<int>& layout) {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);

        if (err == cudaSuccess && device_count > 0) {
            std::cout << "CUDA-enabled GPU found. Using GPU acceleration." << std::endl;
            using_gpu = true;
            gpu_net = std::make_unique<NeuralNetworkGPU>(layout);
        } else {
            std::cout << "No CUDA-enabled GPU found. Using CPU threading." << std::endl;
            using_gpu = false;
            cpu_net = std::make_unique<NeuralNetworkCPU>(layout);
        }
    }
    
    py::array_t<double> forward(py::array_t<double> input_data_py) {
        if (using_gpu) {
            return gpu_net->forward_pass(input_data_py);
        } else {
            return cpu_net->forward_pass(input_data_py);
        }
    }

    void train(py::array_t<double> training_inputs_py, py::array_t<double> training_labels_py, double learning_rate, int epochs) {
        if (using_gpu) {
            gpu_net->train(training_inputs_py, training_labels_py, learning_rate, epochs);
        } else {
            cpu_net->train(training_inputs_py, training_labels_py, learning_rate, epochs);
        }
    }
};

PYBIND11_MODULE(my_nn_module, m) {
    m.doc() = "Custom Neural Network module with GPU acceleration and CPU fallback.";
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const std::vector<int>&>(), py::arg("layout"))
        .def("forward", &NeuralNetwork::forward, py::arg("input_data"))
        .def("train", &NeuralNetwork::train, py::arg("training_inputs"), py::arg("training_labels"), py::arg("learning_rate"), py::arg("epochs"));
}