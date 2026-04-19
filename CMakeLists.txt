cmake_minimum_required(VERSION 3.15...3.28)
project(NeuralNetPythonBinding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and Pybind11 - pybind11 will be used to get include paths etc.
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- Define the C++ Shared Library ---
# The target name 'my_nn_module' MUST match the 'name' in Pybind11Extension in setup.py.
# This creates a shared library.
add_library(my_nn_module SHARED NN.cpp) # <--- CHANGED: Use add_library instead of pybind11_add_module

# Link the generated library with Python and Pybind11 includes/libraries
target_link_libraries(my_nn_module PRIVATE Python::Python pybind11::headers) # Link against pybind11 headers

# --- Link with the backend for std::execution::par ---
# This section remains the same to ensure parallelization.
find_package(TBB CONFIG)
if (TBB_FOUND)
    message(STATUS "Found TBB: ${TBB_VERSION}")
    target_link_libraries(my_nn_module PRIVATE TBB::tbb)
else()
    message(WARNING "TBB not found. Parallel algorithms might fall back to sequential or use OpenMP/compiler-specific backend.")
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP: ${OpenMP_CXX_LIBRARIES}")
        target_link_libraries(my_nn_module PRIVATE OpenMP::OpenMP_CXX)
    else()
        message(FATAL_ERROR "No TBB or OpenMP found. std::execution::par will not be parallelized, and compilation might fail.")
    endif()
endif()

# Additional compile options for optimization (optional but recommended for performance)
target_compile_options(my_nn_module PRIVATE -O3)