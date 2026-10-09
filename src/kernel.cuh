// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace sfkrylov {
constexpr int max_order = 1024;
constexpr int max_iteration_limit = 4096;
constexpr int max_batch = 65535;
enum class Status : int { Converged, MaxIterations, Breakdown, InvalidInput, NonFinite };
struct Result { Status status; int iterations; double relative_residual; };

static __device__ double block_sum(double value, double* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const double sum = scratch[0];
    __syncthreads();
    return sum;
}

static __device__ double block_max(double value, double* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (tid < stride) scratch[tid] = fmax(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    const double maximum = scratch[0];
    __syncthreads();
    return maximum;
}

static __global__ void cg_kernel(int n, int nnz, const int* offsets, const int* columns,
    const double* matrices, const double* rhs, double* solutions, Result* results,
    int limit, double relative_tolerance, double absolute_tolerance) {
    __shared__ double x[max_order], r[max_order], p[max_order], ap[max_order], sums[256];
    __shared__ double rr, norm_b, target, alpha, beta;
    __shared__ int status, iterations;
    const int tid = threadIdx.x;
    const double* values = matrices ? matrices + static_cast<size_t>(blockIdx.x) * nnz : nullptr;
    const double* b = rhs + static_cast<size_t>(blockIdx.x) * n;
    double* output = solutions + static_cast<size_t>(blockIdx.x) * n;
    double invalid = 0.0, initial = 0.0, maximum_b = 0.0;
    for (int i = tid; i < n; i += 256) {
        x[i] = 0.0; r[i] = b[i]; p[i] = b[i];
        initial += b[i] * b[i];
        maximum_b = fmax(maximum_b, fabs(b[i]));
        invalid += !isfinite(b[i]) || offsets[i] < 0 || offsets[i + 1] < offsets[i] || offsets[i + 1] > nnz;
    }
    for (long long k = tid; k < nnz; k += 256)
        invalid += columns[k] < 0 || columns[k] >= n || !isfinite(values[k]);
    if (tid == 0) invalid += offsets[0] != 0 || offsets[n] != nnz;
    const double invalid_count = block_sum(invalid, sums);
    const double initial_squared = block_sum(initial, sums);
    const double largest_b = block_max(maximum_b, sums);
    if (tid == 0) {
        rr = initial_squared; norm_b = sqrt(rr);
        target = fmax(absolute_tolerance, relative_tolerance * norm_b);
        iterations = 0;
        status = invalid_count > 0 ? static_cast<int>(Status::InvalidInput) :
            (!isfinite(rr) || (rr == 0.0 && largest_b != 0.0) ? static_cast<int>(Status::NonFinite) :
            (norm_b <= target ? static_cast<int>(Status::Converged) : static_cast<int>(Status::MaxIterations)));
    }
    __syncthreads();
    for (int iteration = 0; iteration < limit && status == static_cast<int>(Status::MaxIterations); ++iteration) {
        double product = 0.0, invalid_product = 0.0;
        for (int i = tid; i < n; i += 256) {
            double value = 0.0;
            for (int k = offsets[i]; k < offsets[i + 1]; ++k) value = fma(values[k], p[columns[k]], value);
            ap[i] = value; product += p[i] * value;
            invalid_product += !isfinite(p[i]) || !isfinite(ap[i]);
        }
        const double p_ap = block_sum(product, sums);
        const double invalid_product_count = block_sum(invalid_product, sums);
        if (tid == 0) {
            if (invalid_product_count > 0.0 || !isfinite(p_ap) || !isfinite(rr)) status = static_cast<int>(Status::NonFinite);
            else if (!(p_ap > 0.0)) status = static_cast<int>(Status::Breakdown);
            else { alpha = rr / p_ap; if (!isfinite(alpha)) status = static_cast<int>(Status::NonFinite); }
        }
        __syncthreads();
        if (status != static_cast<int>(Status::MaxIterations)) break;
        for (int i = tid; i < n; i += 256) { x[i] = fma(alpha, p[i], x[i]); r[i] = fma(-alpha, ap[i], r[i]); }
        __syncthreads();
        double residual_squared = 0.0, invalid_update = 0.0;
        for (int i = tid; i < n; i += 256) {
            residual_squared += r[i] * r[i];
            invalid_update += !isfinite(x[i]) || !isfinite(r[i]);
        }
        const double next_rr = block_sum(residual_squared, sums);
        const double invalid_update_count = block_sum(invalid_update, sums);
        if (tid == 0) {
            ++iterations;
            if (invalid_update_count > 0.0 || !isfinite(next_rr)) status = static_cast<int>(Status::NonFinite);
            else if (sqrt(next_rr) <= target) status = static_cast<int>(Status::Converged);
            else { beta = next_rr / rr; rr = next_rr; if (!isfinite(beta)) status = static_cast<int>(Status::NonFinite); }
        }
        __syncthreads();
        if (status != static_cast<int>(Status::MaxIterations)) break;
        double invalid_direction = 0.0;
        for (int i = tid; i < n; i += 256) {
            p[i] = fma(beta, p[i], r[i]);
            invalid_direction += !isfinite(p[i]);
        }
        const double invalid_direction_count = block_sum(invalid_direction, sums);
        if (tid == 0 && invalid_direction_count > 0.0) status = static_cast<int>(Status::NonFinite);
        __syncthreads();
    }
    double largest_error = 0.0, invalid_true_residual = 0.0;
    if (status != static_cast<int>(Status::InvalidInput)) {
        for (int i = tid; i < n; i += 256) {
            double value = 0.0;
            for (int k = offsets[i]; k < offsets[i + 1]; ++k) value = fma(values[k], x[columns[k]], value);
            const double error = b[i] - value;
            r[i] = error;
            invalid_true_residual += !isfinite(x[i]) || !isfinite(value) || !isfinite(error);
            if (isfinite(error)) largest_error = fmax(largest_error, fabs(error));
        }
    }
    const double invalid_true_residual_count = block_sum(invalid_true_residual, sums);
    const double error_scale = block_max(largest_error, sums);
    double exact_squared = 0.0;
    if (status != static_cast<int>(Status::InvalidInput) && invalid_true_residual_count == 0.0 && error_scale > 0.0)
        for (int i = tid; i < n; i += 256) { const double scaled = r[i] / error_scale; exact_squared += scaled * scaled; }
    const double accumulated_norm = error_scale * sqrt(block_sum(exact_squared, sums));
    const double exact_norm = invalid_true_residual_count > 0.0 ? nan("") : accumulated_norm;
    if (tid == 0) {
        if (status != static_cast<int>(Status::InvalidInput)) {
            if (invalid_true_residual_count > 0.0 || !isfinite(exact_norm)) status = static_cast<int>(Status::NonFinite);
            else if (status == static_cast<int>(Status::Converged) && exact_norm > target)
                status = static_cast<int>(Status::MaxIterations);
        }
        results[blockIdx.x] = {static_cast<Status>(status), iterations,
            status == static_cast<int>(Status::InvalidInput) ? nan("") : (norm_b > 0.0 ? exact_norm / norm_b : exact_norm)};
    }
    for (int i = tid; i < n; i += 256) output[i] = x[i];
}

inline cudaError_t conjugate_gradient(int n, int nnz, int batch, const int* offsets,
    const int* columns, const double* matrices, const double* rhs, double* solutions, Result* results,
    int max_iterations = 1024, double relative_tolerance = 1e-10, double absolute_tolerance = 0.0,
    cudaStream_t stream = nullptr) {
    if (n <= 0 || n > max_order || nnz < 0 || batch <= 0 || batch > max_batch || max_iterations < 0 || max_iterations > max_iteration_limit ||
        !offsets || (nnz && (!columns || !matrices)) || !rhs || !solutions || !results || rhs == solutions ||
        !std::isfinite(relative_tolerance) || relative_tolerance <= 0.0 || relative_tolerance >= 1.0 ||
        !std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0) return cudaErrorInvalidValue;
    cg_kernel<<<batch, 256, 0, stream>>>(n, nnz, offsets, columns, matrices, rhs, solutions, results,
        max_iterations, relative_tolerance, absolute_tolerance);
    return cudaGetLastError();
}
}
