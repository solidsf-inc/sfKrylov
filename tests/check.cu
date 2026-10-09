// SPDX-License-Identifier: MIT
#include "../src/kernel.cuh"
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void cuda_check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
template<class T> struct Device {
    T* p = nullptr; size_t n;
    explicit Device(size_t count) : n(count) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& data) { if (data.size() != n) throw std::runtime_error("upload size"); cuda_check(cudaMemcpy(p, data.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() { std::vector<T> data(n); cuda_check(cudaMemcpy(data.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return data; }
};
struct Case { std::string name; bool pass; };

int main() {
    std::vector<Case> cases; auto record = [&](const char* name, bool pass) { cases.push_back({name, pass}); };
    cudaStream_t stream = nullptr;
    try {
        cuda_check(cudaStreamCreate(&stream));
        auto parity = [&](int n, int batch, const std::vector<int>& offsets, const std::vector<int>& columns,
            const std::vector<double>& matrices, const std::vector<double>& known) {
            const int nnz = static_cast<int>(columns.size()); std::vector<double> rhs(n * batch, 0.0);
            for (int s = 0; s < batch; ++s) for (int i = 0; i < n; ++i) for (int k = offsets[i]; k < offsets[i + 1]; ++k)
                rhs[s * n + i] += matrices[s * nnz + k] * known[s * n + columns[k]];
            Device<int> d_offsets(offsets.size()), d_columns(columns.size());
            Device<double> d_values(matrices.size()), d_rhs(rhs.size()), d_x(rhs.size()); Device<sfkrylov::Result> d_results(batch);
            d_offsets.put(offsets); d_columns.put(columns); d_values.put(matrices); d_rhs.put(rhs);
            cuda_check(sfkrylov::conjugate_gradient(n, nnz, batch, d_offsets.p, d_columns.p, d_values.p,
                d_rhs.p, d_x.p, d_results.p, 128, 1e-10, 0.0, stream)); cuda_check(cudaStreamSynchronize(stream));
            const auto x = d_x.get(); const auto results = d_results.get(); bool pass = true;
            for (int s = 0; s < batch; ++s) {
                double residual_squared = 0.0, rhs_squared = 0.0;
                for (int i = 0; i < n; ++i) {
                    double ax = 0.0;
                    for (int k = offsets[i]; k < offsets[i + 1]; ++k) ax += matrices[s * nnz + k] * x[s * n + columns[k]];
                    const double error = ax - rhs[s * n + i]; residual_squared += error * error; rhs_squared += rhs[s * n + i] * rhs[s * n + i];
                    pass &= std::isfinite(x[s * n + i]) && std::abs(x[s * n + i] - known[s * n + i]) < 1e-8;
                }
                const double relative = rhs_squared > 0.0 ? std::sqrt(residual_squared / rhs_squared) : std::sqrt(residual_squared);
                pass &= results[s].status == sfkrylov::Status::Converged && relative <= 1e-10;
                pass &= std::isfinite(results[s].relative_residual) && std::abs(relative - results[s].relative_residual) < 1e-12;
            }
            return pass;
        };
        std::vector<int> offsets = {0,2,5,8,11,13}, columns = {0,1, 0,1,2, 1,2,3, 2,3,4, 3,4};
        std::vector<double> matrices;
        for (int system = 0; system < 2; ++system) for (int i = 0; i < 5; ++i) for (int k = offsets[i]; k < offsets[i + 1]; ++k)
            matrices.push_back(columns[k] == i ? 4.0 + 2.0 * system : -1.0 + 0.5 * system);
        record("batched-spd-vs-independent-scalar-residual", parity(5, 2, offsets, columns, matrices, {1,-2,3,-4,5, -0.5,1,-1.5,2,-2.5}));
        const int n = sfkrylov::max_order;
        std::vector<int> large_offsets(n + 1), large_columns(n); std::vector<double> diagonal(n), known(n);
        for (int i = 0; i < n; ++i) { large_offsets[i] = i; large_columns[i] = i; diagonal[i] = 1.0 + i % 3; known[i] = (i % 7 - 3) * 0.25; } large_offsets[n] = n;
        record("maximum-order-1024-spd", parity(n, 1, large_offsets, large_columns, diagonal, known));
        record("zero-rhs-is-converged", parity(5, 1, offsets, columns, std::vector<double>(matrices.begin(), matrices.begin() + 13), std::vector<double>(5, 0.0)));
        Device<int> d_offsets(offsets.size()), d_columns(columns.size()); Device<double> d_values(13), d_rhs(5), d_x(5); Device<sfkrylov::Result> d_result(1);
        d_offsets.put(offsets); d_columns.put(columns); d_values.put(std::vector<double>(matrices.begin(), matrices.begin() + 13)); d_rhs.put({1,2,3,4,5});
        cuda_check(sfkrylov::conjugate_gradient(5, 13, 1, d_offsets.p, d_columns.p, d_values.p, d_rhs.p, d_x.p, d_result.p, 0, 1e-12, 0.0, stream));
        cuda_check(cudaStreamSynchronize(stream));
        record("zero-iteration-budget-status", d_result.get()[0].status == sfkrylov::Status::MaxIterations && d_result.get()[0].iterations == 0 && d_result.get()[0].relative_residual == 1.0);
        cuda_check(sfkrylov::conjugate_gradient(5, 13, 1, d_offsets.p, d_columns.p, d_values.p, d_rhs.p, d_x.p, d_result.p, 1, 1e-12, 0.0, stream));
        cuda_check(cudaStreamSynchronize(stream)); record("one-iteration-budget-status", d_result.get()[0].status == sfkrylov::Status::MaxIterations && d_result.get()[0].iterations == 1);
        columns[0] = 5; d_columns.put(columns);
        cuda_check(sfkrylov::conjugate_gradient(5, 13, 1, d_offsets.p, d_columns.p, d_values.p, d_rhs.p, d_x.p, d_result.p, 128, 1e-12, 0.0, stream));
        cuda_check(cudaStreamSynchronize(stream)); record("invalid-csr-status-before-indexing", d_result.get()[0].status == sfkrylov::Status::InvalidInput);
        columns[0] = 0; d_columns.put(columns); d_rhs.put({1,2,std::numeric_limits<double>::quiet_NaN(),4,5});
        cuda_check(sfkrylov::conjugate_gradient(5, 13, 1, d_offsets.p, d_columns.p, d_values.p, d_rhs.p, d_x.p, d_result.p, 128, 1e-12, 0.0, stream));
        cuda_check(cudaStreamSynchronize(stream)); record("nonfinite-input-status", d_result.get()[0].status == sfkrylov::Status::InvalidInput);
        Device<int> d_one_offsets(2), d_one_column(1); Device<double> d_negative(1), d_one_rhs(1), d_one_x(1);
        d_one_offsets.put({0,1}); d_one_column.put({0}); d_negative.put({-1}); d_one_rhs.put({1});
        cuda_check(sfkrylov::conjugate_gradient(1, 1, 1, d_one_offsets.p, d_one_column.p, d_negative.p, d_one_rhs.p, d_one_x.p, d_result.p, 10, 1e-12, 0.0, stream));
        cuda_check(cudaStreamSynchronize(stream)); record("indefinite-curvature-breakdown", d_result.get()[0].status == sfkrylov::Status::Breakdown);
        Device<int> d_tiny_offsets(3), d_tiny_columns(4); Device<double> d_tiny_values(4), d_huge_rhs(2), d_overflow_x(2);
        d_tiny_offsets.put({0,2,4}); d_tiny_columns.put({0,1,0,1});
        d_tiny_values.put({1e-300,0.0,0.0,1e-300}); d_huge_rhs.put({1e153,1e153});
        cuda_check(sfkrylov::conjugate_gradient(2, 4, 1, d_tiny_offsets.p, d_tiny_columns.p, d_tiny_values.p,
            d_huge_rhs.p, d_overflow_x.p, d_result.p, 10, 1e-10, 0.0, stream));
        cuda_check(cudaStreamSynchronize(stream));
        const auto overflow = d_result.get()[0];
        record("finite-spd-overflow-and-nan-true-residual-refuse-convergence",
            overflow.status == sfkrylov::Status::NonFinite && std::isnan(overflow.relative_residual));
        record("reject-domain-budget-and-alias",
            sfkrylov::conjugate_gradient(1025, 13, 1, d_offsets.p, d_columns.p, d_values.p, d_rhs.p, d_x.p, d_result.p) == cudaErrorInvalidValue &&
            sfkrylov::conjugate_gradient(5, 13, 1, d_offsets.p, d_columns.p, d_values.p, d_rhs.p, d_x.p, d_result.p, 4097) == cudaErrorInvalidValue &&
            sfkrylov::conjugate_gradient(5, 13, 1, d_offsets.p, d_columns.p, d_values.p, d_rhs.p, d_rhs.p, d_result.p) == cudaErrorInvalidValue);
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); record("cuda-runtime", false); }
    if (stream) cudaStreamDestroy(stream);
    bool pass = !cases.empty(); for (const auto& item : cases) pass &= item.pass;
    std::printf("{\"package\":\"sfKrylov\",\"status\":\"%s\",\"cases\":[", pass ? "PASS" : "FAIL");
    for (size_t i = 0; i < cases.size(); ++i) std::printf("%s{\"name\":\"%s\",\"status\":\"%s\"}", i ? "," : "", cases[i].name.c_str(), cases[i].pass ? "PASS" : "FAIL");
    std::puts("]}"); return pass ? 0 : 1;
}
