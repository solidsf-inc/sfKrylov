# sfKrylov

Standalone CUDA C++17 batched real-f64 conjugate gradient for symmetric
positive definite CSR systems, `A x = b`, with a zero initial guess. This is
a fresh MIT mathematical implementation. CUDA runtime is its only library
dependency; no proprietary solver source or private rvllm implementation is
included.

`src/kernel.cuh` exports `sfkrylov::conjugate_gradient(n, nnz, batch, offsets,
columns, matrices, rhs, solutions, results, max_iterations,
relative_tolerance, absolute_tolerance, stream)`. All systems share the same
zero-based signed-32-bit CSR offsets/columns. Each system has its own
contiguous `nnz` f64 values and `n` f64 RHS/solution entries. `results` is a
device array of `sfkrylov::Result`. It reports `Converged`, `MaxIterations`,
`Breakdown`, `InvalidInput`, or `NonFinite`, iterations and true relative
residual. Check the status before using a solution.

Each matrix-vector product, iterate/residual update, search direction and
true-residual component is checked collectively for finiteness. A non-finite
true residual reports `NonFinite` and a NaN residual metric; max reductions
are never allowed to hide NaNs and declare convergence.

This first implementation intentionally supports `1 <= n <= 1024`,
`1 <= batch <= 65535`, and `0 <= max_iterations <= 4096`. One 256-thread CTA
owns each system; vectors remain in shared memory and reductions are block
reductions. There is no host/device exchange inside the iteration loop.
This is a small-system baseline, not a large sparse solver or an optimized
GMRES/BiCGStab implementation. Those methods are not implemented here.

The caller must supply SPD matrices. The kernel validates CSR monotonicity,
offset endpoints, column bounds and finite matrix/RHS entries before indexing
CSR rows. Nonpositive search-direction curvature reports `Breakdown`;
that check does not establish matrix symmetry or positive definiteness.
`Converged` additionally requires an independently recomputed device residual
`||b-Ax||2 <= max(absolute_tolerance, relative_tolerance ||b||2)`.
The reported residual is computed with scaled norm accumulation and refers
to the original supplied matrix. A recurrence estimate alone cannot declare
convergence. A true-residual miss reports `MaxIterations`; no fallback or
silent best-effort success exists. Breakdown/nonfinite returns the finite
prefix iterate where available. Invalid input returns a zero iterate.

Relative tolerance must be finite and in `(0,1)`, absolute tolerance finite
and nonnegative. A zero iteration budget still checks whether the initial
zero solution is exact. Squared recurrence norms and products use f64;
underflow/overflow of those computations can report `NonFinite` or
`Breakdown`. Callers should scale exceptionally small or large systems;
this package does not promise convergence for every finite SPD input.

Buffers must be valid device storage of the stated extents. RHS/solution
overlap is unsupported. Host checks reject invalid dimensions, null pointers,
exact aliasing and invalid budgets/tolerances. The API queues work on the
supplied stream, returns `cudaGetLastError()` and does not synchronize.
The caller checks stream completion and reads the device result status.

Required device targets: H100/H200 `sm_90`, B200 `sm_100`, B300/GB300 `sm_103`,
GB10 `sm_121`. Only portable CUDA core operations are used. f64 performance
varies by device. GB10 timing for n=128, nnz=382 and 17 iterations was
114.400 us versus 845.824 us for a cuSPARSE/cuBLAS host loop (7.394x).
These seven-sample CUDA-event medians include the reference host scalar
round trips and GPU idle gaps, and exclude setup and transfers. This is
solve latency for that workload; other devices and larger workloads remain
unmeasured. Use a toolkit that
recognizes the selected target, and record hardware qualification separately
from architecture compilation.

`tests/check.cu` emits one JSON object, returning nonzero on any failed case.
Manufactured SPD solutions and separate scalar CSR residuals test a batched
tridiagonal system, the 1024-order boundary and zero RHS. Negative controls
cover exhausted iteration budgets, indefinite curvature, invalid CSR,
non-finite inputs, domain limits and exact aliasing. An extreme but finite
SPD matrix with explicit zero entries and an overflowing
solution verifies that NaN true residuals cannot produce `Converged`.
Build with the parent
tooling: `make -j1 ARCHS='90 100 103 121' BUILD_DIR=<durable-path>`, then
`make test` to run `BUILD_DIR/check`. Alternatively compile
`nvcc -std=c++17 -O3 -arch=sm_121 tests/check.cu -o
<retained-staging-path>/check`, under approved resource limits.

## Standalone build

CUDA 13.0 or newer is required for the full target matrix.

```sh
make -j1
make test
# A single device family can be selected explicitly:
make ARCHS=90 BUILD_DIR=build-hopper
```

Default targets: H100/H200 (`sm_90`), B200 (`sm_100`), B300/GB300 (`sm_103`),
GB10 (`sm_121`), plus portable Hopper PTX. Do not add `--use_fast_math`;
that changes the numerical contract. Compile success and numerical
qualification are reported separately from performance benchmarking.

## Published source and GB10 measurements

The standalone source repository is [solidsf-inc/sfKrylov](https://github.com/solidsf-inc/sfKrylov). The [solidSF CUDA kernel catalog](https://solidsf.com/kernels) contains the formal report, all measured workloads and raw timing samples.

| Workload | SF median us | Reference median us | Reference / SF |
| --- | ---: | ---: | ---: |
| cg f64; 128 x 128, nnz=382 | 113.792 | 853.472 | 7.500x |

References: conventional zero-start CG: cuSPARSE f64 CSR ALG2 plus cuBLAS Ddot/Daxpy/Dscal.

These are seven-sample medians of alternating CUDA-event measurements on one GB10 at stock clocks. Setup, allocations, transfers, plan creation and destructive-input resets are excluded; stream timing can include API enqueue gaps. Ratios above one mean SF is faster; near-unity medians do not establish a meaningful advantage. No aggregate or universal speedup is claimed.

All 10 package numerical/API cases passed on GB10. All four cubin targets and compute_90 PTX were verified. H100, H200, B200, B300 and GB300 hardware runtime and performance remain unmeasured.
The CG reference includes host scalar round trips and GPU idle gaps; the ratio measures solve latency.
