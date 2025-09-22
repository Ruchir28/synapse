# Synapse

Synapse is an n-dimensional array engine written in Rust.

- `NDArray<T>` stores elements in a reference-counted `Vec<T>` behind an `Arc`, allowing cheap cloning of views while keeping ownership clear.
- Each array tracks logical `dims`, signed `strides`, and a base `offset`. Slicing and permutation reuse the same allocation by manipulating these metadata fields instead of copying buffers.
- Safe element access is exposed through `get`/`get_mut`, while iterators (`iter`, `indexed_iter`) traverse a view in logical order regardless of stride layout.

## Operation Modules
- `ops::arithmetic` implements element-wise arithmetic operations (`Add`, `Sub`, `Mul`, `Div`) with broadcasting support. Operations delegate to checked routines (`try_add`, `try_mul`, …) that validate shapes and return detailed errors via `NDArrayError`.
- `ops::broadcast` computes compatible shapes and adjusted strides for NumPy-style broadcasting between arrays of different dimensions. It enables element-wise operations on mismatched shapes without copying data or creating intermediate arrays.
- `ops::slice` creates zero-copy sub-views by manipulating strides and offsets rather than copying data. Multiple chained slicing operations remain O(1) and continue sharing the original buffer allocation.
- `ops::transform` provides layout transformations like `permute_axis` (transpose) that reorder dimensions by manipulating strides without data movement. These zero-copy transforms enable efficient matrix operations and tensor reshaping.
- `ops::reduction` provides aggregation operations (`sum`, `sum_axis`, `mean`, `mean_axis`) that collapse arrays along specified axes. Operations use logical iteration over views and produce compact outputs with reduced dimensionality, automatically converting to appropriate types (e.g., `f64` for means).
- `ops::dot` implements matrix multiplication for both 2D matrices and batched higher-dimensional tensors. It uses SIMD acceleration on supported architectures and optimizes memory access patterns through strategic transposition.
- `ops::arch` contains architecture-specific SIMD kernels. On `aarch64`, it uses NEON intrinsics to accelerate element-wise arithmetic and dot products on `f32` arrays, processing multiple data lanes in a single instruction. A scalar fallback handles any remaining elements.
- `ops::shapes` provides `reshape` with validation against element counts and contiguity, returning a view that reuses buffers when possible.


## Repository Layout
- `src/` — Rust implementation of the NDArray core, operation traits, and optional Python bindings.
- `python/synapse/` — Python package that ships the compiled extension and ergonomic wrappers.
- `benches/` — Criterion benchmarks for arithmetic kernels.
