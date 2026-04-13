# Cube–Vector software pipelining

This document describes the **CV Pipelining** pass in HIVM. It optimizes CV-style kernels. Before reading, we recommend [CV Optimization](../CV/CVOptimization.md) for CV terminology.

## Hardware Context

Each Ascend Core contains Cube and Vector units, performing matrix-multiplication and other vectorized compuations respectively. These cores are capable of running in parallel to each other asynchronously whenever there are no data dependencies. In order to achieve better performance, it is essential to leverage this capability for better hardware utilization.

CV-Pipelining optimizes loops in Mix kernels where multiple Cube and Vector instructions depend on each other (e.g. FlashAttention). By running Vector and Cube in parallel, it improves hardware utilization (ILP) and performance.

The pass uses multi-buffering, which can increase UB usage; tune the number of pipeline stages for your workload.

## Algorithm

The pass finds suitable `scf.for` loops, splits Cube and Vector into separate work items, sets up data dependencies, expands tensors that need multi-buffering, unrolls the loop, and places each work item in its own loop.

Before:

```mlir
scf.for 0 to N step S {
    %c = Cube() : tensor<16x16xf32>
    %v = Vector(%c) : tensor<16x16xf32>
    %c1 = Cube(%v) : tensor<16x16xf32>
    %v1 = Vector(%c1) : tensor<16x16xf32>
}
```

After:

```mlir
scf.for 0 to N step 3*S {
    %c = scf.for 0 to 3 -> tensor<3x16x16xf32> {
        Cube();
        tensor.insert_slice
    } {cube_loop}
    %v = scf.for 0 to 3 -> tensor<3x16x16xf32> {
        %c_slice = extract_slice %c
        Vector(%c_slice) : tensor<16x16xf32>
        tensor.insert_slice
    } {vector_loop}
    %c1 = scf.for 0 to 3 -> tensor<3x16x16xf32> {
        %v_slice = extract_slice %v
        Cube(%v_slice) : tensor<16x16xf32>
        tensor.insert_slice
    } {cube_loop}
    // When no other Work Item needs the result, no buffer expansion needed
    %v1 = scf.for 0 to 3 -> tensor<16x16xf32> {
        %c_slice = extract_slice %c1
        Vector(%c_slice) : tensor<16x16xf32>
    } {vector_loop}
}
```

## Interface

### Compile options

| Option | Default | Description |
|--------|--------|-------------|
| `set-workspace-multibuffer` | 2 | Number of software pipeline stages and multi-buffering count |

## Constraints

1. Only `scf.for` and `scf.if` in the pipelined loop may have regions/blocks, and each region must contain only Cube or only Vector instructions.
2. Cross-iteration dependencies must be separable into distinct work items.
   - CV-pipelining is not applied when `v0` and `v1` cannot be in the same work item (due to a Cube between them) but `arg0` is defined in `v1` and used by `v0`.
   - If Cube does not use `v0`, `v0` is placed in the same work item as `v1` and CV-pipelining applies.

```mlir
scf.for iter_args(%arg0 = %init) {
    %v0 = Vector(%arg0)
    %c = Cube(%v0)
    %v1 = Vector(%c)
    yield %v1
}
```
