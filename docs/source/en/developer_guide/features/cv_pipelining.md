# Cube and Vector Software Pipelining Optimization

This document introduces the CV Pipelining Pass in HIVM. This pass optimizes CV-type kernels. Before reading this document, it is recommended to read [CV Optimization](./cv_optimization.md) first to understand the CV compilation-related terminology.

## Hardware Background

The Ascend core includes the Cube core (responsible for matrix multiplication-related operations) and the Vector core (responsible for other vector operations). These two cores can run in parallel and asynchronously when there is no dependency between them. Improving hardware utilization is a particularly important part of performance optimization.

This feature optimizes scenarios in MIX operators where multiple Cube and Vector instructions depend on each other within a loop (for example, operators such as FlashAttention). By parallelizing the Vector and Cube cores, higher hardware utilization (ILP) is achieved, resulting in better performance.

This feature uses Multi-Buffering optimization, which increases UB space usage. Therefore, the number of software pipelining stages needs to be adjusted according to the actual scenario to achieve the best performance.

## Pipeline Mode

CV Pipelining currently supports two main pipeline modes:

| Mode | Feature | Applicable Scenarios |
|------|------|----------|
| Unroll mode | Unrolls the original loop by the configured depth and generates an inner loop for each Work Item | Standard CV pipelining scenarios |
| Preload mode | Uses the Skew approach to achieve cross-iteration overlapping execution through iteration offset | CV fusion scenarios such as FlashAttention |

**Preload mode principle**: Organizes the Cube/Vector computations in the original loop into multiple Preload stages, so that data movement and computation of different iterations are interleaved and scheduled. By executing the data loading of subsequent iterations in advance, it hides memory access and synchronization latency and improves C/V core utilization. Each code segment is wrapped by `scope` and marked with `preload_number`, which indicates the number of physical loop beats by which it is advanced relative to the final consumption stage.

## Algorithm Principles

Locate an appropriate `for` loop, separate the Cube and Vector instructions into independent Work Items, establish the data dependencies between each Work Item, and expand the `tensor` that needs to be extended into a `multi-buffer`.

### Unroll Mode Transformation

After the original loop is `unroll`ed, place each Work Item into a separate loop.

Before transformation:

```mlir
scf.for 0 to N step S {
    %c = Cube() : tensor<16x16xf32>
    %v = Vector(%c) : tensor<16x16xf32>
    %c1 = Cube(%v) : tensor<16x16xf32>
    %v1 = Vector(%c1) : tensor<16x16xf32>
}
```

After transformation:

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

### Preload Mode Transformation

The Skew approach is used to encapsulate Work Items into a `scope` with `preload_number`, enabling cross-iteration interleaved scheduling through iteration offset.

Before transformation:

```mlir
scf.for 0 to N step S {
    %c = Cube() : tensor<16x16xf32>
    %v = Vector(%c) : tensor<16x16xf32>
    %c1 = Cube(%v) : tensor<16x16xf32>
    %v1 = Vector(%c1) : tensor<16x16xf32>
}
```

After transformation: (using Ascend 950PR/Ascend 950DT as an example)

```mlir
scf.for %i = 0 to N+max_preload_num step S {
    if 0 <= %i < N {
        scope.scope {max_preload_num=4, preload_number = 3} {
            %c_i = Cube() -> buffer0[%i % 2]
        }
    }
    %ii = %i - 1
    if 0 <= %ii < N {
        scope.scope {max_preload_num=4, preload_number = 2} {
            %v_ii = Vector(buffer0[%ii % 2]) -> buffer1[%ii % 2]
        }
    }
    %iii = %i -2
    if 0 <= %iii < N {
        scope.scope {max_preload_num=4, preload_number = 1} {
            %c1_iii = Cube(buffer1[%iii % 2]) -> buffer2[%iii % 2]
        }
    }
    %iiii = %i - 3
    if 0 <= %iiii < N {
        scope.scope {max_preload_num=4, preload_number = 0} {
            %v1_iiii = Vector(buffer2[%iiii % 2])
        }
    }
}
```

## Compilation Options

| Option | Default Value | Meaning |
|------|--------|------|
| `set-workspace-multibuffer` | 2 | Number of software pipeline stages, which is also the number of Multi-Buffering. |
| `--enable-lazy-loading` | false | Enables the Lazy Load feature in CV Pipelining, allowing Load ops to be cloned into multiple Work Items to reduce intermediate buffer expansion. |
| `--cv-pipeline-mode` | Unroll | CV pipeline mode: `Off`/`Unroll`/`Skew`. `Skew` corresponds to the Preload mode. |
| `--enable-preload` | false | Enables the Preload mode, which is equivalent to setting `--cv-pipeline-mode=Skew`. |

You can also enable the Lazy Load feature for a specified tensor on the operator side through the `cv_pipeline_lazy_load` compilation hint:

```python
extension.compile_hint(t, "cv_pipeline_lazy_load", True)
```

## Constraints

1. In loops that support pipeline processing, only the `scf.for` and `scf.if` ops may contain regions/blocks, and only cube or vector instructions are allowed inside the regions of these ops.
2. Data dependencies between iterations must be separable into independent Work Items.
    - Scenario where CV-Pipelining cannot be enabled: if `v0` and `v1` cannot be extracted into the same Work Item (because of a Cube dependency in between), while the parameter `arg0` is defined by `v1` but used by `v0`.
    - Scenario where CV-Pipelining takes effect normally: if Cube does not use `v0`, then `v0` can be sunk into the Work Item where `v1` resides, and CV-Pipelining takes effect normally.

Example code:

```mlir
scf.for iter_args(%arg0 = %init) {
    %v0 = Vector(%arg0)
    %c = Cube(%v0)
    %v1 = Vector(%c)
    yield %v1
}
```

## Future Evolution

The currently implemented Preload mode is based on a fixed stage partitioning strategy. More flexible scheduling optimizations will be supported in the future:

- **Dynamic scheduling mode**: Through a Cost Model, Profiling, or user-provided Hints, the optimal pipeline depth and number of buffers are dynamically selected based on factors such as the Cube/Vector load ratio, stage duration, communication volume, and on-chip memory capacity.
- **Adaptive strategy selection**: For different workload characteristics (compute-intensive, memory-intensive, etc.), automatically switch between Unroll and Preload modes to maximize performance gains.
- **Cross-iteration dependency support**: Extend support for more complex dependency patterns to handle scenarios that cannot be enabled under the current constraints.
