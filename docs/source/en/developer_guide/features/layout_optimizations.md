# Reduction Decomposition Optimization

This document describes the `DecomposeReduction` pass in the Triton layer. This pass decomposes a large-scale `tt.ReduceOp` into multiple small-scale reduction operations and inserts `ttg.ConvertLayout` layout conversions between them, so as to leverage the warp synchronization reduction, avoid global memory atomic operations, and improve reduction performance.

## Background

In the Triton programming model, `tt.ReduceOp` is used to reduce a tensor along a certain axis (such as sum, max, min, and so on). On GPU-class hardware, reduction operations are typically implemented in several ways:

- **Intra-thread reduction**: A single thread traverses all elements along the reduction axis without inter-thread collaboration.
- **Warp synchronization reduction**: Threads within the same warp collaborate through shuffle instructions to complete the computation along the reduction axis, reducing redundant computation.
- **Global memory atomic reduction**: Different warps/blocks complete the reduction through global memory atomic operations, which incur higher latency.

In the Triton compilation path of the Ascend NPU, if the tensor size along the reduction axis is large (for example, reducing `[1024, 1024]` along axis 1), intra-thread reduction alone entails excessive computation, whereas warp synchronization reduction can distribute the computation across multiple threads within a warp, with each thread handling a portion of the reduction axis and then merging the results through shuffles. The core idea of this pass is to split the reduction axis into multiple sub-axes through Reshape so that the sub-axes can be covered by warp synchronization, thereby converting the operation into an equivalent multi-level reduction sequence. Meanwhile, since intra-thread reduction is more efficient than shuffle operations within a warp, this pass reduces shuffle operations and improves performance.

## Algorithm Principles

### Overall Approach

The DecomposeReduction pass decomposes a large-scale Reduction into the following pattern:

```plaintext
Original:
  %result = tt.reduce(%input) axis = rAxis { combine_op }

After decomposition:
  // Step 1: Split the reduction axis via Reshape (if needed).
  %reshaped = tt.reshape %input
  // Step 2: Sub-level Reduce (optional, 3D→2D or 2D→2D).
  %sub_reduced = tt.reduce(%reshaped) axis = newAxis { combine_op }
  // Step 3: Convert to a warp-synchronous layout.
  %converted = ttg.convert_layout %sub_reduced
  // Step 4: Final Reduce.
  %result = tt.reduce(%converted) axis = rAxis { combine_op }
```

### Layout Calculation

Within the pass, the `calcVectHighDimLayout` method computes a `BlockedLayout` suitable for warp synchronization reduction. Its core strategy is as follows:

1. **4-way vectorization on the reduction axis**: `sizePerThread[rAxis] = min(shape[rAxis], 4)`, so that each thread on the reduction axis processes at most 4 elements.
2. **Warps allocated preferentially to higher dimensions**: `warpsPerCTA` is allocated to the higher dimensions of the non-reduction axes, ensuring `warpsPerCTA[rAxis] = 1` on the reduction axis (that is, warp synchronization).
3. **Threads allocated preferentially to higher dimensions**: `threadsPerWarp` is allocated to the higher dimensions of the non-reduction axes to improve parallelism.
4. **Tail padding**: When the product of the layout is less than the total number of threads or the total number of warps, padding is applied on the last non-reduction dimension.

```cpp
// Pseudocode illustration.
calcVectHighDimLayout(sizePerThread, threadsPerWarp, warpsPerCTA, dim, srcShape):
  // Initialize all axes to 1.
  // Traverse from higher dimensions to lower dimensions:
  //   If i == rAxis: sizePerThread[i] = min(shape[i], 4); shape[i] /= 4
  //   If i != rAxis: warpsPerCTA[i] = min(shape[i], nw); the remaining warps continue to be allocated.
  //   Then from high to low dimensions: threadsPerWarp[i] = min(shape[i], nt); the remaining threads continue to be allocated.
```

### Decomposition Strategies

The pass adopts different decomposition strategies based on the shape of the reduction tensor and the reduction axis:

#### Strategy 1: Rank 2 axis 0 or Rank 3 axis 1 (with layout convert)

When the input satisfies the following conditions, `createDecomposeConvertReduceSequence` is used:

- The reduction axis is 1 (with rank 3) or the reduction axis is 0 (with rank 2).
- The input is skipped directly when it is already in a warp-synchronous layout.
- The corresponding `convert_layout` exists.

**Transformation example (Rank 3 axis 1)**:

Assume the input shape is `[2, 1024, 64]`, and the reduction is performed on axis=1:

```mlir
// Before decomposition:
%input = ttg.convert_layout %src { blocking = #blocked<...> }
%out = tt.reduce %input axis = 1 { combine_op }
```

At this point, if threadsPerWarp[1] is small, the pass attempts the following transformation:

```mlir
// Reshape to 3D: insert a new axis at axis 0, splitting the original shape[1] into shape[1]/4 and 4.
%reshape = tt.reshape %input { shape = [2, 4, 256, 64] }

// 3DTo2D Reduce: reduce on axis=1.
%sub = tt.reduce %reshape axis = 1 { combine_op }  // shape: [2, 256, 64]

// Convert to warp-synchronous layout.
%cvt = ttg.convert_layout %sub { blocking = #blocked<warps=[...]> }

// Final reduction.
%result = tt.reduce %cvt axis = 1 { combine_op }   // shape: [2, 64]
```

#### Strategy 2: Rank 3 Arbitrary Axis + Reshape Prepositioning

When the input of `tt.ReduceOp` comes from `tt.ReshapeOp`, and the rank is 3 with a corresponding `convert_layout`, the pass recombines the Reduce and Reshape:

```mlir
// Before decomposition:
%reshape = tt.reshape %src { shape = [a, b, c] }
%cvt_in = ttg.convert_layout %reshape
%reduce = tt.reduce %cvt_in axis = rAxis { combine_op }
%cvt_out = ttg.convert_layout %reduce

// After decomposition:
%r_reduced = tt.reduce %src axis = rAxis { combine_op }  // Perform the reduction first
%r_reshaped = tt.reshape %r_reduced { shape = [a, b, c'] }  // Then reshape to the target shape
%cvt_out = ttg.convert_layout %r_reshaped  // Reuse the original convert_layout
```

This leverages the principle that reduction can be reordered with Reshape to reduce the amount of intermediate computation.

#### Strategy 3: Rank 2, Axis 1 (Iterative Decomposition)

For a reduction with Rank 2 and axis=1, an iterative decomposition method is used:

```plaintext
while shape[axis] / numThreads > 1:
    // 1) Reshape: Split axis=1 into [shape[1]/numThreads, numThreads].
    outputShape = [shape[0], shape[1]/numThreads, numThreads]

    // 2) 3DTo2D Reduce: Reduce on axis=2 (the newly inserted thread axis).
    %sub = reduce(%reshaped) axis=2

    // 3) ConvertLayout: Switch to the warp-synchronous layout.
    %cvt = convert_layout(%sub)
```

Iterate repeatedly until the reduction axis size divided by numThreads is less than or equal to 1. After the iteration completes, perform the final reduce and write the result back to the output of the original `convert_layout`.

**Transformation example**:

Input shape `[128, 8192]`, reduction on axis=1, numThreads=32:

```mlir
// Original:
%result = tt.reduce %input axis = 1 { combine_op }
// Corresponding convert_layout:
%cvt = ttg.convert_layout %result

// First iteration: shape[1]/32 = 256 > 1.
%r1 = tt.reshape %input { shape = [128, 256, 32] }
%sub1 = tt.reduce %r1 axis = 2 { combine_op }  // shape: [128, 256]
%cvt1 = ttg.convert_layout %sub1

// Second iteration: 256/32 = 8 > 1.
%r2 = tt.reshape %cvt1 { shape = [128, 8, 32] }
%sub2 = tt.reduce %r2 axis = 2 { combine_op }  // shape: [128, 8]
%cvt2 = ttg.convert_layout %sub2

// Third iteration: 8/32 <= 1, terminate the iteration.
// Final reduction:
%final = tt.reduce %cvt2 axis = 1 { combine_op }  // shape: [128]

// Write back the output of the original convert_layout.
%cvt->setOperand(0, %final)
```

#### Strategy 4: Other Cases

For reductions that do not match the configurations above (for example, rank is not 2/3, or axis is not 0/1), the pass directly inserts `convert_layout` and then leaves the original ReduceOp unchanged without performing decomposition.

### Workflow Summary

The overall execution flow of the Pass is as follows:

1. **Initialization**: Read `ttg.num-warps` and `ttg.threads-per-warp` from ModuleOp.
2. **Pre-check**: Only single-result reductions (`getNumResults() <= 1`) and tensors encoded with `BlockedEncodingAttr` are supported.
3. **Walk through all ReduceOps**, and for each op:
   - If the warp-synchronous layout is already satisfied, skip it.
   - Select a decomposition strategy (Strategies 1 to 4) based on rank and axis.
   - After creating the new operation, redirect all users of the original result to the new result.
   - Add the original ReduceOp to the deletion queue.
4. **Post-processing**: Delete the replaced old ops in reverse insertion order (ensuring that uses are cleaned up before defs).

## Compilation Options

| Option | Description |
|------|------|
| `--decompose-reduction` | Enables the DecomposeReduction Pass, which operates on `ModuleOp` |

It can be invoked directly through `bishengir-opt`:

```bash
bishengir-opt --decompose-reduction input.mlir -o output.mlir
```

## Debugging Methods

This pass embeds `LLVM_DEBUG` log output, which can be enabled as follows:

```bash
bishengir-opt --decompose-reduction --debug-only=decompose-reduction input.mlir
```

Key log messages include:

- `sizePerThread / threadsPerWarp / warpsPerCTA`: Prints the computed per-axis parameters of the BlockedLayout.
- `Failed because encoding is not warp sync at axis: N`: The layout does not satisfy the warp-synchronous requirement.
- `Created ReshapeOp` / `Created ReduceOp` / `Created 3DTo2D ReduceOp`: Shows the newly created instructions.
- `Profitable to decompose axis: N`: The cost model determines that decomposition is beneficial.

## Constraints

- **Only single-result reduction is supported**: Multi-result reduction (reducing multiple input tensors simultaneously) is not yet supported and is skipped when encountered.
- **Only BlockedEncodingAttr is supported**: The encoding of the input tensor must be `BlockedEncodingAttr`; other encodings (such as `MmaEncodingAttr`) are skipped.
- **Only the scenario where numThreads == 32 is supported**: Currently the pass performs decomposition only when the number of threads per warp is 32; for other thread counts, it returns directly (to be generalized).
