# Auto-Subtiling

This document describes the AutoBindSubBlock feature in HIVM. This feature optimizes Cube-Vector (CV) kernels by 1:2-tiling the AIV function. Before reading this document, you are advised to read CV Optimization to understand its terms.

AutoBindSubBlock is the informal name of this feature. The compiler Pipeline option is `--enable-auto-bind-sub-block`, which maps to the `enableAutoBindSubBlock` field in code. The pass that tiles and binds sub-blocks is named `hivm-bind-sub-block`, which is implemented as `TileAndBindSubBlock`. Do not treat the feature name or the option name as the pass name.

## Hardware Background

During Ascend chip evolution, AIC and AIV were separated with a 1:2 core ratio.

![image](../../../images/developer_guide/cvarch.png)

In the current ecosystem, neither user-written kernels nor community operators typically implement Ascend Cube–Vector 1:2 sub-block logic. To improve compute efficiency and Ascend affinity, the compiler needs automatic sub-block (subtiling) capability. This feature applies a Cube–Vector 1:2 subtiling strategy and performs the corresponding data splitting.

## Algorithm Principle

The overall approach is:

![image](../../../images/developer_guide/auto_subtiling2.png)

Effects:

![image](../../../images/developer_guide/auto_subtiling3.png)

### Input/output Example

`TileAndBindSubBlock` runs on AIV functions marked `hivm.part_of_mix`.

Original code:

```mlir
%t0 = hivm.hir.vexp ins(%src: tensor<64xf16>)
                     outs(%init: tensor<64xf16>) -> tensor<64xf16>
%t1 = hivm.hir.vabs ins(%t0: tensor<64xf16>)
                     outs(%init: tensor<64xf16>) -> tensor<64xf16>
hivm.hir.store ins(%t1: tensor<64xf16>) outs(%output : memref<64xf16>)
```

On success, the function body is wrapped in `scf.for` (0 to 2, step 1) with `map_for_to_forall` and `mapping = [#hivm.sub_block<x>]`. Stores are split in half with `extract_slice` or `subview` along the chosen axis; after BubbleUp, Vector ops see the half tile. Tiled stores carry `{tiled_op}`:

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
scf.for %i = %c0 to %c2 step %c1 {
  %off = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%i]
  %src_slice = tensor.extract_slice %src[%off] [32] [1]
      : tensor<64xf16> to tensor<32xf16>
  %init_slice = tensor.empty() : tensor<32xf16>
  %t0 = hivm.hir.vexp ins(%src_slice: tensor<32xf16>)
                       outs(%init_slice: tensor<32xf16>) -> tensor<32xf16>
  %t1 = hivm.hir.vabs ins(%t0: tensor<32xf16>)
                       outs(%init_slice: tensor<32xf16>) -> tensor<32xf16>
  %out_slice = memref.subview %output[%off] [32] [1]
      : memref<64xf16> to memref<32xf16, strided<[1], offset: ?>>
  hivm.hir.store ins(%t1: tensor<32xf16>)
                 outs(%out_slice : memref<32xf16, strided<[1], offset: ?>>)
      {tiled_op}
} {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
```

After a successful tile, remaining untiled store or copy-to-L1 or custom ops may still be wrapped by `limitUniqueSubBlockToStore` in `scf.if(get_sub_block_idx == 0)`.

### Implementation Idea

1. Clone the AIV function and wrap its body in `scf.for` (0 to `kSubBlockDim=2`) with sub-block mapping.
2. Split Store data in half via extract-slice or subview.
3. Bubble up the extract-slice using the BubbleUpExtractSlice pattern.
4. On success, replace the original with the clone; on failure, drop the clone and keep the original.

If subtiling fails, the compiler falls back to 1:1 and limits unsplit write-back ops to sub-block 0.

![image](../../../images/developer_guide/auto_subtiling4.png)

Figure: Auto-subtiling 1:2 implementation

### Design

#### Dimension analyzer (axis selection)

The Dimension Analyzer chooses a parallel axis for splitting by analyzing all operators in the target kernel.

#### Why choose a parallel axis

Vector cores do not share a direct data path. To maximize parallelism and correctness, splitting must avoid cross-tile dependencies. Splitting along a parallel axis allows each tile to be computed independently on a vector unit.

#### Tile and slice store (leaf)

Before each StoreOp or leaf node, an ExtractSliceOp for 1:2 splitting is inserted along the axis chosen by the Dimension Analyzer.

#### BubbleUp Extract Slice

A BubbleUp strategy is implemented per op type. Supported op types include:

BroadcastOp, ReduceOp, ExpandOp (specific shapes), CollapseOp (specific shapes)

ElementwiseOp, LoopOp, ExtractSliceOp (specific cases), InsertSliceOp (specific cases)

Additional op types can be supported by adding matchAndRewrite patterns.

## Interface

Behavior is controlled by:

`--enable-auto-bind-sub-block=True` — enable this feature (default). This is `enableAutoBindSubBlock`, which sets `TileAndBindSubBlock` `enable-tile` to true.

`--enable-auto-bind-sub-block=False` — disable tiling. The pass still runs with `enable-tile=false`, then calls `limitUniqueSubBlockToStore` on AIV functions: Vector compute still runs on both sub-cores; only `store`, copy-to-L1, `custom`, `indirect_store`, and `stride_store` are wrapped in `scf.if(get_sub_block_idx == 0)` so both cores do not write the same buffer.

`--skip-hivm-bind-sub-block-pass=True` — omit `hivm-bind-sub-block` entirely (default: `False`). This is `skipHIVMBindSubBlockPass`. Use it only when that pass itself must not run.

## Constraints and fallback

If subtiling or an intermediate transformation fails, the compiler automatically falls back to 1:1 to preserve correctness.

`TileAndBindSubBlock` tiles a clone. On failure, `failAndRevert` drops the clone, marks the module `hivm.tile_and_bind_subblock_reverted`, keeps the original function, and calls `limitUniqueSubBlockToStore`.

Common reasons for falling back to 1:1:

1. Axis selection fails (no valid parallel axis), or no store or copy was tiled.
2. BubbleUpExtractSlice fails, or post-tile verification fails.
3. A dynamic-shape store cannot be split 1:2, tightly-coupled UB was not tiled, or post-tiling cleanup fails.
4. Tiling is skipped and sub-block 0 limiting is used directly when any of the following is true:
   - `hivm.core_ratio` vector count `< 2`
   - the AIV function has a custom op
   - a regbase AIC function is labeled `batch_matmul`
   - the AIV function has implicit transpose
   - (membase) load and store share an address

### Fallback example

Original code

```mlir
%t0 = hivm.hir.vexp ins(%src: tensor<64xf16>)
                     outs(%init: tensor<64xf16>) -> tensor<64xf16>
%t1 = hivm.hir.vabs ins(%t0: tensor<64xf16>)
                     outs(%init: tensor<64xf16>) -> tensor<64xf16>
hivm.hir.store ins(%t1: tensor<64xf16>) outs(%output : memref<64xf16>)
```

When auto 1:2 tiling fails, the data stays in the unsplit 1:1 shape. Vector compute is not wrapped in `scf.if`; only the write-back `store` gets a `get_sub_block_idx == 0` guard (`limit_sub_block_id0`) so both AIV sub-cores do not store the same buffer:

```mlir
%t0 = hivm.hir.vexp ins(%src: tensor<64xf16>)
                     outs(%init: tensor<64xf16>) -> tensor<64xf16>
%t1 = hivm.hir.vabs ins(%t0: tensor<64xf16>)
                     outs(%init: tensor<64xf16>) -> tensor<64xf16>
%idx = hivm.hir.get_sub_block_idx -> i64
%idx_i = arith.index_cast %idx : i64 to index
%c0 = arith.constant 0 : index
%eq0 = arith.cmpi eq, %idx_i, %c0 : index
scf.if %eq0 {
  hivm.hir.store ins(%t1: tensor<64xf16>) outs(%output : memref<64xf16>)
} {limit_sub_block_id0}
```
