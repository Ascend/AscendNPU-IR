# Tile Cube and Vector Loop

This document describes the **TileCubeVectorLoop** pass in HIVM. It optimizes CV-style kernels. Before reading, we recommend [CV Optimization](CVOptimization) for CV terminology.

---

## Overview

![Effect of using Tile Cube and Vector Loop](TileCubeAndVectorLoop.png)

**After CV Pipelining, this pass performs another tiling pass on the Cube and Vector loops in Mix kernels**: it splits one large iteration into multiple smaller iterations. Goals:

1. **Reduce inter-core sync**: Smaller iterations are more likely to fit in local buffers (L0C, UB), reducing sync cost.
2. **Coarser tiles**: Under hardware limits, larger tile sizes can improve memory and compute efficiency:
   - **Cube**: Matmul results go to L0C; if one iteration exceeds L0C capacity, it cannot be placed in L0C in one shot.
   - **Vector**: Too large an iteration can cause UB overflow.

## Interface

### Compile options

| Option | Default | Description |
|--------|--------|-------------|
| `tile-mix-cube-loop` | 1 | Cube loop trip count; 1 means no tiling. |
| `tile-mix-vector-loop` | 1 | Vector loop trip count; 1 means no tiling. |

---

## Algorithm

The pass walks the IR, finds CopyOut ops for Cube and Vector loops, uses them as anchors to split their producers, and fuses the split into the loop.

Before:

```
scf.for {
  hivm.load A
  hivm.load B
  hivm.hir.mmadL1
  hivm.hir.fixpipe
} {cube_loop}
```

After:

```
scf.for {
  for {
    hivm.load slice_A
    hivm.load slice_B
    hivm.hir.mmadL1
    hivm.hir.fixpipe
  } {sub_tile}
} {cube_loop}
```

---

## Constraints

1. Only `scf.for` with **`hivm.loop_core_type`** is processed; allowed values:
   - **`#hivm.tcore_type<CUBE>`**: Cube loop
   - **`#hivm.tcore_type<VECTOR>`**: Vector loop
2. For Vector: if the tiled block size is smaller than UB alignment, no tiling is applied.
3. For Cube: if the block size before tiling is smaller than total L0C size, no tiling is applied. (*Note: L1 size is not yet considered; L1 Memory Overflow may occur in some cases; Cube tiling will be refined with liveness analysis.*)
