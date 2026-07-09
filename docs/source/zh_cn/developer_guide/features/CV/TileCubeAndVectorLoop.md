# Cube与Vector循环切块（Tile Cube and Vector Loop）

本文介绍HIVM中的TileCubeVectorLoop Pass。该Pass针对CV类`kernel`进行优化。在阅读本文之前，建议先阅读[CV Optimization](./CVOptimization.md)，了解CV编译相关术语。

## 硬件背景

当代昇腾AI加速芯片采取AIC、AIV的分离模式，AIC与AIV核的数据交互需要经过Global Memory。当AIC与AIV核存在数据依赖时，需要核间同步指令保证数据的正确性。然而，频繁的核间同步会导致性能下降。因此，为了提升算子的运行性能，需要尽可能地降低AIC与AIV核的同步频率。

## 功能介绍

![Effect of using Tile Cube and Vector Loop](../../../../images/developer_guide/TileCubeAndVectorLoop.png)

对`MIX`算子中已经完成软件流水（CV Pipelining）的`Cube`循环和`Vector`循环，再做一次`Tiling`切分，把原来一次迭代完成的一整块计算，拆成多次迭代、每次处理更小的一块。这样做的目的主要有两点：

1. 减少核间同步：每次迭代处理的数据更小，更可能被限制在本地buffer（`L0C`、`UB`等）内，从而减少跨核同步的开销。
2. 增大切分粒度：在满足硬件约束的前提下，有机会使用更大的tile size，有利于访存与计算效率：
   - `Cube`侧：矩阵乘法的结果存储在`L0C` Buffer，若单次迭代的数据总大小超过`L0C`容量，就无法一次性放进`L0C`。
   - `Vector`侧：单次迭代若过大，可能会导致`UB`（Unified Buffer）缓冲溢出。

## 接口说明

### 编译选项

| 选项 | 默认值 | 含义 |
|------|--------|------|
| `tile-mix-cube-loop` | 1 | Cube循环目标trip count；为1时不tiling。 |
| `tile-mix-vector-loop` | 1 | Vector循环目标trip count；为1时不tiling。 |

## 算法原理

通过遍历`IR`，寻找`Cube`和`Vector`循环对应的`CopyOut`操作进行切分，并以其为锚点，将数据的producer以此进行切分，并融合至循环内。

Before:

```mlir
scf.for {
  hivm.load A
  hivm.load B
  hivm.hir.mmadL1
  hivm.hir.fixpipe
} {cube_loop}

```

After:

```mlir
scf.for {
  for {
    hivm.load slice_A
    hivm.load slice_B
    hivm.hir.mmadL1
    hivm.hir.fixpipe
  } {sub_tile}
} {cube_loop}

```

## 约束能力

1. 只处理携带`hivm.loop_core_type`属性的`scf.for`，且该属性的值为：
    - `#hivm.tcore_type<CUBE>`：Cube循环
    - `#hivm.tcore_type<VECTOR>`：Vector循环

2. 对于`Vector`计算，假设通过`Tiling`后的切块大小小于`UB`对齐大小，则不做`Tiling`。
3. 对于`Cube`计算，假设`Tiling`前的切块大小小于`L0C`总大小，则不做`Tiling` （注：当前尚未考虑`L1`空间大小约束，部分场景下可能会出现`L1` Memory Overflow的报错；后续会结合生命周期分析决定`Cube`侧的`Tiling`）。
