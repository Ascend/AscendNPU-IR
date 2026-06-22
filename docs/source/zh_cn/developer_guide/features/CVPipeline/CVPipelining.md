# Cube-Vector 软件流水优化

本文介绍 HIVM 中的 **CV Pipelining** Pass。该Pass针对CV类kernel进行优化。在阅读本文之前，建议先阅读 [CV Optimization](../CV/CVOptimization.md)，了解CV编译相关术语。

## 硬件背景

昇腾核心中包括Cube核心（负责矩阵乘相关运算）与Vector核心（负责其他向量运算）。这两个核心可以在没有相互依赖的情况下并行异步运行，提高硬件利用率是性能优化中尤其重要的一部分。

对MIX算子中有多个Cube与Vector指令相互依赖的循环的场景进行优化（例如FlashAttention等算子）。 通过并行Vector与Cube核心，获得更高的硬件利用率（ILP），达到更高的性能。

该功能使用了Multi-Buffering优化，会导致部分UB空间占用更多，因此需要根据实际场景调整软流水的阶段数来达到最好的性能。

## 算法原理

寻找适当的for 循环，将Cube与Vector指令分开成独立的Work Item, 建立每个Work Item之间的数据依赖并将其中需要扩展成multi-buffer的tensor扩展，将原循环unroll后，再将每个Work Item放至单独循环中。

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

## 接口说明

### 编译选项

| 选项 | 默认值 | 含义 |
|------|--------|------|
| `set-workspace-multibuffer` | 2 | 软件流水的阶段数，同时也是Multi-Buffering的数量|

## 约束能力

1. Pipeline的循环只有scf.for与scf.if op拥有region/block, 并且其region内只能有cube或vector指令。
2. 迭代间的数据依赖必须可以被分离至独自的Work Item
    - 以下情况无法开启cv-pipelining：`v0` 与 `v1`无法被提取至同一Work Item（因为中间有Cube依赖），但是`arg0`的定义在`v1`，却被`v0`用到。该情况CV-Pipelining不会开启
    - 若`Cube`没有用到`v0`, 那么`v0`会下沉至`v1`同一个Work Item, CV-Pipelining会生效

```mlir
scf.for iter_args(%arg0 = %init) {
    %v0 = Vector(%arg0)
    %c = Cube(%v0)
    %v1 = Vector(%c)
    yield %v1
}
```
