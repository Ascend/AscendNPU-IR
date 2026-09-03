# Cube与Vector软件流水优化

本文介绍HIVM中的CV Pipelining Pass。该Pass针对CV类kernel进行优化。在阅读本文之前，建议先阅读[CV Optimization](./cv_optimization.md)，了解CV编译相关术语。

## 硬件背景

昇腾核心中包括Cube核心（负责矩阵乘相关运算）与Vector核心（负责其他向量运算）。这两个核心可以在没有相互依赖的情况下并行异步运行，提高硬件利用率是性能优化中尤其重要的一部分。

对MIX算子中有多个Cube与Vector指令相互依赖的循环的场景进行优化（例如FlashAttention等算子）。通过并行Vector与Cube核心，获得更高的硬件利用率（ILP），达到更高的性能。

该功能使用了Multi-Buffering优化，会导致部分UB空间占用更多，因此需要根据实际场景调整软流水的阶段数来达到最好的性能。

## 流水模式

CV Pipelining当前支持两种主要流水模式：

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| Unroll模式 | 按配置depth对原循环unroll，为各Work Item生成内层循环 | 标准CV流水场景 |
| Preload模式 | 采用Skew方式，通过迭代偏移实现跨迭代重叠执行 | FlashAttention等CV融合场景 |

**Preload模式原理**：将原始循环中的Cube/Vector计算组织为多个Preload stage，使不同迭代的数据搬运和计算交错调度。通过提前执行后续迭代的数据加载，隐藏访存与同步延迟，提高C/V核利用率。各代码段通过`scope`封装并标记`preload_number`，表示相对最终消费阶段提前的物理循环拍数。

## 算法原理

寻找适当的`for`循环，将Cube与Vector指令分开成独立的Work Item，建立每个Work Item之间的数据依赖并将其中需要扩展成`multi-buffer`的`tensor`扩展。

### Unroll模式变换

将原循环`unroll`后，再将每个Work Item放至单独循环中。

变换前：

```mlir
scf.for 0 to N step S {
    %c = Cube() : tensor<16x16xf32>
    %v = Vector(%c) : tensor<16x16xf32>
    %c1 = Cube(%v) : tensor<16x16xf32>
    %v1 = Vector(%c1) : tensor<16x16xf32>
}
```

变换后：

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

### Preload模式变换

采用Skew方式，将Work Item封装为带`preload_number`的`scope`，通过迭代偏移实现跨迭代交错调度。

变换前：

```mlir
scf.for 0 to N step S {
    %c = Cube() : tensor<16x16xf32>
    %v = Vector(%c) : tensor<16x16xf32>
    %c1 = Cube(%v) : tensor<16x16xf32>
    %v1 = Vector(%c1) : tensor<16x16xf32>
}
```

变换后：(以Ascend 950PR/Ascend 950DT为例)

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

## 编译选项

| 选项 | 默认值 | 含义 |
|------|--------|------|
| `set-workspace-multibuffer` | 2 | 软件流水的阶段数，同时也是Multi-Buffering的数量 |
| `--enable-lazy-loading` | false | 开启CV Pipelining中的Lazy Load功能，允许将Load op克隆到多个Work Item中，以减少中间buffer扩展 |
| `--cv-pipeline-mode` | Unroll | CV流水模式：`Off`/`Unroll`/`Skew`。`Skew`对应Preload模式 |
| `--enable-preload` | false | 开启Preload模式，等价于设置`--cv-pipeline-mode=Skew`。|

也可以在算子侧通过`cv_pipeline_lazy_load`编译提示为指定tensor开启Lazy Load功能：

```python
extension.compile_hint(t, "cv_pipeline_lazy_load", True)
```

## 使用约束

1. 支持Pipeline处理的循环中，仅`scf.for`与`scf.if` op可包含region/block，且上述算子的region内部仅允许存在cube或vector指令。
2. 迭代间的数据依赖必须能够拆分到各自独立的Work Item中
    - 无法开启CV-Pipelining场景：若`v0`与`v1`无法被提取至同一Work Item（因为中间有Cube依赖），同时参数`arg0`由`v1`定义却被`v0`使用。
    - CV-Pipelining可正常生效场景：若Cube未使用`v0`，则`v0`可下沉至`v1`所在的Work Item，此时CV-Pipelining可正常生效。

示例代码：

```mlir
scf.for iter_args(%arg0 = %init) {
    %v0 = Vector(%arg0)
    %c = Cube(%v0)
    %v1 = Vector(%c)
    yield %v1
}
```

## 未来演进

当前实现的Preload模式基于固定的stage划分策略，未来将支持更灵活的调度优化：

- **动态调度模式**：通过Cost Model、Profiling或用户提供的Hint，根据Cube/Vector负载比例、阶段时长、通信量和片上内存容量等因素，动态选择最优的流水深度和多Buffer数量
- **自适应策略选择**：针对不同workload特征（计算密集型、访存密集型等），自动在Unroll和Preload模式间选择，实现性能收益最大化
- **跨迭代依赖支持**：扩展对更复杂依赖模式的支持，处理当前约束下无法开启的场景
