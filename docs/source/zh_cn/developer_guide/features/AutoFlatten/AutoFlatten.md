# 自动展平（Auto Flatten）

## 概述

Auto Flatten Pass（`HIVMFlattenOps`）自动将多维张量操作折叠为低维等价形式，在保持语义正确性的同时降低秩（rank）。该优化简化了内存访问模式，提升了目标加速器上的硬件利用率。

---

## 硬件背景

现代硬件加速器通常存在有利于低秩张量操作的约束和性能特性：

| 方面 | 高秩的影响 | 展平的收益 |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| **地址计算** | 多维索引需要多次乘加运算 | 简化的线性寻址降低了开销 |
| **内存合并** | 复杂的步长模式可能阻碍高效内存访问 | 连续的展平维度实现更好的合并 |
| **硬件循环** | 硬件循环计数器数量有限 | 维度更少 = 所需循环嵌套更少 |
| **DMA 效率** | 多步长传输可能需要多个 DMA 描述符 | 折叠维度实现批量传输 |
| **寄存器压力** | 更多的索引变量占用寄存器 | 减少簿记开销 |

### 示例场景

考虑一个形状为 `[1, 64, 1, 128, 256]` 的 5D 逐元素操作：
- **展平前**：5 层嵌套循环，复杂的步长计算
- **展平后**：形状变为 `[64, 128, 256]` 甚至 `[64, 32768]`，实现更高效的硬件利用率

---

## 算法原理

展平算法作为**多阶段流水线**运行，在遵守操作特定约束的同时逐步折叠维度。

### 核心概念

#### 重关联映射（Reassociation Maps）

重关联映射定义了原始维度如何映射到折叠后的维度：

```
Original shape: [A, B, C, D, E] (rank 5)
Reassociation:  [[0, 1], [2], [3, 4]]
Result shape:   [A*B, C, D*E] (rank 3)
```

#### 维度分类（三值掩码）

每个维度被分类为以下三种类型之一：

| 类别 | 符号 | 描述 | 折叠行为 |
| ------------------ | ------ | ---------------------------- | ------------------------------------ |
| **单元维度** | `U` | 大小为 1 的维度 | 被吸收到相邻组中 |
| **可折叠维度** | `C` | 可与邻居合并 | 形成组，吸收相邻单元维度 |
| **不可折叠维度** | `N` | 屏障维度 | 独立存在，阻止单元维度吸收 |

#### 屏障维度（Barrier Dimensions）

某些维度由于语义要求不能一起折叠：
- **归约维度**：必须保持独立以保留归约语义
- **广播维度**：形状不匹配阻止折叠
- **转置维度**：置换要求约束分组

### 流水线阶段

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Operation                              │
│            Shape: [1, 64, 1, 128, 1, 256]                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Unit Dimension Collapse                               │
│  ─────────────────────────────────────────────                  │
│  • Identify unit (size-1) dimensions                            │
│  • Build ternary mask considering barriers                      │
│  • Collapse units into adjacent non-barrier groups              │
│                                                                 │
│  Mask:    [U,  C, U,   C, U,   C]                               │
│  Result:  [[0, 1, 2], [3, 4], [5]]  →  Shape: [64, 128, 256]    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Uniform Reassociation Collapse                        │
│  ─────────────────────────────────────────────                  │
│  • Check memory contiguity (stride patterns)                    │
│  • Respect target dimension boundaries                          │
│  • Apply input consistency checks (for broadcast)               │
│                                                                 │
│  Contiguous dims can be further collapsed                       │
│  Result:  [[0], [1, 2]]  →  Shape: [64, 32768]                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Compose Results                                       │
│  ─────────────────────────────────────────────                  │
│  • Combine reassociation maps from all stages                   │
│  • Adjust target dimension indices                              │
│  • Update barrier dimension tracking                            │
│                                                                 │
│  Final: [[0, 1, 2], [3, 4, 5]]  →  Shape: [64, 32768]           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Operation                             │
│  • Insert memref.collapse_shape for each operand                │
│  • Clone operation with collapsed operands                      │
│  • Adjust operation attributes (reduce_dims, broadcast_dims)    │
└─────────────────────────────────────────────────────────────────┘
```

### 特殊情况：可转置的 OTF（On-The-Fly）

对于具有内联转置语义的操作，算法**分别**处理输入和输出重关联映射：

```
Input shape:   [A, B, C, D, E, F]
Permutation:   [2, 3, 0, 4, 1, 5]
Output shape:  [C, D, A, E, B, F]

Step 1: Unit collapse on input (if B, D are unit)
Step 2: Derive permutation blocks from inverse permutation
Step 3: Generate separate input/init reassociation maps
Step 4: Compose results maintaining permutation semantics
```

### 掩码构建逻辑

```cpp
for each dimension i:
    if (strictBarrierWithUnit && isBarrier[i]):
        mask[i] = NonCollapsible    // 严格模式：屏障维度独立
    else if (isUnit[i] && !isBarrier[i]):
        mask[i] = Unit              // 单元维度被吸收
    else:
        mask[i] = Collapsible       // 可形成组
```

### 从掩码生成重关联

```
Input Mask: [U, C, U, N, U, C, U]

Processing:
  Segment 1: [U, C, U] → Group units with collapsible → [[0, 1, 2]]
  Segment 2: [N]       → Isolated non-collapsible    → [[3]]
  Segment 3: [U, C, U] → Group units with collapsible → [[4, 5, 6]]

Result: [[0, 1, 2], [3], [4, 5, 6]]
```

---

## API

### Pass 注册

```cpp
// 创建展平 Pass
std::unique_ptr<Pass> mlir::hivm::createFlattenOpsPass();

// 在 Pass 流水线中使用
pm.addPass(mlir::hivm::createFlattenOpsPass());
```

### FlattenInterface

实现自动展平的操作必须实现 `FlattenInterface`：

```cpp
class FlattenInterface {
public:
  /// 计算该操作的展平结果
  virtual FailureOr<FlattenResult> getFlattened(FlattenOptions options) = 0;

  /// 展平后调整操作属性
  virtual void adjustTargetDimensions(OpBuilder &builder,
                                       const FlattenResult &result) = 0;
};
```

### FlattenOptions

```cpp
struct FlattenOptions {
  /// 为 true 时，即使是单元维度的屏障也变为 NonCollapsible
  bool strictBarrierWithUnit = false;

  /// 检查步长注释的对齐要求
  bool checkMarkStride = false;

  /// 折叠前验证输入形状一致性（用于广播）
  bool checkInputConsistency = false;
};
```

### FlattenResult

```cpp
struct FlattenResult {
  // 核心数据
  Operation *op;                               // 源操作
  SmallVector<ReassociationMap> reassociation; // 折叠映射
  SmallVector<KindTypePair> operandTypes;      // 折叠后的类型
  SmallVector<Value> operandOriginalVal;       // 原始操作数值

  // 维度追踪
  SmallVector<int64_t> originalTargetDims;     // 原始目标维度索引
  SmallVector<int64_t> adjustedTargetDims;     // 折叠后调整的索引
  SmallVector<int64_t> barrierDims;            // 不可折叠的边界维度

  // 查询方法
  bool isIdentityCollapse() const;
  int getRankAfterFlatten() const;
  SmallVector<Type> getOperandTypes(DpsKind kind) const;
  ReassociationMap getInputReassociation() const;
  ReassociationMap getInitReassociation() const;
  bool uniformReassociation() const;  // 所有输入和 init 的重关联是否相同
};
```

### 操作特性（Operation Traits）

```cpp
// 表示操作对所有操作数使用相同的重关联
OpTrait::UniformReassociationFlattenTrait

// 表示连续的目标维度可以被折叠
OpTrait::CollapsibleConsecutiveTargetDimsTrait
```

### 支持的操作调整

每种操作类型均实现了 `adjustTargetDimensions`：

| 操作 | 调整的属性 |
| -------------------------- | --------------------------------------------- |
| `VBrcOp` | `broadcast_dims` |
| `VReduceOp` | `reduce_dims` |
| `VTransposeOp` | `permutation` |
| `VCumsumOp` / `VCumprodOp` | `cum_dims` |
| `VPadOp` | `static_low`、`static_high` |
| `VConcatOp` | `dim` |
| `VFlipOp` | `flip_axis` |
| 逐元素操作 | `iterator_types`（broadcast/transpose 数组） |

---

## 能力与限制

### ✅ 能力

| 特性 | 描述 |
| ------------------------------- | ------------------------------------------------------------ |
| **单元维度折叠** | 自动移除大小为 1 的维度 |
| **连续性感知** | 遵守内存布局；非连续维度保持独立 |
| **操作特定处理** | 针对归约、广播、转置、填充等的自定义逻辑 |
| **流水线组合** | 多个折叠阶段可正确组合 |
| **均匀重关联** | 所有操作数折叠方式相同时的高效处理 |
| **非均匀重关联** | 支持不同的输入/init 重关联（转置 OTF） |
| **屏障保护** | 语义关键维度保持独立 |
| **跳过 Host 函数** | 自动跳过 host 侧函数 |

### ⚠️ 限制

| 限制 | 描述 | 规避方案 |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **仅支持 MemRef 类型** | 只折叠 `MemRefType` 操作数 | 张量必须先进行缓冲化（bufferize） |
| **需要静态形状** | 动态维度可能无法正确折叠 | 优先运行符号方言或形状推断 Pass |
| **严格屏障模式** | `VFlipOp` 需要 `strictBarrierWithUnit=true` | 自动处理 |
| **转置后向维度** | 某些操作的最后一个维度不能进行 OTF 转置 | 算法保留最后一个维度不折叠 |
| **非 HIVMStructuredOp** | 未实现接口的操作返回恒等映射 | 实现 `FlattenInterface` |

### 边界情况

```cpp
// 恒等折叠（无变化）—— Pass 报告匹配失败
if (res->isIdentityCollapse())
  return rewriter.notifyMatchFailure(op, "Identity reassociation");

// 操作无法处理
if (failed(res))
  return rewriter.notifyMatchFailure(op, "Operation cannot be handled");
```

### 调试

使用 `LDBG` 宏启用调试日志，可追踪：
- 每个阶段的重关联映射
- 掩码分类
- 调整后的目标维度
- 组合结果

---

## 变换示例

### 变换前

```mlir
%0 = hivm.vbrc %input broadcast_dims = [3]
     : memref<1x64x1x128x256xf32> -> memref<1x64x16x128x256xf32>
```

### 经过 Flatten Pass 后

```mlir
// 折叠输入：[[0, 1, 2], [3, 4]] → 秩 2
%collapsed_input = memref.collapse_shape %input [[0, 1, 2], [3, 4]]
     : memref<1x64x1x128x256xf32> into memref<64x1x32768xf32>

// 广播后调整维度：[1, 3] → [0]（重映射后）
%0 = hivm.vbrc %collapsed_input broadcast_dims = [1]
     : memref<64x1x32768xf32> -> memref<64x16x32768xf32>

// 注：输出展开由单独的 Pass 处理
```

## 补充示例：带步长的广播操作

### 关于带步长的 MemRef 类型

类型为 `memref<N₀×N₁×…×Nₙ×f32, strides={S[0], S[1], …, S[n]}, offset=O>` 的 memref 将坐标 $[i_0, i_1, \dots, i_n]$ 映射到线性内存地址：

$$\text{address} = \sum_{k=0}^{n} i_k \cdot S[k] \;+\; O$$

例如，`memref<5x6xf32>` 具有步长为 `[6, 1]`、偏移为 `0` 的默认（恒等）布局。访问元素 `[2, 4]` 得到：

$$\text{address} = 2 \times 6 + 4 \times 1 + 0 = 16$$

当未指定显式布局时，MLIR 使用**行主序**。步长从最内维到最外维计算：

$S[n] = 1$

$S[i] = S[i+1] \times N_{i+1}$

这意味着最后一个维度上的元素在内存中相邻，下一个外层维度的每一"行"紧跟在前一行之后——没有间隙。

当一个维度大小为 1 时，其索引始终为 0。该维度的步长对地址的贡献为 $0 \times S[k] = 0$，使得步长值**无关紧要**。这就是为什么展平 Pass 可以自由地将单元维度吸收到相邻组中，而无需考虑其步长值。

在不失一般性且假设使用行主序的情况下，相邻的两个维度 $d_i$ 和 $d_{i+1}$ **连续**当且仅当：

$$S[i] = S[i{+}1] \times N_{i+1}$$

这意味着遍历维度 $i{+}1$ 的所有元素后，维度 $i$ 增加 1 时恰好落在下一个元素上——没有间隙，没有重叠。基础情况是最外层维度（轴 0）按惯例始终被认为是连续的。

**只有连续的相邻维度才能被折叠。** 折叠非连续维度会改变实际访问的内存位置。

以下场景演示了展平 Pass 如何与**带步长的内存布局**交互，这是处理非连续内存视图时的常见情况。这些示例使用 `hivm.hir.vbrc`——一种将 memref 填充为标量值的标量广播操作。

---

### 场景 1：非连续步长阻止所有折叠

**函数：** `@strided_brc`

```mlir
// memref<16x16xf32, strided<[16, 2]>>
//   dim 0: size=16, stride=16
//   dim 1: size=16, stride=2   ← 非连续（连续要求 stride=1）
```

**分析：**

无法合并维度 0 和 1：为满足连续性，维度 1 的步长必须等于 1（元素步长）。这里步长为 2，表示"每隔一个元素"的非连续访问模式。将维度 $[0, 1]$ 折叠为单个维度会产生平坦索引 $i \cdot 16 + j$，但实际内存访问模式为 $i \cdot 16 + j \cdot 2$。两者并不等价——展平会悄悄改变实际访问的内存位置。

**输出（未变化）：**

```mlir
func.func @strided_brc(%arg0: f32, %arg1: memref<16x16xf32, strided<[16, 2]>>) {
  hivm.hir.vbrc ins(%arg0 : f32) outs(%arg1 : memref<16x16xf32, strided<[16, 2]>>)
  return
}
```

---

### 场景 2：部分连续步长允许部分折叠

**函数：** `@strided_brc_collapse_continuous`

```mlir
// memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>
//   dim 0: size=8,  stride=?   ← 动态，无法验证与 dim 1 的连续性
//   dim 1: size=?,  stride=?   ← 动态，无法验证与 dim 2 的连续性
//   dim 2: size=4,  stride=2   ← stride = dim3.size(2) × dim3.stride(1) = 2 ✓
//   dim 3: size=2,  stride=1   ← 最内层，连续
```

**相邻维度对的连续性检验：**

$$\text{contiguous}(d_i, d_{i+1}) \iff \text{stride}(d_i) = \text{size}(d_{i+1}) \times \text{stride}(d_{i+1})$$

| 维度对 | 计算 | 是否连续 |
| -------- | -------------------- | ------------------- |
| 维度 0–1 | $? = ? \times ?$ | ❌ 未知（动态） |
| 维度 1–2 | $? = 4 \times 2 = 8$ | ❌ 未知（动态） |
| 维度 2–3 | $2 = 2 \times 1 = 2$ | ✅ 是 |

**输出（维度 2 和 3 已折叠）：**

```mlir
func.func @strided_brc_collapse_continuous(
    %arg0: f32, %arg1: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>) {
  %collapse_shape = memref.collapse_shape %arg1 [[0], [1], [2, 3]]
      : memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>
        into memref<8x?x8xf32, strided<[?, ?, 1]>>
  hivm.hir.vbrc ins(%arg0 : f32)
      outs(%collapse_shape : memref<8x?x8xf32, strided<[?, ?, 1]>>)
  return
}
```

折叠后的结果：
- 维度 2 和 3 合并：大小 $4 \times 2 = 8$，步长 $= 1$（连续）
- 秩从 4 降为 3

---

### 场景 3：动态内层维度阻止连续性验证

**函数：** `@scalar_brc_cannot_collapse_continuous`

```mlir
// memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>
//   dim 0: size=8,  stride=?
//   dim 1: size=?,  stride=?
//   dim 2: size=4,  stride=2
//   dim 3: size=?,  stride=1   ← 动态大小！
```

**维度 2–3 的连续性检验：**

编译器**无法静态证明** $2 = ?$。若维度 3 的运行时大小为 2，则它们连续；若大小为 3，则不连续。Pass 保守地拒绝折叠。

| 维度对 | 计算 | 是否连续 |
| -------- | -------------------- | ------------------------------------- |
| 维度 0–1 | $? = ? \times ?$ | ❌ 未知 |
| 维度 1–2 | $? = 4 \times 2 = 8$ | ❌ 未知 |
| 维度 2–3 | $2 = ? \times 1 = ?$ | ❌ **未知**（维度 3 大小为动态） |

**输出（未变化）：**

```mlir
func.func @scalar_brc_cannot_collapse_continuous(
    %arg0: f32, %arg1: memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>) {
  hivm.hir.vbrc ins(%arg0 : f32)
      outs(%arg1 : memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>)
  return
}
```
