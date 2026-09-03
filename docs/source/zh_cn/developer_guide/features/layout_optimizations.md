# 规约分解优化

本文档介绍 Triton 层 `DecomposeReduction` Pass。该 Pass 将大规模 `tt.ReduceOp` 分解为多个小规模规约操作，并在其间插入 `ttg.ConvertLayout` 布局转换，以利用 warp 同步规约机制，避免全局内存原子操作，提升规约性能。

## 背景

在 Triton 编程模型中，`tt.ReduceOp` 用于对张量的某一轴进行规约（如 sum、max、min 等）。规约操作在 GPU 类硬件上通常有多种实现方式：

- **线程内规约**：单个线程遍历规约轴上的所有元素，无线程间协作。
- **Warp 同步规约**：同一 warp 内的线程通过 shuffle 指令协作完成规约轴上的计算，减少冗余计算量。
- **全局内存原子规约**：不同 warp/block 之间通过全局内存原子操作完成规约，延迟较高。

在昇腾 NPU 的 Triton 编译路径中，若规约轴的张量规模较大（例如 `[1024, 1024]` 在 axis 1 上规约），单靠线程内规约计算量过大，而 warp 同步规约可以将计算任务分摊到 warp 内多个线程上，各线程负责规约轴的一部分，再通过 shuffle 合并结果。该 Pass 的核心思路即是：通过 Reshape 将规约轴拆分为多个子轴，使子轴可以被 warp 同步覆盖，转换为等效的多级规约序列。同时，单线程内的规约比warp内的shuffle操作效率高，此pass减少了shuffle操作，提示性能。

## 算法原理

### 总体思路

DecomposeReduction Pass 将大规模的 Reduction 分解为以下模式：

```plaintext
原始:
  %result = tt.reduce(%input) axis = rAxis { combine_op }

分解后:
  // 步骤1: 通过 Reshape 将规约轴拆分（若需要）
  %reshaped = tt.reshape %input
  // 步骤2: 子级 Reduce（可选，3D→2D 或 2D→2D）
  %sub_reduced = tt.reduce(%reshaped) axis = newAxis { combine_op }
  // 步骤3: 转换为 warp 同步布局
  %converted = ttg.convert_layout %sub_reduced
  // 步骤4: 最终 Reduce
  %result = tt.reduce(%converted) axis = rAxis { combine_op }
```

### 布局计算

Pass 内部通过 `calcVectHighDimLayout` 方法计算适用于 warp 同步规约的 `BlockedLayout`，其核心策略为：

1. **规约轴 4 路向量化**：`sizePerThread[rAxis] = min(shape[rAxis], 4)`，使规约轴的每个线程处理至多 4 个元素。
2. **Warp 优先分配至高维**：将 `warpsPerCTA` 分配给非规约轴的高维方向，确保规约轴的 `warpsPerCTA[rAxis] = 1`（即 warp 同步）。
3. **线程优先分配至高维**：将 `threadsPerWarp` 分配给非规约轴的高维方向，提高并行度。
4. **尾部填充**：当布局的乘积不足线程总数或 warp 总数时，在最后一个非规约维度上补齐。

```cpp
// 伪代码示意
calcVectHighDimLayout(sizePerThread, threadsPerWarp, warpsPerCTA, dim, srcShape):
  // 初始化所有轴为 1
  // 从高维到低维遍历:
  //   若 i == rAxis: sizePerThread[i] = min(shape[i], 4); shape[i] /= 4
  //   若 i != rAxis: warpsPerCTA[i] = min(shape[i], nw); 剩余 warp 继续分配
  //   再从高维到低维: threadsPerWarp[i] = min(shape[i], nt); 剩余线程继续分配
```

### 分解策略

Pass 根据规约张量的形状和规约轴，采用不同的分解策略：

#### 策略一：Rank 2 axis 0 或 Rank 3 axis 1（带 layout convert）

当输入满足以下条件时，使用 `createDecomposeConvertReduceSequence`：

- 规约轴为 1（且 rank 为 3）或规约轴为 0（且 rank 为 2）
- 输入已经是 warp 同步布局时直接跳过
- 对应的 `convert_layout` 存在

**变换示例 (Rank 3 axis 1)**：

假设输入形状为 `[2, 1024, 64]`，在 axis=1 上规约：

```mlir
// 分解前:
%input = ttg.convert_layout %src { blocking = #blocked<...> }
%out = tt.reduce %input axis = 1 { combine_op }
```

此时若 threadsPerWarp[1] 较小，Pass 将尝试以下变换：

```mlir
// reshape 为 3D: 在 axis 0 处插入新轴，将原 shape[1] 拆分为 shape[1]/4 和 4
%reshape = tt.reshape %input { shape = [2, 4, 256, 64] }

// 3DTo2D Reduce: 在 axis=1 上规约
%sub = tt.reduce %reshape axis = 1 { combine_op }  // shape: [2, 256, 64]

// convert 到 warp 同步布局
%cvt = ttg.convert_layout %sub { blocking = #blocked<warps=[...]> }

// 最终规约
%result = tt.reduce %cvt axis = 1 { combine_op }   // shape: [2, 64]
```

#### 策略二：Rank 3 任意轴 + Reshape 前置

当 `tt.ReduceOp` 的输入来自 `tt.ReshapeOp`，且 rank 为 3、有对应的 `convert_layout` 时，Pass 会将 Reduce 与 Reshape 重组：

```mlir
// 分解前:
%reshape = tt.reshape %src { shape = [a, b, c] }
%cvt_in = ttg.convert_layout %reshape
%reduce = tt.reduce %cvt_in axis = rAxis { combine_op }
%cvt_out = ttg.convert_layout %reduce

// 分解后:
%r_reduced = tt.reduce %src axis = rAxis { combine_op }  // 先做规约
%r_reshaped = tt.reshape %r_reduced { shape = [a, b, c'] }  // 再 reshape 到目标形状
%cvt_out = ttg.convert_layout %r_reshaped  // 复用原有 convert_layout
```

这利用了规约可以与 Reshape 交换顺序、减少中间计算量的原理。

#### 策略三：Rank 2 axis 1（迭代分解）

对于 Rank 2、axis=1 的规约，采用迭代分解方法：

```plaintext
while shape[axis] / numThreads > 1:
    // 1) Reshape: 将 axis=1 拆为 [shape[1]/numThreads, numThreads]
    outputShape = [shape[0], shape[1]/numThreads, numThreads]

    // 2) 3DTo2D Reduce: 在 axis=2 上规约（新插入的线程轴）
    %sub = reduce(%reshaped) axis=2

    // 3) ConvertLayout: 切换到 warp 同步布局
    %cvt = convert_layout(%sub)
```

反复迭代，直到规约轴大小 ÷ numThreads ≤ 1。迭代完成后执行最终的 reduce，并将结果写回原有 `convert_layout` 的输出端。

**变换示例**：

输入形状 `[128, 8192]`，axis=1 规约，numThreads=32：

```mlir
// 原始:
%result = tt.reduce %input axis = 1 { combine_op }
// 对应 convert_layout:
%cvt = ttg.convert_layout %result

// 第一次迭代: shape[1]/32 = 256 > 1
%r1 = tt.reshape %input { shape = [128, 256, 32] }
%sub1 = tt.reduce %r1 axis = 2 { combine_op }  // shape: [128, 256]
%cvt1 = ttg.convert_layout %sub1

// 第二次迭代: 256/32 = 8 > 1
%r2 = tt.reshape %cvt1 { shape = [128, 8, 32] }
%sub2 = tt.reduce %r2 axis = 2 { combine_op }  // shape: [128, 8]
%cvt2 = ttg.convert_layout %sub2

// 第三次迭代: 8/32 <= 1，终止迭代
// 最终规约:
%final = tt.reduce %cvt2 axis = 1 { combine_op }  // shape: [128]

// 写回原有 convert_layout 的输出
%cvt->setOperand(0, %final)
```

#### 策略四：其他情况

对于不符合上述配置的规约（如 rank 非 2/3、axis 非 0/1 等），Pass 直接插入 `convert_layout` 后将原 ReduceOp 保持不变，不执行分解。

### 工作流总结

整个 Pass 的执行流程如下：

1. **初始化**：从 ModuleOp 读取 `ttg.num-warps` 和 `ttg.threads-per-warp`。
2. **前置检查**：仅支持单结果规约（`getNumResults() <= 1`）和 `BlockedEncodingAttr` 编码的张量。
3. **Walk 遍历所有 ReduceOp**，对每个 op：
   - 若 warp 同步布局已满足，跳过。
   - 根据 rank 与 axis 选择分解策略（策略一~四）。
   - 创建新操作后，重定向所有使用原始结果的用户到新结果。
   - 将原始 ReduceOp 加入删除队列。
4. **后处理**：按插入顺序逆序删除被替换的旧 op（保证 use 在 def 之前被清理）。

## 编译选项

| 选项 | 含义 |
|------|------|
| `--decompose-reduction` | 启用 DecomposeReduction Pass，作用于 `ModuleOp` |

可直接通过 `bishengir-opt` 调用：

```bash
bishengir-opt --decompose-reduction input.mlir -o output.mlir
```

## 调试方法

该 Pass 内嵌了 `LLVM_DEBUG` 日志输出，可通过以下方式开启：

```bash
bishengir-opt --decompose-reduction --debug-only=decompose-reduction input.mlir
```

关键日志信息包括：

- `sizePerThread / threadsPerWarp / warpsPerCTA`：打印计算出的 BlockedLayout 各轴参数
- `Failed because encoding is not warp sync at axis: N`：布局不满足 warp 同步要求
- `Created ReshapeOp` / `Created ReduceOp` / `Created 3DTo2D ReduceOp`：显示新创建的指令
- `Profitable to decompose axis: N`：cost model 判定分解有利

## 约束

- **仅支持单结果规约**：多结果（多个输入张量同时规约）暂不支持，遇到时直接跳过。
- **仅支持 BlockedEncodingAttr**：输入张量的编码必须是 `BlockedEncodingAttr`，其他编码（如 `MmaEncodingAttr`）被跳过。
- **仅支持 numThreads == 32 的场景**：当前 Pass 仅在每 warp 线程数为 32 时执行分解；其他线程数时直接返回（待泛化）。
