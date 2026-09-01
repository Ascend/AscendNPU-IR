# 多缓冲

多缓冲（Multi Buffer）是AscendNPU IR中的一项基础性能优化。它把循环体内的某个Buffer扩展为N份物理内存，让相邻迭代的搬运与计算落在不同的内存槽位上，从而把原本串行的多条硬件流水线重叠起来。N等于2时也称Double Buffer（双缓冲、乒乓流水），是最常用的配置。

本文重点说明多缓冲带来性能收益的原因、各内存层次上的收益形态，以及控制该优化的编译选项。变换的实现细节可查阅[-hivm-mark-multi-buffer](../passes/hivm_passes.md#-hivm-mark-multi-buffer)与[-hivm-enable-multi-buffer](../passes/hivm_passes.md#-hivm-enable-multi-buffer)。

## 硬件背景

昇腾AICore内部包含多条相互独立、可并行执行的流水线（PIPE），一条指令只在其归属的流水线上排队执行：

| 流水线 | 承载的典型HIVM操作 | 作用 |
|-----|-----|-----|
| MTE2 | `hivm.hir.load`、`hivm.hir.nd2nz` | GM搬入L1或UB |
| MTE1 | `hivm.hir.mmadL1`的搬入阶段 | L1搬入L0A与L0B |
| M（Cube） | `hivm.hir.mmadL1`的计算阶段 | 矩阵乘计算 |
| V（Vector） | `hivm.hir.vadd`等向量操作 | 向量计算 |
| FIX | `hivm.hir.fixpipe` | L0C结果搬出，含量化、ReLU等后处理 |
| MTE3 | `hivm.hir.store`、`hivm.hir.nz2nd` | UB或L1搬出至GM |

各流水线由硬件异步执行，其先后顺序由编译器插入的`hivm.set_flag`与`hivm.wait_flag`等同步操作建立，详见[自动同步-硬件背景](./auto_sync.md#硬件背景)。片上各级Buffer（UB、L1、L0A、L0B、L0C）的容量有限且需要显式分配地址，详见[内存管理-硬件背景](./plan_memory.md#硬件背景)。

关键点在于：流水线具备并行能力，但数据依赖会把并行度吃掉。

## 性能收益来源

### 消除迭代间的反依赖

循环型算子的典型形态是“搬入→计算→搬出”。若整个循环只使用一份Buffer，迭代之间就会形成反依赖（WAR，Write-After-Read）：第`i+1`次迭代的搬入要向Buffer写入新数据，必须等到第`i`次迭代的计算把这块Buffer读完。于是MTE2、V、MTE3三条本可并行的流水线被迫串行：

```text
单缓冲（1片Buffer）：
iter 0: [MTE2 load][ V compute ][MTE3 store]
iter 1:                                     [MTE2 load][ V compute ][MTE3 store]
iter 2:                                                                          [MTE2 load]...

耗时 ≈ 迭代数 × (T_load + T_compute + T_store)
```

开启多缓冲后，Buffer被扩展为N片，第`i`次迭代使用第`i % N`片。相邻迭代访问的是不同的物理地址，迭代间的WAR依赖随之消失，只保留同一片Buffer内部的真依赖（RAW）：

```text
双缓冲（2片Buffer）：
iter 0: [MTE2 load buf0][ V compute buf0 ][MTE3 store buf0]
iter 1:                 [MTE2 load buf1][ V compute buf1 ][MTE3 store buf1]
iter 2:                                 [MTE2 load buf0][ V compute buf0 ]...

进入稳态后耗时 ≈ 迭代数 × max(T_load, T_compute, T_store)
```

多缓冲把“各阶段耗时之和”变成了“最长阶段的耗时”，本质上是一种用内存换时间的延迟隐藏（Latency Hiding）手段。由此可以推出收益的判断依据：

- 收益上界由流水线的均衡程度决定。搬运与计算耗时相当时，收益最明显；某一条流水线是绝对瓶颈时（例如计算耗时远大于搬运），多缓冲只能隐藏掉非瓶颈部分的耗时，收益有限。
- 收益需要足够的迭代数摊薄。稳态之外的首次搬入与末次搬出无法被隐藏，循环次数很少时收益会被这部分开销抵消。
- 增大N并不等比例增加收益。N从1增到2即可消除相邻迭代的反依赖，继续增大N只在流水阶段数更多（例如CV软件流水）或单次搬运耗时波动较大时才有额外收益，但内存占用是线性增长的。

### 各内存层次上的收益形态

编译器按内存层次识别多缓冲候选，不同层次重叠的流水线不同，收益形态也不同：

| 内存层次 | 触发标记的操作 | 重叠的流水线 | 收益形态 |
|-----|-----|-----|-----|
| GM（Workspace） | 写Workspace的`hivm.hir.store`、`hivm.hir.fixpipe` | Cube核与Vector核跨核并行 | MIX算子中Cube的计算结果经Workspace交给Vector继续处理。多份Workspace让两个核错开一个迭代同时工作，而非互相等待 |
| L1（cbuf） | `hivm.hir.nd2nz` | MTE2与MTE1、M | 矩阵乘左右矩阵的搬入与当前分块的Cube计算重叠 |
| L0C | `hivm.hir.fixpipe` | M与FIX | 上一次矩阵乘结果搬出L0C的同时，下一次矩阵乘可以开始累加 |
| UB | `hivm.hir.load`、`hivm.hir.store` | MTE2、V、MTE3 | 纯Vector算子以及MIX算子的Vector侧，搬入、计算、搬出三段流水并行 |

其中GM Workspace层次的多缓冲是MIX算子（如FlashAttention）实现Cube与Vector核并行的前提，与[Cube与Vector软件流水优化-硬件背景](./cv_pipelining.md#硬件背景)配合使用，软件流水的阶段数即为Workspace的多缓冲数量。

### 与内存复用策略的关系

多缓冲的收益依赖“不同槽位互不干扰”这一前提。当片上内存紧张时，[内存管理-算法原理](./plan_memory.md#算法原理)的Level 1策略会保护这一前提：若同一循环内的单缓冲Buffer复用了多缓冲Buffer的空间，会导致多缓冲实际无法并行、流水被打断，因此该单缓冲会被自动转为多缓冲。这也解释了为什么开启多缓冲后内存占用的增长可能超出直观预期。

## 代价与风险

多缓冲的代价是N倍的内存占用，由此带来两类需要权衡的问题：

- **内存溢出**：UB、L1、L0C容量有限，N倍占用可能导致PlanMemory分配失败并上报overflow。编译器带有溢出回退机制：在Ascend 950PR/Ascend 950DT上，会先只关闭溢出内存空间对应的多缓冲后重试，逐项退让无效时再关闭总开关；在Atlas A3系列产品与Atlas A2系列产品上，则依次关闭`--enable-code-motion`与多缓冲总开关后重试。回退能保证编译成功，但意味着对应层次的收益丢失，此时更合适的做法是减小tiling分块。开启`--enable-tuning-mode`可禁用该重试行为，使溢出直接暴露为编译失败。
- **分块变小反而变慢**：为容纳多份Buffer而缩小tiling分块，会降低单次DMA的搬运效率、增加循环次数与同步开销。因此并非“开得越多越快”，需要结合实测确认。

此外，多缓冲会改变同步结构（核内的flag ID与跨核的event ID都需要随槽位轮转），在CV跨核等复杂场景下对同步求解的要求更高。若开启多缓冲后出现精度异常或卡死，可先用`--enable-auto-multi-buffer=false`定位问题范围，相关排查手段见[调试调测-调试：工具类](../../user_guide/debug_option.md#调试工具类)。

## 算法原理

多缓冲的使能分为四步，分别由不同的Pass承担：

1. **标记**：`-hivm-mark-multi-buffer`按内存层次识别候选Buffer，在其分配点上生成`annotation.mark {hivm.multi_buffer = N}`。本地Buffer的N固定为2，GM Workspace的N由`--set-workspace-multibuffer`指定。
2. **分配地址**：PlanMemory为带标记的Buffer分配N份地址，生成携带多个偏移的`hivm.hir.pointer_cast`。
3. **插入同步**：自动同步按槽位轮转分配flag ID与event ID，保证同一槽位的生产者与消费者成对匹配。
4. **选择槽位**：`-hivm-enable-multi-buffer`引入迭代计数器`hivm.hir.multi_buffer_counter`，对N取模选出当次迭代实际使用的地址。

IR形态示例（以UB上的双缓冲为例）：

```mlir
// 标记后、PlanMemory分配地址后：一个pointer_cast携带2份地址
%p = hivm.hir.pointer_cast(%addr0, %addr1) : memref<1024xf16, #hivm.address_space<ub>>
annotation.mark %p {hivm.multi_buffer = 2 : i32} : memref<1024xf16, #hivm.address_space<ub>>

// -hivm-enable-multi-buffer之后：拆成2个单地址pointer_cast，按迭代计数选择
%p0 = hivm.hir.pointer_cast(%addr0) : memref<1024xf16, #hivm.address_space<ub>>
%p1 = hivm.hir.pointer_cast(%addr1) : memref<1024xf16, #hivm.address_space<ub>>
%counter = hivm.hir.multi_buffer_counter : i64
%slot = arith.remui %counter, %c2_i64 : i64
%is1 = arith.cmpi eq, %slot, %c1_i64 : i64
%active = arith.select %is1, %p1, %p0 : memref<1024xf16, #hivm.address_space<ub>>
```

排查多缓冲是否真正生效时，可在IR中检索`hivm.multi_buffer`标记与上述`arith.select`结构；两者缺失说明标记阶段未命中候选。

## 编译选项

| 选项名 | 描述 | 类型 | 默认值 |
|--------|------|------|--------|
| --enable-auto-multi-buffer | 自动多缓冲总开关。关闭时下述三个选项均不生效 | bool | true |
| --limit-auto-multi-buffer-only-for-local-buffer | 限定多缓冲仅对片上本地Buffer（UB、L1、L0C）生效。置为`true`时跳过GM Workspace的多缓冲标记，即关闭CV跨核流水 | bool | false |
| --limit-auto-multi-buffer-of-local-buffer=\<value> | 限定本地Buffer的多缓冲范围。`no-l0c`表示不对L0C开启多缓冲；`no-limit`表示不做限制 | enum | no-l0c |
| --limit-auto-multi-buffer-buffer=\<value> | 限定MIX算子中多缓冲的作用侧。`only-cube`表示仅Cube侧（L1、L0C）；`only-vector`表示仅Vector侧（UB）；`no-limit`表示不做限制 | enum | 见下方说明 |

`--limit-auto-multi-buffer-buffer`的默认值与目标硬件相关：

- Ascend 950PR/Ascend 950DT：`no-limit`
- Atlas A3训练系列产品/Atlas A3推理系列产品、Atlas A2训练系列产品/Atlas A2推理系列产品：`only-cube`

显式传入该选项时以用户取值为准。

相关的多缓冲数量选项：`--set-workspace-multibuffer`指定GM Workspace的多缓冲数量，在Ascend 950PR/Ascend 950DT上默认为2，在Atlas A3系列产品与Atlas A2系列产品上默认为4。

配置示例：

```bash
# 关闭全部多缓冲，用于性能对比或问题定位
bishengir-compile input.mlir --enable-auto-multi-buffer=false

# 仅保留片上多缓冲，关闭GM Workspace上的CV跨核流水
bishengir-compile input.mlir --limit-auto-multi-buffer-only-for-local-buffer=true

# 片上全开，包含L0C
bishengir-compile input.mlir --limit-auto-multi-buffer-of-local-buffer=no-limit

# MIX算子仅对Vector侧的UB开启多缓冲
bishengir-compile input.mlir --limit-auto-multi-buffer-buffer=only-vector
```

## 后续演进

上述三个`limit`类选项各自约束一部分内存层次，语义相互交叠且都以“限制”方式表达，可读性较差：既无法直接表达“只对L1开启多缓冲”，也无法为不同层次指定不同的缓冲数量。

后续计划废弃`--limit-auto-multi-buffer-only-for-local-buffer`、`--limit-auto-multi-buffer-of-local-buffer`、`--limit-auto-multi-buffer-buffer`三个选项，合并为一个正向表达的`--multibuffer-mode`选项，直接指定GM、L1、L0C、UB四个内存层次各自开启的多缓冲数量：

```bash
--multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]"
```

取值约定：

- 数量大于1：该内存层次开启多缓冲，数量即为物理槽位数。
- 数量等于1：该内存层次关闭多缓冲（只有一份Buffer）。
- 数量等于0：非法配置，编译报错。

例如只对L1开启多缓冲：

```bash
--multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 1), (ub, 1)]"
```

新旧选项的等价关系如下，可用于迁移现有配置。下表以Ascend 950PR/Ascend 950DT为基准，即目标为MIX算子、其余选项保持默认（GM数量取`--set-workspace-multibuffer`的默认值2）：

| 现有配置 | 等价的`--multibuffer-mode` |
|-----|-----|
| 全部默认 | `[(gm, 2), (l1, 2), (l0c, 1), (ub, 2)]` |
| `--enable-auto-multi-buffer=false` | `[(gm, 1), (l1, 1), (l0c, 1), (ub, 1)]` |
| `--limit-auto-multi-buffer-only-for-local-buffer=true` | `[(gm, 1), (l1, 2), (l0c, 1), (ub, 2)]` |
| `--limit-auto-multi-buffer-of-local-buffer=no-limit` | `[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]` |
| `--limit-auto-multi-buffer-buffer=only-cube` | `[(gm, 2), (l1, 2), (l0c, 1), (ub, 1)]` |
| `--limit-auto-multi-buffer-buffer=only-vector` | `[(gm, 2), (l1, 1), (l0c, 1), (ub, 2)]` |

> 说明：`--multibuffer-mode`尚未提供，本节仅用于说明演进方向。选项切换时会保留一个版本的兼容期，届时在版本说明中同步。

## 使用约束

- `--limit-auto-multi-buffer-only-for-local-buffer`与`--limit-auto-multi-buffer-buffer`仅对核类型为MIX的函数生效。纯Cube或纯Vector算子不存在GM Workspace交接，也不存在Cube侧与Vector侧的划分，配置这两个选项不会改变行为。
- 多缓冲候选必须位于循环内，且从Buffer分配点到最外层的所有祖先循环都必须是`scf.for`或`scf.while`。位于`scf.parallel`、`scf.forall`等循环内的Buffer不满足“每次迭代轮转一个槽位”的前提，不会被标记。
- 本地Buffer的多缓冲数量固定为2，不可通过编译选项调整；仅GM Workspace的数量可通过`--set-workspace-multibuffer`配置。
- 用户需保证N倍Buffer占用不超过对应内存空间的容量上限，否则PlanMemory会上报overflow。报错形式与规避方式见[内存管理-使用约束](./plan_memory.md#使用约束)。
