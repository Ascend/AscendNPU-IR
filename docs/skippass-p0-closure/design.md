# HIVM 规约化簇冗余 Canonicalizer 移除 — 设计方案

## 背景与目标

HIVM 规约化簇（`canonicalizationHIVMPipeline` / `hivm-canonicalization-pipeline`）
在多个流水线位置被反复调用，每个调用点都会启动多个完整 Canonicalizer 实例，
是编译耗时的主要来源之一。本变更的目标：在**不改变任何可观察输出**的前提下，
移除簇内被证明冗余的 Canonicalizer 实例，直接减少 canonicalizer 调用次数。

## 变更前后的簇结构

```
变更前（10 个 pass）:
  ArithToAffineConversion, CanonicalizeIterArg, Canonicalizer,      <- 移除
  SCFForLoopCanonicalization, CSE, Canonicalizer,                    <- 移除
  HIVMOptSinglePoint, Canonicalizer,
  memref::DeadStoreElimination, Canonicalizer

变更后（8 个 pass）:
  ArithToAffineConversion, CanonicalizeIterArg, Canonicalizer,
  SCFForLoopCanonicalization, CSE,
  HIVMOptSinglePoint,
  memref::DeadStoreElimination, Canonicalizer
```

## 移除依据

1. **CSE 与 HIVMOptSinglePoint 之间的 Canonicalizer**：其全部折叠均可被簇尾
   Canonicalizer 重现（CSE 产生的 redex 在尾部 pass 中同样被折叠）；全量语料
   等价验证输出逐字节一致，而该实例在典型 kernel 上每簇触发 20 次调用。
2. **HIVMOptSinglePoint 与 DeadStoreElimination 之间的 Canonicalizer**：其重写
   仅针对全 1 形状 memref，DSE 匹配不依赖该重写（index-free 匹配）；其产生的
   redex 同样被簇尾 Canonicalizer 支配。

## 同步修改的两个站点

两处定义必须锁步（lockstep）修改，保持语义一致：

- `bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp`
  — RegBase 生产流水线中的 `canonicalizationHIVMPipeline`；
- `bishengir/lib/Dialect/HIVM/Transforms/HIVMCanonicalizationPipeline.cpp`
  — 独立 pass `hivm-canonicalization-pipeline`（`--hivm-canonicalization-pipeline`）。

注意：`HIVMPipelines.cpp` 使用的是另一个 pass（`createExtendedCanonicalizerPass`），
不属于本簇，**未做修改**。

## 验收判据

- **正确性**：414 个可运行编译语料输入（含 RegBase 全流水线 80 个真实入口）
  输出逐字节/逐结构一致；在最新 master（`37c6ebc33`）上冒烟验证同样一致。
- **确定性收益**：canonicalizer 调用数直接测量下降（典型入口 94→54）。
- **回归护栏**：4 个新增 lit 测试文件固定被移除 pass 原本保证的可观察行为
  （见 `code-changes.md`）。

## 已否决的替代方案

曾评估在 `hivm-inline-otf-broadcast` 前用目标化 broadcast 清理替换整个簇，
证伪：最宽泛的超集规则在 24/73 个站点输入上仍产生分歧，且分歧随不动点
迭代结构性放大。因此采用本变更的"逐 pass 移除"路线。

## 非目标（明确不在本变更范围）

- 簇内存活成员（如首尾 Canonicalizer、CSE 等）的合并 — 属后续 P1 工作，
  需要独立设计与验证；
- PassManager 指纹缓存、新框架 — 未引入。
