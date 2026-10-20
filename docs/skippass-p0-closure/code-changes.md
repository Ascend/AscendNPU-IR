# 代码修改说明

## 修改文件总览

| 文件 | 类型 | 说明 |
|---|---|---|
| `bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp` | 生产 | 移除 2 个冗余 `createCanonicalizerPass` 调用 + 简短理由注释 |
| `bishengir/lib/Dialect/HIVM/Transforms/HIVMCanonicalizationPipeline.cpp` | 生产 | 同上（与上一文件锁步） |
| `bishengir/test/Dialect/HIVM/regbase/canonicalization-cse-cleanup.mlir` | 测试 | 新增：CSE→SinglePoint 区域行为回归 |
| `bishengir/test/Dialect/HIVM/regbase/single-point-dse-cleanup.mlir` | 测试 | 新增：SinglePoint/DSE 区域行为回归 |
| `bishengir/test/Dialect/HIVM/regbase/broadcast-cleanup-pre-inline-otf.mlir` | 测试 | 新增：InlineOTF 前 broadcast 清理行为回归 |
| `bishengir/test/Dialect/HIVM/regbase/canonicalization-pipeline-pre-inline-otf.mlir` | 测试 | 新增：簇→InlineOTF 契约回归 |
| `docs/skippass-p0-closure/` | 文档 | 本交接文档 + `corpus-manifest.csv`（557 条编译语料清单） |

## 生产代码改动细节

两个文件的改动完全对称：在 pass 序列中删除两行 `createCanonicalizerPass`，
并各保留两处 2–3 行注释说明删除理由（冗余性由簇尾 Canonicalizer 支配）。
除注释外无任何代码、选项、pass 名称变化；对外接口、命令行参数均不变。

```
- pm.addPass(createCanonicalizerPass(canonicalizerOptions));   // CSE 之后
  pm.nest<func::FuncOp>().addPass(createHIVMOptSinglePointPass());
- pm.addPass(createCanonicalizerPass(canonicalizerOptions));   // SinglePoint 之后
  pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
```

## 新增测试说明

4 个测试文件均只断言**可观察结果**（哪些 op 存活/消失、最终内存访问形态），
不断言 pass 列表本身，因此对簇的后续重构（如合并存活成员）保持稳健：

- `canonicalization-cse-cleanup.mlir` — CSE 合并相同 `tensor.empty` 后产生的
  自拷贝必须被折叠并转发操作数；非冗余 copy 必须保留（阴性对照）；rank-reducing
  `collapse_shape` 链必须合并；1x1 `hivm.hir.vadd` 仍被 SinglePoint 标量化。
- `single-point-dse-cleanup.mlir` — SinglePoint 标量化 + load/store 转发、
  死存储删除等可观察行为；头部注释给出变更后的 8 成员簇结构。
- `broadcast-cleanup-pre-inline-otf.mlir` — 1:1 冗余 `vbrc` 必须消失、真实
  扩维 `vbrc` 必须保留、内联进 `vmul` 后 `broadcast` 维度正确、动态形状
  单位维保持。
- `canonicalization-pipeline-pre-inline-otf.mlir` — 簇输出作为
  `hivm-inline-otf-broadcast` 前置条件的契约（alloc/load/store/地址空间与
  broadcast 决策不变）。

每个文件的第 2 条 RUN 线以 `-hacc-append-device-spec=target=Ascend950PR_957b`
（A5 RegBase）重复同样检查；该 target 是生产流水线实际执行本簇的唯一路径。

## 构建与使用影响

- 无 pass 改名、无选项变化；下游对 `--hivm-canonicalization-pipeline` 的使用
  方式完全不变，仅每次调用少跑 2 个 Canonicalizer 实例。
- 格式化：仅对触碰的 2 个 C++ 文件按仓库根 `.clang-format` 校验，未重排
  无关代码。
