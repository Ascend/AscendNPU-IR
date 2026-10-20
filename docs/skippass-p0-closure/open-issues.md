# 遗留问题与交接说明

## 1. 基线 rebase（建议合入前完成）

本变更基于 master `9601a8ce1` 开发；当前 `origin/master` 已前进至
`37c6ebc33`（约领先 5.6 天）。开发阶段已在该 master 上做过 cherry-pick
验证（当时以 4 个提交形式，零冲突），squash 为单提交后预计同样可干净
应用，但**合入前请 rebase 到最新 master 并重跑**：

```bash
git rebase origin/master
ninja -C build bishengir-opt bishengir-compile
cd build && ./bin/llvm-lit -s tools/bishengir/bishengir/test/Dialect/HIVM/regbase/
```

## 2. P1：簇内存活成员合并（未开始）

本次仅移除 2 个被支配的冗余 Canonicalizer。消融敏感性矩阵已证明首尾
Canonicalizer、CSE 等存活成员均有独立贡献，不存在更多可直接移除的单点。
进一步收益需要**合并存活簇成员**（减少簇调用点本身），这需要：

- 独立的设计评审（合并顺序、pass 选项语义、失败回退）；
- 重新执行与本次同规格的 L1/L2 等价与性能验证。

## 3. device-link 环境（阻塞完整 E2E 验证）

本开发机缺少 NPU 链接环境，`bishengir-compile` 在 device-link 阶段失败
（exit 1/6/134 不等，基线与候选对称）。因此本变更的正确性结论止于
pre-link IR 等价。在具备链接环境的机器上补一轮完整 E2E 编译是合入后
建议的补强项。

## 4. dump 流非确定性（工具层问题，供后续参考）

验证过程发现：`--mlir-print-ir-after-all` 的 dump 流在多线程下顺序不稳定，
**同一二进制**重复运行之间也会出现字节级差异（80 输入中 3 个可复现）。
因此本次 L2 判据改用确定性的**逐 pass 调用直方图**（20 次运行恒定），字节级
比对仅用于 L1/冒烟的单次确定性输入。后续若有人依赖 dump 流做字节级回归，
需注意该性质（或先做 dump 排序归一化）。

## 5. 语料清单再生成

`corpus-manifest.csv` 由脚本生成（开发机
`build/rb-canon-evidence/p0-closure/make_manifest.py`，未入库）。若需要扩展
语料或换机器复验，可按 manifest 中 `compile_command` 列逐条复跑；63 条不可
运行输入的原因分类见 manifest 的 `expected_output`/备注列。

## 6. 交付物位置

- 本目录：设计方案、代码修改说明、测试报告（本文件）、语料清单。
- 历轮过程性证据/评审记录未随本提交入库（开发机
  `build/rb-canon-evidence/` 下保留）。
