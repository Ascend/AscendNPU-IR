# 常见问题（FAQ）

本文汇总 AscendNPU IR 使用与开发中的常见问题，按类别与编号整理，便于快速查找。更多构建细节见 :doc:`安装与构建 <sources/introduction/quickStart/installingGuide>`，贡献流程见 :doc:`贡献指南 <sources/contributing-guide/contribute>`。

---

## 构建与安装

**Q1.1** 执行 build.sh 时报错「ninja: error: loading 'build.ninja': No such file or directory」怎么办？

在调用 `build-tools/build.sh` 时添加 `-r` 选项，重新执行 CMake 并生成新的 `build.ninja`，例如：

```bash
./build-tools/build.sh -o ./build -r --build-type Debug
```

**Q1.2** 构建时报错「Too many open files」怎么办？

系统对单进程打开文件数有限制，可临时提高上限后重新构建，例如：

```bash
ulimit -n 65535
```

**Q1.3** 首次构建为什么要加 `--apply-patches`？

`--apply-patches` 用于使能 AscendNPU IR 对 LLVM/MLIR 等三方仓库的扩展（patch），首次编译时必须启用；非首次增量构建可不再加该参数。

---

## 运行与调试

**Q2.1** 如何运行测试？

在**构建目录**下可执行：

- **bishengir 测试**：`ninja check-bishengir` 或 `cmake --build . --target check-bishengir`
- **LIT 测试套**：`./bin/llvm-lit ../bishengir/test`（路径以实际仓库与构建目录为准）

详见 :doc:`安装与构建 <sources/introduction/quickStart/installingGuide>` 中的「运行测试」。

**Q2.2** 上板运行需要什么环境？

端到端在 NPU 上运行算子需要：**CANN**（安装并 source set_env.sh）、**bishengir-compile** 生成的设备端二进制（如 kernel.o）、以及使用 CANN runtime 的 Host 程序完成注册与调用。参见 :doc:`快速开始示例 <sources/introduction/quickStart/examples>` 与 :doc:`快速开始 <sources/introduction/quickStart/index>`。

**Q2.3** 如何获取各层 MLIR 的中间编译态（如 HFusion、HIVM）？

1. **构建时**：在构建脚本中将 `ENABLE_IR_PRINT` 与 `BISHENGIR_PUBLISH` 设为 ON（以 `build-tools/build.sh` 及文档为准）。
2. **运行时**：使用 `bishengir-compile` 的打印选项，在指定 pass 前后导出 MLIR，例如：

```bash
bishengir-compile your.mlir --bishengir-print-ir-before=hivm-inject-block-sync --bishengir-print-ir-after=hivm-inject-block-sync
```

可替换为其他 pass 名称。更多选项见 :doc:`编译选项 <sources/user-guide/compileOption>`、:doc:`调试调测 <sources/user-guide/debugOption>`。

**Q2.4** 如何用 bishengir-compile 将 MLIR 编译为设备端二进制？

使用 `-enable-hivm-compile` 等选项将高层 MLIR 编译为可在 NPU 上执行的二进制，例如：

```bash
bishengir-compile input.mlir -enable-hivm-compile -o kernel.o
```

具体选项与 pipeline 见 :doc:`编译选项 <sources/user-guide/compileOption>` 与 :doc:`架构设计 <sources/introduction/architecture>`。

**Q2.5** LIT 或 check-bishengir 测试失败如何排查？

根据失败用例名称定位到对应测试文件与断言，查看是 IR 变换、数值结果还是环境（CANN、路径等）问题；可结合「如何获取各层 MLIR」（Q2.3）查看中间态。调试选项见 :doc:`调试调测 <sources/user-guide/debugOption>`。

---

## 性能调优

**Q3.1** 如何定位算子性能瓶颈？

*TODO：本节内容待补充。建议涵盖：profiling 手段、与参考实现/竞品对比、关键 pass 对性能的影响等。*

**Q3.2** 编译选项或优化 pass 对性能有何影响？

*TODO：本节内容待补充。建议涵盖：常用编译选项、HFusion/HIVM 中与性能相关的 pass、Release/Debug 构建差异等。*

**Q3.3** 如何开启或关闭某项优化（如 CVPipeline、AutoSubTiling）？

*TODO：本节内容待补充。建议涵盖：bishengir-compile 或 pass 的开关选项、对应文档入口等。*

---

## 精度定位

**Q4.1** 算子结果与参考（如 CPU/GPU 或参考实现）不一致时如何排查？

*TODO：本节内容待补充。建议涵盖：逐层对比中间结果、数据类型与舍入、与调试选项的结合等。*

**Q4.2** 如何对比各层 MLIR 或中间表示的数值结果？

*TODO：本节内容待补充。建议涵盖：插桩/打印中间结果、与 Q2.3 的配合、常用调试流程等。*

**Q4.3** 常见精度问题有哪些（如 BF16/FP16 精度损失、累加顺序）？

*TODO：本节内容待补充。建议涵盖：常见场景与缓解方式、文档或最佳实践链接等。*

---

## 贡献与社区

**Q5.1** 如何参与贡献？

参与前需签署 Ascend 社区贡献者许可协议（CLA），并遵循 [ascend-community](https://gitcode.com/ascend/community) 行为准则。贡献流程包括：通过 Issue 反馈或认领任务、Fork 仓库开发、自测（如 `ninja check-bishengir`）、提交 PR，以及通过门禁（编译、静态检查、CI）。合入需 2 位 Reviewer 的 `/lgtm` 与 1 位 Approver 的 `/approve`。完整说明见 :doc:`贡献指南 <sources/contributing-guide/contribute>`。

**Q5.2** PR 门禁失败（编译失败、静态检查失败、CI 未通过）如何排查？

根据 CI 提示信息逐项修复：**编译失败**时检查报错与构建环境；**静态检查失败**时按提示修改代码风格或逻辑；**CI 测试未通过**时定位失败用例并修复后重新触发 CI。详见 :doc:`贡献指南 <sources/contributing-guide/contribute>` 中的「门禁异常处理」。

**Q5.3** 提交 PR 前有哪些注意事项？

避免在 PR 中引入与本次修改无关的变更；保持提交历史简洁（可适当 squash/rebase）；创建 PR 前将分支 rebase 到上游最新 master；若为错误修复类 PR，请在描述中关联相关 Issue 与 PR。详见 :doc:`贡献指南 <sources/contributing-guide/contribute>` 中的「注意事项」。
