# 常见问题（FAQ）

本文汇总 AscendNPU IR 使用与开发中的常见问题，按类别与编号整理，便于快速查找。更多构建细节见 [安装与构建](../introduction/quick_start/installing_guide_zh.md)，贡献流程见 [贡献指南](../contributing_guide/contribute_zh.md)。

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

详见 [安装与构建](../introduction/quick_start/installing_guide_zh.md) 中的「运行测试」。

**Q2.2** 上板运行需要什么环境？

端到端在 NPU 上运行算子需要：**CANN**（安装并 source set_env.sh）、**bishengir-compile** 生成的设备端二进制（如 kernel.o）、以及使用 CANN runtime 的 Host 程序完成注册与调用。参见 [快速开始示例](../introduction/quick_start/examples_zh.md) 与 [快速开始](../introduction/quick_start/index_zh.rst)。

**Q2.3** 如何获取各层 MLIR 的中间编译态（如 HFusion、HIVM）？

1. **构建时**：在构建脚本中将 `ENABLE_IR_PRINT` 与 `BISHENGIR_PUBLISH` 设为 ON（以 `build-tools/build.sh` 及文档为准）。
2. **运行时**：使用 `bishengir-compile` 的打印选项，在指定 pass 前后导出 MLIR，例如：

```bash
bishengir-compile your.mlir --bishengir-print-ir-before=hivm-inject-block-sync --bishengir-print-ir-after=hivm-inject-block-sync
```

可替换为其他 pass 名称。更多选项见 [编译选项](../user_guide/compile_option_zh.md)、[调试调测](../user_guide/debug_option_zh.md)。

**Q2.4** 如何用 bishengir-compile 将 MLIR 编译为设备端二进制？

使用 `-enable-hivm-compile` 等选项将高层 MLIR 编译为可在 NPU 上执行的二进制，例如：

```bash
bishengir-compile input.mlir -enable-hivm-compile -o kernel.o
```

具体选项与 pipeline 见 [编译选项](../user_guide/compile_option_zh.md) 与 [架构设计](../introduction/architecture_zh.md)。

**Q2.5** LIT 或 check-bishengir 测试失败如何排查？

根据失败用例名称定位到对应测试文件与断言，查看是 IR 变换、数值结果还是环境（CANN、路径等）问题；可结合「如何获取各层 MLIR」（Q2.3）查看中间态。调试选项见 [调试调测](../user_guide/debug_option_zh.md)。

---

## 性能调优

**Q3.1** 如何定位算子性能瓶颈？

### MindStudio工具
链接：https://www.hiascend.com/developer/software/mindstudio
在华为昇腾平台上，使用 MindStudio 调试 Triton Kernel 的性能主要依赖于其内置的性能分析工具（Profiler），它可以采集硬件运行时的关键指标，帮助开发者定位 kernel 执行的瓶颈

### Torch NPU profiler
torch_npu.profiler.profile
是昇腾AI处理器上用于PyTorch训练/推理任务性能分析的核心API接口。它的主要功能是采集并解析模型运行时的性能数据，帮助开发者定位瓶颈并进行优化

核心功能与定位:
该接口通过代码注入的方式，在模型执行过程中全面采集CPU和NPU（昇腾AI处理器）的性能数据。采集的数据非常丰富，主要包括：
PyTorch层信息：框架侧算子调用、内存占用、调用栈等。
CANN层信息：昇腾计算语言接口层的调度和执行情况。
硬件层信息：NPU上的算子执行时间、AI Core性能指标（如流水线利用率）、缓存命中率等。

它是连接你的PyTorch训练脚本与后面用于可视化分析的工具（如MindStudio Insight或TensorBoard插件）之间的桥梁。

代码样例
```python
@triton.jit
def triton_example(in_ptr0, in_ptr1, out_ptr0, x0_numel, r1_numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    ...

dtype = torch.float16
torch.manual_seed(0)

input0 = rand_strided((86, 64, 130), (8320, 130, 1), device='npu:0', dtype=dtype)
input1 = rand_strided((1, 64, 1), (64, 1, 1), device='npu:0', dtype=dtype)
output = empty_strided((86, 1), (1, 86), device='npu', dtype=dtype)
triton_example[6,1,1](input0, input1, output, 86, 64, XBLOCK=16, XBLOCK_SUB=16)

experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    )
with torch_npu.profiler.profile(
    activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=False, #采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
    record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
    profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
    schedule=torch_npu.profiler.schedule(wait=1,
                                         warmup=1,
                                         active=10,
                                         repeat=1,
                                         skip_first=1),
    # schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, skip_first=6),
    # warmup默认为0，老版本torch_npu包该参数为必填项
    experimental_config=experimental_config,  # 该参数选填，默认为Level0
    # 产生的profling文件的位置
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
    # 导出tensorboard可呈现的数据形式，可指定worker_name, 默认为：{host名称}_{进程id}
) as prof:
    for i in range(20):
        triton_example[6,1,1](input0, input1, output, 86, 64, XBLOCK=16, XBLOCK_SUB=16)
        prof.step()
```
---

## 精度定位

**Q4.1** 算子结果与参考（如 CPU/GPU 或参考实现）不一致时如何排查？
在 Triton kernel 中调试精度问题时，tl.device_print 是必不可少的工具。
它允许你在 NPU 运行时直接打印张量或标量的中间值，从而定位误差出现的具体位置。使用指南如下。
```python
# 使用时需要打开环境变量 TRITON_DEVICE_PRINT=1
tl.device_print("前缀字符串",  value)
```

精度问题排查策略
1. 分段打印：在关键计算步骤（如矩阵乘加、归约、激活函数）前后插入 tl.device_print，观察数值变化。
2. 对比预期值：打印中间结果后，与手工计算或 CPU 参考实现的结果比对，快速定位误差源头。
3. 关注异常值：若发现数值突然变为 NaN 或 Inf，可在相应位置前后打印更多上下文。

代码样例
```python
import triton
import triton.language as tl

@triton.jit
def triton_add(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        # 在 NPU 运行时直接打印tmp0的数据
        tl.device_print("tmp0",  tmp0)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```

**Q4.2** 如何使用bishengir-opt对比各层 MLIR？
bishengir-opt 是类似于 mlir-opt 的工具，主要用于加载、优化和转换（降级）MLIR代码。你可以把它理解为一个“瑞士军刀”式的测试和调试工具，
它读取一个.mlir文件，对其应用一系列由用户指定的编译过程（pass），然后将结果输出，因此可以用于对 AscendNPU IR 进行独立的 Pass 调试。
通过它，开发者可以单独应用某个 Pass，并对比应用前后的 IR 差异，从而验证该 Pass 是否达到预期功能。

基本语法：
`bishengir-opt xx.mlir --{pass名称}`

示例：
test.mlir
```c++
// before hfusion-normalize-ops
func.func @test_normalize_rec_i32_to_f32(%arg0 : tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = tensor.empty() : tensor<1x2xi32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi32>) outs(%0 : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
}
```

单独执行 hfusion-normalize-ops pass
bishengir-opt test.mlir --hfusion-normalize-ops

```c++
// after hfusion-normalize-ops
module {
  func.func @test_normalize_rec_i32_to_f32(%arg0: tensor<1x2xi32>) -> tensor<1x2xi32> {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x2xf32>
    %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<1x2xi32>) outs(%0 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %2 = tensor.empty() : tensor<1x2xf32>
    %3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%1 : tensor<1x2xf32>) outs(%2 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %4 = tensor.empty() : tensor<1x2xi32>
    %5 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%3 : tensor<1x2xf32>) outs(%4 : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %5 : tensor<1x2xi32>
  }
}
```

**Q4.3** 常见精度问题有哪些（如 BF16/FP16 精度损失、累加顺序）？

如何判断精度损失是否符合标准：使用三方对比方案验证精度损失 (NPU, GPU, CPU)

这是一个非常经典且必要的验证流程，尤其在将算法从CPU移植到NPU或GPU时，用于确保硬件加速没有引入不可接受的精度损失。
以CPU的float64作为“真值”基准，对比三者float32的输出，是衡量精度损失的黄金标准。

为什么会有精度损失？
计算机使用二进制表示小数，很多十进制小数（如 0.1）无法被有限长度的二进制精确表示，只能取近似值。float32 和 float64 的精度差异巨大：
float32 (单精度)：约 7 位有效数字。内存占用 4 字节。
float64 (双精度)：约 15-16 位有效数字。内存占用 8 字节。

为什么需要三方对比？
CPU (float64)：作为参考基准，提供最高精度的计算结果。
CPU (float32)：用于隔离“精度损失”的来源。对比 float32 CPU 结果与 float64 结果，可以观察到单纯因“单精度”带来的理论损失。
GPU/NPU (float32)：用于观察在特定硬件加速器上，由于硬件指令集差异、算子实现算法不同、中间结果保留精度不同（
                   如某些NPU可能使用float16进行累加）或驱动/库的优化策略所引入的额外误差

核心对比逻辑
由于浮点数不能直接用 == 比较，必须使用容差比较。常用的方法有两种：
绝对误差 (Absolute Error)：|a - b|
相对误差 (Relative Error)：|a - b| / max(|a|, |b|)，适用于大数比较。
混合容差：结合两者，如 np.isclose() 的实现。
---

## 贡献与社区

**Q5.1** 如何参与贡献？

参与前需签署 Ascend 社区贡献者许可协议（CLA），并遵循 [ascend-community](https://gitcode.com/ascend/community) 行为准则。贡献流程包括：通过 Issue 反馈或认领任务、Fork 仓库开发、自测（如 `ninja check-bishengir`）、提交 PR，以及通过门禁（编译、静态检查、CI）。合入需 2 位 Reviewer 的 `/lgtm` 与 1 位 Approver 的 `/approve`。完整说明见 [贡献指南](../contributing_guide/contribute_zh.md)。

**Q5.2** PR 门禁失败（编译失败、静态检查失败、CI 未通过）如何排查？

根据 CI 提示信息逐项修复：**编译失败**时检查报错与构建环境；**静态检查失败**时按提示修改代码风格或逻辑；**CI 测试未通过**时定位失败用例并修复后重新触发 CI。详见 [贡献指南](../contributing_guide/contribute_zh.md) 中的「门禁异常处理」。

**Q5.3** 提交 PR 前有哪些注意事项？

避免在 PR 中引入与本次修改无关的变更；保持提交历史简洁（可适当 squash/rebase）；创建 PR 前将分支 rebase 到上游最新 master；若为错误修复类 PR，请在描述中关联相关 Issue 与 PR。详见 [贡献指南](../contributing_guide/contribute_zh.md) 中的「注意事项」。
