# 常见问题（FAQ）

本文汇总AscendNPU IR使用与开发中的常见问题，按类别与编号整理，便于快速查找。更多构建细节见[安装与构建](../introduction/quick_start/installing_guide.md)，贡献流程见[贡献指南](../contributing_guide/contribute.md)。

## 构建与安装

**Q：执行`build-tools/build.sh`时报错「ninja: error: loading 'build.ninja': No such file or directory」怎么办？**

在调用`build-tools/build.sh`时添加`-r`选项，重新执行CMake并生成新的`build.ninja`，例如：

```bash
./build-tools/build.sh -o ./build -r --build-type Debug
```

**Q：构建时报错「Too many open files」怎么办？**

文件同时打开数量超过了系统中配置的上限，可以通过`ulimit -n xxx`来修改文件同时打开数量上限，如`ulimit -n 65535`。

**Q：构建时遇到以下报错如何处理？**

```bash
 The CMAKE_CXX_COMPILER:

 clang++

 is not a full path and was not found in the PATH.
```

未指定C++编译器或C++编译器二进制存在问题，首先尝试通过`--cxx-compiler=${CXX-COMPILER-PATH}`指定要使用的C++编译器，如果已经指定了C++编译器仍然报错，则尝试重新安装或使用其他版本的C++编译器，如使用推荐的clang++-15。

## 运行与调试

**Q：如何运行测试？**

在构建目录下可执行：

- **bishengir测试**：`ninja check-bishengir`或`cmake --build . --target check-bishengir`
- **LIT测试套**：`./bin/llvm-lit ../bishengir/test`（路径以实际仓库与构建目录为准）

详见[安装与构建-运行测试](../introduction/quick_start/installing_guide.md#运行测试)。

**Q：上板运行需要什么环境？**

端到端在NPU上运行算子，需准备以下三项依赖：

- `CANN`：完成安装并执行`source set_env.sh`配置环境变量。
- `bishengir-compile`：编译生成设备端二进制文件（例：`kernel.o`）。
- CANN Runtime的Host程序：用于算子注册与调用。

完整操作流程与实操案例，可查阅：[快速开始示例](../introduction/quick_start/examples.md)、[快速开始](../introduction/quick_start/index.rst)

**Q：如何获取各层MLIR的中间编译态（如HFusion、HIVM）？**

- 构建阶段：修改构建脚本，将`ENABLE_IR_PRINT`、`BISHENGIR_PUBLISH`配置为ON，具体配置方式以`build-tools/build.sh`及配套文档为准。

- 运行阶段：通过`bishengir-compile`提供的打印参数，在指定Pass前后导出对应MLIR，示例命令如下：

  ```bash
  bishengir-compile your.mlir --bishengir-print-ir-before=hivm-inject-block-sync --bishengir-print-ir-after=hivm-inject-block-sync
  ```

​  参数内的Pass名称可按需替换，完整参数说明可查阅：[编译选项](../user_guide/compile_option.md)、[调试调测](../user_guide/debug_option.md)

**Q：如何用bishengir-compile将MLIR编译为设备端二进制？**

使用`-enable-hivm-compile`等选项将高层MLIR编译为可在NPU上执行的二进制，例如：

```bash
bishengir-compile input.mlir -enable-hivm-compile -o kernel.o
```

具体选项与pipeline见[编译选项](../user_guide/compile_option.md)与[架构设计](../introduction/architecture.md)。

**Q：LIT或check-bishengir测试失败如何排查？**

1. 根据失败用例名称定位对应测试文件与断言信息，区分故障类型：IR变换、数值结果、运行环境异常（CANN版本、文件路径配置等）。

2. 如需定位IR变换问题，可参考上文「如何获取各层MLIR的中间编译态（如HFusion、HIVM）」的方法查看中间态。

调试选项可查阅：[调试调测](../user_guide/debug_option.md)

## 性能调优

**Q：如何定位算子性能瓶颈？**

1. `MindStudio`：可使用[MindStudio](https://www.hiascend.com/developer/software/mindstudio)调试Triton Kernel性能，工具内置Profiler性能分析组件，能够采集硬件运行时的关键指标，帮助开发者定位kernel执行的瓶颈。

2. `torch_npu.profiler.profile`：昇腾AI处理器上用于PyTorch训练 / 推理任务性能分析的核心API接口。它的主要功能是采集并解析模型运行时的性能数据，帮助开发者定位瓶颈并进行优化。该接口通过代码注入的方式，在模型执行过程中全面采集CPU和NPU（昇腾AI处理器）的性能数据。可采集多维度数据，主要包括：

   - PyTorch层信息：框架侧算子调用、内存占用、调用栈等。
   - CANN层信息：昇腾计算语言接口层的调度和执行情况。
   - 硬件层信息：NPU上的算子执行时间、AI Core性能指标（如流水线利用率）、缓存命中率等。

   它是连接你的PyTorch训练脚本与用于可视化分析的工具（如MindStudio Insight或TensorBoard插件）之间的桥梁。

   **示例**：

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
       # 产生的profiling文件的位置
       on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
       # 导出tensorboard可呈现的数据形式，可指定worker_name, 默认为：{host名称}_{进程id}
   ) as prof:
       for i in range(20):
           triton_example[6,1,1](input0, input1, output, 86, 64, XBLOCK=16, XBLOCK_SUB=16)
           prof.step()
   ```

## 精度定位

**Q：算子结果与参考（如CPU/GPU或参考实现）不一致时如何排查？**

在Triton kernel中调试精度问题时，`tl.device_print`是必不可少的工具。它允许你在NPU运行时直接打印张量或标量的中间值，从而定位误差出现的具体位置。使用指南如下。

```python
# 使用时需要打开环境变量 TRITON_DEVICE_PRINT=1
tl.device_print("前缀字符串",  value)
```

**精度问题排查策略**：

1. 分段打印：在关键计算步骤（如矩阵乘加、归约、激活函数）前后插入`tl.device_print`，观察数值变化。
2. 对比预期值：打印中间结果后，与手工计算或CPU参考实现的结果比对，快速定位误差源头。
3. 关注异常值：若发现数值突然变为`NaN`或`Inf`，可在相应位置前后打印更多上下文。

**示例**：

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

**Q：如何使用bishengir-opt对比各层MLIR？**

`bishengir-opt`是类似于`mlir-opt`的工具，是一个用于加载、优化和降级转换MLIR代码的综合测试调试工具。该工具读取`.mlir`文件，执行用户指定的编译Pass并输出变换后的IR，支持对AscendNPU IR进行独立的Pass调试。开发者可单独执行指定Pass，对比变换前后的IR差异，验证该Pass是否达到预期功能。

**基本语法**：

`bishengir-opt xx.mlir --{Pass名称}`

**示例**：

变换前输入IR（`test.mlir`）：

```mlir
// before hfusion-normalize-ops
func.func @test_normalize_rec_i32_to_f32(%arg0 : tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = tensor.empty() : tensor<1x2xi32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi32>) outs(%0 : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
}
```

执行`bishengir-opt test.mlir --hfusion-normalize-ops`转换后输出IR：

```mlir
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

**Q：常见精度问题有哪些（如BF16/FP16精度损失、累加顺序）？**

**典型精度问题场景**：

常见精度偏差场景包含BF16/FP16低精度计算带来的数值损失、张量累加顺序不同引发的误差累积等。

**浮点数精度损失产生原理**：

计算机使用二进制表示小数，多数十进制小数（如0.1）无法被有限长度的二进制精确表示，只能取近似值。不同浮点位宽的数值表达能力存在明显差距：

- float32（单精度）：有效数字约7位，占用4字节存储空间。
- float64（双精度）：有效数字约15~16位，占用8字节存储空间。

**精度判定方案**：

采用CPU、GPU、NPU三方结果对照的验证方案校验精度偏差，该流程是算法从CPU移植到NPU/GPU迁移时的标准必做验证，可保障硬件加速没有引入不可接受的精度损失。

校验基准规则：以CPU float64高精度计算结果作为真值基准，对比三类硬件float32输出，以此量化整体精度误差。

**三方对比的作用**：

1. CPU (float64)：作为参考基准，提供最高精度的计算结果。
2. CPU (float32)：用于隔离“精度损失”的来源。对比float32 CPU结果与float64结果，可以观察到单纯因“单精度”带来的理论损失。
3. GPU/NPU (float32)：定位硬件额外误差，误差来源包含硬件指令集、算子实现逻辑、中间计算存储位宽（如部分NPU采用FP16累加）、驱动/库的优化策略等。

**浮点数对比核心逻辑**：

浮点数无法直接用`==`比较，需基于容差阈值判定。常用判别方式分为两类：

- 绝对误差：`|a - b|`
- 相对误差：`|a - b| / max(|a|, |b|)`，适用于大数比较。

混合容差是融合绝对误差与相对误差两种判别逻辑，典型实现如`np.isclose()`。

## 贡献与社区

**Q：如何参与贡献？**

1. 前置要求：

   参与前需签署Ascend社区贡献者许可协议（CLA），并遵循[ascend-community](https://gitcode.com/ascend/community)行为准则。

2. 标准贡献流程：

   通过Issue反馈或认领任务；Fork目标仓库，本地完成功能开发；本地执行自测（例如运行`ninja check-bishengir` ）；提交PR；通过门禁校验（包含编译、静态检查、CI）。

3. PR合入条件：

   PR需获得2位评审者的`/lgtm`，且至少1位审批者的`/approve` ，方可合入主干。

完整贡献规范详见文档：[贡献指南](../contributing_guide/contribute.md)。

**Q：PR门禁失败（编译失败、静态检查失败、CI未通过）如何排查？**

根据CI输出提示逐项处理门禁异常：

- 编译失败：查看构建报错日志，检查代码与构建环境。
- 静态检查失败：按照工具提示修正代码格式、编码规范或逻辑问题。
- CI测试不通过：定位失败测试用例，修复代码后重新触发CI。

详细方案参考文档：[贡献指南-门禁异常处理](../contributing_guide/contribute.md#门禁异常处理)

**Q：提交PR前有哪些注意事项？**

- 避免在PR中引入与本次修改无关的变更。
- 保持提交历史简洁（可适当squash/rebase）。
- 创建PR前将分支rebase到上游最新master。
- 若为错误修复类PR，请在描述中关联相关Issue与PR。

详见[贡献指南-注意事项](../contributing_guide/contribute.md#注意事项)。
