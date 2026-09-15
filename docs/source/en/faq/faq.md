# FAQ

This document summarizes common issues in using and developing AscendNPU IR, organized by category and number for quick reference. For more build details, see [Build and Installation](../introduction/quick_start/installing_guide.md). For the contribution process, see [Contribution Guide](../contributing_guide/contribute.md).

## Build and Installation

**Q: When I run `build-tools/build.sh`, I get the error "ninja: error: loading 'build.ninja': No such file or directory". What should I do?**

Add the `-r` option when invoking `build-tools/build.sh` to re-run CMake and generate a new `build.ninja`, for example:

```bash
./build-tools/build.sh -o ./build -r --build-type Debug
```

**Q: What should I do if the build reports "Too many open files"?**

The number of simultaneously open files exceeds the limit configured on the system. You can change the limit on the number of simultaneously open files by running `ulimit -n xxx`, for example, `ulimit -n 65535`.

**Q: How do I handle the following error during the build?**

```bash
 The CMAKE_CXX_COMPILER:

 clang++

 is not a full path and was not found in the PATH.
```

No C++ compiler is specified, or the C++ compiler binary is faulty. First, try specifying the C++ compiler to use via `--cxx-compiler=${CXX-COMPILER-PATH}`. If the error persists even after specifying a C++ compiler, try reinstalling it or using a different version of the C++ compiler, such as the recommended clang++-15.

## Running and Debugging

**Q: How do I run tests?**

In the build directory, you can execute the following:

- **bishengir tests**: `ninja check-bishengir` or `cmake --build . --target check-bishengir`
- **LIT test suite**: `./bin/llvm-lit ../bishengir/test` (the path depends on the actual repository and build directory)

For details, see [Build and Installation - Running Tests](../introduction/quick_start/installing_guide.md#running-tests).

**Q: What environment is required for on-board deployment?**

To run operators end-to-end on the NPU, the following three dependencies are required:

- `CANN`: Complete the installation and run `source set_env.sh` to configure the environment variables.
- `bishengir-compile`: Compiles and generates the device-side binary file (for example, `kernel.o`).
- Host program of CANN Runtime: used for operator registration and invocation.

For the complete workflow and hands-on examples, see [Compilation and Execution Example](../introduction/quick_start/examples.md) and [Quick Start](../introduction/quick_start/index.rst).

**Q: How do I obtain the intermediate compilation states of MLIR at each layer (such as HFusion and HIVM)?**

- Build phase: Modify the build script to set `ENABLE_IR_PRINT` and `BISHENGIR_PUBLISH` to ON. For the specific configuration method, refer to `build-tools/build.sh` and its accompanying documentation.

- Run phase: Use the print parameters provided by `bishengir-compile` to export the corresponding MLIR before and after a specified Pass. An example command is as follows:

  ```bash
  bishengir-compile your.mlir --bishengir-print-ir-before=hivm-inject-block-sync --bishengir-print-ir-after=hivm-inject-block-sync
  ```

  The Pass name in the parameter can be replaced as needed. For the complete parameter description, see [Compilation Options](../user_guide/compile_option.md) and [Debugging and Tuning](../user_guide/debug_option.md).

**Q: How do I use bishengir-compile to compile MLIR into a device-side binary?**

Use options such as `-enable-hivm-compile` to compile high-level MLIR into a binary that can be executed on the NPU, for example:

```bash
bishengir-compile input.mlir -enable-hivm-compile -o kernel.o
```

For specific options and pipelines, see [Compilation Options](../user_guide/compile_option.md) and [Architecture Design](../introduction/architecture.md).

**Q: How do I troubleshoot LIT or check-bishengir test failures?**

1. Locate the corresponding test file and assertion information based on the name of the failed test case, and distinguish the failure type: IR transformation, numerical results, or runtime environment exceptions (CANN version, file path configuration, etc.).

2. To locate IR transformation issues, refer to the method described above in "How to obtain the intermediate compilation states of MLIR at each layer (such as HFusion and HIVM)" to view the intermediate states.

For debugging options, see [Debugging and Tunning](../user_guide/debug_option.md)

## Performance Tuning

**Q: How do I locate operator performance bottlenecks?**

1. `MindStudio`: You can use [MindStudio](https://www.hiascend.com/developer/software/mindstudio) to debug Triton Kernel performance. The tool has a built-in Profiler performance analysis component that can collect key metrics during hardware runtime, helping developers locate kernel execution bottlenecks.

2. `torch_npu.profiler.profile`: A core API for performance analysis of PyTorch training/inference tasks on Ascend AI processors. Its main function is to collect and parse performance data during model runtime, helping developers locate bottlenecks and optimize them. Through code injection, this API comprehensively collects CPU and NPU (Ascend AI processor) performance data during model execution. It can collect multi-dimensional data, mainly including:

   - PyTorch layer information: framework-side operator invocation, memory usage, call stacks, and so on.
   - CANN layer information: scheduling and execution of the Ascend Computing Language interface layer.
   - Hardware layer information: operator execution time on the NPU, AI Core performance metrics (such as pipeline utilization), cache hit rate, and so on.

   It serves as a bridge between your PyTorch training script and tools used for visual analysis (such as MindStudio Insight or the TensorBoard plugin).

   **Example**:

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
       with_stack=False, # Switch for collecting the function call stack of torch operators. This parameter is optional and disabled by default.
       record_shapes=False,  # Switch for collecting the input shape and input type of torch operators. This parameter is optional and disabled by default.
       profile_memory=False,  # Switch for collecting memory-related data. This parameter is optional and disabled by default.
       schedule=torch_npu.profiler.schedule(wait=1,
                                            warmup=1,
                                            active=10,
                                            repeat=1,
                                            skip_first=1),
       # schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, skip_first=6),
       # warmup defaults to 0. In older versions of the torch_npu package, this parameter is required.
       experimental_config=experimental_config,  # This parameter is optional and defaults to Level0.
       # Location of the generated profiling files.
       on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
       # Export data in a format that TensorBoard can present. You can specify worker_name, which defaults to {host name}_{process ID}.
   ) as prof:
       for i in range(20):
           triton_example[6,1,1](input0, input1, output, 86, 64, XBLOCK=16, XBLOCK_SUB=16)
           prof.step()
   ```

## Precision Debugging

**Q: How to troubleshoot when the operator result is inconsistent with the reference (such as CPU/GPU or a reference implementation)?**

When debugging precision issues in a Triton kernel, `tl.device_print` is an indispensable tool. It allows you to directly print intermediate values of tensors or scalars at NPU runtime, thereby locating the exact position where the error occurs. The usage guide is as follows.

```python
# Enable the TRITON_DEVICE_PRINT=1 environment variable before use.
tl.device_print("prefix string",  value)
```

**Precision Troubleshooting Strategy**:

1. Print in segments: Insert `tl.device_print` before and after key computation steps (such as matrix multiply-add, reduction, and activation functions) to observe value changes.
2. Compare with expected values: After printing intermediate results, compare them with manually computed results or a CPU reference implementation to quickly locate the source of the error.
3. Pay attention to abnormal values: If a value suddenly becomes `NaN` or `Inf`, print more context before and after the corresponding position.

**Example**:

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
        # Print the data of tmp0 directly at NPU runtime.
        tl.device_print("tmp0",  tmp0)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```

**Q: How do I use bishengir-opt to compare the MLIR of each layer?**

`bishengir-opt` is a tool similar to `mlir-opt`. It is a comprehensive testing and debugging tool for loading, optimizing, and lowering MLIR code. The tool reads `.mlir` files, executes the compilation passes specified by the user, and outputs the transformed IR. It supports independent pass debugging for AscendNPU IR. Developers can execute a specified pass independently, compare the IR differences before and after the transformation, and verify whether the pass achieves the expected functionality.

**Basic syntax**:

`bishengir-opt xx.mlir --{Pass name}`

**Example**:

Input IR before transformation (`test.mlir`):

```mlir
// before hfusion-normalize-ops
func.func @test_normalize_rec_i32_to_f32(%arg0 : tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = tensor.empty() : tensor<1x2xi32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi32>) outs(%0 : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
}
```

Output IR after executing `bishengir-opt test.mlir --hfusion-normalize-ops`:

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

**Q: What are the common precision issues (such as BF16/FP16 precision loss and accumulation order)?**

**Typical precision problem scenarios**:

Common precision deviation scenarios include numerical loss caused by BF16/FP16 low-precision computation, and error accumulation caused by different tensor accumulation orders.

**Principle of Floating-Point Precision Loss**:

Computers use binary to represent decimals. Most decimal fractions (such as 0.1) cannot be represented exactly by a finite-length binary number and can only be approximated. Different floating-point bit widths differ significantly in their numerical representation capability:

- float32 (single precision): approximately 7 significant digits, occupying 4 bytes of storage.
- float64 (double precision): approximately 15 to 16 significant digits, occupying 8 bytes of storage.

**Precision Verification Scheme**:

A verification scheme that compares the results from CPU, GPU, and NPU is adopted to verify precision deviation. This process is a standard mandatory verification when migrating an algorithm from CPU to NPU/GPU, and it ensures that hardware acceleration does not introduce unacceptable precision loss.

Baseline rule for verification: the high-precision float64 computation result on CPU is used as the ground-truth baseline, and the float32 outputs of the three types of hardware are compared against it to quantify the overall precision error.

**Purpose of the Three-Way Comparison**:

1. CPU (float64): serves as the reference baseline, providing the highest-precision computation result.
2. CPU (float32): Used to isolate the source of "Precision Loss". By comparing the float32 CPU result with the float64 result, the theoretical loss caused purely by "single precision" can be observed.
3. GPU/NPU (float32): Used to locate additional hardware errors. The error sources include the hardware instruction set, operator implementation logic, intermediate computation storage bit width (for example, some NPUs use FP16 accumulation), and driver/library optimization strategies.

**Core logic of floating-point comparison**:

Floating-point numbers cannot be compared directly with `==`; they must be judged based on tolerance thresholds. Common judgment methods fall into two categories:

- Absolute error: `|a - b|`
- Relative error: `|a - b| / max(|a|, |b|)`, suitable for comparing large numbers.

Mixed tolerance combines the two judgment logics of absolute error and relative error, with a typical implementation such as `np.isclose()`.

## Contribution and Community

**Q: How can I contribute?**

1. Prerequisites:

   Before participating, you must sign the Ascend Community Contributor License Agreement (CLA) and follow the [ascend-community](https://gitcode.com/ascend/community) code of conduct.

2. Standard contribution process:

   Provide feedback or claim tasks through Issues; fork the target repository and complete feature development locally; run local self-tests (for example, run `ninja check-bishengir`); submit a PR; and pass the gate check (including compilation, static check, and CI).

3. PR merge conditions:

   A PR must receive `/lgtm` from two reviewers and `/approve` from at least one approver before it can be merged into the main branch.

For the complete contribution specifications, see [Contribution Guide](../contributing_guide/contribute.md).

**Q: How to troubleshoot PR gate failures (compilation failure, static check failure, or CI failure)?**

Handle each gate exception based on the CI output prompts:

- Compilation failure: Check the build error logs, and inspect the code and build environment.
- Static check failure: Correct the code format, coding standards, or logic issues as prompted by the tool.
- CI test failure: Locate the failing test cases, fix the code, and then re-trigger CI.

For detailed solutions, see [Contribution Guide - Gate Exception Handling](../contributing_guide/contribute.md#gate-exception-handling)

**Q: What precautions should be taken before submitting a PR?**

- Avoid introducing changes unrelated to this modification in the PR.
- Keep the commit history concise (squash/rebase as appropriate).
- Rebase the branch onto the latest upstream master before creating the PR.
- For bug-fix PRs, associate the relevant Issue and PR in the description.

For details, see [Contribution Guide - Notes](../contributing_guide/contribute.md#notes).
