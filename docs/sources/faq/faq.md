# FAQ

This page collects common questions about using and developing AscendNPU IR, grouped by topic. For build details see [Install and build](../introduction/quick_start/installing_guide.md); for contribution flow see [Contributing guide](../contributing_guide/contribute.md).



## Build and installation

**Q1.1** When running build.sh I get "ninja: error: loading 'build.ninja': No such file or directory". What should I do?

Add the `-r` option to `build-tools/build.sh` to rerun CMake and regenerate `build.ninja`, e.g.:

```bash
./build-tools/build.sh -o ./build -r --build-type Debug
```

**Q1.2** Build fails with "Too many open files". What should I do?

The system limits open files per process. Raise the limit and rebuild, e.g.:

```bash
ulimit -n 65535
```

**Q1.3** Why is `--apply-patches` required on first build?

`--apply-patches` enables AscendNPU IR extensions (patches) for LLVM/MLIR and other third-party repos; it is required on the first build. You can omit it for later incremental builds.



## Running and debugging

**Q2.1** How do I run tests?

From the **build directory**:

- **bishengir tests**: `ninja check-bishengir` or `cmake --build . --target check-bishengir`
- **LIT test suite**: `./bin/llvm-lit ../bishengir/test` (adjust paths to your repo and build dir)

See "Running tests" in [Install and build](../introduction/quick_start/installing_guide.md).

**Q2.2** What is needed to run on device?

You need **CANN** (installed and `set_env.sh` sourced), a device binary built with **bishengir-compile** (e.g. `kernel.o`), and a Host program that uses CANN runtime to register and launch the kernel. See [Quick start example](../introduction/quick_start/examples.md) and [Quick start](../introduction/quick_start/index.rst).

**Q2.3** How do I get intermediate MLIR (e.g. HFusion, HIVM)?

1. **At build time**: set `ENABLE_IR_PRINT` and `BISHENGIR_PUBLISH` to ON in the build script (see `build-tools/build.sh` and docs).
2. **At run time**: use `bishengir-compile` print options to dump MLIR before/after a pass, e.g.:

```bash
bishengir-compile your.mlir --bishengir-print-ir-before=hivm-inject-block-sync --bishengir-print-ir-after=hivm-inject-block-sync
```

You can use other pass names. See [Compile options](../user_guide/compile_option.md) and [Debug and tune](../user_guide/debug_option.md).

**Q2.4** How do I compile MLIR to a device binary with bishengir-compile?

Use options such as `-enable-hivm-compile` to compile high-level MLIR to an NPU binary, e.g.:

```bash
bishengir-compile input.mlir -enable-hivm-compile -o kernel.o
```

See [Compile options](../user_guide/compile_option.md) and [Architecture](../introduction/architecture.md) for options and pipeline.

**Q2.5** How do I debug LIT or check-bishengir failures?

Use the failing test name to find the test file and assertions; check whether the issue is IR transformation, numerical result, or environment (CANN, paths, etc.). Use "How do I get intermediate MLIR" (Q2.3) to inspect intermediate state. See [Debug and tune](../user_guide/debug_option.md).



## Performance tuning

**Q3.1** How do I find operator performance bottlenecks?

### MindStudio

Link: https://www.hiascend.com/developer/software/mindstudio  
On Ascend, MindStudio’s Profiler collects runtime metrics to help locate kernel bottlenecks when debugging Triton kernels.

### Torch NPU profiler

`torch_npu.profiler.profile` is the main API for profiling PyTorch training/inference on Ascend. It collects and parses runtime performance data to help locate and fix bottlenecks.

It injects instrumentation to collect CPU and NPU data, including: PyTorch-layer info (ops, memory, call stacks), CANN-layer scheduling and execution, and hardware-layer info (operator time, AI Core metrics such as pipeline utilization, cache hit rate). It bridges your training script and visualization tools (e.g. MindStudio Insight or TensorBoard).

**Example**

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
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=False,
    record_shapes=False,
    profile_memory=False,
    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=10, repeat=1, skip_first=1),
    experimental_config=experimental_config,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
) as prof:
    for i in range(20):
        triton_example[6,1,1](input0, input1, output, 86, 64, XBLOCK=16, XBLOCK_SUB=16)
        prof.step()
```

## Accuracy and debugging

**Q4.1** Results differ from reference (CPU/GPU or reference implementation). How do I debug?
When debugging precision issues in Triton kernels, `tl.device_print` is an indispensable tool.
It allows you to directly print intermediate values of tensors or scalars at NPU runtime, 
thereby pinpointing where the error occurs. Usage guide:
```python
# Usage requires setting the environment variable TRITON_DEVICE_PRINT=1
tl.device_print("prefix string", value)
```

Precision issue troubleshooting strategy:
1. Segmented printing: Insert tl.device_print before and after key computation steps (e.g., matrix multiply-add, reduction, activation functions) to observe numerical changes.
2. Compare with expected values: After printing intermediate results, compare them with manual calculations or the CPU reference implementation to quickly locate the source of error.
3. Watch for abnormal values: If values suddenly become NaN or Inf, print more context around the corresponding positions.

**Example**

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
        # Print tmp0 data directly at NPU runtime
        tl.device_print("tmp0",  tmp0)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```

**Q4.2** How do I compare numerical results across MLIR layers or intermediate representations?

**bishengir-opt** is a tool similar to **mlir-opt**, primarily used for loading, optimizing, and transforming (lowering) MLIR code. 
You can think of it as a "Swiss Army knife" testing and debugging tool: it reads an .mlir file, applies a series of user-specified compilation passes, and outputs the result. 
Hence, it can be used for independent pass debugging of AscendNPU IR.
Through it, developers can apply a specific pass individually and compare the IR differences before and after application, thereby verifying whether the pass achieves the intended functionality.

**Basic syntax:**
`bishengir-opt xx.mlir --{pass name}`

**Example**
test.mlir
```c++
// before hfusion-normalize-ops
func.func @test_normalize_rec_i32_to_f32(%arg0 : tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = tensor.empty() : tensor<1x2xi32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi32>) outs(%0 : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
}
```
Execute the hfusion-normalize-ops pass alone:
`bishengir-opt test.mlir --hfusion-normalize-ops`
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

**Q4.3** What are common accuracy issues (e.g. BF16/FP16 loss, reduction order)?

How to determine if precision loss meets standards: Use a three-way comparison scheme to verify precision loss (NPU, GPU, CPU)
This is a classic and necessary verification process, especially when porting algorithms from CPU to NPU or GPU, to ensure hardware acceleration does not introduce unacceptable precision loss.
Taking CPU float64 as the "ground truth" benchmark and comparing the float32 outputs of all three is the gold standard for measuring precision loss.

Why does precision loss occur?
Computers use binary to represent decimal numbers; many decimal fractions (e.g., 0.1) cannot be exactly represented by finite binary length and can only be approximated. The precision difference between float32 and float64 is huge:
float32 (single precision): approximately 7 significant digits. Memory footprint: 4 bytes.
float64 (double precision): approximately 15–16 significant digits. Memory footprint: 8 bytes.

Why is a three-way comparison needed?
CPU (float64): Serves as the reference benchmark, providing the highest precision computation results.
CPU (float32): Isolates the source of "precision loss." Comparing float32 CPU results with float64 results reveals the theoretical loss caused purely by "single precision."
GPU/NPU (float32): Observes additional errors introduced by specific hardware accelerators due to differences in instruction sets, operator implementation algorithms, intermediate result retention precision (e.g., some NPUs may use float16 for accumulation), or driver/library optimization strategies.

Core comparison logic
Since floating-point numbers cannot be directly compared with ==, tolerance-based comparison must be used. Common methods are:
Absolute Error: |a - b|
Relative Error: |a - b| / max(|a|, |b|) – suitable for comparing large numbers.
Mixed tolerance: Combines both, e.g., the implementation of np.isclose().


## Contributing and community

**Q5.1** How do I contribute?

You must sign the Ascend Community CLA and follow the [ascend-community](https://gitcode.com/ascend/community) code of conduct. Flow: open or claim an Issue, fork and develop, self-test (e.g. `ninja check-bishengir`), open a PR, and pass CI (build, static check, tests). Merge requires 2 Reviewers’ `/lgtm` and 1 Approver’s `/approve`. See [Contributing guide](../contributing_guide/contribute.md).

**Q5.2** PR fails CI (build, static check, or tests). How do I fix it?

Follow the CI log: fix **build failures** (errors and environment), **static check** (style/logic as indicated), and **failing tests** (fix and re-run CI). See "Dealing with CI failures" in [Contributing guide](../contributing_guide/contribute.md).

**Q5.3** What should I do before opening a PR?

Avoid unrelated changes; keep history clear (squash/rebase if needed); rebase your branch onto the latest upstream master; for bug-fix PRs, reference related Issues and PRs in the description. See "Notes" in [Contributing guide](../contributing_guide/contribute.md).
