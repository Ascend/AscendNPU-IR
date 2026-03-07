# FAQ

This page collects common questions about using and developing AscendNPU IR, grouped by topic. For build details see [Install and build](../introduction/quick_start/installing_guide.md); for contribution flow see [Contributing guide](../contributing_guide/contribute.md).

---

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

---

## Running and debugging

**Q2.1** How do I run tests?

From the **build directory**:

- **bishengir tests**: `ninja check-bishengir` or `cmake --build . --target check-bishengir`
- **LIT test suite**: `./bin/llvm-lit ../bishengir/test` (adjust paths to your repo and build dir)

See "Running tests" in [Install and build](../introduction/quick_start/installing_guide.md).

**Q2.2** What is needed to run on device?

You need **CANN** (installed and `set_env.sh` sourced), a device binary built with **bishengir-compile** (e.g. `kernel.o`), and a Host program that uses CANN runtime to register and launch the kernel. See [Quick start example](../introduction/quick_start/examples.md) and [Quick start](../introduction/quick_start/index.md).

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

---

## Performance tuning

**Q3.1** How do I find operator performance bottlenecks?

### MindStudio

Link: https://www.hiascend.com/developer/software/mindstudio  
On Ascend, MindStudio’s Profiler collects runtime metrics to help locate kernel bottlenecks when debugging Triton kernels.

### Torch NPU profiler

`torch_npu.profiler.profile` is the main API for profiling PyTorch training/inference on Ascend. It collects and parses runtime performance data to help locate and fix bottlenecks.

It injects instrumentation to collect CPU and NPU data, including: PyTorch-layer info (ops, memory, call stacks), CANN-layer scheduling and execution, and hardware-layer info (operator time, AI Core metrics such as pipeline utilization, cache hit rate). It bridges your training script and visualization tools (e.g. MindStudio Insight or TensorBoard).

Example:

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

*TODO: Expand with more profiling methods, comparison with reference/competitors, and impact of key passes.*

**Q3.2** How do compile options or optimization passes affect performance?

*TODO: Add common options, performance-related HFusion/HIVM passes, Release vs Debug build differences.*

**Q3.3** How do I enable or disable an optimization (e.g. CVPipeline, AutoSubTiling)?

*TODO: Add bishengir-compile and pass toggles and links to relevant docs.*

---

## Accuracy and debugging

**Q4.1** Results differ from reference (CPU/GPU or reference implementation). How do I debug?

*TODO: Add layer-by-layer comparison, data types and rounding, use of debug options.*

**Q4.2** How do I compare numerical results across MLIR layers or intermediate representations?

*TODO: Add instrumentation/printing, combination with Q2.3, common debug workflow.*

**Q4.3** What are common accuracy issues (e.g. BF16/FP16 loss, reduction order)?

*TODO: Add common cases, mitigations, and links to best practices.*

---

## Contributing and community

**Q5.1** How do I contribute?

You must sign the Ascend Community CLA and follow the [ascend-community](https://gitcode.com/ascend/community) code of conduct. Flow: open or claim an Issue, fork and develop, self-test (e.g. `ninja check-bishengir`), open a PR, and pass CI (build, static check, tests). Merge requires 2 Reviewers’ `/lgtm` and 1 Approver’s `/approve`. See [Contributing guide](../contributing_guide/contribute.md).

**Q5.2** PR fails CI (build, static check, or tests). How do I fix it?

Follow the CI log: fix **build failures** (errors and environment), **static check** (style/logic as indicated), and **failing tests** (fix and re-run CI). See "Dealing with CI failures" in [Contributing guide](../contributing_guide/contribute.md).

**Q5.3** What should I do before opening a PR?

Avoid unrelated changes; keep history clear (squash/rebase if needed); rebase your branch onto the latest upstream master; for bug-fix PRs, reference related Issues and PRs in the description. See "Notes" in [Contributing guide](../contributing_guide/contribute.md).
