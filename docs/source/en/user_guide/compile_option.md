# Compilation Options

The BiShengIR compiler controls the compilation process, feature switches, optimization strategies, and target hardware adaptation through a series of command-line compilation options. The following sections describe the specific configuration rules and parameter meanings of each option by category.

## BiShengIR Compilation Function Control Options

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --enable-triton-kernel-compile | Enables the Triton kernel compilation function. | bool | false |
| --enable-torch-compile | Enables the Torch-MLIR compilation function. | bool | false (this option is provided only when the `BISHENGIR_ENABLE_TORCH_CONVERSIONS` compilation macro is defined) |
| --enable-hivm-compile | Enables the BiShengHIR HIVM compilation function. | bool | true |
| --enable-hfusion-compile | Enables the BiShengHIR HFusion compilation function. | bool | false |
| --enable-symbol-analysis | Enables the symbol analysis function. | bool | false |
| --enable-multi-kernel | When disabled, the computation graph must be fused into a single kernel; when enabled, supports outlining to generate multiple kernels. | bool | false |
| --enable-manage-host-resources | Enables the resource management function for host functions. | bool | false |
| --ensure-no-implicit-broadcast | Whether to ensure that no implicit broadcast semantics exist. If a broadcast from a dynamic dimension to a dynamic dimension exists, a runtime error is triggered. | bool | false (this option is provided only when the `BISHENGIR_ENABLE_TORCH_CONVERSIONS` compilation macro is defined) |
| --disable-auto-inject-block-sync | Disables the `injectBlockSync` pass from automatically generating block synchronization `wait/set` instructions. | bool | false |
| --enable-hivm-graph-sync-solver | Uses the HIVM computation graph synchronization solver to replace the synchronization injection mechanism. | bool | false |
| --disable-auto-cv-work-space-manage | Must be used together with the `disableAutoInjectBlockSync` option. | bool | false |
| --disable-hivm-auto-inject-sync | Disables automatic injection of synchronization operations within the kernel. | bool | false |
| --disable-hivm-tensor-compile | Disables the BiShengHIR HIVM tensor compilation function. | bool | false |

## BiShengIR General Optimization Options

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --enable-auto-multi-buffer | Enables automatic multi-buffering optimization. For details, see [multi-buffering](../developer_guide/features/multi_buffer.md). | bool | true |
| --limit-auto-multi-buffer-only-for-local-buffer | After auto multi-buffering is enabled, restricts this optimization to on-chip local buffers (UB, L1, L0C) only, that is, skips multi-buffering of the GM Workspace. Takes effect only for functions whose core type is MIX. | bool | false |
| --enable-tuning-mode | Enables tuning mode. When memory planning fails, compilation is not retried multiple times. | bool | false |
| --block-dim=\<uint> | Specifies the number of blocks to use. | unsigned | 1 |

## BiShengIR HFusion Optimization Options

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --enable-deterministic-computing | When enabled, the computation result is deterministic. When disabled, additional optimizations are enabled to improve performance (for example, binding reduction operations to multiple cores for execution), but the computation result is no longer deterministic. | bool | true |
| --enable-ops-reorder | Enables operator reordering optimization in the optimization pipeline. | bool | true |
| --hfusion-max-horizontal-fusion-size=\<int> | Maximum number of attempts for horizontal fusion (unlimited by default). | int32_t | -1 |
| --hfusion-max-buffer-count-tuning=\<long> | Tuning threshold for the maximum number of buffers in HFusion automatic scheduling. | int64_t | 0 |
| --cube-tiling-tuning=\<long> | Tuning parameter for the Cube tiling size in HFusion automatic scheduling. | list int64_t | "" |
| --enable-hfusion-count-buffer-dma-opt | When enabled, buffers used by DMA operations are not reused by vector operations. | bool | false |

## BiShengIR HIVM Optimization Options

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --limit-auto-multi-buffer-of-local-buffer=\<value> | After enabling auto multi-buffering, limits the multi-buffering scope of local buffers. `no-l0c` means that multi-buffering is not enabled for L0C; `no-limit` means no restriction. | enum | no-l0c |
| --limit-auto-multi-buffer-buffer=\<value> | After enabling auto multi-buffering, limits the side on which multi-buffering takes effect in MIX operators. `only-cube` means only the Cube side (L1, L0C); `only-vector` means only the Vector side (UB); `no-limit` means no restriction. | enum | `no-limit` for Ascend 950PR/Ascend 950DT; `only-cube` for Atlas A3 series products and Atlas A2 series products |
| --set-workspace-multibuffer=\<uint> | Specifies the number of multi-buffers for GM Workspace. | unsigned | 2 for Ascend 950PR/Ascend 950DT; 4 for Atlas A3 series products and Atlas A2 series products |

## BiShengIR Target Platform Options

The compilation option format is `--target=Ascend<Name>`, which specifies the target hardware platform for MLIR compilation. Here, `<Name>` is a placeholder that must be replaced with the actual value obtained through the corresponding query command based on the actual AI processor model.

The AI processor models and their corresponding query methods are as follows:

**Method 1: Query using the `npu-smi info -t board -i <id> -c <chip_id>` command**

**Applicable products**:

- Ascend 950PR/Ascend 950DT
- Atlas A3 training products/Atlas A3 inference products

Run this command on the server where the AI processor is installed to obtain the **Chip Name** and **NPU Name** information. The actual configuration value is `<Chip Name>_<NPU Name>`. For example, if the Chip Name is `Ascendxxx` and the NPU Name is `yyy`, the configuration value is `Ascendxxx_yyy`.

Command parameter description:

- `id`: Device ID. `NPU ID` obtained through the `npu-smi info -l` command is the device ID.
- `chip_id`: Chip ID. `hip ID` obtained through the `npu-smi info -m` command is the chip ID.

**Method 2: Query through the `npu-smi info` command**

**Applicable Products**:

- Atlas A2 training products/Atlas A2 inference products

Run this command on a server where the AI processor is installed to obtain the corresponding value of `<Name>`. The complete configuration value is `Ascend<Name>`. Example: If the value of `<Name>` is `xxx`, the configuration value is `Ascendxxx`.
