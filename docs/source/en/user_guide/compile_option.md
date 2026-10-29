# Compilation Options

The BiShengIR compiler controls the compilation process, feature switches, optimization strategies, and target hardware adaptation through a series of command-line compilation options. The following sections describe the specific configuration rules and parameter meanings of each option by category.

## BiShengIR Feature Control Options

**Applicable products**:

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 training products/Atlas A2 inference products
<!-- end id2 -->
<!-- npu="A3" id3 -->
- Atlas A3 training products/Atlas A3 inference products
<!-- end id3 -->

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --enable-triton-kernel-compile | Enables Triton kernel compilation. | bool | false |
| --enable-torch-compile | Enables Torch-MLIR compilation (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products). | bool | false (this option is provided only when the `BISHENGIR_ENABLE_TORCH_CONVERSIONS` compilation macro is defined) |
| --enable-hivm-compile | Enables BiShengHIR HIVM compilation. | bool | true |
| --enable-hfusion-compile | Enables BiShengHIR HFusion compilation. | bool | false |
| --enable-symbol-analysis | Enables symbol analysis (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products). | bool | false |
| --enable-multi-kernel | When disabled, the computation graph must be fused into a single kernel; when enabled, supports hoisting to generate multiple kernels (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products). | bool | false |
| --enable-manage-host-resources | Enables resource management for host functions (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products). | bool | false |
| --ensure-no-implicit-broadcast | Determines whether to ensure that no implicit broadcast semantics exist. If a broadcast from a dynamic dimension to a dynamic dimension exists, a runtime error is triggered (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products). | bool | false (this option is provided only when the `BISHENGIR_ENABLE_TORCH_CONVERSIONS` compilation macro is defined) |
| --enable-hivm-graph-sync-solver | Uses the HIVM computation graph synchronization solver instead of the synchronization injection mechanism. | bool | true |
| --disable-auto-cv-work-space-manage | Must be used together with the `disableAutoInjectBlockSync` option (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products). | bool | false |
| --disable-hivm-auto-inject-sync | Disables automatic intra-core synchronization injection. | bool | false |
| --disable-hivm-tensor-compile | Disables BiShengHIR HIVM tensor compilation (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products). | bool | false |

## BiShengIR General Optimization Options

**Applicable products**:

<!-- npu="950" id4 -->
- Ascend 950PR/Ascend 950DT
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 training products/Atlas A2 inference products
<!-- end id5 -->
<!-- npu="A3" id6 -->
- Atlas A3 training products/Atlas A3 inference products
<!-- end id6 -->

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --enable-auto-multi-buffer | Enables automatic multi-buffering optimization. For details, see [Multi-buffering](../developer_guide/features/multi_buffer.md) | bool | true |
| --limit-auto-multi-buffer-only-for-local-buffer | After enabling automatic multi-buffering, restricts this optimization to on-chip local buffers (UB, L1, L0C) only, that is, skips multi-buffering of the GM Workspace. This takes effect only for functions whose core type is MIX | bool | false |
| --enable-tuning-mode | Enables tuning mode. When memory planning fails, compilation is not retried multiple times | bool | false |
| --block-dim=\<uint> | Specifies the number of blocks to use | unsigned | 1 |

## BiShengIR HFusion Optimization Options

**Applicable products**:

<!-- npu="950" id7 -->
- Ascend 950PR/Ascend 950DT
<!-- end id7 -->
<!-- npu="910b" id8 -->
- Atlas A2 training products/Atlas A2 inference products
<!-- end id8 -->
<!-- npu="A3" id9 -->
- Atlas A3 training products/Atlas A3 inference products
<!-- end id9 -->

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --enable-deterministic-computing | When enabled, the computation results are deterministic; when disabled, additional optimizations are enabled to improve performance (for example, binding reduction operations to multiple cores for execution), but the computation results are not deterministic (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products) | bool | true |
| --enable-ops-reorder | Enables operator reordering optimization in the optimization pipeline | bool | true |
| --hfusion-max-horizontal-fusion-size=\<int> | Maximum number of attempts for horizontal fusion (unlimited by default) (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products) | int32_t | -1 |
| --hfusion-max-buffer-count-tuning=\<long> | Maximum buffer count tuning threshold in HFusion automatic scheduling (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products) | int64_t | 0 |
| --cube-tiling-tuning=\<long> | Cube tiling size tuning parameter in HFusion automatic scheduling (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products) | list int64_t | "" |
| --enable-hfusion-count-buffer-dma-opt | When enabled, the buffers used by DMA operations are not reused by vector operations (only supports Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products) | bool | false |

## BiShengIR HIVM Optimization Options

**Applicable products**:

<!-- npu="950" id10 -->
- Ascend 950PR/Ascend 950DT
<!-- end id10 -->
<!-- npu="910b" id11 -->
- Atlas A2 training products/Atlas A2 inference products
<!-- end id11 -->
<!-- npu="A3" id12 -->
- Atlas A3 training products/Atlas A3 inference products
<!-- end id12 -->

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --limit-auto-multi-buffer-of-local-buffer=\<value> | After enabling automatic multi-buffering, limits the multi-buffering scope of the local buffer. `no-l0c` means multi-buffering is not enabled for L0C; `no-limit` means no restriction | enum | no-l0c |
| --limit-auto-multi-buffer-buffer=\<value> | After enabling automatic multi-buffering, limits the side on which multi-buffering takes effect in MIX operators. `only-cube` means only the Cube side (L1, L0C); `only-vector` means only the Vector side (UB); `no-limit` means no restriction | enum | `no-limit` for Ascend 950PR/Ascend 950DT; `only-cube` for Atlas A3 training products/Atlas A3 inference products and Atlas A2 training products/Atlas A2 inference products |
| --set-workspace-multibuffer=\<uint> | Specifies the number of multi-buffers for the GM Workspace | unsigned | 2 for Ascend 950PR/Ascend 950DT; 4 for Atlas A3 training products/Atlas A3 inference products and Atlas A2 training products/Atlas A2 inference products |

## BiShengIR Target Platform Options

The compilation option format is `--target=Ascend<Name>`, which specifies the target hardware platform for MLIR compilation. Here, `<Name>` is a placeholder that must be replaced with the actual value obtained through the corresponding query command based on the actual AI processor model.

The AI processor models and their corresponding query methods are as follows:

<!-- npu="950,A3,910b" id13 -->
**Method 1: Query using the `npu-smi info -t board -i <id> -c <chip_id>` command**

**Applicable products**:

- Ascend 950PR/Ascend 950DT
- Atlas A2 training products/Atlas A2 inference products
- Atlas A3 training products/Atlas A3 inference products

Run this command on the server where the AI processor is installed to obtain the **Chip Name** and **NPU Name** information. The actual configuration value is `<Chip Name>_<NPU Name>`. For example, if the Chip Name is `Ascendxxx` and the NPU Name is `yyy`, the configuration value is `Ascendxxx_yyy`.

Command parameter description:

- `id`: Device ID. The `NPU ID` obtained through the `npu-smi info -l` command is the device ID.
- `chip_id`: Chip ID. The `Chip ID` obtained through the `npu-smi info -m` command is the chip ID.
<!-- end id13 -->

<!-- npu="910b" id14 -->
**Method 2: Query through the `npu-smi info` command**

**Applicable products**:

- Atlas A2 training products/Atlas A2 inference products

Run this command on the server where the AI processor is installed to obtain the corresponding value of `<Name>`. The complete configuration value is `Ascend<Name>`. Example: If the value of `<Name>` is `xxx`, the configuration value is `Ascendxxx`.
<!-- end id14 -->
