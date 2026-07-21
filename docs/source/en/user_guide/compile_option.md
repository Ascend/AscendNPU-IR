# Compile Options

## bishengir-compile Options

### BiShengIR Feature Control Options

| Option | Description | Type| Default| Status|
|--------|------|------|--------|--------|
| --enable-triton-kernel-compile | Enable Triton kernel compilation. | bool | false | In use  |
| --enable-torch-compile| Enable Torch-MLIR compilation. | bool | false (when BISHENGIR_ENABLE_TORCH_CONVERSIONS)| In use  |
| --enable-hivm-compile | Enable BiShengHIR HIVM compilation. | bool | true | In use  |
| --enable-hfusion-compile | Enable BiShengHIR HFusion compilation. | bool | false | In use  |
| --enable-symbol-analysis | Enable symbol analysis. | bool | false | In use  |
| --enable-multi-kernel | When disabled, graph must fuse as single kernel; when enabled, outline multiple kernels. | bool | false | In use  |
| --enable-manage-host-resources | Enable managing resource for Host functions. | bool | false | In use  |
| --ensure-no-implicit-broadcast | Whether to ensure that there is no implicit broadcast semantics. If there is a dynamic to dynamic dim broadcast, raise a runtime error. | bool | false (when BISHENGIR_ENABLE_TORCH_CONVERSIONS)| In use  |
| --disable-auto-inject-block-sync | Disable generating blocksync wait/set by injectBlockSync pass. | bool | false | In use  |
| --enable-hivm-graph-sync-solver | Use hivm graph-sync-solver instead of inject-sync. | bool | false | In use  |
| --disable-auto-cv-work-space-manage | In combination with the disableAutoInjectBlockSync option. | bool | false | In use  |
| --disable-hivm-auto-inject-sync | Disable auto inject sync intra core. | bool | false | In use  |
| --disable-hivm-tensor-compile | Disable BiShengHIR HIVM Tensor compilation. | bool | false | In use  |

### BiShengIR General Optimization Options

| Option | Description | Type| Default| Status|
|--------|------|------|--------|--------|
| --enable-auto-multi-buffer | Enable auto multi buffer. | bool | false | In use  |
| --limit-auto-multi-buffer-only-for-local-buffer | When enable-auto-multi-buffer = true, limit it only to work for local buffer | bool | true | In use  |
| --enable-tuning-mode | Enable tuning mode and will not try compile multi times in case of plan memory failure | bool | false | In use  |
| --block-dim=\<uint> | Number of blocks to use | unsigned | 1 | In use  |

### BiShengIR HFusion Optimization Options

| Option | Description | Type| Default| Status|
|--------|------|------|--------|--------|
| --enable-deterministic-computing | If enabled, the computation result is deterministic. If disabled, we will enable extra optimizations that might boost performance, e.g. bind reduce to multiple cores. However, the result will be non-deterministic. | bool | true | In use  |
| --enable-ops-reorder | Enable ops reorder to opt pipeline. | bool | true | In use  |
| --hfusion-max-horizontal-fusion-size=\<int> | Number of horizontal fusion attempt (Default: unlimited). | int32_t | -1 | In use  |
| --hfusion-max-buffer-count-tuning=\<long>  | Max buffer count tuning in HFusion auto schedule. | int64_t | 0 | In use  |
| --cube-tiling-tuning=\<long> | Cube block size tuning in HFusion auto schedule | list int64_t | "" | In use  |
| --enable-hfusion-count-buffer-dma-opt | If enabled, the buffer used by DMA operations will not be reused by Vector operations. | bool | false | In use  |

### BiShengIR Target Options

The compilation option `--target=Ascend<Name>` is used to specify the target platform for MLIR compilation. `<Name>` is a placeholder, and its specific content varies by AI processor model, which can be acquired through dedicated query commands.

The AI processor models and corresponding query methods are described below:

**Method 1: Query with the command `npu-smi info`**

**Applicable Products**

- Atlas A2 training products/Atlas A2 inference products
- Atlas 200I/500 A2 inference products
- Atlas inference products
- Atlas training products

Run this command on the server equipped with the AI processor to get the value of `<Name>`. The full configuration value is `Ascend<Name>`. Example: If `<Name>` is `xxx`, the configuration value is `Ascendxxx`.

**Method 2: Query with the command `npu-smi info -t board -i id -c chip_id`**

**Applicable Products**

- Atlas 350 accelerator card
- Atlas A3 training products and Atlas A3 inference products

Execute this command on the server with the AI processor to obtain **Chip Name** and **NPU Name**. The actual configuration value is formatted as `<Chip Name>_<NPU Name>`. Example: If **Chip Name** is `Ascendxxx` and the **NPU Name** is `yyy`, the configuration value is `Ascendxxx_yyy`.

Command parameter description:

- `id`: Device ID, namely the NPU ID queried by the `npu-smi info -l` command.
- `chip_id`: Chip ID, namely the Chip ID queried by the `npu-smi info -m` command.
