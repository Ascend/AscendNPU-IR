# Compile options

## bishengir-compile options

### BiShengIR Feature Control Options

| Option                              | Description                                                                              | Type | Default                                         | Status |
| ----------------------------------- | ---------------------------------------------------------------------------------------- | ---- | ----------------------------------------------- | ------ |
| --enable-triton-kernel-compile      | Enable Triton kernel compilation.                                                        | bool | false                                           | In use |
| --enable-torch-compile              | Enable Torch-MLIR compilation.                                                           | bool | false (when BISHENGIR_ENABLE_TORCH_CONVERSIONS) | In use |
| --enable-hivm-compile               | Enable BiShengHIR HIVM compilation.                                                      | bool | true                                            | In use |
| --enable-hfusion-compile            | Enable BiShengHIR HFusion compilation.                                                   | bool | false                                           | In use |
| --enable-symbol-analysis            | Enable symbol analysis.                                                                  | bool | false                                           | In use |
| --enable-multi-kernel               | When disabled, graph must fuse as single kernel; when enabled, outline multiple kernels. | bool | false                                           | In use |
| --enable-manage-host-resources      | Enable managing resource for Host functions.                                             | bool | false                                           | In use |
| --ensure-no-implicit-broadcast      | If set, no implicit broadcast; dynamic-to-dynamic dim broadcast raises a runtime error.  | bool | false (when BISHENGIR_ENABLE_TORCH_CONVERSIONS) | In use |
| --disable-auto-inject-block-sync    | Disable generating blocksync wait/set by injectBlockSync pass.                           | bool | false                                           | In use |
| --enable-hivm-graph-sync-solver     | Use hivm graph-sync-solver instead of inject-sync.                                       | bool | false                                           | In use |
| --disable-auto-cv-work-space-manage | Used with disableAutoInjectBlockSync.                                                    | bool | false                                           | In use |
| --disable-hivm-auto-inject-sync     | Disable auto inject sync intra core.                                                     | bool | false                                           | In use |
| --disable-hivm-tensor-compile       | Disable BiShengHIR HIVM Tensor compilation.                                              | bool | false                                           | In use |

### BiShengIR General Optimization Options

| Option                                          | Description                                                                     | Type     | Default | Status |
| ----------------------------------------------- | ------------------------------------------------------------------------------- | -------- | ------- | ------ |
| --enable-auto-multi-buffer                      | Enable auto multi buffer.                                                       | bool     | false   | In use |
| --limit-auto-multi-buffer-only-for-local-buffer | When enable-auto-multi-buffer = true, limit to local buffer only.               | bool     | true    | In use |
| --enable-tuning-mode                            | Enable tuning mode; do not retry compile multiple times on plan memory failure. | bool     | false   | In use |
| --block-dim=                                    | Number of blocks to use.                                                        | unsigned | 1       | In use |

### BiShengIR HFusion Optimization Options

| Option                                | Description                                                                                                                                               | Type         | Default | Status |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------- | ------ |
| --enable-deterministic-computing      | If enabled, result is deterministic; if disabled, extra optimizations (e.g. bind reduce to multiple cores) may apply but result can be non-deterministic. | bool         | true    | In use |
| --enable-ops-reorder                  | Enable ops reorder in opt pipeline.                                                                                                                       | bool         | true    | In use |
| --hfusion-max-horizontal-fusion-size= | Max horizontal fusion attempts (default: unlimited).                                                                                                      | int32_t      | -1      | In use |
| --hfusion-max-buffer-count-tuning=    | Max buffer count tuning in HFusion auto schedule.                                                                                                         | int64_t      | 0       | In use |
| --cube-tiling-tuning=                 | Cube block size tuning in HFusion auto schedule.                                                                                                          | list int64_t | ""      | In use |
| --enable-hfusion-count-buffer-dma-opt | If enabled, buffer used by DMA is not reused by Vector ops.                                                                                               | bool         | false   | In use |

### BiShengIR Target Options

The compilation option `--target=Ascend<Name>` is used to specify the target platform for MLIR compilation. `<Name>` is a placeholder, and its specific content varies by AI processor model, which can be acquired through dedicated query commands.

The AI processor models and corresponding query methods are described below:

**Method 1: Query with the command `npu-smi info`**

**Applicable Products**:

- Atlas A2 Training Series / Atlas A2 Inference Series
- Atlas 200I/500 A2 Inference Product
- Atlas Inference Series
- Atlas Training Series

Run this command on the server equipped with the AI processor to get the value of `<Name>`. The full configuration value is `Ascend<Name>`.
Example: If `<Name>` is `xxx`, the configuration value is `Ascendxxx`.

**Method 2: Query with the command `npu-smi info -t board -i id -c chip_id`**

**Applicable Products**:

- Atlas 350 Accelerator Card
- Atlas A3 Training Series / Atlas A3 Inference Series

Execute this command on the server with the AI processor to obtain the **Chip Name** and **NPU Name**. The actual configuration value is formatted as `<Chip Name>_<NPU Name>`.
Example: If the Chip Name is `Ascendxxx` and the NPU Name is `yyy`, the configuration value is `Ascendxxx_yyy`.

Command Parameter Description:

- `id`: Device ID, namely the NPU ID queried by the `npu-smi info -l` command.
- `chip_id`: Chip ID, namely the Chip ID queried by the `npu-smi info -m` command.