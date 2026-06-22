# 编译选项

## bishengir-compile编译选项

### BiShengIR Feature Control Options

| 选项名 | 描述 | 类型 | 默认值 | 状态 |
|--------|------|------|--------|--------|
| --enable-triton-kernel-compile | Enable Triton kernel compilation. | bool | false | In use  |
| --enable-torch-compile| Enable Torch-MLIR compilation. | bool | false（仅当 BISHENGIR_ENABLE_TORCH_CONVERSIONS 时存在） | In use  |
| --enable-hivm-compile | Enable BiShengHIR HIVM compilation. | bool | true | In use  |
| --enable-hfusion-compile | Enable BiShengHIR HFusion compilation. | bool | false | In use  |
| --enable-symbol-analysis | Enable symbol analysis. | bool | false | In use  |
| --enable-multi-kernel | When disabled, graph must fuse as single kernel; when enabled, outline multiple kernels. | bool | false | In use  |
| --enable-manage-host-resources | Enable managing resource for Host functions. | bool | false | In use  |
| --ensure-no-implicit-broadcast | Whether to ensure that there is no implicit broadcast semantics. If there is a dynamic to dynamic dim broadcast, raise a runtime error. | bool | false（仅当 BISHENGIR_ENABLE_TORCH_CONVERSIONS 时存在） | In use  |
| --disable-auto-inject-block-sync | Disable generating blocksync wait/set by injectBlockSync pass. | bool | false | In use  |
| --enable-hivm-graph-sync-solver | Use hivm graph-sync-solver instead of inject-sync. | bool | false | In use  |
| --disable-auto-cv-work-space-manage | In combination with the disableAutoInjectBlockSync option. | bool | false | In use  |
| --disable-hivm-auto-inject-sync | Disable auto inject sync intra core. | bool | false | In use  |
| --disable-hivm-tensor-compile | Disable BiShengHIR HIVM Tensor compilation. | bool | false | In use  |

### BiShengIR General Optimization Options

| 选项名 | 描述 | 类型 | 默认值 | 状态 |
|--------|------|------|--------|--------|
| --enable-auto-multi-buffer | Enable auto multi buffer. | bool | false | In use  |
| --limit-auto-multi-buffer-only-for-local-buffer | When enable-auto-multi-buffer = true, limit it only to work for local buffer | bool | true | In use  |
| --enable-tuning-mode | Enable tuning mode and will not try compile multi times in case of plan memory failure | bool | false | In use  |
| --block-dim=\<uint> | Number of blocks to use | unsigned | 1 | In use  |

### BiShengIR HFusion Optimization Options

| 选项名 | 描述 | 类型 | 默认值 | 状态 |
|--------|------|------|--------|--------|
| --enable-deterministic-computing | If enabled, the computation result is deterministic. If disabled, we will enable extra optimizations that might boost performance, e.g. bind reduce to multiple cores. However, the result will be non-deterministic. | bool | true | In use  |
| --enable-ops-reorder | Enable ops reorder to opt pipeline. | bool | true | In use  |
| --hfusion-max-horizontal-fusion-size=\<int> | Number of horizontal fusion attempt (Default: unlimited). | int32_t | -1 | In use  |
| --hfusion-max-buffer-count-tuning=\<long>  | Max buffer count tuning in HFusion auto schedule. | int64_t | 0 | In use  |
| --cube-tiling-tuning=\<long> | Cube block size tuning in HFusion auto schedule | list int64_t | "" | In use  |
| --enable-hfusion-count-buffer-dma-opt | If enabled, the buffer used by DMA operations will not be reused by Vector operations. | bool | false | In use  |

### BiShengIR Target Options

编译选项名 `--target=Ascend<Name>`，用于指定将MLIR编译至的平台，其中`<Name>`为占位标识，具体内容需结合AI处理器型号，通过对应查询命令获取。

AI处理器型号及对应查询方式如下：

**方式一：通过 `npu-smi info` 命令查询**

**适用产品**：

- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas 200I/500 A2 推理产品
- Atlas 推理系列产品
- Atlas 训练系列产品

在安装AI处理器的服务器执行命令，查询获取`<Name>`对应内容，完整配置值为`Ascend<Name>`。示例：若`<Name>`取值为`xxx`，则配置值为`Ascendxxx`。

**方式二：通过 `npu-smi info -t board -i id -c chip_id` 命令查询**

**适用产品**：

- Atlas 350 加速卡
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

在安装AI处理器的服务器执行命令，获取**Chip Name**和**NPU Name**信息，实际配置值为`<Chip Name>_<NPU Name>`。示例：若Chip Name为`Ascendxxx`、NPU Name为`yyy`，则配置值为`Ascendxxx_yyy`。

命令参数说明：

- `id`：设备id，通过`npu-smi info -l`命令查出的NPU ID即为设备id。
- `chip_id`：芯片id，通过`npu-smi info -m`命令查出的Chip ID即为芯片id。
