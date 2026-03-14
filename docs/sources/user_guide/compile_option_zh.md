# 编译选项

## bishengir-compile编译选项
### BiShengIR Feature Control Options:

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


### BiShengIR General Optimization Options:

| 选项名 | 描述 | 类型 | 默认值 | 状态 |
|--------|------|------|--------|--------|
| --enable-auto-multi-buffer | Enable auto multi buffer. | bool | false | In use  |
| --limit-auto-multi-buffer-only-for-local-buffer | When enable-auto-multi-buffer = true, limit it only to work for local buffer | bool | true | In use  |
| --enable-tuning-mode | Enable tuning mode and will not try compile multi times in case of plan memory failure | bool | false | In use  |
| --block-dim=<uint> | Number of blocks to use | unsigned | 1 | In use  |

### BiShengIR HFusion Optimization Options:
| 选项名 | 描述 | 类型 | 默认值 | 状态 |
|--------|------|------|--------|--------|
| --enable-deterministic-computing | If enabled, the computation result is deterministic. If disabled, we will enable extra optimizations that might boost performance, e.g. bind reduce to multiple cores. However, the result will be non-deterministic. | bool | true | In use  |
| --enable-ops-reorder | Enable ops reorder to opt pipeline. | bool | true | In use  |
| --hfusion-max-horizontal-fusion-size=<int> | Number of horizontal fusion attempt (Default: unlimited). | int32_t | -1 | In use  |
| --hfusion-max-buffer-count-tuning=<long>  | Max buffer count tuning in HFusion auto schedule. | int64_t | 0 | In use  |
| --cube-tiling-tuning=<long> | Cube block size tuning in HFusion auto schedule | list int64_t | "" | In use  |
| --enable-hfusion-count-buffer-dma-opt | If enabled, the buffer used by DMA operations will not be reused by Vector operations. | bool | false | In use  |

### BiShengIR Target Options:
| 选项名 | 数值 | 状态 |
|--------|------|------|
| --target=<value>  |Target device name. | In use  |
| =Ascend910B1 | Ascend910B1 | In use  |
| =Ascend910B2 | Ascend910B2 | In use  |
| =Ascend910B3 | Ascend910B3 | In use  |
| =Ascend910B4 | Ascend910B4 | In use  |
| =Ascend910B4-1 | Ascend910B4-1 | In use  |
| =Ascend910B2C | Ascend910B2C | In use  |
| =Ascend910_9362 | Ascend910_9362 | In use  |
| =Ascend910_9372 | Ascend910_9372 | In use  |
| =Ascend910_9381 | Ascend910_9381 | In use  |
| =Ascend910_9382 | Ascend910_9382 | In use  |
| =Ascend910_9391 | Ascend910_9391 | In use  |
| =Ascend910_9392 | Ascend910_9392 | In use  |
| =Ascend910_950z | Ascend910_950z | In use  |
| =Ascend910_9579 | Ascend910_9579 | In use  |
| =Ascend910_957b | Ascend910_957b | In use  |
| =Ascend910_957d | Ascend910_957d | In use  |
| =Ascend910_9581 | Ascend910_9581 | In use  |
| =Ascend910_9589 | Ascend910_9589 | In use  |
| =Ascend910_958a | Ascend910_958a | In use  |
| =Ascend910_958b | Ascend910_958b | In use  |
| =Ascend910_9599 | Ascend910_9599 | In use  |
| =Unknown | Unknown | In use  |


## bishengir-hivm-compile编译选项
### BiShengIR HIVM Optimization Options
| 选项名 | 描述 | 类型 | 默认值 | 状态 |
|--------|------|------|--------|--------|
| --limit-auto-multi-buffer-of-local-buffer=<value> | When enable-auto-multi-buffer = true, limit local buffer mode. Value=<no-limit/no-l0c> | MultiBufferStrategy  | no-l0c  | In use  |
| --limit-auto-multi-buffer-buffer=<value> | When enable-auto-multi-buffer = true, limit it to only cube, only vector or no limit.Value=<no-limit/only-cube/only-vector> | MultiBufferStrategy  | only-cube | In use  |
| --enable-auto-bind-sub-block | Enable auto bind sub block. | bool | true | In use  |
| --enable-code-motion | Enable code-motion/subset-hoist. | bool | true | In use  |
| --enable-hivm-unit-flag-sync | Enable inject sync pass to use unit-flag modes for synchronization. | bool | false | In use  |
| --enable-hivm-assume-alive-loops | Assume that all loops (forOp whileOp) will execute at least once. | bool | false | In use |
| --enable-hivm-inject-block-all-sync | Enable inject all block sync for HIVM inject block sync. | bool | false | In use  |
| --enable-hivm-inject-barrier-all-sync | Enable barrier all mode for HIVM inject sync. | bool | false | In use  |
| --set-workspace-multibuffer=<uint> | Override number of multibuffers for workspace, defaults to 2. | unsigned | 2 | In use  |
| --enable-hivm-global-workspace-reuse | Enable global workspace reuse. | bool | false | In use  |
| --enable-hivm-auto-cv-balance  | Enable balancing during cv-pipelining. | bool | false | Deprecated  |
| --enable-hivm-auto-storage-align | Enable mark/enable storage align. | bool | true | In use |
| --enable-hivm-nd2nz-on-vector | Enable nd2nz on vector. | bool | false | In development |
| --enable-auto-blockify-loop  | Enable auto loop on blocks for all parallel. | bool | false | In use |
| --tile-mix-vector-loop=<uint> | The trip count of the tiled vector loop for mix kernels. | unsigned | 1 | In use |
| --tile-mix-cube-loop=<uint> | The trip count of the tiled cube loop for mix kernels. | unsigned | 1 | In use |



### Options Shared with bishengir-hivm-compile:
| 选项名 | 描述 | 类型 | 默认值 | 状态 |
|--------|------|------|--------|----------|
| --enable-static-bare-ptr | Enable generating bare ptr calling convention for static shaped kernels. | bool | true | In use |
| --enable-bin-relocation | Enable binary relocation. | bool | true | In use |
| --enable-lir-compile | Enable BiShengLIR compilation. | bool | true | In use |
| --enable-sanitizer | Enable ascend sanitizer. | bool | false | In use |
| --enable-debug-info| Enable debug info. | bool | false | In use |
| --enable-hivm-inject-barrier-all-sync  | Enable barrier all mode for HIVM inject sync. | bool | false | In use |

