# 编译选项

## 1 bishengir-compile编译选项
### 1.1 Feature Control Options（FeatureControlOptions，未设置 shared bit）

| 选项名 | 描述 | 类型 | 默认值 |
|--------|------|------|--------|
| EnableTritonKernelCompile | Enable Triton kernel compilation. | bool | false |
| EnableTorchCompile | Enable Torch-MLIR compilation. | bool | false（仅当 BISHENGIR_ENABLE_TORCH_CONVERSIONS 时存在） |
| EnableHIVMCompile | Enable BiShengHIR HIVM compilation. | bool | true |
| EnableHfusionCompile | Enable BiShengHIR HFusion compilation. | bool | false |
| EnableSymbolAnalysis | Enable symbol analysis. | bool | false |
| EnableMultiKernel | When disabled, graph must fuse as single kernel; when enabled, outline multiple kernels. | bool | false |
| EnableManageHostResources | Enable managing resource for Host functions. | bool | false |
| ConvertHIRToLIR | Enable converting from BiShengHIR to BiShengLIR. | bool | false（hidden） |
| EnsureNoImplicitBroadcast | Whether to ensure that there is no implicit broadcast semantics. If there is a dynamic to dynamic dim broadcast, raise a runtime error. | bool | false（仅当 BISHENGIR_ENABLE_TORCH_CONVERSIONS 时存在） |
| DisableAutoInjectBlockSync | Disable generating blocksync wait/set by injectBlockSync pass. | bool | false |
| EnableHIVMGraphSyncSolver | Use hivm graph-sync-solver instead of inject-sync. | bool | false |
| DisableAutoCVWorkSpaceManage | In combination with the disableAutoInjectBlockSync option. | bool | false |
| DisableHIVMAutoInjectSync | Disable auto inject sync intra core. | bool | false |
| DisableHIVMTensorCompile | Disable BiShengHIR HIVM Tensor compilation. | bool | false |

### 1.2 General Optimization Control Options

| 选项名 | 描述 | 类型 | 默认值 |
|--------|------|------|--------|
| EnableAutoMultiBuffer | Enable auto multi buffer. | bool | false |
| LimitAutoMultiBufferOnlyForLocalBuffer | When enable-auto-multi-buffer = true, limit it only to work for local buffer | bool | true |
| EnableTuningMode | Enable tuning mode and will not try compile multi times in case of plan memory failure | bool | false |
| BlockDim | Number of blocks to use | unsigned | 1 |

### 1.3 HFusion Optimization Control Options

| 选项名 | 描述 | 类型 | 默认值 |
|--------|------|------|--------|
| EnableDeterministicComputing | If enabled, the computation result is deterministic. If disabled, we will enable extra optimizations that might boost performance, e.g. bind reduce to multiple cores. However, the result will be non-deterministic. | bool | true |
| EnableOpsReorder | Enable ops reorder to opt pipeline. | bool | true |
| HfusionMaxHorizontalFusionSize | Number of horizontal fusion attempt (Default: unlimited). | int32_t | -1 |
| HfusionMaxBufferCountTuning | Max buffer count tuning in HFusion auto schedule. | int64_t | 0 |
| CubeTilingTuning | Cube block size tuning in HFusion auto schedule | list int64_t | "" |
| enableHfusionCountBufferDmaOpt | If enabled, the buffer used by DMA operations will not be reused by Vector operations. | bool | false |

### 1.4 HIVM Optimization Control Options（除 EnableHIVMInjectBarrierAllSync 外均不传递）

| 选项名 | 描述 | 类型 | 默认值 |
|--------|------|------|--------|
| LimitAutoMultiBufferOfLocalBuffer | When enable-auto-multi-buffer = true, limit local buffer mode. | MultiBufferStrategy | CUBE_NO_L0C |
| LimitAutoMultiBufferBuffer | When enable-auto-multi-buffer = true, limit it to only cube, only vector or no limit. | MultiBufferStrategy | ONLY_CUBE |
| EnableAutoBindSubBlock | Enable auto bind sub block. | bool | true |
| EnableCodeMotion | Enable code-motion/subset-hoist. | bool | true |
| EnableHIVMUnitFlagSync | Enable inject sync pass to use unit-flag modes for synchronization. | bool | false |
| EnableHIVMAssumeAliveLoops | Assume that all loops (forOp whileOp) will execute at least once. | bool | false |
| EnableHIVMInjectBlockAllSync | Enable inject all block sync for HIVM inject block sync. | bool | false |
| SetWorkspaceMultibuffer | Override number of multibuffers for workspace, defaults to 2. | unsigned | 2 |
| EnableHIVMGlobalWorkspaceReuse | Enable global workspace reuse. | bool | false |
| EnableHIVMAutoCVBalance | Enable balancing during cv-pipelining. | bool | false |
| EnableHIVMAutoStorageAlign | Enable mark/enable storage align. | bool | true |
| EnableHivmNd2nzOnVector | Enable nd2nz on vector. | bool | false |
| EnableAutoBlockifyLoop | Enable auto loop on blocks for all parallel. | bool | false |
| TileMixVectorLoop | The trip count of the tiled vector loop for mix kernels. | unsigned | 1 |
| TileMixCubeLoop | The trip count of the tiled cube loop for mix kernels. | unsigned | 1 |


## 2 bishengir-hivm-compile编译选项
| 选项名 | 描述 | 类型 | 默认值 | 
|--------|------|------|--------|----------|
| **EnableStaticBarePtr** | Enable generating bare ptr calling convention for static shaped kernels. | bool | true | 
| **EnableBinRelocation** | Enable binary relocation. | bool | true |
| **EnableLIRCompile** | Enable BiShengLIR compilation. | bool | true |
| **EnableSanitizer** | Enable ascend sanitizer. | bool | false |
| **EnableDebugInfo** | Enable debug info. | bool | false | 
| **EnableCpuTraceIntrinsic** | Enable to generate host-accepted IR by eliminating HIVM special traits. | bool | false |
| **EnableHIVMInjectBarrierAllSync** | Enable barrier all mode for HIVM inject sync. | bool | false | 
| **MlirPrintIrAfterAll** | Print IR after each pass. | bool | false | 
| **MlirPrintIrBeforeAll** | Print IR before each pass. | bool | false | 

