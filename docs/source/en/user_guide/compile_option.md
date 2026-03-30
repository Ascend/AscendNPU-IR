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

| Option          | Value               | Status |
| --------------- | ------------------- | ------ |
| --target=       | Target device name. | In use |
| =Ascend910B1    | Ascend910B1         | In use |
| =Ascend910B2    | Ascend910B2         | In use |
| =Ascend910B3    | Ascend910B3         | In use |
| =Ascend910B4    | Ascend910B4         | In use |
| =Ascend910B4-1  | Ascend910B4-1       | In use |
| =Ascend910B2C   | Ascend910B2C        | In use |
| =Ascend910_9362 | Ascend910_9362      | In use |
| =Ascend910_9372 | Ascend910_9372      | In use |
| =Ascend910_9381 | Ascend910_9381      | In use |
| =Ascend910_9382 | Ascend910_9382      | In use |
| =Ascend910_9391 | Ascend910_9391      | In use |
| =Ascend910_9392 | Ascend910_9392      | In use |
| =Ascend910_950z | Ascend910_950z      | In use |
| =Ascend910_9579 | Ascend910_9579      | In use |
| =Ascend910_957b | Ascend910_957b      | In use |
| =Ascend910_957d | Ascend910_957d      | In use |
| =Ascend910_9581 | Ascend910_9581      | In use |
| =Ascend910_9589 | Ascend910_9589      | In use |
| =Ascend910_958a | Ascend910_958a      | In use |
| =Ascend910_958b | Ascend910_958b      | In use |
| =Ascend910_9599 | Ascend910_9599      | In use |
| =Unknown        | Unknown             | In use |

## bishengir-hivm-compile options

### BiShengIR HIVM Optimization Options

| Option                                     | Description                                                                                                                | Type                | Default   | Status         |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------- | -------------- |
| --limit-auto-multi-buffer-of-local-buffer= | When enable-auto-multi-buffer = true, limit local buffer mode. Value=<no-limit/no-l0c>                                     | MultiBufferStrategy | no-l0c    | In use         |
| --limit-auto-multi-buffer-buffer=          | When enable-auto-multi-buffer = true, limit to only cube, only vector, or no limit. Value=<no-limit/only-cube/only-vector> | MultiBufferStrategy | only-cube | In use         |
| --enable-auto-bind-sub-block               | Enable auto bind sub block.                                                                                                | bool                | true      | In use         |
| --enable-code-motion                       | Enable code-motion/subset-hoist.                                                                                           | bool                | true      | In use         |
| --enable-hivm-unit-flag-sync               | Use unit-flag modes for sync in inject sync pass.                                                                          | bool                | false     | In use         |
| --enable-hivm-assume-alive-loops           | Assume all loops (forOp, whileOp) execute at least once.                                                                   | bool                | false     | In use         |
| --enable-hivm-inject-block-all-sync        | Enable inject all block sync for HIVM inject block sync.                                                                   | bool                | false     | In use         |
| --enable-hivm-inject-barrier-all-sync      | Enable barrier-all mode for HIVM inject sync.                                                                              | bool                | false     | In use         |
| --set-workspace-multibuffer=               | Override number of multibuffers for workspace (default 2).                                                                 | unsigned            | 2         | In use         |
| --enable-hivm-global-workspace-reuse       | Enable global workspace reuse.                                                                                             | bool                | false     | In use         |
| --enable-hivm-auto-cv-balance              | Enable balancing during cv-pipelining.                                                                                     | bool                | false     | Deprecated     |
| --enable-hivm-auto-storage-align           | Enable mark/enable storage align.                                                                                          | bool                | true      | In use         |
| --enable-hivm-nd2nz-on-vector              | Enable nd2nz on vector.                                                                                                    | bool                | false     | In development |
| --enable-auto-blockify-loop                | Enable auto loop on blocks for all parallel.                                                                               | bool                | false     | In use         |
| --tile-mix-vector-loop=                    | Trip count of tiled vector loop for mix kernels.                                                                           | unsigned            | 1         | In use         |
| --tile-mix-cube-loop=                      | Trip count of tiled cube loop for mix kernels.                                                                             | unsigned            | 1         | In use         |

### Options shared with bishengir-hivm-compile

| Option                                | Description                                                   | Type | Default | Status |
| ------------------------------------- | ------------------------------------------------------------- | ---- | ------- | ------ |
| --enable-static-bare-ptr              | Enable bare ptr calling convention for static-shaped kernels. | bool | true    | In use |
| --enable-bin-relocation               | Enable binary relocation.                                     | bool | true    | In use |
| --enable-lir-compile                  | Enable BiShengLIR compilation.                                | bool | true    | In use |
| --enable-sanitizer                    | Enable ascend sanitizer.                                      | bool | false   | In use |
| --enable-debug-info                   | Enable debug info.                                            | bool | false   | In use |
| --enable-hivm-inject-barrier-all-sync | Enable barrier-all mode for HIVM inject sync.                 | bool | false   | In use |
