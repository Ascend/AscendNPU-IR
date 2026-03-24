# IR Interface Overview

## Multi-level IR Abstraction

- High-level abstract interfaces that hide low-level details and map hardware-agnostic expressions to low-level instructions, improving operator development usability
- Fine-grained performance control interfaces for precise control of on-chip memory addresses, pipeline sync insertion, and ping-pong pipeline optimization
- Multi-level interfaces support flexible integration of custom DSLs and frameworks for high-performance custom operators on Ascend NPU

```
  Torch-MLIR / Triton       (Framework/DSL layer)
         |
         v
  Linalg / Tensor            (General tensor algebra layer)
         |
         v
  HFusion                    (Hardware-aware fusion & scheduling layer)
         |
         v
  HIVM                       (NPU instruction layer)
         |
         v
  LIR -> Binary              (Machine code generation)
```

- **Linalg / Tensor layer**: Standard MLIR dialects for operator semantics (Elemwise, Broadcast, Reduce, Transpose, Concat, etc.); HFusion performs fusion, tiling, and scheduling automatically
- **HFusion layer**: Ascend-NPU-aware named ops (e.g. `hfusion.elemwise_unary`, `hfusion.cast`, `hfusion.select`, `hfusion.reduce_with_index`), tensor semantics, automatic bufferization, tiling, and scheduling
- **HIVM layer**: Direct mapping to NPU instructions, explicit control of memory hierarchy (GM/UB/L1/L0), compute pipelines (Vector/Cube/MTE), and sync primitives for fine-grained tuning

These layers allow custom DSLs and frameworks to integrate. Triton and PyTorch connect via IR conversion for high-performance custom operators on Ascend NPU.

## Compile Options and Function Attributes

### Compile options

`bishengir-compile` provides these common options:

| Option | Default | Description |
|--------|---------|-------------|
| `-target` | `Ascend910B1` | Target device (core count, on-chip memory, etc.) |
| `-block-dim` | `1` | Number of blocks; compiled kernel carries `hacc.block_dim` |
| `-enable-hfusion-compile` | `false` | Enable HFusion pipeline (fusion, scheduling, tiling) |
| `-enable-hivm-compile` | `true` | Enable HIVM pipeline (lower to HIVM and optimize) |
| `-enable-torch-compile` | `false` | Enable Torch-MLIR pipeline |
| `-enable-triton-kernel-compile` | `false` | Enable Triton kernel pipeline |

Supported targets:

- Ascend910B: `Ascend910B1`, `Ascend910B2`, `Ascend910B3`, `Ascend910B4`, etc.
- Ascend910_93: `Ascend910_9362`, `Ascend910_9372`, `Ascend910_9381`, etc.
- Ascend910_95: `Ascend910_950z`, `Ascend910_9579`, `Ascend910_9589`, etc.

### Function attributes

The following attributes mark kernel entry functions and are shared across all integration paths:

| Attribute | Description |
|-----------|-------------|
| `hacc.entry` | Marks the current function as the kernel entry |
| `hacc.function_kind = #hacc.function_kind<DEVICE>` | Function runs on the DEVICE side |
| `hacc.function_kind = #hacc.function_kind<HOST>` | Function runs on the HOST side; HFusion will outline device kernels |

Example:

```
func.func @kernel(...) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  ...
}
```

## Triton integration

Triton is a widely used language for high-performance kernel development. [Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) converts Triton kernels to MLIR for the AscendNPU IR stack and enables running Triton kernels on Ascend NPU. See [Triton interface](triton_interface.md) for installation, environment, op mapping, and Ascend extensions.

## TileLang integration

TileLang (tilelang-ascend) is a domain-specific language for Ascend NPU kernel development, built on tile-lang's Pythonic syntax and [TVM](https://tvm.apache.org/). It supports GEMM, vector operations, and attention mechanisms, compiling kernels to AscendNPU IR (HIVM) for execution on Ascend NPU. See [TileLang interface](tile_lang_interface.md) for installation and usage details.

## Framework integration

AscendNPU IR supports framework integration (PyTorch/TensorFlow/MindSpore) in two ways:

- **DSL integration**: e.g., Triton and TileLang
- **IR integration**: e.g., Torch IR, Linalg/HFusion IR, and HIVM IR

See [Framework interface](framework_interface.md) for details.

---
