# IR Interface Overview

## Multi-level IR Abstraction

- High-level abstract interfaces: hide low-level details and map hardware-agnostic expressions to low-level instructions, improving operator development usability.
- Fine-grained performance control interfaces: precisely control on-chip memory addresses, pipeline sync insertion, and ping-pong pipeline optimization.
- Multi-level interfaces: support flexible integration of custom DSLs and frameworks for high-performance custom operators on Ascend NPU.

```text
  Torch-MLIR/Triton       (Framework/DSL layer)
         |
         v
  Linalg/Tensor           (General tensor algebra layer)
         |
         v
  HFusion                   (Hardware-aware fusion & scheduling layer)
         |
         v
  HIVM                      (NPU instruction layer)
         |
         v
  LIR -> Binary             (Machine code generation)
```

- **Linalg / Tensor layer**: Uses standard MLIR dialects for operator semantics (Elemwise, Broadcast, Reduce, Transpose, Concat, etc.); HFusion performs fusion, tiling, and scheduling automatically.
- **HFusion layer**: Provides Ascend NPU-aware named operators (such as `hfusion.elemwise_unary`, `hfusion.cast`, `hfusion.select` and `hfusion.reduce_with_index`), supports tensor semantics, and automatically performs bufferization, tiling, and scheduling.
- **HIVM layer**: Directly maps to NPU instructions, explicitly controls memory hierarchy (GM/UB/L1/L0), compute pipelines (Vector/Cube/MTE), and sync primitives for fine-grained tuning.

These layers allow custom DSLs and frameworks to integrate. Triton and PyTorch connect via IR conversion for high-performance custom operators on Ascend NPU.

## Compile Options and Function Attributes

### Compile Options

`bishengir-compile` provides these common compile options:

| Option| Default Value| Description|
|------|--------|------|
| `-target` | `Ascend<Name>` | Target device (retrieving core count, on-chip memory size and other hardware specifications). It can be queried via `npu-smi info`.|
| `-block-dim` | `1` | Number of blocks. Compiled kernel carries `hacc.block_dim`.|
| `-enable-hfusion-compile` | `false` | Enables the HFusion pipeline (fusion, scheduling, tiling).|
| `-enable-hivm-compile` | `true` | Enables the HIVM pipeline (lower ops to HIVM and optimize them).|
| `-enable-torch-compile` | `false` | Enables the Torch-MLIR pipeline.|
| `-enable-triton-kernel-compile` | `false` | Enables the Triton kernel pipeline.|

Supported target devices include the Atlas A2/A3 and Ascend 950PR/Ascend 950DT series.

### Function attributes

The following attributes mark kernel entry functions and are shared across all integration paths:

| Attribute| Description|
|------|------|
| `hacc.entry` | Marks the current function as the kernel entry.|
| `hacc.function_kind = #hacc.function_kind<DEVICE>` | Function runs on the DEVICE side.|
| `hacc.function_kind = #hacc.function_kind<HOST>` | Function runs on the HOST side; HFusion will outline device kernels.|

Example:

```mlir
func.func @kernel(...) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  ...
}
```

## Triton integration

Triton is a widely used language for high-performance kernel development. [Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) converts Triton kernels to MLIR for the AscendNPU IR stack and enables running Triton kernels on Ascend NPU. See [Triton Integration](triton_interface.md) for installation, environment, op mapping, and Ascend extensions.

## TileLang integration

TileLang (tilelang-ascend) is a domain-specific language for Ascend NPU kernel development, built on tile-lang's Pythonic syntax and [TVM](https://tvm.apache.org/). It supports GEMM, vector operations, and attention mechanisms, compiling kernels to AscendNPU IR (HIVM) for execution on Ascend NPU. See [TileLang Integration](tile_lang_interface.md) for installation and usage details.

## Framework integration

AscendNPU IR supports framework integration (PyTorch/TensorFlow/MindSpore) in two ways:

- **DSL integration**: e.g., Triton and TileLang
- **IR integration**: e.g., Torch IR, Linalg/HFusion IR, and HIVM IR

See [Framework Integration](framework_interface.md) for installation and usage details.
