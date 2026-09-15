# IR Integration Overview

## Multi-level IR Abstraction Architecture

- Provides a series of high-level abstraction interfaces that hide underlying details and map hardware-independent expressions to low-level instructions, improving the ease of operator development.
- Provides fine-grained performance control interfaces that can precisely control on-chip memory addresses, pipeline synchronization insertion positions, and whether to enable ping-pong pipeline optimization.
- Supports flexible integration of custom DSLs and ecosystem frameworks based on the multi-level interfaces, enabling high-performance execution of custom operators on Ascend NPUs.

```text
  Torch-MLIR / Triton      (Framework/DSL layer)
          |
          v
  Linalg / Tensor           (General tensor algebra layer)
          |
          v
  HFusion                   (Hardware-aware fusion scheduling layer)
          |
          v
  HIVM                      (NPU instruction layer)
          |
          v
  LIR -> Binary             (Machine code generation)
```

- **Linalg / Tensor layer**: Uses standard MLIR dialects to express operator semantics, supporting operations such as `Elemwise`, `Broadcast`, `Reduce`, `Transpose`, and `Concat`. HFusion automatically performs operator fusion, tiling, and scheduling.
- **HFusion layer**: Provides Ascend NPU-aware named ops (such as `hfusion.elemwise_unary`, `hfusion.cast`, `hfusion.select`, and `hfusion.reduce_with_index`), supports `tensor` semantics, and automatically performs bufferization, tiling, and scheduling.
- **HIVM layer**: Directly maps NPU hardware instructions, explicitly controls the memory hierarchy (GM/UB/L1/L0), compute pipelines (Vector/Cube/MTE), and synchronization primitives, and supports fine-grained performance tuning.

The multi-level interfaces described above support flexible integration with custom DSLs and ecosystem frameworks. Frameworks such as Triton and PyTorch are integrated into the aforementioned process through IR transformation, enabling custom operators to run with high performance on Ascend NPUs.

## Compilation Options and Function Attributes

### Compilation Options

`bishengir-compile` provides the following public compilation options:

| Option | Default Value | Description |
|------|--------|------|
| `-target` | `Ascend<Name>` | Target device, used to obtain hardware specifications such as the number of cores and on-chip memory size, which can be queried via `npu-smi info`. |
| `-block-dim` | 1 | Specifies the number of blocks to use. After compilation, the kernel carries the `hacc.block_dim` attribute. |
| `-enable-hfusion-compile` | `false` | Enables the HFusion compilation process (fusion, scheduling, and tiling). |
| `-enable-hivm-compile` | `true` | Enables the HIVM compilation process (conversion to HIVM instructions and optimization). |
| `-enable-torch-compile` | `false` | Enables the Torch-MLIR compilation process. |
| `-enable-triton-kernel-compile` | `false` | Enables the Triton kernel compilation process. |

Supported target devices include:

- Ascend 950PR/Ascend 950DT
- Atlas A3 training products/Atlas A3 inference products
- Atlas A2 training products/Atlas A2 inference products

### Function Attributes

The following attributes annotate the kernel entry function and are common to all integration paths:

| Attribute | Description |
|------|------|
| `hacc.entry` | Marks the current function as the kernel entry. |
| `hacc.function_kind = #hacc.function_kind<DEVICE>` | Indicates that the function runs on the DEVICE side. |
| `hacc.function_kind = #hacc.function_kind<HOST>` | Indicates that the function runs on the HOST side, and HFusion automatically outlines the device kernel. |

Example:

```mlir
func.func @kernel(...) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  ...
}
```

## Triton Integration

Triton is a mainstream high-performance operator development programming language. [Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) converts Triton operators into MLIR and integrates them into the AscendNPU IR ecosystem, enabling Triton kernels to run on Ascend NPUs. For a detailed Triton integration guide (including installation, environment, operator mapping, and Ascend extensions), see [Triton Integration](triton_interface.md).

## TileLang Integration

TileLang (tilelang-ascend) is a domain-specific language for Ascend NPUs. It is built on the Python syntax of tile-lang and [TVM](https://tvm.apache.org/), and supports operators such as GEMM, vector operations, and attention mechanisms. It can compile operators into AscendNPU IR (HIVM) for execution on Ascend NPUs. For detailed TileLang integration instructions, see [TileLang Integration](tile_lang_interface.md).

## Framework Integration

AscendNPU IR supports framework integration (PyTorch/TensorFlow/MindSpore) through two approaches:

- **DSL integration**: such as Triton and TileLang
- **IR integration**: such as Torch IR, Linalg/HFusion IR, and HIVM IR

For detailed framework integration instructions, see [Framework Integration](framework_interface.md).
