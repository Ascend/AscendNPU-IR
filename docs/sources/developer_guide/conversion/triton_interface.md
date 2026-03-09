# Triton Integration

[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) is the component that connects Triton to the Ascend platform. After building and installing Triton Ascend, you can run Triton kernels with Ascend as the backend.

## Installation

### Environment

#### Python

Triton-Ascend requires **Python 3.9–3.11**.

#### Ascend CANN

CANN (Compute Architecture for Neural Networks) is Ascend’s heterogeneous compute stack for AI. It supports frameworks such as MindSpore, PyTorch, and TensorFlow and targets Ascend AI processors.

Install CANN from the [Ascend community site](https://www.hiascend.com/developer/download/community/result?module=cann) and follow the [installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit).

Choose a CANN **{version}** compatible with your Triton-Ascend version:

**Commercial CANN:**

| Triton-Ascend | CANN commercial | CANN release |
|---------------|-----------------|--------------|
| 3.2.0         | CANN 8.5.0      | 2026/01/16   |
| 3.2.0rc4      | CANN 8.3.RC2    | 2025/11/20   |
|               | CANN 8.3.RC1    | 2025/10/30   |

**Community CANN:**

| Triton-Ascend | CANN community   | CANN release |
|---------------|------------------|--------------|
| 3.2.0         | CANN 8.5.0       | 2026/01/16   |
| 3.2.0rc4     | CANN 8.3.RC2     | 2025/11/20   |
|               | CANN 8.5.0.alpha001 | 2025/11/12 |
|               | CANN 8.3.RC1     | 2025/10/30   |

Set **{arch}** to your CPU architecture (aarch64 or x86_64) and **{version}** to the chosen CANN version.

Recommended: install 8.5.0. Package names:

| Type    | Description      | Package name |
|---------|------------------|----------------------------------|
| Toolkit | CANN toolkit    | Ascend-cann-toolkit_**{version}**_linux-**{arch}**.run |
| Ops     | CANN operator binaries | Ascend-cann-A3-ops_**{version}**_linux-**{arch}**.run |

Note: A2 Ops packages use a different naming pattern (e.g. Ascend-cann-910b-ops_**{version}**_linux-**{arch}**.run). For CANN before 8.5.0, Ops may be named like Atlas-A3-cann-kernels_**{version}**_linux-**{arch}**.run.

**CANN install script (example for 8.5.0 A3):**

```bash
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
sudo ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
sudo ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pyyaml
```

Default install path: root → `/usr/local/Ascend`; non-root → `${HOME}/Ascend`. Add `source .../set_env.sh` to your shell config (e.g. `.bashrc`) if needed.

#### torch_npu

Use the matching torch_npu version (e.g. 2.7.1):

```bash
pip install torch_npu==2.7.1
```

If you see `ERROR: No matching distribution found for torch==2.7.1+cpu`, install PyTorch first:

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Install Triton-Ascend via pip

**Stable:**

```bash
pip install triton-ascend
```

If you have upstream Triton installed, uninstall it first:

```bash
pip uninstall triton
pip install triton-ascend
```

**Nightly:**

```bash
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```

See [history](https://test.pypi.org/project/triton-ascend/#history) for nightly builds. For SSL errors, add `--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org`.

### Build Triton-Ascend from source

Use source build when you need to modify or extend Triton-Ascend.

**System:** GCC >= 9.4.0, GLIBC >= 2.27.

**Dependencies:** zlib1g-dev, clang (>= 15), lld (>= 15); optional: ccache. Example (Ubuntu):

```bash
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15 ccache
```

For yum: `sudo yum install -y zlib-devel`.

**Python build deps:**

```bash
pip install ninja cmake wheel pybind11
```

**LLVM:** Triton uses LLVM; Ascend’s Bisheng compiler also uses LLVM for NPU code. Build LLVM once (choose one method).

Checkout LLVM:

```bash
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
```

**Option 1 – Build with Clang:**

```bash
export LLVM_INSTALL_PREFIX=/path/to/llvm-install
cd $HOME/llvm-project && mkdir build && cd build
cmake ../llvm -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
  -DCMAKE_LINKER=/usr/bin/lld-15 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
ninja install
```

**Option 2 – Build with GCC:** Set `LLVM_INSTALL_PREFIX`, then run cmake with `-DLLVM_ENABLE_PROJECTS="mlir;llvm"` (and optionally `-DLLVM_ENABLE_LLD=ON` if you see `ld.lld: error: undefined symbol`; use `-DLLVM_CCACHE_BUILD=ON` only if ccache is installed and working).

**Clone and build Triton-Ascend:**

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend/python
```

```bash
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install
```

If GCC < 9.4.0 and you see “ld.lld: error: unable to find library -lstdc++fs”, uncomment the `link_libraries(stdc++fs)` block in `triton-ascend/CMakeLists.txt` and rebuild.

**Run an example:**

```bash
cd triton-ascend && pip install -r requirements_dev.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

You should see tensors and “The maximum difference between torch and triton is 0.0”.

## Calling a Triton kernel

After installing Triton-Ascend, you can run Triton kernels on NPU. Use `pytest -sv <file>.py` to verify. Example pattern:

```python
import triton
import triton.language as tl
import torch
import torch_npu

@triton.jit
def triton_lt(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    for loop1 in range(XBLOCK // XBLOCK_SUB):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, None)
        tmp1 = tl.load(in_ptr1 + x_index, None)
        tmp2 = tmp0 < tmp1
        tl.store(out_ptr0 + x_index, tmp2, None)

# Invoke: grid from [] config, XBLOCK/XBLOCK_SUB control tiling
triton_lt[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
```

**Dynamic tiling:** Configure parallelism via the grid in `[]` and tiling size via `XBLOCK` and `XBLOCK_SUB`. **Dynamic shape:** The kernel adapts to 1D tensors of arbitrary length; pass the actual shape at runtime.

## Triton op to Ascend NPU IR mapping

Triton Ascend lowers Triton dialect ops through Linalg, HFusion, and HIVM to optimized IR for Ascend NPU. Summary of main mappings:

| Triton op | Ascend NPU IR op | Description |
|-----------|------------------|-------------|
| **Memory** | | |
| `triton::StoreOp` | `memref::copy` | Store to memory. |
| `triton::LoadOp` | `memref::copy` + `bufferization::ToTensorOp` | Load from memory. |
| `triton::AtomicRMWOp` | `hivm::StoreOp` or `hfusion::AtomicXchgOp` | Atomic read-modify-write. |
| `triton::AtomicCASOp` | `linalg::GenericOp` | Atomic compare-and-swap. |
| `triton::GatherOp` | `func::CallOp` → `hfusion::GatherOp` | Gather by index. |
| **Pointer** | | |
| `triton::AddPtrOp` | `memref::ReinterpretCast` | Pointer offset. |
| `triton::PtrToIntOp` | `arith::IndexCastOp` | Pointer to integer. |
| `triton::IntToPtrOp` | `hivm::PointerCastOp` | Integer to pointer. |
| `triton::AdvanceOp` | `memref::ReinterpretCastOp` | Advance pointer. |
| **Program** | | |
| `triton::GetProgramIdOp` | Function parameter | Program ID. |
| `triton::GetNumProgramsOp` | Function parameter | Number of programs. |
| `triton::AssertOp` | `func::CallOp` → `hfusion::AssertOp` | Assert. |
| `triton::PrintOp` | `func::CallOp` → `hfusion::PrintOp` | Print. |
| **Tensor** | | |
| `triton::ReshapeOp` | `tensor::ReshapeOp` | Reshape. |
| `triton::ExpandDimsOp` | `tensor::ExpandShapeOp` | Expand dims. |
| `triton::BroadcastOp` | `linalg::BroadcastOp` | Broadcast. |
| `triton::TransOp` | `linalg::TransposeOp` | Transpose. |
| `triton::SplitOp` | `tensor::ExtractSliceOp` | Split. |
| `triton::JoinOp` / `triton::CatOp` | `tensor::InsertSliceOp` | Join / concat. |
| `triton::MakeRangeOp` | `linalg::GenericOp` | Range tensor. |
| `triton::SplatOp` | `linalg::FillOp` | Fill with scalar. |
| `triton::SortOp` | `func::CallOp` → `hfusion::SortOp` | Sort. |
| **Math** | | |
| `triton::DotOp` | `linalg::MatmulOp` | Matrix multiply. |
| `triton::DotScaledOp` | `linalg::MatmulOp` | Scaled matmul. |
| **Reduce** | | |
| `triton::ArgMinOp` / `triton::ArgMaxOp` | `linalg::ReduceOp` | Argmin/argmax. |
| `triton::ReduceOp` | `linalg::ReduceOp` | Reduction. |
| `triton::ScanOp` | `func::CallOp` → `hfusion::CumsumOp` / `CumprodOp` | Scan (cumsum/cumprod). |

## Triton Ascend extensions

Import the Ascend extension:

```python
import triton.language.extra.cann.extension as al
```

These APIs are Ascend-specific and not guaranteed compatible across versions.

### Sync and debug

#### debug_barrier

Vector pipeline sync for debugging and tuning. Parameter: `sync_mode` (e.g. `al.SYNC_IN_VF.VV_ALL`). Use inside `al.scope(core_mode="vector")`:

```python
with al.scope(core_mode="vector"):
    al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
    # ... vector ops
```

#### sync_block_set / sync_block_wait

Set or wait for sync events between Cube and Vector. Parameters: `sender`, `receiver`, `event_id`, `sender_pipe_value`, `receiver_pipe_value`. Example (after dot, before reload/exp):

```python
extension.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
extension.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
```

#### sync_block_all

Global block sync. Parameters: `mode` (`"all_cube"`, `"all_vector"`, `"all"`, `"all_sub_vector"`), `event_id` (0–15). Example: `al.sync_block_all("all", 10)`.

### Hardware query

#### sub_vec_id / sub_vec_num

`al.sub_vec_id()` returns the current Vector core index on the AI Core. `al.sub_vec_num()` returns the number of Vector cores per AI Core (910B supports up to 2). Use for splitting work across sub-vectors.

#### parallel

Ascend extends iteration with **parallel** semantics: `al.parallel(start, end, step)`, optional `num_stages`, `loop_unroll_factor`. Example: `for _ in extension.parallel(2, 10, 3): ...`.

### Compile hints

#### compile_hint

Pass optimization hints: `al.compile_hint(tensor, "hint_name", hint_val)`. Example: `al.compile_hint(mask, "bitwise_mask")` for mask; `tl.compile_hint(res, "tile_cube_loop", 2)` for cube sub-tiling.

#### multibuffer

Double (or multi) buffering: `al.multibuffer(src_tensor, size)` (e.g. `size=2`).

#### scope

Scope manager for core type: `with al.scope(core_mode="vector"):` or `core_mode="cube"`.

### Tensor slice

#### extract_slice / insert_slice

`al.extract_slice(ful, offsets, sizes, strides)` extracts a slice. `al.insert_slice(ful, sub, offsets, sizes, strides)` inserts a slice into a full tensor.

#### get_element

`al.get_element(src, indice)` reads a single element at the given index tuple.

### Tensor compute

#### sort

`al.sort(ptr, dim=-1, descending=False)` sorts along a dimension.

#### flip

`al.flip(ptr, dim)` flips along a dimension.

#### cast

`al.cast` / `tl.cast` with optional `fp_downcast_rounding`, `bitcast`, `overflow_mode` (e.g. `"trunc"`, `"saturate"`).

### Index and gather

#### _index_select

Gather from GM into UB by index tensor. Parameters: `src`, `index`, `dim`, `bound`, `end_offset`, `start_offset`, `src_stride`, optional `other`, `out`. Supports 2D–5D.

#### index_put

Scatter from UB to GM by index. Parameters: `ptr`, `index`, `value`, `dim`, `index_boundary`, `end_offset`, `start_offset`, `dst_stride`. Rules: dim 0/1/2 place by index on that dimension; `ptr` and `value` same rank; `ptr.dtype` float16/bfloat16/float32; `index.numel` = `value.shape[dim]`.

#### gather_out_to_ub

Gather from GM to UB with boundary check. Parameters: `src`, `index`, `index_boundary`, `dim`, `src_stride`, `end_offset`, `start_offset`, optional `other`. Returns UB tensor with shape of `index`. Constraints: same rank for `src` and `index`; integer index; valid `dim`.

#### scatter_ub_to_out

Scatter from UB to GM. Parameters: `ptr`, `value`, `index`, `index_boundary`, `dim`, `dst_stride`, `end_offset`, `start_offset`. Same rank for `ptr`, `index`, `value`; integer index; valid `dim`.

#### index_select_simd

Parallel index select from GM to UB. Parameters: `src`, `dim`, `index`, `src_shape`, `src_offset`, `read_shape`. Constraint: `read_shape[dim]` = -1; `src_offset[dim]` can be -1. No bounds check on `index`. Returns UB tensor with `dim` size equal to `index` length.

## Custom ops

Triton-Ascend custom ops are user-defined operations that lower to device-side function calls (library or user-provided source/bytecode).

### Registration

Use `@al.register_custom_op` on a class. Minimum attributes: `name`, `core` (e.g. `al.CORE.VECTOR`), `pipe` (e.g. `al.PIPE.PIPE_V`), `mode` (e.g. `al.MODE.SIMT`).

### Usage

Call with `al.custom('op_name', arg1, arg2, ..., out=dst)`. Output can be specified via `out`; if so, the return value is that output (or a tuple for multiple outputs).

### Built-in custom ops

Names start with `"__builtin_"`; no registration needed. Example: `al.custom('__builtin_indirect_load', src, index, mask, other, out=dst)`.

### Parameter validation

Implement `__init__(self, ...)` on the registered class with the same parameter list as the call. Use asserts to check types and shapes; violations surface at compile time.

### Symbol name

Set `symbol` on the class for the device function name. Can be static (string) or set in `__init__` (e.g. from dtype/shape) for dynamic dispatch.

### Source and compile

Set `source` (path to .cce or source) and `compile` (command; `%<` = source, `%@` = target) for ops that are compiled from source.

### Parameter mapping

Argument order matches Python; `out` is last. Tuples/lists are flattened to separate arguments. Integer constants map to int32_t by default; use `al.int64(value)` or type hints (e.g. `bound: tl.int64`) or `self.arg_type['bound'] = idx.dtype` for int64 or type-dependent constants.

### Wrapping

Use `@al.builtin` and `al.custom_semantic` to wrap a custom op so callers can use it like a normal function (e.g. `x = my_custom_op(src, index)`); pass `_builder` through to semantic helpers.

## Extension enums

**SYNC_IN_VF:** `VV_ALL`, `VST_VLD`, `VLD_VST`, `VST_VST`, `VS_ALL`, `VST_LD`, `VLD_ST`, `VST_ST`, `SV_ALL`, `ST_VLD`, `LD_VST`, `ST_VST` (vector/scalar load/store sync modes).

**FixpipeDMAMode:** `NZ2DN`, `NZ2ND`, `NZ2NZ`.

**FixpipeDualDstMode:** `NO_DUAL`, `COLUMN_SPLIT`, `ROW_SPLIT`.

**FixpipePreQuantMode:** `NO_QUANT`, `F322BF16`, `F322F16`, `S322I8`.

**FixpipePreReluMode:** `LEAKY_RELU`, `NO_RELU`, `NORMAL_RELU`, `P_RELU`.

**CORE:** `VECTOR`, `CUBE`, `CUBE_OR_VECTOR`, `CUBE_AND_VECTOR`.

**MODE:** `SIMD`, `SIMT`, `MIX`.

**PIPE:** `PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_ALL`, `PIPE_FIX`.
