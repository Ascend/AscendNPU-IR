# Triton Integration

[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) is an important component that helps Triton access the Ascend platform. After the Triton Ascend is built and installed, you can use the Ascend as the backend when executing the Triton operator.

## Installation and Execution

### Environment setup

#### Python version

Currently, the Python version required by Triton-Ascend is **py3.9-py3.11**.

#### Installing Ascend CANN

The end-to-end operation of the AscendNPU-IR depends on the CANN environment.

1. Download the CANN package: Download the toolkit package and the ops package corresponding to the hardware. You can download the toolkit package from the [Ascend Community CANN download page](https://www.hiascend.com/cann/download).

2. Install the CANN package.

    ```bash
    # In the x86 A3 environment, {version} indicates the CANN version, for example, 9.0.0.
    chmod +x Ascend-cann_{version}_linux-x86_64.run
    chmod +x Ascend-cann-A3-ops_{version}_linux-x86_64.run
    ./Ascend-cann_{version}_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
    ./Ascend-cann-A3-ops_{version}_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
    # Install the Python dependencies of CANN.
    pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pyyaml
    ```

3. Set environment variables.

    ```bash
    # If the version is earlier than 8.5.0, the path is ${PATH-TO-CANN}/ascend-toolkit/set_env.sh.
    source ${PATH-TO-CANN}/cann/set_env.sh
    ```

#### Installing torch_npu & Triton-Ascend

Currently, the torch_npu version is 2.7.1.

```bash
pip install torch_npu==2.7.1
pip install triton-ascend
```

### Calling Triton Kernel

After installing Triton-Ascend, you can call the related Triton Kernel. For details, see the following source code. You can run `pytest -sv <file>.py` to verify the functions after the installation. If the function is correct, the terminal displays `PASS`.

```python
from typing import Optional
import pytest
import triton
import triton.language as tl
import torch
import torch_npu

def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    elif dtype == 'uint8':
        return torch.randint(low=0, high=255, size=shape, dtype=torch.uint8)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None):
    y_cal=y_cal.npu()
    y_ref=y_ref.npu()
    if overflow_mode == "saturate":
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32),  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'uint8' or dtype == 'uint16' or dtype == 'uint32' or dtype == 'uint64':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def torch_lt(x0, x1):
    return x0 < x1

@triton.jit
def triton_lt(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, None)
        tmp1 = tl.load(in_ptr1 + x_index, None)
        tmp2 = tmp0 < tmp1
        tl.store(out_ptr0 + x_index, tmp2, None)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (32,), 1, 32, 32],
                         ])
def test_lt(param_list):
    # Generate data
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = generate_tensor(shape, dtype).npu()
    x1 = generate_tensor(shape, dtype).npu()
    # Torch result
    torch_res = torch_lt(x0, x1).to(eval('torch.' + dtype))
    # Triton result
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_lt[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # Compare results
    validate_cmp(dtype, triton_res, torch_res)
```

**Dynamic tiling support**: The parallel granularity is configured by the grid parameter in [], and the tiling size is controlled by the XBLOCK and XBLOCK_SUB parameters. You can adjust the size as required.

**Dynamic shape support**: The kernel automatically adapts 1D tensors of any length. You only need to transfer the actual shape data.

## Conversion from Triton Op to AscendNPU IR Op

Triton Ascend degrades the advanced GPU abstraction operations of the Triton dialect to target dialects such as Linalg, HFusion, and HIVM, resulting in an optimized intermediate representation that can be efficiently executed on the Ascend NPU. The following table details the various Triton operations and their corresponding AscendNPU IR operations in the fall process.

| Triton Op | Target AscendNPU IR Op| Description |
| :--- | :--- | :--- |
| **Storage access ops**| | |
| `triton::StoreOp` | `memref::copy` | Stores data to the memory.|
| `triton::LoadOp` | `memref::copy` + `bufferization::ToTensorOp` | Loads data from the memory.|
| `triton::AtomicRMWOp` | `hivm::StoreOp` or `hfusion::AtomicXchgOp`| Performs atomic read-modify-write operations.|
| `triton::AtomicCASOp` | `linalg::GenericOp` | Performs atomic compare-and-swap operations.|
| `triton::GatherOp` | Converts to `func::CallOp` first (calling `triton_gather`)<br>Then converts to `hfusion::GatherOp`| Collects data based on indexes.|
| **Pointer operation class ops**| | |
| `triton::AddPtrOp` | `memref::ReinterpretCast` | Performs an offset operation on a pointer.|
| `triton::PtrToIntOp` | `arith::IndexCastOp` | Converts a pointer to an integer.|
| `triton::IntToPtrOp` | `hivm::PointerCastOp` | Converts an integer to a pointer.|
| `triton::AdvanceOp` | `memref::ReinterpretCastOp` | Pushes the pointer position.|
| **Program information ops**| | |
| `triton::GetProgramIdOp` | Parameters of `functionOp`| Obtains the ID of the current program.|
| `triton::GetNumProgramsOp` | Parameters of `functionOp`| Obtains the total number of programs.|
| `triton::AssertOp` | Converts to `func::CallOp` first (calling `triton_assert`)<br>Then converts to `hfusion::AssertOp`.| Performs an assertion.|
| `triton::PrintOp` | Converts to `func::CallOp` first (calling `triton_print`)<br>Then converts to `hfusion::PrintOp`.| Performs a print.|
| **Tensor operation Ops**| | |
| `triton::ReshapeOp` | `tensor::ReshapeOp` | Changes the tensor shape.|
| `triton::ExpandDimsOp` | `tensor::ExpandShapeOp` | Extends the tensor dimension.|
| `triton::BroadcastOp` | `linalg::BroadcastOp` | Broadcasts a tensor.|
| `triton::TransOp` | `linalg::TransposeOp` | Transposes a tensor.|
| `triton::SplitOp` | `tensor::ExtractSliceOp` | Splits a tensor.|
| `triton::JoinOp` | `tensor::InsertSliceOp` | Joins a tensor.|
| `triton::CatOp` | `tensor::InsertSliceOp` | Concatenates a tensor.|
| `triton::MakeRangeOp` | `linalg::GenericOp` | Creates a tensor containing consecutive integers.|
| `triton::SplatOp` | `linalg::FillOp` | Fills a tensor with scalar values.|
| `triton::SortOp` | Converts to `func::CallOp` first (calling `triton_sort`).<br>Then converts to `hfusion::SortOp`.| Sorts tensors.|
| **Numeric computation ops**| | |
| `triton::MulhiUIOp` | `arith::MulSIExtendedOp` | Multiplies unsigned integers, returning high-order results|
| `triton::PreciseDivFOp` | `arith::DivFOp` | Performs high-precision floating-point division|
| `triton::PreciseSqrtOp` | `math::SqrtOp` | Performs high-precision floating-point square root|
| `triton::BitcastOp` | `arith::BitcastOp` | Reinterprets bits between different types|
| `triton::ClampFOp` | `tensor::EmptyOp` + `linalg::FillOp` | Limits floating-point numbers to a specified range|
| `triton::DotOp` | `linalg::MatmulOp` | Executes general matrix multiplication|
| `triton::DotScaledOp` | `linalg::MatmulOp` | Executes matrix multiplication with scaling factors|
| `triton::ascend::FlipOp` | Converts to `func::CallOp` first (calling `triton_flip`).<br>Then converts to `hfusion::FlipOp`.| Executes matrix multiplication with scaling factors|
| **Reduction ops**| | |
| `triton::ArgMinOp` | `linalg::ReduceOp` | Returns the index of the smallest value in the tensor|
| `triton::ArgMaxOp` | `linalg::ReduceOp` | Returns the index of the largest value in the tensor|
| `triton::ReduceOp` | `linalg::ReduceOp` | Performs a general reduction operation|
| `triton::ScanOp` | Converts to `func::CallOp` first (calling `triton_cumsum` or `triton_cumprod`)<br>Then converts to `hfusion::CumsumOp` and `hfusion::CumprodOp`| Performs a scan operation (such as cumulative sum and cumulative product)|

## Triton Extended Operations

AscendNPU-IR provides language features. Triton-Ascend extends some operations based on NPU IR. To enable the capabilities, you need to import the following modules:

```python
import triton.language.extra.cann.extension as al
```

The relevant Ascend Language-specific (al) interface can then be used. In addition, the Ascend Language provides bottom-layer interfaces, and the interfaces are not compatible.

### Synchronization and Debugging Operations

#### debug_barrier

Ascend provides multiple synchronization modes and supports the internal synchronization mode of the vector pipeline for fine-grained synchronization control during debugging and performance optimization.

##### Parameters

| Parameter| Type| Description |
|--------|------|------|
| `sync_mode` | [Enumerated value of SYNC_IN_VF](#sync_in_vf)| Vector pipeline synchronization mode|

**Example**:

```python
@triton.jit
def kernel_debug_barrier():
    # ...
    with al.scope(core_mode="vector"):
        al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
    # ...
```

#### sync_block_set & sync_block_wait

Ascend supports the setting of synchronization events between computing units and vector units. `sync_block_set` and `sync_block_wait` must be used together.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `sender` | str | Type of the sending unit|
| `receiver` | str | Type of the receive unit|
| `event_id` | int | Event identifier|
| `sender_pipe_value` | [Enumerated value of PIPE](#pipe)| Value of the sending pipe|
| `receiver_pipe_value` | [Enumerated value of PIPE](#pipe)| Value of the receiving pipe|

**Example**:

```python
@triton.jit
def triton_matmul_exp():
    # ...
    tbuff_ptrs = TBuff_ptr + (row + offs_i) * N + (col + offs_j)
    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)
    
    extension.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    extension.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    
    acc_11_reload = tl.load(tbuff_ptrs)
    c_ptrs = C_ptr + (row + offs_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
    # ...
```

#### sync_block_all

Ascend supports global synchronization of the entire computing block, ensuring that all computing cores of a specified type complete the current operation.

**Parameters**:

| Parameter| Type| Description | Valid Value|
|--------|------|------|--------|
| `mode` | str | Synchronization mode, which specifies the core type to be synchronized.| `"all_cube"`, `"all_vector"`, `"all"`, `"all_sub_vector"` |
| `event_id` | int | Synchronization event identifier.| `0` ~ `15` |

**Synchronization mode details**:

| Mode| Description | Synchronization Range|
|------|------|----------|
| `"all_cube"` | Synchronize all Cube cores.| All cube cores on the current AI core|
| `"all_vector"` | Synchronize all vector cores.| All vector cores on the current AI core|
| `"all"` | Synchronize all cores.| All computing cores (Cube+Vector) on the current AI core|
| `"all_sub_vector"` | Synchronize all subvector cores.| All vector cores on the current AI core|

**Example**:

```python
@triton.jit
def test_sync_block_all():
    # ...
    al.sync_block_all("all_cube", 8)
    al.sync_block_all("all_sub_vector", 9)
    # ...
```

### Hardware Query and Control Operations

#### sub_vec_id & sub_vec_num

Ascend provides APIs to query hardware information.
Calling `sub_vec_id` obtains the vector core index on the current AI core.
Calling `sub_vec_num` obtains the number of vector cores on a single AI core.

**Example**:

```python
@triton.jit
def triton_matmul_exp():
    # ...
    sub_vec_id = al.sub_vec_id()
    row_exp = row_matmul + (M // al.sub_vec_num()) * sub_vec_id
    offs_exp_i = tl.arange(0, M // al.sub_vec_num())[:, None]
    tbuff_exp_ptrs = TBuff_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    # ...
```

#### parallel

Ascend extends the standard `range` function of Python, adding `parallel` iterators with parallel execution semantics.

**Parameters**:

| Parameter| Type| Description| Example|
|------|------|------|------|
| `arg1` | int | Start or end value| `parallel(10)` |
| `arg2` | int | (Optional) End value| `parallel(0, 10)` |
| `step` | int | (Optional) Stride| `parallel(0, 10, 2)` |
| `num_stages` | int | (Optional) Number of pipeline stages| `parallel(0, 10, num_stages=3)` |
| `loop_unroll_factor` | int | (Optional) Loop unrolling factor| `parallel(0, 10, loop_unroll_factor=4)` |

**Restrictions:**

Currently, Atlas A2 supports a maximum of two Vector cores.

**Example**:

```python
@triton.jit
def triton_add():
    # ...
    for _ in al.parallel(2, 5, 2):
        ret = ret + x1
    for _ in al.parallel(2, 10, 3):
        ret = ret + x0
    tl.store(out_ptr0, ret)
    # ...
```

### Compilation Optimization Hints

#### compile_hint

Ascend can pass optimization hints to the compiler to guide code generation and performance tuning.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `ptr` | tensor | Pointer to the target tensor.|
| `hint_name` | str | Hint name.|
| `hint_val` | Multiple types| (Optional) Hint value.|

**Example**:

```python
@triton.jit
def triton_where_lt_case1():
    # ...
    mask = tl.where(cond, in1, in0)
    al.compile_hint(mask, "bitwise_mask")
    # ...
```

#### multibuffer

`multibuffer` sets up double buffering for existing tensors, optimizing data flow and computational overlap through compiler hints.

**Parameters**:

| Parameter| Type| Description|
|------|------|------|
| `src` | tensor | Tensor to be double buffered.|
| `size` | int | Number of buffer copies.|

**Example**:

```python
@triton.jit
def triton_compile_hint():
    # ...
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    al.multibuffer(tmp0, 2)
    tl.store(out_ptr0 + (xindex), tmp0, xmask)
    # ...
```

#### scope

Ascend supports scope managers, adding hint information to a section of locale code, one use of which is to specify the cube or vector type via `core_mode`.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `core_mode` | str | Only the "cube" or "vector" mode is accepted. Core type, which specifies the computing core used by operations in a block. Only `"cube"` and `"vector"` are supported.|

**Core mode options**:

| Mode| Description |
|------|------|
| `"cube"` | Use the Cube core for computation.|
| `"vector"` | Use the Vector core for computation.|

**Example**:

```python
@triton.jit
def kernel_debug_barrier():
    # ...
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
    # ...
```

### Tensor slicing operation

#### insert_slice & extract_slice

Ascend supports inserting a tensor into another tensor based on the offset, size, and step parameters of the operation (i.e. `insert_slice`) or extract the specified slice from another tensor (i.e. `extract_slice`).

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `ful` | Tensor | Receive the inserted target tensor.|
| `sub` | Tensor | Source tensor to be inserted.|
| `offsets` | Integer tuple| Start offset of the insert operation.|
| `sizes` | Integer tuple| Size range of the insert operation.|
| `strides` | Integer tuple| Stride of the insert operation.|

**Example**:

```python
@triton.jit
def triton_kernel():
    # ...
    x_sub = al.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = al.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    output = tl.load(output_ptr + offsets, mask=mask)
    output = al.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
    # ...
```

#### get_element

Ascend reads a single element value from a tensor at a specified index position.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `src` | tensor | Source tensor to access.|
| `indice` | int tuple| Index position of the element to obtain.|

**Example**:

```python
@triton.jit
def index_select_manual_kernel():
    # ...
    gather_offset = al.get_element(indices, (i,)) * g_stride
    val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
    # ...
```

### Tensor Computing Operation

#### sort

Ascend sorts input tensors along the specified dimension.

**Parameters**:

| Parameter| Type| Description | Default Value|
|--------|------|------|--------|
| `ptr` | tensor | Input tensor.| - |
| `dim` | int or tl.constexpr[int]| Dimension to be sorted.| `-1` |
| `descending` | bool or tl.constexpr[bool]| Sorting direction. `True` indicates descending, and `False` indicates ascending.| `False` |

**Example**:

```python
@triton.jit
def sort_kernel_2d():
    # ...
    x = tl.load(X + off2d)
    x = al.sort(x, descending=descending, dim=1)
    tl.store(Z + off2d, x)
    # ...
```

#### flip

Ascend flips input tensors along the specified dimension.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `ptr` | tensor | Input tensor.|
| `dim` | int or tl.constexpr[int]| Dimension to be flipped.|

**Example**:

```python
@triton.jit
def flip_kernel_2d():
    # ...
    input = tl.load(input_ptr + offset)
    flipped_input = flip(input, dim=2)
    # ...
```

#### cast

Ascend converts tensors to specified data types, supporting numerical conversion, bit conversion, and overflow handling.

**Parameters**:

| Parameter| Type| Description | Default Value|
|--------|------|------|--------|
| `input` | tensor | Input tensor.| - |
| `dtype` | dtype | Target data type| - |
| `fp_downcast_rounding` | str, optional| Rounding mode when a floating point number is converted down| `None` |
| `bitcast` | bool, optional| Whether to perform bit conversion (rather than numerical conversion)| `False` |
| `overflow_mode` | str, optional| Overflow handling mode| `None` |

**Example**:

```python
@triton.jit
def cast_to_bool():
    # ...
    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int1, overflow_mode=overflow_mode)
    tl.store(output_ptr + idx, ret)
    # ...
```

### Indexing and Collection Operations

#### _index_select

Ascend collects data in specified dimensions based on the index UB tensor from the source GM tensor and uses the SIMT template to collect values to the output UB tensor. This operation supports 2D to 5D tensors.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `src` | pointer type | Source tensor pointer (in GM).|
| `index` | tensor | Index tensor for collection (in UB).|
| `dim` | int | Dimension along which the collection is performed.|
| `bound` | int | Upper bound of the index value.|
| `end_offset` | int tuple| End offset of each dimension of the index tensor.|
| `start_offset` | int tuple| Start offset of each dimension of the source tensor.|
| `src_stride` | int tuple| Stride of each dimension of the source tensor.|
| `other` (optional)| scalar value | Default value (in UB) when the index is out of bounds.|
| `out` | tensor | Output tensor (in UB).|

**Example**:

```python
@triton.jit
def select_index():
    # ...
    tmp_buf = al._index_select(
        src=src_3d_ptr,
        index=index_2d_tile,
        dim=1,
        bound=50,
        end_offset=(2, 4, 64),
        start_offset=(0, 8, 0),
        src_stride=(256, 64, 1),
        other=0.0
    )
    # ...
```

#### index_put

Ascend places the value tensor into the target tensor based on the index tensor.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `ptr` | tensor (pointer type)| Target tensor pointer (in GM).|
| `index` | tensor | Index for placement (in UB).|
| `value` | tensor | Value to be stored (in UB).|
| `dim` | int32 | Dimension along which the index is placed.|
| `index_boundary` | int64 | Upper bound of the index value.|
| `end_offset` | int tuple| End offset of the placement area for each dimension.|
| `start_offset` | int tuple| Start offset of the placement area for each dimension.|
| `dst_stride` | int tuple| Stride of each dimension of the target tensor.|

**Index placement rules**:

- **2D index placement**
    - dim = 0: `out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]`

- 3D index placement
    - dim = 0: `out[index[i]][start_offset[1]:end_offset[1]][start_offset[2]:end_offset[2]]  = value[i][0:end_offset[1]-start_offset[1]][0:end_offset[2]-start_offset[2]]`
    - dim = 1: `out[start_offset[0]:end_offset[0]][index[j]][start_offset[2]:end_offset[2]] = value[0:end_offset[0]-start_offset[0]][j][0:end_offset[2]-start_offset[2]]`

**Constraints**:

- `ptr` and `value` must have the same rank.
- Currently, `ptr.dtype` supports only `float16`, `bfloat16`, and `float32`.
- `index` must be an integer tensor. If `index.rank` is not equal to 1, it will be reshaped to 1D.
- `index.numel` must be equal to `value.shape[dim]`.
- `value` supports 2D to 5D tensors.
- `dim` must be valid (0 ≤ dim < rank(value) – 1).

**Example**:

```python
@triton.jit
def put_index():
    # ...
    tmp_buf = al.index_put(
        ptr=dst_ptr,
        index=index_tile,
        value=value_tile,
        dim=0,
        index_boundary=4,
        end_offset=(2, 2),
        start_offset=(0, 0),
        dst_stride=(2, 1)
    )
    # ...
```

#### gather_out_to_ub

Ascend can collect data from scatterpoints in the GM and save the data to the UB in a specified dimension. This operation supports index bounds check, ensuring efficient and secure data transfer.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `src` | tensor (pointer type)| Source tensor pointer (in GM).|
| `index` | tensor | Index tensor for collection (in UB).|
| `index_boundary` | int64 | Upper bound of the index value.|
| `dim` | int32 | Dimension along which the collection is performed.|
| `src_stride` | int64 tuple| Stride of each dimension of the source tensor.|
| `end_offset` | int32 tuple| End offset of each dimension of the index tensor.|
| `start_offset` | int32 tuple| Start offset of each dimension of the index tensor.|
| `other` | Scalar value (optional)| Default value (in UB) when the index is out of bounds.|

**Returns**:

- **Type**: tensor
- **Description**: Result tensor located in UB. Its shape is the same as that of `index.shape`.

**Scatter collection rules**:

- 1D index collection
    - dim = 0: `out[i] = src[start_offset[0] + index[i]]`

- 2D index collection
    - dim = 0: `out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]`
    - dim = 1: `out[i][j] = src[start_offset[0] + i][start_offset[1] + index[i][j]]`

- 3D index collection
    - dim = 0: `out[i][j][k] = src[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k]`
    - dim = 1: `out[i][j][k] = src[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k]`
    - dim = 2: `out[i][j][k] = src[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]]`

**Constraints**:

- `src` and `index` must have the same rank.
- Currently, `src.dtype` supports only `float16`, `bfloat16`, and `float32`.
- `index` must be an integer tensor with a rank between 1 and 5.
- `dim` must be valid (0 ≤ dim < rank(index)).
- `other` must be a scalar value.
- For each dimension `i` that is not equal to `dim`, `index.size[i]` ≤ `src.size[i]`.
- The output shape is the same as that of `index.shape`. If `index` is None, the output tensor will be an empty tensor with the same shape as `index`.

**Example**:

```python
@triton.jit
def gather():
    # ...
    tmp_buf = al.gather_out_to_ub(
        src=src_ptr,
        index=index,
        index_boundary=4,
        dim=0,
        src_stride=(2, 1),
        end_offset=(2, 2),
        start_offset=(0, 0)
    )
    # ...
```

#### scatter_ub_to_out

Ascend stores data from scatterpoints in UB to GM along a specified dimension. This operation supports index bounds check, ensuring efficient and secure data transfer.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `ptr` | tensor (pointer type)| Target tensor pointer (in GM).|
| `value` | tensor | Tile value to be stored (in UB).|
| `index` | tensor | Index used for scatter storage (in UB).|
| `index_boundary` | int64 | Upper bound of the index value.|
| `dim` | int32 | Dimension along which the scatter storage is performed.|
| `dst_stride` | int64 tuple| Stride of each dimension of the target tensor.|
| `end_offset` | int32 tuple| End offset of each dimension of the index tensor.|
| `start_offset` | int32 tuple| Start offset of each dimension of the index tensor.|

**Scatter storage rules**:

- 1D index scatter
    - dim = 0: `out[start_offset[0] + index[i]] = value[i]`

- 2D index scatter
    - dim = 0: `out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]`
    - dim = 1: `out[start_offset[0] + i][start_offset[1] + index[i][j]] = value[i][j]`

- 3D index scatter
    - dim = 0: `out[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k] = value[i][j][k]`
    - dim = 1: `out[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k] = value[i][j][k]`
    - dim = 2: `out[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]] = value[i][j][k]`

**Constraints**:

- `ptr`, `index`, and `value` must have the same rank.
- Currently, `ptr.dtype` supports only `float16`, `bfloat16`, and `float32`.
- `index` must be an integer tensor with a rank between 1 and 5.
- `dim` must be valid (0 ≤ dim < rank(index)).
- For each dimension `i` that is not equal to `dim`, `index.size[i]` ≤ `ptr.size[i]`.
- The output shape is the same as that of `index.shape`. If `index` is None, the output tensor will be an empty tensor with the same shape as `index`.

**Example**:

```python
@triton.jit
def scatter():
    # ...
    tmp_buf = al.scatter_ub_to_out(
        ptr=dst_ptr,
        value=value,
        index=index,
        index_boundary=4,
        dim=0,
        dst_stride=(2, 1),
        end_offset=(2, 2),
        start_offset=(0, 0)
    )
    # ...
```

#### index_select_simd

Ascend supports parallel index selection. Data is directly loaded to the UB from GM points, implementing zero copy and efficient read.

**Parameters**:

| Parameter| Type| Description |
|--------|------|------|
| `src` | tensor (pointer type) | Source tensor pointer (in GM).|
| `dim` | int or constexpr| Dimension along which the index is selected.|
| `index` | tensor | One-dimensional tensor of the index to be selected (in UB).|
| `src_shape` | List[Union[int, tensor]] | Full shape of the source tensor (which can be an integer or a tensor).|
| `src_offset` | List[Union[int, tensor]] | Start offset of the read operation (which can be an integer or a tensor).|
| `read_shape` | List[Union[int, tensor]] | Size to read (tile shape, which can be an integer or a tensor).|

**Constraints**:

- `read_shape[dim]` must be `-1`.
- `src_offset[dim]` can be `-1` (which will be ignored).
- Boundary handling: When `src_offset + read_shape > src_shape`, the data will be automatically truncated to the `src_shape` boundary.
- **No check is performed** on whether there `index` contains out-of-bounds values.

**Returns**:

- **Return type**: tensor
- **Description**: Resulting tensor in UB. The dimension `dim` in its shape is replaced with the length of `index`.

**Example**:

```python
@triton.jit
def index_select_simd():
    # ...
    tmp_buf = al.index_select_simd(
        src=in_ptr,
        dim=dim,
        index=indices,
        src_shape=(other_numel, g_stride),
        src_offset=(-1, 0),
        read_shape=(-1, other_block)
    )
    # ...
```

## Triton Extended CustomOp

In the A5 architecture, Triton-Ascend Custom Op allows you to customize and use operations. During runtime, a custom operation is converted into a call to the implementation function on the device. You can call an existing library function or an implementation function generated by compiling the source code or bytecode provided by the user.

### Basic Usage

#### Registering a custom operation

The functions related to custom operations are provided by the triton Ascend extension package. A custom operation defined by the user must be registered before it can be used. You can decorate a class via `register_custom_op` provided by the extension package to define and register a custom operation.

```python
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

```

Registering a simple custom operation requires the following basic attributes: name, core, pipe, and mode.

- **name**: operation name, which is the unique identifier of the custom operation. If omitted, the class name is used by default.
- **core**: Ascend core on which the operation runs.
- **pipe**: corresponding pipeline.
- **mode**: programming mode.

#### Using a custom operation

A registered custom operation can be called using the `custom()` function provided by the Ascend extension package. When calling the function, you need to provide the name and parameters of the custom operation.

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def my_kernel(...):
    ...
    res = al.custom('my_custom_op', src, index, dim=0, out=dst)
    ...

```

The parameters of `custom()` include the operation name, input parameters, and optional output parameters.

- **Operation name**: must be the same as the registered operation name.
- **Input parameters**: vary depending on the operation.
- (Optional) **Output parameters**: specified by `out`, indicating the output of the operation.

If the output variable is specified using the `out` parameter, the return value of the custom operation is the same as the output variable. Otherwise, the return value of the operation is unavailable.

### Builtin custom operations

The names of builtin custom operations start with `"__builtin_"`. These operations are predefined in Triton-Ascend and can be used directly without registration. For example:

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def my_kernel(...):
    ...
    dst = tl.full(dst_shape, 0, tl.float32)
    x = al.custom('__builtin_indirect_load', src, index, mask, other, out=dst)
    ...

```

The specific builtin custom operations vary with versions. For details, see the document of the corresponding version.

### Parameter validity check

Without constraints, users may pass any parameters to the `al.custom()` function. Incorrect parameter count or types will cause an error during runtime.

To avoid this issue and improve user experience, we can provide a constructor for the custom registration class to describe the parameter list and perform parameter validity checks. For example:

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, index, dim, out=None):
        assert index.dtype.is_int(), "index must be an integer tensor"
        assert isinstance(dim, int), "dim must be an integer"
        assert out, "out is required"
        assert out.shape == index.shape, "out should have same shape as index"
        ...

```

The constructor parameter list of the registration class is exactly the parameter list required for calling the custom operations. Provide valid parameters that match the requirements during the calling. For example:

```python
    res = al.custom('my_custom_op', src_ptr, index, dim=1, out=dst)
```

If the provided parameters are incorrect, an error will be reported during compilation. For example, the dim parameter must be an integer constant. If a floating-point number is provided, the following error will be reported:

```text
    ...
    res = al.custom('my_custom_op', src_ptr, index, dim=1.0, out=dst)
          ^
AssertionError('dim must be an integer')
```

### Output parameters and return values

`al.custom` returns the output parameters specified by `out`. For example:

```python
x = al.custom('my_custom_op', src, index, out=dst)
```

`dst` is returned to `x`.

The `out` parameter can specify multiple output parameters. `al.custom` returns a tuple containing them:

```python
x, y = al.custom('my_custom_op', src, index, out=(dst1, dst2))
```

`dst1` is returned to `x`, and `dst2` is returned to `y`.

If there is no `out` parameter, `al.custom` returns `None`.

### Symbol name of the called function

Custom operations are eventually converted into calls to device-side implementation functions. You can configure the symbol name of the function by registering the `symbol` attribute in the custom operation class. If the `symbol` attribute is not set, the name of the custom operation is used as the function name by default.

#### Static symbol name

If a custom operation always calls a device-side function, you can statically set the symbol name:

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = '_my_custom_op_symbol_name_'

```

In this way, `al.custom('my_custom_op', ...)` will fix the `_my_custom_op_symbol_name_(...)` function on the corresponding device.

#### Dynamic symbol name

In most cases, the same custom operation needs to call different device-side functions based on the dimensions and types of input parameters. In this case, dynamic symbol names are required. Similar to parameter validity check, you can dynamically set the symbol name in the constructor of the custom operation class. For example:

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, index, dim, out=None):
        ...
        self.symbol = f"my_func_{len(index.shape)}d_{src.dtype.element_ty.cname}_{index.dtype.cname}"
        ...

```

When the input src is a pointer to the float32 type and index is a 3D tensor of the int32 type, the symbol name of the device-side function corresponding to the preceding custom operation is "my_func_3d_float_int32_t". Different input parameters correspond to different symbol names.

Note that the type name here uses `cname`, which indicates the name of the corresponding type in the Ascend C language. For example, the cname corresponding to int32 is `int32_t`. Because we usually declare these functions using macros and embed the related type names into the function names, `cname` is commonly used.

### Source code and compilation

If the function that implements the custom operation needs to be generated from source code or bytecode, you need to configure the `source` and `compile` attributes when registering the custom operation class.

- `source` defines the path of the source code or bytecode file that implements the custom operation function.
- `compile` defines the compile command for implementing the custom operation function. In the command, `%<` and `%@` can be used to represent the source file and target file, respectively (similar to Makefile).

Similar to symbol names, these two attributes can also be statically or dynamically configured in the constructor of the registration class. For example:

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    ...
    source = "workspace/my_custom_op.cce"
    compile = "bisheng -std=c++17 -O2 -o $@ -c $<"

```

### Parameter conversion rules

#### Parameter Sequencing

Custom operations are converted into corresponding function calls. The parameter sequence is the same as that on the Python side. The output parameter (out, if any) is always placed at the end. For example, the following Python code is used:

```python
al.custom('my_custom_op', src, index, dim, out=dst)
```

Converting to a function call, it is equivalent to:

```cpp
my_custom_op(src, index, dim, dst);
```

#### List and tuple parameters

The tuple or list parameter on the Python side is flattened. For example:

```python
al.custom('my_custom_op', src, index, offsets=(1, 2, 3), out=dst)
```

When converted to a function call, the offsets parameter is flattened:

```cpp
my_custom_op(src, index, 1, 2, 3, dst);
```

### Constant parameter types

Custom operations support constant parameter types of integers and floating points. However, the integer and floating point types in Python do not distinguish bit widths. Therefore, by default, only integers are mapped to the int32_t type, and floating-point numbers are mapped to the float type. If the constant parameter of the implementation function is of another bit width type (for example, int64_t), an error occurs due to a mismatch in the function signature.

For example, the implementation function signature of a custom operation is as follows:

```cpp
custom_op_impl_func(memref_t<...> *src, memref_t<...> *idx, int64_t bound);
```

The bound parameter must be an integer of the int64_t type.

When a custom operation is called in Python, the value of the bound constant parameter is provided:

```python
al.custom('my_custom_op', src, idx, bound=1024)
```

Because Python integer constants do not distinguish bit widths, we can only map bound to int32_t by default. As a result, the signature does not match the implementation function, causing an error.

To avoid such issues, we recommend that all integer parameters of the implementation function use int32_t and all floating-point parameters use float. In some specific scenarios, we provide the following methods to specify the exact type:

#### Specify the integer bit width using al.int64

By default, integer constants are mapped to the int32_t type. If the implementation function requires an int64_t type, you can use al.int64 to wrap the integer. For example:

```python
al.custom('my_custom_op', src, idx, bound=al.int64(1024))
```

#### Specify the type using type hint

In the constructor of the registered class, you can add type annotations to the corresponding parameters. For example:

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, idx, bound: tl.int64):
        ...

```

In this way, the bound parameter is always mapped to the int64_t type.

#### Dynamically specifying the parameter types

Another extreme case is that the parameter type varies depending on other parameters. For example, the bound type must be the same as that of idx. You can use `arg_type` to dynamically specify the type in the constructor. For example:

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, idx, bound):
        ...
        self.arg_type['bound'] = idx.dtype

```

### Encapsulating custom operations

Directly using `al.custom` to call custom operations can sometimes be cumbersome, especially when there are output parameters. Before calling the operation, you need to prepare the output parameters. For example:

```python
dst = tl.full(index.shape, 0, tl.float32)
x = al.custom('my_custom_op', src, index, out=dst)
```

We can encapsulate the custom operation into an operation function for easier use. For example:

```python
@al.builtin
def my_custom_op(src, index, _builder=None):
    dst = tl.semantic.full(index.shape, 0, src.dtype.element_ty, _builder)
    return al.custom_semantic(_my_custom_op.name, src, index, out=dst, _builder=_builder)
```

The encapsulated operation function needs to be decorated with `al.builtin` and the custom operation needs to be called through `al.custom_semantic`. You can use the functions provided by `tl.semantic` to prepare output parameters. Note that when encapsulating the operation function, you need to provide an additional `_builder` parameter and pass it to all semantic functions.

The encapsulated operation function can be directly called like a native operation:

```python
@triton.jit
def my_kernel(...):
    ...
    x = my_custom_op(src, index)
    ...
```

## Triton Extended Enumerations

### SYNC_IN_VF

| Enumerated Value| Description |
|--------|----------|
| `VV_ALL` | Blocks the execution of vector load/store instructions until all vector load/store instructions are complete.|
| `VST_VLD` | Blocks the execution of vector load instructions until all vector store instructions are complete.|
| `VLD_VST` | Blocks the execution of vector store instructions until all vector load instructions are complete.|
| `VST_VST` | Blocks the execution of vector store instructions until all vector store instructions are complete.|
| `VS_ALL` | Blocks the execution of scalar load/store instructions until all vector load/store instructions are complete.|
| `VST_LD` | Blocks the execution of scalar load instructions until all vector store instructions are complete.|
| `VLD_ST` | Blocks the execution of scalar store instructions until all vector load instructions are complete.|
| `VST_ST` | Blocks the execution of scalar store instructions until all vector store instructions are complete.|
| `SV_ALL` | Blocks the execution of vector load/store instructions until all scalar load/store instructions are complete.|
| `ST_VLD` | Blocks the execution of vector load instructions until all scalar store instructions are complete.|
| `LD_VST` | Blocks the execution of vector store instructions until all scalar load instructions are complete.|
| `ST_VST` | Blocks the execution of vector store instructions until all scalar store instructions are complete.|

### PIPE

| Enumerated Value| Description |
|--------|------|
| `PIPE_S` | Scalar compute pipeline|
| `PIPE_V` | Vector compute pipeline|
| `PIPE_M` | Memory operation pipeline|
| `PIPE_MTE1` | Memory transfer engine 1 pipeline|
| `PIPE_MTE2` | Memory transfer engine 2 pipeline|
| `PIPE_MTE3` | Memory transfer engine 3 pipeline|
| `PIPE_ALL` | All pipelines|
| `PIPE_FIX` | Fixed-function pipeline|
