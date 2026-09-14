# Triton Integration

[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) is an important component that facilitates the integration of Triton into the Ascend platform. After the build and installation of Triton Ascend are completed, users can select Ascend as the backend when executing Triton operators.

## Installation and Execution

### Environment Preparation

Triton-Ascend requires Python 3.9 to 3.11 (inclusive). Its runtime depends on the Ascend CANN environment, the torch_npu package, and the triton-ascend package.

1. Install Ascend CANN

    Download the Toolkit package and the ops package corresponding to your hardware from the [Ascend Community CANN download page](https://www.hiascend.com/cann/download).

    ```bash
    # Take the CANN installation for Atlas A3 series products in an x86 environment as an example, where {version} is replaced with the actual CANN version number, such as 9.0.0.
    chmod +x Ascend-cann_{version}_linux-x86_64.run
    chmod +x Ascend-cann-A3-ops_{version}_linux-x86_64.run
    ./Ascend-cann_{version}_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
    ./Ascend-cann-A3-ops_{version}_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
    # Install the Python dependencies of CANN.
    pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pyyaml
    ```

2. Set environment variables:

    ```bash
    # For version 8.5.0 and earlier, the path is ${PATH-TO-CANN}/ascend-toolkit/set_env.sh.
    source ${PATH-TO-CANN}/cann/set_env.sh
    ```

3. Install torch_npu and triton-ascend

   Use the fixed matching version torch_npu==2.7.1. The installation command is as follows:

   ```bash
   pip install torch_npu==2.7.1
   pip install triton-ascend
   ```

### Triton Kernel Invocation and Verification

After Triton-Ascend is successfully installed, you can try to invoke the relevant Triton Kernel and verify its functionality by running `pytest -sv <file>.py`. If the functionality is correct, `PASS` is output.

Code example:

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
    # Generate data.
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = generate_tensor(shape, dtype).npu()
    x1 = generate_tensor(shape, dtype).npu()
    # torch result.
    torch_res = torch_lt(x0, x1).to(eval('torch.' + dtype))
    # triton result.
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_lt[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # Compare the results.
    validate_cmp(dtype, triton_res, torch_res)
```

**Dynamic tiling support**: The parallel granularity is configured through the grid parameter in `[]`, and the tiling size is controlled through the `XBLOCK` and `XBLOCK_SUB` parameters. Users can adjust them as needed.

**Dynamic shape support**: The kernel automatically adapts to 1D tensors of any length. Users only need to pass in data with the actual shape.

## Conversion from Triton Ops to Ascend NPU IR Ops

Triton Ascend progressively lowers the high-level GPU abstraction operations of the Triton dialect into target dialects such as Linalg, HFusion, and HIVM, ultimately generating an optimized intermediate representation that can be executed efficiently on the Ascend NPU. The following table lists in detail the various Triton operations and their corresponding Ascend NPU IR operations during the lowering process.

**Memory access Ops**:

| Triton Op             | Target Ascend NPU IR Op                                      | Description                     |
| :-------------------- | :----------------------------------------------------------- | :------------------------------ |
| `triton::StoreOp`     | `memref::copy`                                               | Stores data to memory.          |
| `triton::LoadOp`      | `memref::copy` + `bufferization::ToTensorOp`                 | Loads data from memory.         |
| `triton::AtomicRMWOp` | `hivm::StoreOp` or `hfusion::AtomicXchgOp`                   | Performs an atomic read-modify-write operation. |
| `triton::AtomicCASOp` | `linalg::GenericOp`                                          | Performs an atomic compare-and-swap operation. |
| `triton::GatherOp`    | First converted to `func::CallOp` (calling `triton_gather`)<br>then converted to `hfusion::GatherOp` | Collects data by index.         |

**Pointer arithmetic Ops**:

| Triton Op            | Target Ascend NPU IR Op        | Description               |
| :------------------- | :---------------------------- | :------------------------ |
| `triton::AddPtrOp`   | `memref::ReinterpretCast`     | Performs offset arithmetic on a pointer. |
| `triton::PtrToIntOp` | `arith::IndexCastOp`          | Converts a pointer to an integer. |
| `triton::IntToPtrOp` | `hivm::PointerCastOp`         | Converts an integer to a pointer. |
| `triton::AdvanceOp`  | `memref::ReinterpretCastOp`   | Advances the pointer position. |

**Program information Ops**:

| Triton Op                  | Target Ascend NPU IR Op                                       | Description             |
| :------------------------- | :------------------------------------------------------------ | :---------------------- |
| `triton::GetProgramIdOp`   | Parameter of `functionOp`                                     | Obtains the ID of the current program. |
| `triton::GetNumProgramsOp` | Parameter of `functionOp`                                     | Obtains the total number of programs. |
| `triton::AssertOp`         | First converted to `func::CallOp` (calling `triton_assert`)<br>then converted to `hfusion::AssertOp` | Assertion operation. |
| `triton::PrintOp`          | First converted to `func::CallOp` (calling `triton_print`)<br>then converted to `hfusion::PrintOp` | Print operation. |

**Tensor operation Ops**:

| Triton Op              | Target Ascend NPU IR Op                                       | Description                       |
| :--------------------- | :------------------------------------------------------------ | :-------------------------------- |
| `triton::ReshapeOp`    | `tensor::ReshapeOp`                                           | Changes the tensor shape.         |
| `triton::ExpandDimsOp` | `tensor::ExpandShapeOp`                                       | Expands the tensor dimensions.    |
| `triton::BroadcastOp`  | `linalg::BroadcastOp`                                         | Broadcasts the tensor.            |
| `triton::TransOp`      | `linalg::TransposeOp`                                         | Transposes the tensor.            |
| `triton::SplitOp`      | `tensor::ExtractSliceOp`                                      | Splits the tensor.                |
| `triton::JoinOp`       | `tensor::InsertSliceOp`                                       | Joins the tensor.                 |
| `triton::CatOp`        | `tensor::InsertSliceOp`                                       | Concatenates the tensor.          |
| `triton::MakeRangeOp`  | `linalg::GenericOp`                                           | Creates a tensor containing consecutive integers. |
| `triton::SplatOp`      | `linalg::FillOp`                                              | Fills the tensor with a scalar value. |
| `triton::SortOp`       | First converted to `func::CallOp` (calling `triton_sort`)<br>then converted to `hfusion::SortOp` | Sorts the tensor.                 |

**Numerical computation Ops**:

| Triton Op                | Target Ascend NPU IR Op                                         | Description                         |
| :----------------------- | :----------------------------------------------------------- | :--------------------------- |
| `triton::MulhiUIOp`      | `arith::MulSIExtendedOp`                                     | Unsigned integer multiplication, returning the high-order result. |
| `triton::PreciseDivFOp`  | `arith::DivFOp`                                              | Performs high-precision floating-point division.           |
| `triton::PreciseSqrtOp`  | `math::SqrtOp`                                               | Performs high-precision floating-point square root.         |
| `triton::BitcastOp`      | `arith::BitcastOp`                                           | Reinterprets bits between different types. |
| `triton::ClampFOp`       | `tensor::EmptyOp` + `linalg::FillOp`                         | Clamps a floating-point number to a specified range.     |
| `triton::DotOp`          | `linalg::MatmulOp`                                           | Performs general matrix multiplication.             |
| `triton::DotScaledOp`    | `linalg::MatmulOp`                                           | Performs matrix multiplication with a scaling factor.     |
| `triton::ascend::FlipOp` | First converted to `func::CallOp` (calling `triton_flip`)<br>then converted to `hfusion::FlipOp` | Flips a tensor along a specified dimension.     |

**Reduction Ops**:

| Triton Op | Target Ascend NPU IR Op | Description |
| :--- | :--- | :--- |
| `triton::ArgMinOp` | `linalg::ReduceOp` | Returns the index of the minimum value in the tensor. |
| `triton::ArgMaxOp` | `linalg::ReduceOp` | Returns the index of the maximum value in the tensor. |
| `triton::ReduceOp` | `linalg::ReduceOp` | General reduction operation. |
| `triton::ScanOp` | First converted to `func::CallOp` (calling `triton_cumsum` or `triton_cumprod`)<br>then converted to `hfusion::CumsumOp` and `hfusion::CumprodOp` | Performs a scan operation (such as cumulative sum and cumulative product). |

## Triton Extension Operations

The Ascend NPU IR increment provides language features, based on which Triton-Ascend extends a set of operations on the NPU IR. To enable the related capabilities, you need to import the following module.

```python
import triton.language.extra.cann.extension as al
```

Thereafter, you can use the related Ascend Language exclusive interfaces. In addition, since Ascend Language provides low-level interfaces, these interfaces are not compatible.

### Synchronization and Debugging Operations

#### debug_barrier

**Function**: Ascend provides multiple synchronization modes, including the internal synchronization mode of the vector pipeline, for fine-grained synchronization control during debugging and performance optimization.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `sync_mode` | [SYNC_IN_VF enumeration value](#sync_in_vf) | Vector pipeline synchronization mode |

**Code example**:

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

**Function**: Ascend supports setting synchronization events between the compute unit and the vector unit. `sync_block_set` and `sync_block_wait` must be used together.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `sender` | str | Sender unit type. |
| `receiver` | str | Receiver unit type. |
| `event_id` | int | Event identifier. |
| `sender_pipe_value` | [PIPE enum value](#pipe) | Sender pipe value. |
| `receiver_pipe_value` | [PIPE enum value](#pipe) | Receiver pipe value. |

**Code example**:

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

**Function**: Ascend supports global synchronization for the entire compute block, ensuring that all compute cores of the specified type complete the current operation.

**Parameters**:

| Parameter | Type | Description | Valid Values |
|--------|------|------|--------|
| `mode` | str | Synchronization mode, specifying the type of cores to synchronize. | "all_cube", "all_vector", "all", "all_sub_vector" |
| `event_id` | int | Synchronization event identifier. | 0 ~ 15 |

**Synchronization Mode Details**:

| Mode | Description | Synchronization Scope |
|------|------|----------|
| "all_cube" | Synchronizes all Cube cores. | All Cube cores on the current AI Core |
| "all_vector" | Synchronizes all Vector cores. | All Vector cores on the current AI Core |
| "all" | Synchronizes all cores. | All compute cores (Cube+Vector) on the current AI Core |
| "all_sub_vector" | Synchronizes all sub-Vector cores. | All sub-Vector cores on the current AI Core |

**Code example**:

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

**Function**: Ascend provides interfaces for querying hardware information. By calling the `sub_vec_id` interface, the Vector core index on the current AI Core can be obtained. By calling the `sub_vec_num` interface, the number of Vector cores on a single AI Core can be obtained.

**Code example**:

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

**Function**: Ascend extends the standard Python `range` functionality by adding the `parallel` iterator with parallel execution semantics.

**Parameters**:

| Parameter | Type | Description | Example |
|------|------|------|------|
| `arg1` | int | Start value or stop value. | `parallel(10)` |
| `arg2` | int | Stop value (optional). | `parallel(0, 10)` |
| `step` | int | Step (optional). | `parallel(0, 10, 2)` |
| `num_stages` | int | Number of pipeline stages (optional). | `parallel(0, 10, num_stages=3)` |
| `loop_unroll_factor` | int | Loop unrolling factor (optional). | `parallel(0, 10, loop_unroll_factor=4)` |

**Constraints**:

Currently, Atlas A2 training products/Atlas A2 inference products support a maximum of two Vector cores.

**Code example**:

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

### Compile Optimization Hints

#### compile_hint

**Function**: Ascend supports passing optimization hint information to the compiler to guide code generation and performance optimization.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `ptr` | tensor | Pointer to the target tensor. |
| `hint_name` | str | Hint name. |
| `hint_val` | Multiple types | Hint value (optional). |

**Code example**:

```python
@triton.jit
def triton_where_lt_case1():
    # ...
    mask = tl.where(cond, in1, in0)
    al.compile_hint(mask, "bitwise_mask")
    # ...
```

#### multibuffer

**Function**: `multibuffer` is a function used to set up multiple buffering (double buffering) for an existing tensor, optimizing data flow and computation overlap through compiler hints.

**Parameters**:

| Parameter | Type | Description |
|------|------|------|
| `src` | tensor | Tensor to be multi-buffered. |
| `size` | int | Number of buffer copies. |

**Code example**:

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

**Function**: Ascend supports a scope manager that adds hint information to a block of code. One usage is to specify the cube or vector type through `core_mode`.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `core_mode` | str | Core type, which specifies the compute core used by operations within the block. Only "cube" or "vector" modes are accepted. |

**Core mode options**:

| Mode | Description |
|------|------|
| "cube" | Uses the Cube core for computation. |
| "vector" | Uses the Vector core for computation. |

**Code example**:

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

### Tensor Slicing Operation

#### insert_slice & extract_slice

**Function**: Ascend supports inserting a tensor into another tensor (that is, `insert_slice`) or extracting a specified slice from another tensor (that is, `extract_slice`) based on the offset, size, and stride parameters of the operation.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `ful` | Tensor | Target tensor that receives the insertion. |
| `sub` | Tensor | Source tensor to be inserted. |
| `offsets` | integer tuple | Starting offset of the insertion operation. |
| `sizes` | integer tuple | Size range of the insertion operation. |
| `strides` | integer tuple | Stride parameter of the insertion operation. |

**Code example**:

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

**Function**: Ascend supports reading a single element value at a specified index position from a tensor.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `src` | tensor | Source tensor to be accessed. |
| `indice` | integer tuple | Index position of the element to be obtained. |

**Code example**:

```python
@triton.jit
def index_select_manual_kernel():
    # ...
    gather_offset = al.get_element(indices, (i,)) * g_stride
    val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
    # ...
```

### Tensor Computation Operation

#### sort

**Function**: Ascend supports sorting the input tensor along a specified dimension.

**Parameters**:

| Parameter | Type | Description | Default Value |
|--------|------|------|--------|
| `ptr` | `tensor` | Input tensor. | - |
| `dim` | `int` or `tl.constexpr[int]` | Dimension to sort. | -1 |
| `descending` | `bool` or `tl.constexpr[bool]` | Sort direction, where `True` indicates descending order and `False` indicates ascending order. | `False` |

**Code example**:

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

**Function**: Ascend supports flipping the input tensor along a specified dimension.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `ptr` | tensor | Input tensor. |
| `dim` | int or tl.constexpr[int] | Dimension to flip. |

**Code example**:

```python
@triton.jit
def flip_kernel_2d():
    # ...
    input = tl.load(input_ptr + offset)
    flipped_input = flip(input, dim=2)
    # ...
```

#### cast

**Function**: Ascend converts a tensor to a specified data type, supporting numeric conversion, bitcast, and overflow handling.

**Parameters**:

| Parameter | Type | Description | Default Value |
|--------|------|------|--------|
| `input` | tensor | Input tensor. | - |
| `dtype` | dtype | Target data type. | - |
| `fp_downcast_rounding` | str, optional | Rounding mode for floating-point downcast. | `None` |
| `bitcast` | bool, optional | Whether to perform bitcast (instead of numeric conversion). | `False` |
| `overflow_mode` | str, optional | Overflow handling mode. | `None` |

**Code example**:

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

### Index and Collection Operations

#### _index_select

**Function**: Ascend supports collecting data from a source GM tensor into an output UB tensor along a specified dimension based on an index UB tensor, using the SIMT template. This operation supports 2D–5D tensors.

**parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `src` | pointer type | Source tensor pointer (in GM). |
| `index` | tensor | Index tensor used for collection (in UB). |
| `dim` | int | Dimension along which collection is performed. |
| `bound` | int | Upper bound of index values. |
| `end_offset` | integer tuple | Ending offset of each dimension of the index tensor. |
| `start_offset` | integer tuple | Starting offset of each dimension of the source tensor. |
| `src_stride` | integer tuple | Stride of each dimension of the source tensor. |
| `other` (optional) | scalar value | Default value when the index is out of bounds (in UB). |
| `out` | tensor | Output tensor (in UB). |

**Code example**:

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

**Function**: Ascend supports placing the value tensor into the target tensor according to the index tensor.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `ptr` | tensor (pointer type) | Target tensor pointer (in GM). |
| `index` | tensor | Index used for placement (in UB). |
| `value` | tensor | Value to be stored (in UB). |
| `dim` | int32 | Dimension along which index placement is performed. |
| `index_boundary` | int64 | Upper bound of index values. |
| `end_offset` | int tuple | Ending offset of the placement region in each dimension. |
| `start_offset` | int tuple | Starting offset of the placement region in each dimension. |
| `dst_stride` | int tuple | Stride of each dimension of the target tensor. |

**Index placement rules**:

- Two-dimensional index placement

    `dim = 0: out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]`

- Three-dimensional index placement

    `dim = 0: out[index[i]][start_offset[1]:end_offset[1]][start_offset[2]:end_offset[2]]  = value[i][0:end_offset[1]-start_offset[1]][0:end_offset[2]-start_offset[2]]`

    `dim = 1: out[start_offset[0]:end_offset[0]][index[j]][start_offset[2]:end_offset[2]] = value[0:end_offset[0]-start_offset[0]][j][0:end_offset[2]-start_offset[2]]`

**Constraints**:

- `ptr` and `value` must have the same rank.
- `ptr.dtype` currently supports only `float16`, `bfloat16`, and `float32`.
- `index` must be an integer tensor. If `index.rank` != 1, it is reshaped to 1D.
- `index.numel` must be equal to `value.shape[dim]`.
- `value` supports 2- to 5-dimensional tensors.
- `dim` must be valid (`0 ≤ dim < rank(value) - 1`).

**Code example**:

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

**Function**: Ascend supports scatter collection of data from GM to UB along a specified dimension. This operation supports index boundary checking to ensure efficient and safe data transfer.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `src` | tensor (pointer type) | Source tensor pointer (located in GM). |
| `index` | tensor | Index tensor used for collection (located in UB). |
| `index_boundary` | int64 | Upper bound of index values. |
| `dim` | int32 | Dimension along which collection is performed. |
| `src_stride` | int64 tuple | Stride of each dimension of the source tensor. |
| `end_offset` | int32 tuple | Ending offset of each dimension of the index tensor. |
| `start_offset` | int32 tuple | Starting offset of each dimension of the index tensor. |
| `other` | Scalar value (optional) | Default value used when the index is out of bounds (located in UB). |

**Return Value**:

- **Type**: tensor
- **Description**: Result tensor located in UB, with the same shape as `index.shape`.

**Scatter Collection Rules**:

- One-dimensional index collection

    `dim = 0: out[i] = src[start_offset[0] + index[i]]`

- Two-dimensional index collection

    `dim = 0: out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]`

    `dim = 1: out[i][j] = src[start_offset[0] + i][start_offset[1] + index[i][j]]`

- Three-dimensional index collection

    `dim = 0: out[i][j][k] = src[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k]`

    `dim = 1: out[i][j][k] = src[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k]`

    `dim = 2: out[i][j][k] = src[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]]`

**Constraints**:

- `src` and `index` must have the same rank.
- `src.dtype` currently supports only `float16`, `bfloat16`, and `float32`.
- `index` must be an integer tensor with a rank between 1 and 5.
- `dim` must be valid (`0 ≤ dim < rank(index)`).
- `other` must be a scalar value.
- For each dimension `i` that is not equal to `dim`, `index.size[i]` ≤ `src.size[i]`.
- The output shape is the same as `index.shape`. If `index` is `None`, the output tensor is an empty tensor with the same shape as `index`.

**Code example**:

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

**Function**: Ascend supports scatter storage of data from UB to GM along a specified dimension. This operation supports index boundary checking to ensure efficient and safe data transfer.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `ptr` | tensor (pointer type) | Target tensor pointer (located in GM). |
| `value` | tensor | Tile value to be stored (located in UB). |
| `index` | tensor | Index used for scatter storage (located in UB). |
| `index_boundary` | int64 | Upper bound of index values. |
| `dim` | int32 | Dimension along which scatter storage is performed. |
| `dst_stride` | int64 tuple | Stride of each dimension of the target tensor. |
| `end_offset` | int32 tuple | Ending offset of each dimension of the index tensor. |
| `start_offset` | int32 tuple | Starting offset of each dimension of the index tensor. |

**Scatter storage rules**:

- One-dimensional index scatter

    `dim = 0: out[start_offset[0] + index[i]] = value[i]`

- Two-dimensional index scatter

    `dim = 0: out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]`

    `dim = 1: out[start_offset[0] + i][start_offset[1] + index[i][j]] = value[i][j]`

- Three-dimensional index scatter.

    `dim = 0: out[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k] = value[i][j][k]`

    `dim = 1: out[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k] = value[i][j][k]`

    `dim = 2: out[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]] = value[i][j][k]`

**Constraints**:

- `ptr`, `index`, and `value` must have the same rank.
- `ptr.dtype` currently supports only `float16`, `bfloat16`, and `float32`.
- `index` must be an integer tensor with a rank between 1 and 5.
- `dim` must be valid (`0 ≤ dim < rank(index)`).
- For each dimension `i` that is not equal to `dim`, `index.size[i]` ≤ `ptr.size[i]`.
- The output shape is the same as `index.shape`. If `index` is `None`, the output tensor is an empty tensor with the same shape as `index`.

**Code example**:

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

**Description**: Ascend supports the parallel index selection operation, which directly loads data selected from multiple points in GM into UB, achieving zero-copy efficient reading.

**Parameters**:

| Parameter | Type | Description |
|--------|------|------|
| `src` | tensor (pointer type) | Source tensor pointer (located in GM). |
| `dim` | int or constexpr | Dimension along which indices are selected. |
| `index` | tensor | One-dimensional tensor of indices to be selected (located in UB). |
| `src_shape` | List[Union[int, tensor]] | Complete shape of the source tensor (can be integers or tensors). |
| `src_offset` | List[Union[int, tensor]] | Starting offset for reading (can be integers or tensors). |
| `read_shape` | List[Union[int, tensor]] | Size to be read (tile shape, can be integers or tensors). |

**Constraints**:

- `read_shape[dim]` must be -1.
- `src_offset[dim]` can be -1 (which will be ignored).
- Boundary handling: when `src_offset + read_shape > src_shape`, it is automatically truncated to the `src_shape` boundary.
- No check is performed on whether `index` contains out-of-bounds values.

**Return Value**:

- **Type**: tensor
- **Description**: Result tensor located in UB, where the `dim` dimension in its shape is replaced by the length of `index`.

**Code example**:

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

## Triton-Exclusive Custom Operations

On the Ascend 950PR/Ascend 950DT architecture, the Custom Op of Triton-Ascend allows users to customize operations and use them. At runtime, a custom operation is converted into a call to a device-side implementation function, which can call existing library functions or implementation functions generated by compiling source code or bytecode provided by the user.

### Registering and Using Custom Operations

**Registering a custom operation**:

Custom operation-related functionality is provided by the Triton Ascend extension package. A user-defined custom operation must be registered before it can be used. You can define and register a custom operation by decorating a class with the `register_custom_op` decorator provided by the extension package:

```python
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

```

Registering the simplest custom operation requires at least the following basic attributes: `name`, `core`, `pipe`, and `mode`, where:

- `name` indicates the operation name, which is the unique identifier of this custom operation. If omitted, the class name is used by default.
- `core` indicates the type of Ascend core on which the operation runs.
- `pipe` indicates the corresponding pipeline.
- `mode` indicates the programming mode used.

**Using a custom operation**:

A registered custom operation can be invoked through the `custom()` function provided by the Ascend extension package. When invoking it, you need to provide the name of the custom operation and its arguments:

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

The parameters of `custom()` consist of three parts: the operation name, the input parameters, and the optional output parameters:

- **Operation name**: Must be consistent with the registered operation name.
- **Input parameters**: Different operations have different input parameters.
- **Output parameters (optional)**: The output parameters are specified by `out`, indicating the output of the operation.

If the output variable is specified through the `out` parameter, the return value of the custom operation is consistent with the output variable; otherwise, the return value of the operation is unavailable.

### Built-in Custom Operations

The names of built-in custom operations all start with *__builtin_*. They are customized operations built into triton-ascend and can be used directly without registration. For example:

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

Specific built-in custom operations vary by version. Refer to the documentation of the corresponding version.

### Parameter Validity Check

Without constraints, users can pass any arguments to the `al.custom()` function. If the number or types of the passed arguments do not match the expected ones, a runtime error occurs.

To avoid this situation and improve the user experience of custom operations, a constructor can be provided for the registered custom class to describe the argument list and perform parameter validity checks. For example:

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

The constructor argument list of the registered class is exactly the argument list required by the custom operation. When calling the operation, arguments that meet the requirements must be provided. For example:

```python
    res = al.custom('my_custom_op', src_ptr, index, dim=1, out=dst)
```

If the provided arguments are incorrect, a compilation error is reported. For example, the `dim` argument here is required to be an integer constant. If a floating-point number is provided, the following error is reported:

```text
    ...
    res = al.custom('my_custom_op', src_ptr, index, dim=1.0, out=dst)
          ^
AssertionError('dim must be an integer')
```

### Output Parameters and Return Value

`al.custom` returns the output parameters specified by the out parameter, for example:

```python
x = al.custom('my_custom_op', src, index, out=dst)
```

returns **dst** to **x**.

The out parameter can specify multiple output parameters, and `al.custom` returns a tuple containing these output parameters:

```python
x, y = al.custom('my_custom_op', src, index, out=(dst1, dst2))
```

returns **dst1** to **x** and **dst2** to **y**.

When no out parameter is specified, `al.custom` has no return value (returns `None`).

### Symbol Name of the Called Function

A custom operation is ultimately converted into a call to the device-side implementation function. You can configure the symbol name of this function by registering the `symbol` attribute in the custom operation class. If the `symbol` attribute is not set, the name of the custom operation is used as the function name by default.

**Static symbol name**:

If a custom operation always calls a fixed device-side function, the symbol name can be set statically:

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = '_my_custom_op_symbol_name_'

```

In this way, `al.custom('my_custom_op', ...)` is fixed to correspond to the `_my_custom_op_symbol_name_(...)` function on the device side.

**Dynamic symbol name**:

In many cases, the same custom operation needs to call different device-side functions based on the dimensions, types, and other attributes of the input parameters, in which case the symbol name must be set dynamically. Similar to parameter validity checking, the symbol name can be set dynamically in the constructor of the registered custom operation class, for example:

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

When the input `src` is a pointer to the `float32` type and `index` is a 3-dimensional tensor of the `int32` type, the device-side function symbol name corresponding to the above custom operation is `"my_func_3d_float_int32_t"`. Different input parameters correspond to different symbol names.

Note that the type name used here is `cname`, which represents the name of the corresponding type in the AscendC language. For example, the `cname` corresponding to `int32` is `int32_t`. Because these functions are usually declared using macros and the relevant type names are embedded into the function names, `cname` is commonly used.

### Source Code and Compilation

If the function that implements a custom operation needs to be generated by compiling source code or bytecode, the `source` and `compile` attributes must be configured separately when registering the custom operation class:

- `source`: Path to the source code or bytecode file that implements the custom operation function.
- `compile`: Compilation command that implements the custom operation function, where `%<` and `%@` represent the source file and the target file, respectively (similar to Makefile).

Similar to the symbol name, these two attributes can also be configured statically or dynamically in the constructor of the registration class, for example:

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

### Parameter Conversion Rules

**Parameter order**:

A custom operation is converted into a corresponding function call, and its parameter order remains consistent with that on the Python side, where the output parameter `out` (if present) is uniformly placed at the end of the parameter list.

Code example:

```python
al.custom('my_custom_op', src, index, dim, out=dst)
```

The conversion to a function call is equivalent to:

```cpp
my_custom_op(src, index, dim, dst);
```

**List and tuple parameters**:

A `tuple` or `list` parameter on the Python side is flattened, for example:

```python
al.custom('my_custom_op', src, index, offsets=(1, 2, 3), out=dst)
```

When converted into a function call, the `offsets` parameter is flattened:

```cpp
my_custom_op(src, index, 1, 2, 3, dst);
```

### Constant Parameter Types

Custom operations support constant parameter types of integers and floating-point numbers. However, Python integer and floating-point types do not distinguish bit width, so we can only map integers to the `int32_t` type and floating-point numbers to the `float` type by default. When the constant parameter of the implementation function is of another bit width (such as `int64_t`), an error occurs due to a function signature mismatch.

For example, the following is the implementation function signature of a custom operation:

```cpp
custom_op_impl_func(memref_t<...> *src, memref_t<...> *idx, int64_t bound);
```

Its `bound` parameter requires an integer of type `int64_t`.

When calling the custom operation on the Python side, the value of the `bound` constant parameter is provided:

```python
al.custom('my_custom_op', src, idx, bound=1024)
```

Because Python integer constants do not distinguish bit width, we can only map `bound` to `int32_t` by default, which causes a mismatch with the implementation function signature and results in an error.

To avoid such issues, we recommend that the parameters of the implementation function all use `int32_t` for integers and `float` for floating-point numbers. In certain specific scenarios, we also provide the following methods to specify the exact type:

- **Specify the integer bit width through al.int64**

  By default, integer constants are mapped to the `int32_t` type. If the implementation function requires an `int64_t` type, you can wrap the integer with `al.int64`, for example:

  ```python
  al.custom('my_custom_op', src, idx, bound=al.int64(1024))
  ```

- **Specify the type through a type hint**

  In the constructor of the registered class, type annotations can be added to the corresponding parameters, for example:

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

  In this way, the `bound` parameter is always mapped to the `int64_t` type.

- **Dynamically specify the parameter type**:

  There is also a more extreme case where the parameter type varies depending on other parameters. For example, the type of `bound` needs to be consistent with the data type of `idx`. In this case, the type can be dynamically specified in the constructor through `arg_type`, for example:

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

### Wrapping Custom Operations

Calling a custom operation directly with `al.custom` can sometimes be cumbersome, especially when there are output parameters, because the output parameters must be prepared before the call. For example:

```python
dst = tl.full(index.shape, 0, tl.float32)
x = al.custom('my_custom_op', src, index, out=dst)
```

You can wrap the custom operation as an operation function for convenience. For example:

```python
@al.builtin
def my_custom_op(src, index, _builder=None):
    dst = tl.semantic.full(index.shape, 0, src.dtype.element_ty, _builder)
    return al.custom_semantic(_my_custom_op.name, src, index, out=dst, _builder=_builder)
```

The wrapped operation function must be decorated with `al.builtin` and invoke the custom operation through `al.custom_semantic`. At the same time, you can use the capabilities provided by `tl.semantic` to prepare the output parameters.

> **Note**:
>
> When wrapping an operation function, you need to provide an additional `_builder` parameter and pass it to all `semantic` functions.

The wrapped operation function can be called directly just like a native operation:

```python
@triton.jit
def my_kernel(...):
    ...
    x = my_custom_op(src, index)
    ...
```

## Triton Exclusive Extension Enums

### SYNC_IN_VF

| Enumeration Value | Description |
|--------|----------|
| `VV_ALL` | Blocks the execution of vector load/store instructions until all vector load/store instructions complete. |
| `VST_VLD` | Blocks the execution of vector load instructions until all vector store instructions complete. |
| `VLD_VST` | Blocks the execution of vector store instructions until all vector load instructions complete. |
| `VST_VST` | Blocks the execution of vector store instructions until all vector store instructions complete. |
| `VS_ALL` | Blocks the execution of scalar load/store instructions until all vector load/store instructions complete. |
| `VST_LD` | Blocks the execution of scalar load instructions until all vector store instructions complete. |
| `VLD_ST` | Blocks the execution of scalar store instructions until all vector load instructions complete. |
| `VST_ST` | Blocks the execution of scalar store instructions until all vector store instructions complete. |
| `SV_ALL` | Blocks the execution of vector load/store instructions until all scalar load/store instructions complete. |
| `ST_VLD` | Blocks the execution of vector load instructions until all scalar store instructions complete. |
| `LD_VST` | Blocks the execution of vector store instructions until all scalar load instructions complete. |
| `ST_VST` | Blocks the execution of vector store instructions until all scalar store instructions complete. |

### PIPE

| Enumeration Value | Description |
|--------|------|
| `PIPE_S` | Scalar computation pipeline. |
| `PIPE_V` | Vector computation pipeline. |
| `PIPE_M` | Memory operation pipeline. |
| `PIPE_MTE1` | Memory transfer engine 1 pipeline. |
| `PIPE_MTE2` | Memory transfer engine 2 pipeline. |
| `PIPE_MTE3` | Memory transfer engine 3 pipeline. |
| `PIPE_ALL` | All pipelines. |
| `PIPE_FIX` | Fixed-function pipeline. |
