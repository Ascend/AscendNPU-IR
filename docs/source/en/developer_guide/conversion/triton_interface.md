# Triton Integration

[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) is an important component that helps Triton access the Ascend platform. After the Triton Ascend is built and installed, you can use the Ascend as the backend when executing the Triton operator.

## Installation and Execution

### Installation

#### Python

Currently, the Python version required by Triton-Ascend is py3.9-py3.11.

#### Ascend CANN

The end-to-end operation of the Ascend NPU-IR depends on the CANN environment.

1. Download the CANN package: Download the toolkit package and the ops package corresponding to the hardware. You can download the toolkit package from[Ascend Community CANN Download Page](https://www.hiascend.com/cann/download) Get.
2. Install the CANN package.

    ```bash
    #In the x86 A3 environment, {version} indicates the CANN version, for example, 9.0.0.
    chmod +x Ascend-cann_{version}_linux-x86_64.run
    chmod +x Ascend-cann-A3-ops_{version}_linux-x86_64.run
    ./Ascend-cann_{version}_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
    ./Ascend-cann-A3-ops_{version}_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
    #Installing the Python Dependency of CANN
    pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pyyaml
    ```

3. Set environment variables:

    ```bash
    #If the version is earlier than 8.5.0, the path is ${PATH-TO-CANN}/ascend-toolkit/set_env.sh.
    source ${PATH-TO-CANN}/cann/set_env.sh
    ```

#### torch_npu & triton-ascend

Currently, the torch_npu version is 2.7.1.

```bash
pip install torch_npu==2.7.1
pip install triton-ascend
```

### Execution

After installing Triton-Ascend, you can call the related Triton Kernel. For details, see the following source code. You can run the`pytest -sv <file>.py`Verify the functions after the installation. If the function is correct, the terminal displays`PASS`.

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
def triton_lt(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr,XBLOCK_SUB: tl.constexpr):
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
    #Generate Data
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = generate_tensor(shape, dtype).npu()
    x1 = generate_tensor(shape, dtype).npu()
    #Torch results
    torch_res = torch_lt(x0, x1).to(eval('torch.' + dtype))
    #triton results
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_lt[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    #Compare Results
    validate_cmp(dtype, triton_res, torch_res)
```

**Dynamic tiling support**: The parallel granularity is configured by the grid parameter in \[\], and the tiling size is controlled by the XBLOCK and XBLOCK_SUB parameters. Users can adjust the size as required.

**Dynamic shape support**: The kernel automatically adapts 1D tensors of any length. You only need to transfer the actual shape data.

## Triton Op to AscendNPU IR Op Conversion

Triton Ascend degrades the advanced GPU abstraction operations of the Triton dialect to target dialects such as Linalg, HFusion, and HIVM, resulting in an optimized intermediate representation that can be efficiently executed on the Ascend NPU. The following table details the various Triton operations and their corresponding Ascend NPU IR operations in the fall process.

| Triton Op                         | Target Ascend NPU IR Op                                                                                                                          | Description                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| **Storage Access Op**             |                                                                                                                                                  |                                                                    |
| `triton::StoreOp`                 | `memref::copy`                                                                                                                                   | Store data in memory                                               |
| `triton::LoadOp`                  | `memref::copy`\+`bufferization::ToTensorOp`                                                                                                      | Load data from memory                                              |
| `triton::AtomicRMWOp`             | `hivm::StoreOp`or the`hfusion::AtomicXchgOp`                                                                                                     | Perform atomic read-modify-write operations.                       |
| `triton::AtomicCASOp`             | `linalg::GenericOp`                                                                                                                              | Perform atomic compare and swap operations                         |
| `triton::GatherOp`                | First convert to`func::CallOp`(Invokes`triton_gather`) and then converted into`hfusion::GatherOp`                                                | Collecting Data Based on Indexes                                   |
| **Pointer operation class Op.**   |                                                                                                                                                  |                                                                    |
| `triton::AddPtrOp`                | `memref::ReinterpretCast`                                                                                                                        | Performs an offset operation on a pointer.                         |
| `triton::PtrToIntOp`              | `arith::IndexCastOp`                                                                                                                             | Convert a pointer to an integer                                    |
| `triton::IntToPtrOp`              | `hivm::PointerCastOp`                                                                                                                            | Convert an integer to a pointer                                    |
| `triton::AdvanceOp`               | `memref::ReinterpretCastOp`                                                                                                                      | Push pointer position                                              |
| **Program information operation** |                                                                                                                                                  |                                                                    |
| `triton::GetProgramIdOp`          | `functionOp`Parameters for                                                                                                                       | Obtains the ID of the current program.                             |
| `triton::GetNumProgramsOp`        | `functionOp`Parameters of                                                                                                                        | Obtain the total number of programs.                               |
| `triton::AssertOp`                | First convert to`func::CallOp`(Invokes`triton_assert`) and then convert to`hfusion::AssertOp`                                                    | Assertion operation                                                |
| `triton::PrintOp`                 | First convert to`func::CallOp`(Invokes`triton_print`) and then convert to`hfusion::PrintOp`                                                      | Print operation                                                    |
| **Tensor Operation Op**           |                                                                                                                                                  |                                                                    |
| `triton::ReshapeOp`               | `tensor::ReshapeOp`                                                                                                                              | Changing the Tensor Shape                                          |
| `triton::ExpandDimsOp`            | `tensor::ExpandShapeOp`                                                                                                                          | Extended tensor dimension                                          |
| `triton::BroadcastOp`             | `linalg::BroadcastOp`                                                                                                                            | broadcast tensor                                                   |
| `triton::TransOp`                 | `linalg::TransposeOp`                                                                                                                            | transposed tensor                                                  |
| `triton::SplitOp`                 | `tensor::ExtractSliceOp`                                                                                                                         | division tensor                                                    |
| `triton::JoinOp`                  | `tensor::InsertSliceOp`                                                                                                                          | join tensor                                                        |
| `triton::CatOp`                   | `tensor::InsertSliceOp`                                                                                                                          | splicing tensor                                                    |
| `triton::MakeRangeOp`             | `linalg::GenericOp`                                                                                                                              | Creates a tensor that contains consecutive integers.               |
| `triton::SplatOp`                 | `linalg::FillOp`                                                                                                                                 | Filling Tensors with Scalar Values                                 |
| `triton::SortOp`                  | First convert to`func::CallOp`(Invokes`triton_sort`) and then convert to`hfusion::SortOp`                                                        | Sorting Tensors                                                    |
| **Value calculation type**        |                                                                                                                                                  |                                                                    |
| `triton::MulhiUIOp`               | `arith::MulSIExtendedOp`                                                                                                                         | Multiply unsigned integers, returning high-order results           |
| `triton::PreciseDivFOp`           | `arith::DivFOp`                                                                                                                                  | Perform high precision floating-point division                     |
| `triton::PreciseSqrtOp`           | `math::SqrtOp`                                                                                                                                   | Perform high precision floating-point square root                  |
| `triton::BitcastOp`               | `arith::BitcastOp`                                                                                                                               | Bit reinterpretation between different types                       |
| `triton::ClampFOp`                | `tensor::EmptyOp`\+`linalg::FillOp`                                                                                                              | Restricts floating point numbers to a specified range              |
| `triton::DotOp`                   | `linalg::MatmulOp`                                                                                                                               | Perform General Matrix Multiplication                              |
| `triton::DotScaledOp`             | `linalg::MatmulOp`                                                                                                                               | Performing Matrix Multiplication with Scaling Factor               |
| `triton::ascend::FlipOp`          | First convert to`func::CallOp`(Invokes`triton_flip`) and then converted to`hfusion::FlipOp`                                                      | Performing Matrix Multiplication with Scaling Factor               |
| **Reduced Op**                    |                                                                                                                                                  |                                                                    |
| `triton::ArgMinOp`                | `linalg::ReduceOp`                                                                                                                               | Returns the index of the smallest value in a tensor.               |
| `triton::ArgMaxOp`                | `linalg::ReduceOp`                                                                                                                               | Returns the index of the maximum value in a tensor.                |
| `triton::ReduceOp`                | `linalg::ReduceOp`                                                                                                                               | general reduction operation                                        |
| `triton::ScanOp`                  | First convert to`func::CallOp`(Invokes`triton_cumsum`or the`triton_cumprod`) and then converted to`hfusion::CumsumOp`and the`hfusion::CumprodOp` | Perform scanning operations (e.g., cumulative sum, cumulative sum) |

## Triton extended operation

Ascend NPU-IR provides language features. Triton-Ascend extends some operations based on NPU IR. To enable the capabilities, you need to import the following modules:

```python
import triton.language.extra.cann.extension as al
```

The relevant Ascend Language (al) unique interface can then be used. In addition, the Ascend Language provides bottom-layer interfaces, and the interfaces are not compatible.

### Synchronization and Debugging Operations

#### debug_barrier

The Ascend provides multiple synchronization modes and supports the internal synchronization mode of the vector pipeline for fine-grained synchronization control during debugging and performance optimization.

##### Parameter Description

| Parameter name | Type                                           | Description                          |
| -------------- | ---------------------------------------------- | ------------------------------------ |
| `sync_mode`    | [Enumerated value of SYNC_IN_VF](#sync_in_vf)  | Vector pipeline synchronization mode |

**Example**:

```python
@triton.jit
def kernel_debug_barrier():
    #...
    with al.scope(core_mode="vector"):
        al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
    #...
```

#### sync_block_set & sync_block_wait

The Ascend supports the setting of synchronization events between computing units and vector units. The sync_block_set and sync_block_wait must be used together.

**Parameter Description**:

| Parameter name        | Type                                                   | Description         |
| --------------------- | ------------------------------------------------------ | ------------------- |
| `sender`              | str                                                    | Sending unit type   |
| `receiver`            | str                                                    | Receiving unit type |
| `event_id`            | int                                                    | Event Identifier    |
| `sender_pipe_value`   | [Specifies the enumerated values of Pipe.](#pipe)      | Send Pipe Value     |
| `receiver_pipe_value` | [Specifies the enumerated values of the pipe.](#pipe)  | Receive Pipe Value  |

**Example**:

```python
@triton.jit
def triton_matmul_exp():
    #...
    tbuff_ptrs = TBuff_ptr + (row + offs_i) * N + (col + offs_j)
    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)
    
    extension.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    extension.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    
    acc_11_reload = tl.load(tbuff_ptrs)
    c_ptrs = C_ptr + (row + offs_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
    #...
```

#### sync_block_all

Ascend supports global synchronization of the entire computing block, ensuring that all computing cores of a specified type complete the current operation.

**Parameter Description**:

| Parameter name | Type | Description                                            | Valid Value                                            |
| -------------- | ---- | ------------------------------------------------------ | ------------------------------------------------------ |
| `mode`         | str  | Sync mode, specifying the core type to be synchronized | `"all_cube"`,`"all_vector"`,`"all"`,`"all_sub_vector"` |
| `event_id`     | int  | synchronization event identifier                       | `0`\-`15`                                              |

**Synchronization mode details**:

| mode               | Description                           | Synchronization Range                                    |
| ------------------ | ------------------------------------- | -------------------------------------------------------- |
| `"all_cube"`       | Synchronizing All Cube Cores          | All Cube cores on the current AI core                    |
| `"all_vector"`     | Synchronizing All Vector Cores        | All vectoring cores on the current AI core               |
| `"all"`            | Synchronize all cores                 | All computing cores (Cube+Vector) on the current AI core |
| `"all_sub_vector"` | Synchronizing all sub vectoring cores | All sub vectoring cores on the current AI core           |

**Example**:

```python
@triton.jit
def test_sync_block_all():
    #...
    al.sync_block_all("all_cube", 8)
    al.sync_block_all("all_sub_vector", 9)
    #...
```

### Hardware query and control operations

#### sub_vec_id & sub_vec_num

The Ascend provides an interface to query hardware information by calling the`sub_vec_id`Obtain the vector core index on the current AI core by calling.`sub_vec_num`Number of vectoring cores on a single AI core supported by the interface.

**Example**:

```python
@triton.jit
def triton_matmul_exp():
    #...
    sub_vec_id = al.sub_vec_id()
    row_exp = row_matmul + (M //al.sub_vec_num()) * sub_vec_id
    offs_exp_i = tl.arange(0, M //al.sub_vec_num())[:, None]
    tbuff_exp_ptrs = TBuff_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    #...
```

#### parallel

Ascend extends the Python standard`range`capability, adding parallel execution semantics`parallel`Iterator.

**Parameter Description**:

| Parameters           | Type | Description                          | Example                                 |
| -------------------- | ---- | ------------------------------------ | --------------------------------------- |
| `arg1`               | int  | Start or End Value                   | `parallel(10)`                          |
| `arg2`               | int  | End Value (Optional)                 | `parallel(0, 10)`                       |
| `step`               | int  | Step (Optional)                      | `parallel(0, 10, 2)`                    |
| `num_stages`         | int  | Number of pipeline phases (optional) | `parallel(0, 10, num_stages=3)`         |
| `loop_unroll_factor` | int  | Cycle spread factor (optional)       | `parallel(0, 10, loop_unroll_factor=4)` |

**Restriction**:

Currently, the OptiX RTN `Atlas A2` supports a maximum of two vectoring cores.

**Example**:

```python
@triton.jit
def triton_add():
    #...
    for _ in al.parallel(2, 5, 2):
        ret = ret + x1
    for _ in al.parallel(2, 10, 3):
        ret = ret + x0
    tl.store(out_ptr0, ret)
    #...
```

### Compilation Optimization Tips

#### compile_hint

Ascend supports passing optimization prompts to the compiler to guide code generation and performance tuning.

**Parameter Description**:

| Parameter name | Type           | Description                   |
| -------------- | -------------- | ----------------------------- |
| `ptr`          | tensor         | Pointer to the target tensor. |
| `hint_name`    | str            | Prompt name                   |
| `hint_val`     | Multiple types | Prompt Value (Optional)       |

**Example**:

```python
@triton.jit
def triton_where_lt_case1():
    #...
    mask = tl.where(cond, in1, in0)
    al.compile_hint(mask, "bitwise_mask")
    #...
```

#### multibuffer

`multibuffer`is a function used to set up Double Buffering for existing tensors, optimizing data flow and computational overlap through compiler hints.

**Parameter Description**:

| Parameters | Type   | Description                    |
| ---------- | ------ | ------------------------------ |
| `src`      | tensor | Tensor to be multiple buffered |
| `size`     | int    | Number of buffered copies      |

**Example**:

```python
@triton.jit
def triton_compile_hint():
    #...
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    al.multibuffer(tmp0, 2)
    tl.store(out_ptr0 + (xindex), tmp0, xmask)
    #...
```

#### scope

Ascend supports scope managers, adding hint information to a section of locale code, one use of which is through`core_mode`Specifies the cube or vector type.

**Parameter Description**:

| Parameter name | Type | Description                                                                                                                                   |
| -------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `core_mode`    | str  | Core type, which specifies the computing core used by operations in a block. Only `"cube"` and `"vector"` are supported. |

**Core Mode Options**:

| mode       | Description                          |
| ---------- | ------------------------------------ |
| `"cube"`   | Use the Cube core for calculation.   |
| `"vector"` | Use the vector core for calculation. |

**Example**:

```python
@triton.jit
def kernel_debug_barrier():
    #...
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
    #...
```

### Tensor slice operation

#### insert_slice & extract_slice

Ascend supports inserting a tensor into another tensor based on the offset, size, and step parameters of the operation (i.e.`insert_slice`) or extract the specified slice from another tensor (i.e.`extract_slice`).

**Parameter Description**:

| Parameter name | Type          | Description                             |
| -------------- | ------------- | --------------------------------------- |
| `ful`          | Tensor        | Receive the inserted target tensor      |
| `sub`          | Tensor        | Source tensor to be inserted            |
| `offsets`      | Integer Tuple | Start offset of the insert operation.   |
| `sizes`        | Integer Tuple | Size range of insert operations         |
| `strides`      | Integer Tuple | Step parameter of the insert operation. |

**Example**:

```python
@triton.jit
def triton_kernel():
    #...
    x_sub = al.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = al.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    output = tl.load(output_ptr + offsets, mask=mask)
    output = al.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
    #...
```

#### get_element

Ascend supports reading a single element value at a specified index position from a tensor.

**Parameter Description**:

| Parameter name | Type      | Description                                                 |
| -------------- | --------- | ----------------------------------------------------------- |
| `src`          | tensor    | Source Tensor to Access                                     |
| `indice`       | int tuple | Specifies the index location of the element to be obtained. |

**Example**:

```python
@triton.jit
def index_select_manual_kernel():
    #...
    gather_offset = al.get_element(indices, (i,)) * g_stride
    val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
    #...
```

### Tensor Calculation Operations

#### sort

Ascend supports the sorting operation on input tensors along the specified dimension.

**Parameter Description**:

| Parameter name | Type                         | Description                                                                              | Default value |
| -------------- | ---------------------------- | ---------------------------------------------------------------------------------------- | ------------- |
| `ptr`          | tensor                       | Input Tensor                                                                             | -             |
| `dim`          | int or tl.constexpr\[int\]   | Dimension to sort                                                                        | `-1`          |
| `descending`   | bool or tl.constexpr\[bool\] | Sorting direction,`True`Indicates the descending order,`False`Indicates ascending order. | `False`       |

**Example**:

```python
@triton.jit
def sort_kernel_2d():
    #...
    x = tl.load(X + off2d)
    x = al.sort(x, descending=descending, dim=1)
    tl.store(Z + off2d, x)
    #...
```

#### flip

Ascend supports the flip operation on the input tensor along the specified dimension.

**Parameter Description**:

| Parameter name | Type                       | Description       |
| -------------- | -------------------------- | ----------------- |
| `ptr`          | tensor                     | Input Tensor      |
| `dim`          | int or tl.constexpr\[int\] | Dimension to flip |

**Example**:

```python
@triton.jit
def flip_kernel_2d():
    #...
    input = tl.load(input_ptr + offset)
    flipped_input = flip(input, dim=2)
    #...
```

#### cast

The Ascend supports the conversion of tensors to specified data types, including numerical conversion, bit conversion, and overflow processing.

**Parameter Description**:

| Parameter name         | Type           | Description                                                  | Default value |
| ---------------------- | -------------- | ------------------------------------------------------------ | ------------- |
| `input`                | tensor         | Input Tensor                                                 | -             |
| `dtype`                | dtype          | Target Data Type                                             | -             |
| `fp_downcast_rounding` | str, optional  | Rounding mode when a floating point number is converted down | `None`        |
| `bitcast`              | bool, optional | Whether to perform bit conversion (not numeric conversion)   | `False`       |
| `overflow_mode`        | str, optional  | Overflow handling mode                                       | `None`        |

**Example**:

```python
@triton.jit
def cast_to_bool():
    #...
    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int1, overflow_mode=overflow_mode)
    tl.store(output_ptr + idx, ret)
    #...
```

### Indexing and Collection Operations

#### _index_select

The Ascend collects data in specified dimensions based on the index UB tensor from the source GM tensor and uses the SIMT template to collect values to the output UB tensor. This operation supports 2D-5D tensors.

**Parameter Description**:

| Parameter name    | Type         | Description                                          |
| ----------------- | ------------ | ---------------------------------------------------- |
| `src`             | pointer type | Source Tensor Pointer (in GM)                        |
| `index`           | tensor       | Index tensor used for collection (on UB)             |
| `dim`             | int          | Dimension along which the collection takes place     |
| `bound`           | int          | Upper boundary of the index value                    |
| `end_offset`      | int tuple    | End offset of each dimension of the index tensor.    |
| `start_offset`    | int tuple    | Start offset of each dimension of the source tensor. |
| `src_stride`      | int tuple    | Step size of each dimension of the source tensor.    |
| `other`(Optional) | scalar value | Default value when index is out of bounds (in UB)    |
| `out`             | tensor       | Output Tensor (on UB)                                |

**Example**:

```python
@triton.jit
def select_index():
    #...
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
    #...
```

#### index_put

Ascend allows you to place the value tensor in the target tensor based on the index tensor.

**Parameter Description**:

| Parameter name   | Type                  | Description                                           |
| ---------------- | --------------------- | ----------------------------------------------------- |
| `ptr`            | Tensor (pointer type) | Target Tensor Pointer (in GM)                         |
| `index`          | tensor                | Index for placement (on UB)                           |
| `value`          | tensor                | Value to store (on UB)                                |
| `dim`            | int32                 | Dimension along which the index is placed             |
| `index_boundary` | int64                 | Indicates the upper boundary of the index value.      |
| `end_offset`     | int tuple             | End offset of the placement area in each dimension.   |
| `start_offset`   | int tuple             | Start offset of the placement area of each dimension. |
| `dst_stride`     | int tuple             | Step size of each dimension of the target tensor.     |

**Index Placement Rules**:

 * 2D Index Placement
    
     * dim = 0:`out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]`
 * 3D Index Placement
    
     * dim = 0:`out[index[i]][start_offset[1]:end_offset[1]][start_offset[2]:end_offset[2]] = value[i][0:end_offset[1]-start_offset[1]][0:end_offset[2]-start_offset[2]]`
     * dim = 1:`out[start_offset[0]:end_offset[0]][index[j]][start_offset[2]:end_offset[2]] = value[0:end_offset[0]-start_offset[0]][j][0:end_offset[2]-start_offset[2]]`

**Constraints**:

 * `ptr` and `value` must have the same rank.
 * `ptr.dtype` Currently, only supports `float16`,`bfloat16`,`float32`.
 * `index`Must be an integer tensor. If`index.rank`! = 1, will be remodeled as 1D.
 * `index.numel`Must be equal to`value.shape[dim]`.
 * `value`Supports two to five-dimensional tensors.
 * `dim`Must be valid (0 ≤ dim < rank(value) - 1).

**Example**:

```python
@triton.jit
def put_index():
    #...
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
    #...
```

#### gather_out_to_ub

Ascend can collect data from scatterpoints in the GM and save the data to the UB in a specified dimension. This operation supports index boundary check, ensuring efficient and secure data transfer.

**Parameter Description**:

| Parameter name   | Type                    | Description                                         |
| ---------------- | ----------------------- | --------------------------------------------------- |
| `src`            | Tensor (pointer type)   | Source Tensor Pointer (in GM)                       |
| `index`          | tensor                  | Index tensor for collection (on UB)                 |
| `index_boundary` | int64                   | Indicates the upper boundary of the index value.    |
| `dim`            | int32                   | Dimension along which the collection takes place    |
| `src_stride`     | int64 tuple             | Step size of each dimension of the source tensor.   |
| `end_offset`     | int32 tuple             | End offset of each dimension of the index tensor.   |
| `start_offset`   | int32 tuple             | Start offset of each dimension of the index tensor. |
| `other`          | Scalar Value (Optional) | Default value used when index out of bounds (on UB) |

**Return Value**:

 * **Type: tensor**
 * **Description: Result tensor in UB, shape vs.**`index.shape`The same.

**Scatter collection rule**:

 * One-dimensional index collection
    
     * dim = 0:`out[i] = src[start_offset[0] + index[i]]`
 * 2D Index Collection
    
     * dim = 0:`out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]`
     * dim = 1:`out[i][j] = src[start_offset[0] + i][start_offset[1] + index[i][j]]`
 * 3D Index Collection
    
     * dim = 0:`out[i][j][k] = src[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k]`
     * dim = 1:`out[i][j][k] = src[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k]`
     * dim = 2:`out[i][j][k] = src[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]]`

**Constraints**:

 * `src`And to the`index`Must have the same rank.
 * `src.dtype`Currently, only the`float16`,`bfloat16`,`float32`.
 * `index`Must be an integer tensor with a rank between 1 and 5.
 * `dim`Must be valid (0 ≤ dim < rank(index)).
 * `other`Must be a scalar value.
 * For each not equal to`dim`Dimension of`i`,`index.size[i]`≤`src.size[i]`.
 * Output Shape vs.`index.shape`Same. if`index`None, the output tensor will be the same as the`index`Empty tensors of the same shape.

**Example**:

```python
@triton.jit
def gather():
    #...
    tmp_buf = al.gather_out_to_ub(
        src=src_ptr,
        index=index,
        index_boundary=4,
        dim=0,
        src_stride=(2, 1),
        end_offset=(2, 2),
        start_offset=(0, 0)
    )
    #...
```

#### scatter_ub_to_out

Ascend stores data from scatterpoints in UB to GM along a specified dimension. This operation supports index boundary check, ensuring efficient and secure data transfer.

**Parameter Description**:

| Parameter name   | Type                  | Description                                         |
| ---------------- | --------------------- | --------------------------------------------------- |
| `ptr`            | Tensor (pointer type) | Target Tensor Pointer (in GM)                       |
| `value`          | tensor                | Block value to store (on UB)                        |
| `index`          | tensor                | Index used by scatter storage (in UB)               |
| `index_boundary` | int64                 | Indicates the upper boundary of the index value.    |
| `dim`            | int32                 | Dimension along which scatter-stored                |
| `dst_stride`     | int64 tuple           | Step size of each dimension of the target tensor.   |
| `end_offset`     | int32 tuple           | End offset of each dimension of the index tensor.   |
| `start_offset`   | int32 tuple           | Start offset of each dimension of the index tensor. |

**Scatter storage rule**:

 * one-dimensional index scatter
    
     * dim = 0:`out[start_offset[0] + index[i]] = value[i]`
 * 2D Index Scatter
    
     * dim = 0:`out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]`
     * dim = 1:`out[start_offset[0] + i][start_offset[1] + index[i][j]] = value[i][j]`
 * 3D Index Scatter
    
     * dim = 0:`out[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k] = value[i][j][k]`
     * dim = 1:`out[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k] = value[i][j][k]`
     * dim = 2:`out[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]] = value[i][j][k]`

**Constraints**:

 * `ptr`,`index`And to the`value`Must have the same rank.
 * `ptr.dtype`Currently, only the`float16`,`bfloat16`,`float32`.
 * `index`Must be an integer tensor with a rank between 1 and 5.
 * `dim`Must be valid (0 ≤ dim < rank(index)).
 * For each not equal to`dim`Dimension of`i`,`index.size[i]`≤`ptr.size[i]`.
 * Output Shape vs.`index.shape`Same. if`index`None, the output tensor will be the same as the`index`Empty tensors of the same shape.

**Example**:

```python
@triton.jit
def scatter():
    #...
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
    #...
```

#### index_select_simd

The Ascend supports parallel index selection. Data is directly loaded to the UB from GM points, implementing zero copy and efficient read.

**Parameter Description**:

| Parameter name | Type                         | Description                                                     |
| -------------- | ---------------------------- | --------------------------------------------------------------- |
| `src`          | tensor (pointer type)        | Source Tensor Pointer (in GM)                                   |
| `dim`          | int or constexpr             | Dimension along which the index is selected                     |
| `index`        | tensor                       | One-dimensional tensor of the index to select (in UB)           |
| `src_shape`    | List\[Union\[int, tensor\]\] | Full shape of the source tensor (can be an integer or a tensor) |
| `src_offset`   | List\[Union\[int, tensor\]\] | Read start offset (can be an integer or a tensor)               |
| `read_shape`   | List\[Union\[int, tensor\]\] | Size to read (Block shape, which can be an integer or a tensor) |

**Constraints**:

 * `read_shape[dim]` must be `-1`.
 * `src_offset[dim]` can be `-1`(which  will be ignored).
 * Boundary processing: when `src_offset + read_shape > src_shape`, the data will be automatically truncated to the `src_shape` boundary.
 * **No check is performed**  on whether the `index` contains out-of-bounds values.

**Return Value**:

 * **Return type: tensor**
 * **Description: Resulting tensor in UB, whose shape**`dim`Dimension is replaced with`index`Length of.

**Example**:

```python
@triton.jit
def index_select_simd():
    #...
    tmp_buf = al.index_select_simd(
        src=in_ptr,
        dim=dim,
        index=indices,
        src_shape=(other_numel, g_stride),
        src_offset=(-1, 0),
        read_shape=(-1, other_block)
    )
    #...
```

## Triton extended CustomOp

In the A5 architecture, the Custom Op of Triton-Ascend allows users to customize operations and use it. Customization operations are converted into calling the implementation functions on the device side during running. The functions can call the existing library functions or the implementation functions generated by the source code or bytecode compilation provided by the user.

### Basic Usage

#### Registering Customized Operations

The functions related to customization operations are provided by the triton Ascend extension package. User-defined customization operations can be used only after registration. You can use the`register_custom_op`Decorate a class to define and register the custom action:

```python
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
```

To register a simplest customization operation, at least the basic attributes such as name, core, pipe, and mode must be provided.

 * The name operation name, which is the unique identifier for this custom operation. If omitted, the class name is used by default.
 * core indicates the type of Ascend core on which the.
 * pipe indicates the pipeline.
 * mode indicates the programming mode used.

#### Use custom actions

Registered custom actions are available through the Ascend expansion pack`custom()`The function is invoked. The name and parameters of the customized operation must be provided.

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

The parameters of the `custom()` include the operation name, input parameters, and optional output parameters.

 * **Operation name: The value must be the same as the registered operation name.**
 * **Input parameter: Different operations have different input parameters.**
 * **Output parameter (optional): The output parameter is defined by the**`out`Specifies the output of the operation.

If it's passed`out`If the parameter specifies the output variable, the return value of the customization operation is the same as that of the output variable. Otherwise, the return value of the operation is unavailable.

### Built-in Customization Operations

The name of the built-in customization operation starts with`"__builtin_"`Start with the customized operations built in triton-ascend, which can be directly used without registration. For example:

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

For details about the built-in customization operations, see the documentation of the related version.

### Parameter Validity Check

Without constraint, the user can give`al.custom()`If the number of parameters and parameter types are not the expected ones, an error occurs during the runtime.

To avoid this problem and improve the user experience of the customization operation, we can provide constructors for the registered customization class to describe the parameter list and check the parameter validity. For example:

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

The parameter list of the constructor function of the registration class is the parameter list required for invoking the customization operation. When invoking the custom operation, you need to provide the parameters that meet the requirements. For example:

```python
res = al.custom('my_custom_op', src_ptr, index, dim=1, out=dst)
```

If the provided parameter is incorrect, an error will be reported during compilation. For example, the dim parameter must be an integer constant. If the dim parameter is a floating point number, the following error message is displayed:

```text
...
    res = al.custom('my_custom_op', src_ptr, index, dim=1.0, out=dst)
          ^
AssertionError('dim must be an integer')
```

### Output Parameters and Return Values

`al.custom`The output parameter specified by the out parameter is returned. For example:

```python
x = al.custom('my_custom_op', src, index, out=dst)
```

dst is returned to x.

The out parameter can specify multiple output parameters,`al.custom`Returns a tuple containing these output parameters:

```python
x, y = al.custom('my_custom_op', src, index, out=(dst1, dst2))
```

dst1 is returned to x and dst2 is returned to y.

Without the out parameter,`al.custom`No value is returned (None is returned).

### Symbolic name of the called function

The customization operation will eventually be converted into calling the implementation function on the device side. We can register the custom action class`symbol`Property to configure the symbolic name of the function; if not set`symbol`Property, the name of the custom operation is used as the function name by default.

#### Static Symbol Name

If a custom operation always calls a device-side function, you can set the symbol name statically.

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = '_my_custom_op_symbol_name_'
```

In this way,`al.custom('my_custom_op', ...)`will fix the corresponding device side`_my_custom_op_symbol_name_(...)`function.

#### Dynamic symbol name

In most cases, the same customization operation needs to invoke different device functions based on the dimension and type of the input parameter. In this case, the symbol name needs to be set dynamically. Similar to the parameter validity check, you can dynamically set the symbol name in the constructor of the registered custom operation class. For example:

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

When the input src is a pointer pointing to the float32 type and the index is a 3-dimensional tensor of the int32 type, the device-side function symbol corresponding to the preceding customization operation is named as follows:`"my_func_3d_float_int32_t"`; Different input parameters correspond to different symbol names.

Note that the type name is used here`cname`, indicates the name of the corresponding type in the AscendC language. For example, the cname corresponding to int32 is`int32_t`. Because we usually declare these functions as macros and embed the related type name into the function name,`cname`It will be more common.

### Source code and compilation

If the functions for implementing customized operations need to be compiled from source code or bytecode, configure the functions when registering the customized operation class.`source`And to the`compile`Property:

 * source: indicates the source code or bytecode file path for implementing the custom operation function.
 * The compile command implements the compilation command of the customized operation function. You can use the`%<`And to the`%@`Indicates the source and target files, respectively (similar to Makefile).

Similar to symbol names, these two attributes can be configured statically or dynamically in the registration class constructor, for example:

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

### Parameter Conversion Rule

#### Parameter Sequence

Customized operations are converted into corresponding function invoking. The parameter sequence is the same as that on the Python side. The output parameter (out, if any) is always placed at the end. For example, the following Python code is used:

```python
al.custom('my_custom_op', src, index, dim, out=dst)
```

Converting to a function call is equivalent to:

```cpp
my_custom_op(src, index, dim, dst);
```

#### List and Tuple Parameters

The tuple or list parameter on the Python side is flattened. For example:

```python
al.custom('my_custom_op', src, index, offsets=(1, 2, 3), out=dst)
```

When converted to a function call, the offsets parameter is flattened:

```cpp
my_custom_op(src, index, 1, 2, 3, dst);
```

### Constant Parameter Type

Customization operations support the constant parameter types of integers and floating points. However, the integer and floating point types of Python do not distinguish the bit widths. Therefore, by default, only integers are mapped to the int32_t type and floating point numbers are mapped to the float type. When the constant parameter of the implementation function is of other type, for example, int64_t, the function signature does not match, causing errors.

For example, the implementation function signature for the following customized operations is available:

```cpp
custom_op_impl_func(memref_t<...> *src, memref_t<...> *idx, int64_t bound);
```

The bound parameter must be an integer of the int64_t type.

When the customization operation is invoked on the Python side, the value of the bound constant parameter is provided.

```python
al.custom('my_custom_op', src, idx, bound=1024)
```

Because the integer constants of Python do not distinguish the bit width, we can only map bound to int32_t by default. As a result, the signature does not match the implementation function and an error occurs.

To avoid this problem, you are advised to implement the function parameters. For integers, use int32_t, and for floating point numbers, use float. In some specific scenarios, the following methods are provided to specify the type:

#### Specify the integer bit width by using al.int64

By default, integer constants are mapped to the int32_t type. If the implementation function requires an int64_t type, you can use the`al.int64`Wrap an integer, for example:

```python
al.custom('my_custom_op', src, idx, bound=al.int64(1024))
```

#### Specify the type by using the type hint

In the constructor function of the registered class, you can add type annotations to the corresponding parameters. For example:

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

#### Dynamically Specifying Parameter Types

Another extreme case is that the parameter type varies depending on the other parameters. For example, the bound type must be the same as the idx data type. In this case, you can use arg_type to dynamically specify the type in the constructor. For example:

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

### Encapsulation Customization Operations

Direct use`al.custom`Invoking a customized operation is a little troublesome, especially when there are output parameters. Therefore, you need to prepare the output parameters before invoking the operation. For example:

```python
dst = tl.full(index.shape, 0, tl.float32)
x = al.custom('my_custom_op', src, index, out=dst)
```

We can encapsulate the customized operation into an operation function for easy use. For example:

```python
@al.builtin
def my_custom_op(src, index, _builder=None):
    dst = tl.semantic.full(index.shape, 0, src.dtype.element_ty, _builder)
    return al.custom_semantic(_my_custom_op.name, src, index, out=dst, _builder=_builder)
```

Encapsulated operation functions need to be`al.builtin`Decorate, and pass through`al.custom_semantic`Invoke the customization operation. It's also possible to use`tl.semantic`Provide the function preparation output parameters. Note: When encapsulating the operation function, you need to give an additional`_builder`Parameter, and passed to all senmtic functions.

The encapsulated operation function can be directly invoked like the native operation.

```python
@triton.jit
def my_kernel(...):
    ...
    x = my_custom_op(src, index)
    ...
```

## Triton extended Enumeration

### SYNC_IN_VF

| Enumerated Value | Description                                                                                              |
| ---------------- | -------------------------------------------------------------------------------------------------------- |
| `VV_ALL`         | Blocks execution of vector load/store instructions until all vector load/store instructions are complete |
| `VST_VLD`        | Blocks execution of vector load instructions until all vector store instructions are complete.           |
| `VLD_VST`        | Blocks execution of vector store instructions until all vector load instructions are complete.           |
| `VST_VST`        | Blocks execution of vector storage instructions until all vector storage instructions are complete       |
| `VS_ALL`         | Block execution of scalar load/store instructions until all vector load/store instructions are complete  |
| `VST_LD`         | Blocks execution of scalar load instructions until all vector store instructions are complete.           |
| `VLD_ST`         | Blocks execution of scalar store instructions until all vector load instructions are complete.           |
| `VST_ST`         | Blocks execution of scalar storage instructions until all vector storage instructions are complete.      |
| `SV_ALL`         | Blocks execution of vector load/store instructions until all scalar load/store instructions are complete |
| `ST_VLD`         | Blocks execution of the vector load instruction until all scalar store instructions are complete.        |
| `LD_VST`         | Blocks execution of vector store instructions until all scalar load instructions are complete.           |
| `ST_VST`         | Blocks execution of vector store instructions until all scalar store instructions are complete.          |

### PIPE

| Enumerated Value | Description                       |
| ---------------- | --------------------------------- |
| `PIPE_S`         | scalar computing pipeline         |
| `PIPE_V`         | vector computing pipeline         |
| `PIPE_M`         | memory operation pipeline         |
| `PIPE_MTE1`      | Memory transfer engine 1 pipeline |
| `PIPE_MTE2`      | Memory transfer engine 2 pipeline |
| `PIPE_MTE3`      | Memory transfer engine 3 pipeline |
| `PIPE_ALL`       | All pipelines                     |
| `PIPE_FIX`       | Fixed functional pipeline         |
