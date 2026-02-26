# 性能优化案例
## 1. tiling策略(计算)
### 待补充
## 2. kernel昇腾亲和改写(计算)
### 待补充
## 3. 性能优化选项使用(编译器)
### 常见案例
## 4. compile_hint使用(编译器)
### 常见案例

## 5. CV算子核间优化分析(编译器)
### 流水图
### 流水分析
## 功能或精度案例

本章节介绍常见功能或精度案例

## 3.卡死类问题

### 3.1 定界
- **现象** 算子选项规避超时报错,导致算子卡死的部分原因是与硬件同步相关，其中可能涉及核内/间同步，或涉及流水同步。若遇上算子卡死的情况，你可以尝试在调用Kernel时，传入以下入参，修改二进制的同步逻辑，以规避算子卡死的问题。
- **示例**

| 编译选项 | 数值 | 说明 |
|--------|------|------|
| **inject_barrier_all** | false(default). | 前端尝试打开为true,如果卡死问题消失，证明核内同步有问题,适用mix/aic/aiv三类kernel |
| **inject_block_all**|  false(default). | 前端尝试打开为true,如果卡死问题消失，证明核间同步有问题,适用mix类kernel | 


以GDN网络的`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`算子为例，原代码调用为

```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    k=k,
    v=u,
    w=w,
    v_new=v_new,
    g=g,
    gk=gk,
    h=h,
    h0=initial_state,
    ht=final_state,
    cu_seqlens=cu_seqlens,
    chunk_offsets=chunk_offsets,
    T=T,
    H=H,
    K=K,
    V=V,
    BT=BT,
)
```

关闭CV流水后的调用则为

```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    k=k,
    v=u,
    w=w,
    v_new=v_new,
    g=g,
    gk=gk,
    h=h,
    h0=initial_state,
    ht=final_state,
    cu_seqlens=cu_seqlens,
    chunk_offsets=chunk_offsets,
    T=T,
    H=H,
    K=K,
    V=V,
    BT=BT,
    inject_block_all = True # 开启核间同步
    inject_barrier_all = True # 开启核内同步
)
```

### 3.2 参数入参不合理
对于varlen类的算子，通常会在seqlen中随机采样indice，需要保证indice的入参合理性。例如严格递增且再[0, seqlen]范围内。


## 4. ub overflow类问题(计算)
### 4.1. tiling值
### 4.2. stride_align类
### 4.3. 待补充

## 5. d-cache类

### 5.1. 无效地址访问
- **现象** 算子输入合法且均为同一个deviceID, 实际算子的deviceID设置不正确，导致无法取到数据，出现D-cache读写错误
- **示例**
错误示例
```python
A=torch.empty(shape, dtype)
```
正确示例
```python
A=torch.empty(shape, dtype).npu()
or
DEVICE="npu:0"
A=torch.empty(shape, dtype, device=DEVICE).npu()
```

### 5.2. 可能是offset负数
- **现象** ossfet数值ir中是一个计算数值。
- **示例**
1. offset算出来是一个负数，导致读取地址不正确。
2. 算子的offset按照int32表示，实际数值超出这个数据表示范围，导致i32溢出。

### 5.3. 使用非负数iter arg作为访存索引
- **现象** 由于编译过程会对访存操作进行分析并优化编译结果，若访存操作的索引涉及到复杂的控制流（如for循环索引引入的访问越界），目前编译器或许没有能力完全覆盖，因此建议使用非负数的for循环iter参数作为访存索引。
## 4.load行为(编译器)
### 4.1.load非预期引入vtranspose op导致ub overflow(编译器)
### 4.2.Load隐式转置(编译器)
- **现象** “隐式转置”是指在加载或存储数据的同时完成矩阵转置操作，避免单独执行一个转置内核或额外的显式数据重排。
它通常通过调整指针的步长和形状来实现，使得内存访问模式隐含地完成维度交换。
这种技术可以节省全局内存带宽、减少内核启动开销，并提高计算效率。

`tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`
order参数指定内存中元素的迭代顺序，可以用来实现转置。或者，通过设置strides参数来指示转置后的步长。
实际上，对于矩阵转置，如果我们有一个输入矩阵A (M, K) 和输出矩阵B (K, M)，我们可以让每个线程块处理B的一个块，
并从A中加载对应的转置块。加载时，可以使用make_block_ptr从A中加载，但步长设置为导致转置加载的步长？
或者，更常见的做法是加载一个正常的A块，然后使用tl.trans转置后再存储到B。

#### example
```python
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    矩阵转置内核：Y = X^T, 其中 X 形状 (M, N)，Y 形状 (N, M)。
    每个程序块处理 Y 的一个 (BLOCK_N, BLOCK_M) 子块。
    通过交换输入指针的步长，实现隐式转置加载。
    """
    pid_n = tl.program_id(0)  # 输出矩阵的行块索引（原列块）
    pid_m = tl.program_id(1)  # 输出矩阵的列块索引（原行块）

    bn = pid_n * BLOCK_N  # 输出矩阵的行起始 = 原列起始
    bm = pid_m * BLOCK_M  # 输出矩阵的列起始 = 原行起始

    # 构建输入指针：使用交换后的步长，形状 (N, M) 以匹配转置访问
    x_ptr_t = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, M),
        strides=(stride_xn, stride_xm),
        offsets=(bn, bm),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0)
    )

    # 构建输出指针：正常行主序步长，形状 (N, M)
    y_ptr_b = tl.make_block_ptr(
        base=y_ptr,
        shape=(N, M),
        strides=(stride_ym, stride_yn),
        offsets=(bn, bm),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0)
    )

    # 加载输入块（已隐式转置），边界检查防止越界
    x_tile = tl.load(x_ptr_t, boundary_check=(0, 1))

    # 存储到输出矩阵
    tl.store(y_ptr_b, x_tile, boundary_check=(0, 1))
```
### 4.3.数据load方式不合理
### 4.4.待补充


以GDN网络的`causal_conv1d_fwd_kernel`算子为例，原代码逻辑为

```python
for i_w in tl.static_range(-W + 1, 1):
  p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
  b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
  if HAS_WEIGHT:
    b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
```

由于`i_w`可为负数，以上算子需改写为

```python
for i_w in tl.static_range(W):
  p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w - W + 1, i_d * BD), (BT, BD), (1, 0))
  b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
  if HAS_WEIGHT:
    b_yi *= tl.sum(b_w * (o_w == i_w), 1)
```

## 6. 访存类
### 6.1. load非预期引入vtranspose op导致ub overflow
- **现象** 算子编译或者精度报错，隐式转置明显特征最内轴stride不为1，外轴stride为1.
- **示例**
错误示例
```python
K_block_ptr=tl.make_block_ptr(
    base = K,
    shape =(HEAD_DIM, N_CTX),
    stride=(kk, kn)
    offsets=(0, 0),
    block_shape=(HEAD_DIM, BLOCK_N),
    order=(0, 1),
)
k=tl.load(K_block_ptr)
```
正确示例
```python
K_block_ptr=tl.make_block_ptr(
    base = K,
    shape =(N_CTX, HEAD_DIM),
    stride=(kn, kk)
    offsets=(0, 0),
    block_shape=(BLOCK_N, HEAD_DIM),
    order=(1, 0),
)
k=tl.load(K_block_ptr)
trans_k=tl.trans(k)
```

### 6.2.Load隐式转置
- **现象** “隐式转置”是指在加载或存储数据的同时完成矩阵转置操作，避免单独执行一个转置内核或额外的显式数据重排。
它通常通过调整指针的步长和形状来实现，使得内存访问模式隐含地完成维度交换。
这种技术可以节省全局内存带宽、减少内核启动开销，并提高计算效率。

`tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`
order参数指定内存中元素的迭代顺序，可以用来实现转置。或者，通过设置strides参数来指示转置后的步长。
实际上，对于矩阵转置，如果我们有一个输入矩阵A (M, K) 和输出矩阵B (K, M)，我们可以让每个线程块处理B的一个块，
并从A中加载对应的转置块。加载时，可以使用make_block_ptr从A中加载，但步长设置为导致转置加载的步长？
或者，更常见的做法是加载一个正常的A块，然后使用tl.trans转置后再存储到B。

- **示例**
```python
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    矩阵转置内核：Y = X^T, 其中 X 形状 (M, N)，Y 形状 (N, M)。
    每个程序块处理 Y 的一个 (BLOCK_N, BLOCK_M) 子块。
    通过交换输入指针的步长，实现隐式转置加载。
    """
    pid_n = tl.program_id(0)  # 输出矩阵的行块索引（原列块）
    pid_m = tl.program_id(1)  # 输出矩阵的列块索引（原行块）

    bn = pid_n * BLOCK_N  # 输出矩阵的行起始 = 原列起始
    bm = pid_m * BLOCK_M  # 输出矩阵的列起始 = 原行起始

    # 构建输入指针：使用交换后的步长，形状 (N, M) 以匹配转置访问
    x_ptr_t = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, M),
        strides=(stride_xn, stride_xm),
        offsets=(bn, bm),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0)
    )

    # 构建输出指针：正常行主序步长，形状 (N, M)
    y_ptr_b = tl.make_block_ptr(
        base=y_ptr,
        shape=(N, M),
        strides=(stride_ym, stride_yn),
        offsets=(bn, bm),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0)
    )

    # 加载输入块（已隐式转置），边界检查防止越界
    x_tile = tl.load(x_ptr_t, boundary_check=(0, 1))

    # 存储到输出矩阵
    tl.store(y_ptr_b, x_tile, boundary_check=(0, 1))
```

### 6.3.使用mayDiscretememaccess规避UB overflow
- **现象** 导致UB overflow的成因各异，除了本身张量数据类型过大，导致超出192KB的UB限制，另一个可能的原因是非连续搬运导致UB内扩轴。以`<Nx1xf32>`数据类型为例，由于硬件在尾轴需要32B对齐，而`1xf32`只有4B大小，因此`<Nx1xf32>`在硬件上的实际大小会被扩轴至`<Nx8xf32>`以确保32B对齐。无论因为什么原因导致的UB overflow，都可以通过加上`mayDiscretememaccess`的编译提示，使张量操作退化为标量操作，从而避免UB overflow。
- **示例**
改写算子时，只需在load/store操作的数据上加上`compile_hint`即可，参考以下代码段：

```python
# 若为load操作，compile_hint需加在加载出的value中
value = tl.load(pointer)
tl.compile_hint(value, "mayDiscretememaccess")

# 若为store操作，compile_hint需加在被存入的value中
tl.compile_hint(value, "mayDiscretememaccess")
tl.store(pointer, value)
```

- **示例1**
```python
b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
```

通过增加编译提示，张量访存会被退化为标量访存，避免UB overflow，参考以下代码段：

```python
b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
tl.compile_hint(b_x, "mayDiscretememaccess")
```

- **示例2**
```diff
import triton
import triton.language as tl
+ import triton.language.extra.cann.extension as extension

@triton.jit
def copy_column_major_to_row_major(
    A_ptr, B_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # 获取程序ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算块起始位置
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # 创建A的块指针 (列主序: strides=(1, M))，此时最后一维不连续，会自动扩轴
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, N),
        strides=(1, M),
        offsets=(start_m, start_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1),  # 最内层维度是行（索引0），因为列主序
    )

    # 创建B的块指针 (行主序: strides=(N, 1))
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(start_m, start_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),  # 最内层维度是列（索引1），因为行主序
    )

    # 加载A的块，进行边界检查（超出部分填充0）
    a = tl.load(A_block_ptr, boundary_check=(0, 1))
+   # npu
+   extension.compile_hint(a, "mayDiscretememaccess")

    # 存储到B
    tl.store(B_block_ptr, a, boundary_check=(0, 1))
```

- **示例2使用compile hint前后的ir对比**
```c++
// before using tl.compile_hint(a, "mayDiscretememaccess")
module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  func.func @copy_column_major_to_row_major(%arg0: memref<?xi8> , %arg1: memref<?xi8> , %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} , %arg4: i32 {tt.divisibility = 16 : i32} , %arg5: i32 {tt.divisibility = 16 : i32} , %arg6: i32 , %arg7: i32 , %arg8: i32 , %arg9: i32 , %arg10: i32 , %arg11: i32 ) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index 
    %c0 = arith.constant 0 : index 
    %c0_i32 = arith.constant 0 : i32 
    %c64_i32 = arith.constant 64 : i32 
    %0 = arith.muli %arg9, %c64_i32 : i32 
    %1 = arith.muli %arg10, %c64_i32 : i32 
    %2 = arith.maxsi %0, %c0_i32 : i32 
    %3 = arith.index_cast %2 : i32 to index 
    %4 = arith.maxsi %1, %c0_i32 : i32 
    %5 = arith.index_cast %4 : i32 to index 
    %6 = arith.index_cast %arg5 : i32 to index 
    %7 = arith.muli %3, %6 : index 
    %8 = arith.index_cast %arg4 : i32 to index 
    %9 = arith.addi %7, %5 : index 
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [64, 64], strides: [%6, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>> 
    %10 = arith.muli %5, %8 : index 
    %11 = arith.addi %10, %3 : index 
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [64, 64], strides: [%8, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>> 
    %alloc = memref.alloc() : memref<64x64xf32> 
    %12 = arith.divsi %11, %8 : index 
    %13 = arith.subi %6, %12 : index 
    %14 = arith.maxsi %13, %c0 : index 
    %15 = arith.minsi %14, %c64 : index 
    %16 = arith.remsi %11, %8 : index 
    %17 = arith.subi %8, %16 : index 
    %18 = arith.maxsi %17, %c0 : index 
    %19 = arith.minsi %18, %c64 : index 
    %20 = arith.subi %c0_i32, %1 : i32 
    %21 = arith.maxsi %20, %c0_i32 : i32 
    %22 = arith.index_cast %21 : i32 to index 
    %23 = arith.minsi %22, %15 : index 
    %24 = arith.subi %15, %23 : index 
    %25 = arith.subi %c0_i32, %0 : i32 
    %26 = arith.maxsi %25, %c0_i32 : i32 
    %27 = arith.index_cast %26 : i32 to index 
    %28 = arith.minsi %27, %19 : index 
    %29 = arith.subi %19, %28 : index 
    %subview = memref.subview %reinterpret_cast_0[0, 0] [%24, %29] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>> 
    %subview_1 = memref.subview %alloc[%23, %28] [%24, %29] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>> 
    memref.copy %subview, %subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>> 
    %30 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> 
    %31 = tensor.empty() : tensor<64x64xf32> 
    %transposed = linalg.transpose ins(%30 : tensor<64x64xf32>) outs(%31 : tensor<64x64xf32>) permutation = [1, 0]  
    %32 = arith.divsi %9, %6 : index 
    %33 = arith.subi %8, %32 : index 
    %34 = arith.maxsi %33, %c0 : index 
    %35 = arith.minsi %34, %c64 : index 
    %36 = arith.remsi %9, %6 : index 
    %37 = arith.subi %6, %36 : index 
    %38 = arith.maxsi %37, %c0 : index 
    %39 = arith.minsi %38, %c64 : index 
    %40 = arith.minsi %27, %35 : index 
    %41 = arith.subi %35, %40 : index 
    %42 = arith.minsi %22, %39 : index 
    %43 = arith.subi %39, %42 : index 
    %extracted_slice = tensor.extract_slice %transposed[%40, %42] [%41, %43] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32> 
    %subview_2 = memref.subview %reinterpret_cast[0, 0] [%41, %43] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>> 
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> () 
    return 
  } 
} 
```

```c++
// after using tl.compile_hint(a, "mayDiscretememaccess")
module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  func.func @copy_column_major_to_row_major(%arg0: memref<?xi8> , %arg1: memref<?xi8> , %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} , %arg4: i32 {tt.divisibility = 16 : i32} , %arg5: i32 {tt.divisibility = 16 : i32} , %arg6: i32 , %arg7: i32 , %arg8: i32 , %arg9: i32 , %arg10: i32 , %arg11: i32 ) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.muli %arg9, %c64_i32 : i32
    %1 = arith.muli %arg10, %c64_i32 : i32
    %2 = arith.extsi %arg5 : i32 to i64
    %3 = arith.maxsi %1, %c0_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.maxsi %0, %c0_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.index_cast %arg4 : i32 to index
    %8 = arith.muli %4, %7 : index
    %9 = arith.index_cast %arg5 : i32 to index
    %10 = arith.addi %8, %6 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [%7, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xf32>
    %11 = arith.divsi %10, %7 : index
    %12 = arith.subi %9, %11 : index
    %13 = arith.maxsi %12, %c0 : index
    %14 = arith.minsi %13, %c64 : index
    %15 = arith.remsi %10, %7 : index
    %16 = arith.subi %7, %15 : index
    %17 = arith.maxsi %16, %c0 : index
    %18 = arith.minsi %17, %c64 : index
    %19 = arith.subi %c0_i32, %1 : i32
    %20 = arith.maxsi %19, %c0_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.minsi %21, %14 : index
    %23 = arith.subi %14, %22 : index
    %24 = arith.subi %c0_i32, %0 : i32
    %25 = arith.maxsi %24, %c0_i32 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = arith.minsi %26, %18 : index
    %28 = arith.subi %18, %27 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%23, %28] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[%22, %27] [%23, %28] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1], offset: ?>>
    memref.copy %subview, %subview_0 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
    %29 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %30 = tensor.empty() : tensor<64x64xf32>
    %transposed = linalg.transpose ins(%29 : tensor<64x64xf32>) outs(%30 : tensor<64x64xf32>) permutation = [1, 0] 
    %31 = arith.index_cast %arg4 : i32 to index
    %32 = arith.minsi %31, %c64 : index
    scf.for %arg12 = %c0 to %32 step %c1 {
      %33 = arith.index_cast %arg5 : i32 to index
      %34 = arith.minsi %33, %c64 : index
      scf.for %arg13 = %c0 to %34 step %c1 {
        %35 = arith.index_cast %arg12 : index to i64
        %36 = arith.extsi %0 : i32 to i64
        %37 = arith.muli %2, %36 : i64
        %38 = arith.muli %2, %35 : i64
        %39 = arith.addi %37, %38 : i64
        %40 = arith.index_cast %arg13 : index to i64
        %41 = arith.extsi %1 : i32 to i64
        %42 = arith.addi %39, %41 : i64
        %43 = arith.addi %42, %40 : i64
        %44 = arith.index_cast %43 : i64 to index
        %extracted = tensor.extract %transposed[%arg12, %arg13] {DiscreteMemAccess} : tensor<64x64xf32>
        %45 = tensor.empty() : tensor<1xf32>
        %inserted = tensor.insert %extracted into %45[%c0] : tensor<1xf32>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_1 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
      } {ExtractedLoadOrStore}
    } {ExtractedLoadOrStore}
    return
  }
}

```
<hr>

## 7.baseline(计算) 	 
### 7.1.TRITON_INTERPRET模式(计算)	 
### 7.2.GPU特有运算逻辑(计算)	 
### 7.3.待补充

## 8.场景化调试举例	 

本章节介绍Triton NPU算子性能优化指南。
### 8.1.使用bitwise_mask优化访存掩码

#### 问题描述

在昇腾硬件上，布尔类型（i1）的张量在全局内存（GM）中实际是按i8（一个字节）存储的。当Triton Ascend处理以i1张量作为输入的运算时
，它会将i1视为i8搬入，但某些情况下（例如作为tl.where的条件掩码）又需要将结果转换回i1，导致不必要的类型转换，带来性能损耗。

为了解决这个问题，提供了compile_hint: "bitwise_mask:。通过该提示，编译器可以识别出该i1张量是作为位掩码使用的，从而直接按位操作，避免中间的类型转换，提升性能

具体使用方法只需在where后的结果加上`compile_hint("bitwise_mask")`即可，参考以下代码段：

```
mask = tl.where(cond, value1, value2)
tl.compile_hint(cond, "bitwise_mask")
```

需留意，由于mask以bitmask的形式表达，因此对应的mask指针偏移量也需正确运算。

![image](figs/performance/87d47d3c-bbbf-49a1-a37e-fcd73c87026f.png)

![image](figs/performance/e771ee98-44ab-4be7-9dca-4b528910d6f5.png)

#### 算子示例

参考 [Ascend where 算子](https://gitcode.com/Ascend/triton-ascend/blob/master/ascend/examples/pytest_ut/test_where_lt.py)进行改写，
若用户需要输入bitwise的i8掩码作为算子入参，只需为tl.where的结果加上compile_hint即可，见以下代码：

```python
@triton.jit
def triton_where_lt_case1(in_ptr0, in_ptr1, cond_ptr, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        in0 = tl.load(in_ptr0 + xindex, xmask)
        in1 = tl.load(in_ptr1 + xindex, xmask)
        cond = tl.load(cond_ptr + xindex, xmask)
        res = tl.where(cond, in1, in0)
        tl.extra.cann.extension.compile_hint(cond, "bitwise_mask")
        tl.store(out_ptr0 + (xindex), res, xmask)
```

由于bitwise mask将8个i8类型的True/False压缩至一个i8类型的数据，因此mask组装逻辑也需相关更新，可参考以下代码：

```python
def test_where_lt_case1(param_list):
       dtype, shape, ncore, xblock, xblock_sub = param_list
       if shape[-1] %8 != 0:
           raise ValueError("The last dimension should be a multiple of 8")
       x0 = test_common.generate_tensor(shape, dtype).npu()
       x1 = test_common.generate_tensor(shape, dtype).npu()
       # Run triton with i8 bitwise mask
       cond_i8 = test_common.generate_tensor(shape, 'uint8').npu()
       y_cal = test_common.generate_tensor(shape, dtype).npu()
       triton_where_lt_case1[ncore, 1, 1](x0, x1, cond_i8, y_cal, x0.numel(), xblock, xblock_sub)
       # Run torch with i1 mask
       flatten_cond_i8 = cond_i8.flatten()
       numel = flatten_cond_i8.shape[-1]
       num_sub_block = numel // xblock_sub
       flatten_cond_bool = torch.zeros(flatten_cond_i8.shape, dtype=torch.bool).npu()
       for sub_block_id in range(num_sub_block):
           for i in range(min(numel, xblock_sub) // 8):
               byte_value = flatten_cond_i8[xblock_sub * sub_block_id + i]
               for bit in range(8):
                   flatten_cond_bool[..., xblock_sub * sub_block_id + i*8 + bit] = (byte_value & (1 << bit)) != 0
       cond_bool = flatten_cond_bool.view(shape)
       y_ref = torch_where_lt_case1(x0, x1, cond_bool)
       # Precision test
       test_common.validate_cmp(dtype, y_cal, y_ref)
```

#### 限制

只支持i8类型的mask

- 由于Triton前端会将i1转换为i8，如果对其他类型如i16/i32等进行bitwise_mask操作反而会带来性能损耗，因此此功能只支持i8类型的mask

---

### 8.2.动态生成mask类

#### 问题描述

经常出现range后cmp生成下三角的mask，我们硬件上的指令不支持i32/i64的比较，转scalar

![image](figs/performance/aa3ea0d8-698d-4327-ad22-cfb823049c37.png)

![image](figs/performance/79d71bc5-7851-4a0d-b64f-852d5b2195ca.png)

#### 算子示例

diffusion_attention类的

```python
for idx_ingroup in range(GROUP_SIZE):
    idx_n = idx_group * GROUP_SIZE + idx_ingroup
    offs_r_local = tl.arange(0, BLOCK_C)[:, None]
            offs_c_local = tl.arange(0, BLOCK_C)[None, :]
            chunk_idx_r = offs_r_local // BLOCK_SIZE
            chunk_idx_c = offs_c_local // BLOCK_SIZE

        block_mask_bool = (
                (chunk_idx_r > chunk_idx_c)
                & (seq_st + idx_c * BLOCK_C + offs_r_local < seq_ed)
                & (seq_st + idx_c * BLOCK_C + offs_c_local < seq_ed)
            )
```

## 9.CV类

---

### 9.1.使用tile_cube_loop规避L1越界

#### 问题描述

由于编译器目前只能对单个matmul进行切分需求分析，并不考虑其他matmul的生命周期，因此当matmul被多次触发时（例如执行逻辑为`cube -> vector -> cube`时），若上一个matmul的生命周期和当前的matmul生命周期有所重叠，算子运行时可能会导致L1越界。后续编译器会对切分的生命周期分析进行增强，目前则需通过加上 `tile_cubloop` 编译提示，令编译器可以感知是否需要对相关的matmul操作进行sub tiling。

#### 算子示例

改寫算子時，只需为dot操作结果加上`tile_cube_loop`的编译提示即可，參考一下代碼段：

```python
res = tl.dot(lhs, rhs)
tl.compile_hint(res, "tile_cube_loop", 2)
```

以Flash Attention的`_attn_fwd_inner`算子为例，原代码的QKV矩阵乘法逻辑大致为

```python
qk = tl.dot(q, trans_k)
# softmax calculation in between
qk = ...
p = tl.math.exp(qk)
pv = tl.dot(p, v)
```

参考以上代码，`qk`是cube操作，而softmax等计算属于vector操作，最后vector计算出的结果又再导入到第二次的cube操作中执行矩阵乘法。在以上场景下，编译器无法监控第二次cube操作中的切分逻辑，代码或许会在L1缓存中越界。因此，需要为第二次的dot操作结果加上`tile_cube_loop`的编译提示，令编译器对该操作进行sub tiling，见以下代码段：

```python
qk = tl.dot(q, trans_k)
# softmax calculation in between
qk = ...
p = tl.math.exp(qk)
pv = tl.dot(p, v)
tl.compile_hint(pv, "tile_cube_loop", 2)
```

---

### 9.2.参考：编译优化选项

| 编译选项| 含义| 取值范围|
| --- | --- | --- |
| multibuffer | 设置是否启用乒乓流水 | False(默认),True |
| limit_auto_multi_buffer_of_local_buffer | 设置乒乓流水在片中(L1, L0, 及UB)的作用范围"no-limit"表示不限乒乓流水范围"no-l0c"表示只允许L0缓存外启用乒乓流水 | "no-limit","no-l0c"(默认) |
| unit_flag | 设置cube搬出时是否按照block搬出，仅限数据对齐场景下使用 | False(默认),True |
| limit_auto_multi_buffer_only_for_local_buffer | 设置是否在GM workspace中启用CV流水并行，False表示启用  后续会整改接口，提供更可读的选项 | False(默认),True |
| set_workspace_multibuffer | 仅在limit_auto_multi_buffer_only_for_local_buffer=false的场景下生效设置CV并行的并行度使用时需确保数据没有依赖若设置为N，则N个CV操作并行执行 | 2 (默认),4 |
| tile_mix_vector_loop | 仅在limit_auto_multi_buffer_only_for_local_buffer=false的场景下生效设置当前vector的切分数量，数值可由autotuning得出，均可为最优 | 1 (默认),2,4 |
| tile_mix_cube_loop | 仅在limit_auto_multi_buffer_only_for_local_buffer=false的场景下生效设置当前cube的切分数量，数值可由autotuning得出，均可为最优 | 1 (默认),2,4 |

### - 算子分核逻辑

#### 问题描述

对于attention的注意力机制，存在负载不均衡的问题。不同核计算的任务量不同。因为attetnionmask一般是存在倒三角，所以越到后面的核计算的任务量越重，因此我们尽可能把计算少的和计算多的放在同一个核上。

![](figs/performance/5541cae3-7d7d-4979-aacb-45f9f06f2abd.png)

#### 算子示例

[mmad](https://gitee.com/guangpengz/triton-ascend/blob/master/ascend/examples/tutorials/13-matrix-multiplication-optimized.py)

---

### 9.3.算子选项规避超时报错

#### 问题描述

导致算子卡死的部分原因是与硬件同步相关，其中可能涉及核内/间同步，或涉及流水同步。若遇上算子卡死的情况，你可以尝试在调用Kernel时，传入以下入参，修改二进制的同步逻辑，以规避算子卡死的问题。

```python
# 核同步选项
inject_block_all = True # 开启核间同步
inject_barrier_all = True # 开启核内同步
# 流水选项
limit_auto_multi_buffer_only_for_local_buffer = True # 关闭(GM space) CV流水
multibuffer = False # 关闭乒乓流水
```

#### 算子示例

以GDN网络的`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`算子为例，原代码调用为

```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    k=k,
    v=u,
    w=w,
    v_new=v_new,
    g=g,
    gk=gk,
    h=h,
    h0=initial_state,
    ht=final_state,
    cu_seqlens=cu_seqlens,
    chunk_offsets=chunk_offsets,
    T=T,
    H=H,
    K=K,
    V=V,
    BT=BT,
)
```

关闭CV流水后的调用则为

```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    k=k,
    v=u,
    w=w,
    v_new=v_new,
    g=g,
    gk=gk,
    h=h,
    h0=initial_state,
    ht=final_state,
    cu_seqlens=cu_seqlens,
    chunk_offsets=chunk_offsets,
    T=T,
    H=H,
    K=K,
    V=V,
    BT=BT,
    limit_auto_multi_buffer_only_for_local_buffer = True,
)
```

---

## 10.Triton NPU 编程案例
Triton NPU 编程请参考：
[https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/README.zh.md](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/README.zh.md)

