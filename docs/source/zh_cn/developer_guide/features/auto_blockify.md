# 自动块化

## 背景

Auto Blockify Pass通过高效地将逻辑块映射到硬件物理块，是昇腾兼容算子执行链路的核心优化手段。当前架构下调度效率直接决定算子性能，逻辑块与物理块一对一映射可消除调度开销，实现性能提升。

结合AscendNPU IR架构使用实践，可用物理块数量通常远少于计算所需的逻辑块数量（物理块 < 50，逻辑块可能达到500+）。在这种10倍差距的场景下，加速效果可超过原始速度的两倍。

在运行Triton内核（通过triton-ascend）时，激活Auto Blockify逻辑的方式是添加以下标志：`TRITON_ALL_PARALLEL`。

对于AscendNPU IR开发者，可在`bishengir-compile`命令中添加以下标志：`--enable-auto-blockify-loop`。

![image](../../../images/developer_guide/AutoBlockify.jpg)

## SIMD模式

### 算法原理

Auto Blockify Pass（AutoBlockifyParallelLoop）通过引入额外的循环层来变换IR，具体逻辑如下：

```plaintext
for outer from 0,...,ceildiv(logical_block_dim, physical_block_dim)
    for inner from 0,...,physical_block_dim  <- 作为 block.idx 使用
        use(min(outer * physical_block_dim + inner, logical_block_dim))
```

**逻辑说明**：

1. 原始调度

   原始模式通常如下所示：

   ```plaintext
   block.idx = hivm.get_block_idx
   use(block.idx)
   -------等价于--------------
   for block.idx from 0,...,logical_block_num
       use(block.idx)
   ```

2. 使用`TRITON_ALL_PARALLEL`的示例

   当用户在triton adapter中添加`TRITON_ALL_PARALLEL`标志时，内核将被限制为仅使用最大物理块数量启动（假设逻辑块数 > 物理块数）。因此执行被限制为：

   ```plaintext
   for block.idx from 0,...,physical_block_num   <- 来自 get_block_idx
       use(block.idx)
   ```

   仅依靠该循环逻辑无法覆盖全部计算索引，存在索引缺失问题。这也是引入Auto Blockify Pass补齐逻辑的原因：通过自动添加一层外部循环/块化来完善。

   > 注：若不通过triton adapter接入，需要自行确保块维度的设置与上述一致。

3. 使用Auto Blockify后的最终逻辑

   经Auto Blockify Pass自动补全循环结构，最终执行逻辑如下：

   ```plaintext
   for outer from 0,...,ceildiv(logical_block_dim, physical_block_dim)
       for inner from 0,...,physical_block_dim  <- 作为 block.idx 使用
           use(min(outer * physical_block_dim + inner, logical_block_dim))
   ```

**接口说明**：

该功能通过bishengir-compile中的`--enable-auto-blockify-loop`标志控制，也可通过bishengir-opt的`--auto-blockify-parallel-loop`标志直接调用。

为正确使用此功能，需注意以下几点：

- Pass获取逻辑块数量的方式是查找标有`kLogicalBlockNumAttr`属性（IR中为`logical_block_num`）的值，用户需确保该值可用，否则Pass调用时将失败。

- Pass还需要找到一个`hivm get_block_idx`操作，该操作返回从0到块维度的块索引。使用AutoBlockify时，用户需要在调用设备内核时修改块维度（以最大物理块维度启动，与上述算法一致），使得`blockidx`操作返回0到`physical_block_num`范围内的值。

**Triton Adapter**：

该Pass已在triton adapter流水线中广泛使用。在此情况下正确使用AutoBlockify特性的方式是从前端（triton）通过`TRITON_ALL_PARALLEL=1`启用，该环境变量会同时完成准备工作（锁定块数量），然后自动以正确的标志调用相应的编译器命令。在triton流水线中有一个名为`TritonGlobalKernelArgsToHIVMOpPass`的Pass，会自动确保存在标记了`logical_block_num`的值，并创建所需的`get_block_idx`操作。

输入示例：

```mlir
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @add_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32 // 此 logical_block_num 是原始的大数值
    %2 = hivm.hir.get_block_idx -> i64   // for block.idx from 0,...,block_num
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg8, %arg7 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg6 : i32
    %7 = arith.muli %6, %c1024_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xf32>
    %9 = arith.addi %8, %c1024 : index
    %10 = arith.index_cast %arg5 : i32 to index
    %11 = arith.maxsi %8, %10 : index
    %12 = arith.minsi %9, %11 : index
    %13 = arith.subi %12, %8 : index
    %subview = memref.subview %reinterpret_cast[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%13] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_0 : memref<?xf32, strided<[1]>>) left_padding_num = %c0 : index init_out_buffer = false
    %14 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%13] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_4 : memref<?xf32, strided<[1]>>) left_padding_num = %c0 : index init_out_buffer = false
    %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32>
    %16 = tensor.empty() : tensor<1024xf32>
    %17 = hivm.hir.vadd ins(%14, %15 : tensor<1024xf32>, tensor<1024xf32>) outs(%16 : tensor<1024xf32>) -> tensor<1024xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %17[0] [%13] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>>)
    return
  }
}
```

输出示例：

```mlir
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 20 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 40 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @add_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32  // 此 logical_block_num 是原始的大数值
    %c0_i32 = arith.constant 0 : i32
    %c40_i32 = arith.constant 40 : i32 // 40 为此处的物理块数
    %2 = arith.ceildivsi %1, %c40_i32 : i32 // ceildiv(logical_block_num, physical_block_dim)
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg9 = %c0_i32 to %2 step %c1_i32  : i32 { // 外层循环
      %c1024 = arith.constant 1024 : index
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      hivm.hir.set_mask_norm
      %3 = hivm.hir.get_block_idx -> i64 // 内层循环（锁定到物理块）
      %4 = arith.trunci %3 : i64 to i32
      %5 = arith.muli %arg9, %c40_i32 : i32 // outer_i * physical_block_num
      %6 = arith.addi %5, %4 : i32 // outer_i * physical_block_num + inner
      %7 = arith.minsi %6, %1 : i32 // (min(outer*physical_block_dim + inner, logical_block_num))
      %8 = arith.extsi %7 : i32 to i64
      %9 = arith.trunci %8 : i64 to i32
      %10 = arith.muli %arg8, %arg7 : i32
      %11 = arith.divsi %9, %10 : i32
      %12 = arith.remsi %11, %arg6 : i32
      %13 = arith.muli %12, %c1024_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<1024xf32>
      %15 = arith.addi %14, %c1024 : index
      %16 = arith.index_cast %arg5 : i32 to index
      %17 = arith.maxsi %14, %16 : index
      %18 = arith.minsi %15, %17 : index
      %19 = arith.subi %18, %14 : index
      %subview = memref.subview %reinterpret_cast[0] [%19] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %alloc[0] [%19] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_0 : memref<?xf32, strided<[1]>>) left_padding_num = %c0 : index
      %20 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<1024xf32>
      %subview_3 = memref.subview %reinterpret_cast_1[0] [%19] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_4 = memref.subview %alloc_2[0] [%19] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_4 : memref<?xf32, strided<[1]>>) left_padding_num = %c0 : index
      %21 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32>
      %22 = tensor.empty() : tensor<1024xf32>
      %23 = hivm.hir.vadd ins(%20, %21 : tensor<1024xf32>, tensor<1024xf32>) outs(%22 : tensor<1024xf32>) -> tensor<1024xf32>
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %23[0] [%19] [1] : tensor<1024xf32> to tensor<?xf32>
      %subview_6 = memref.subview %reinterpret_cast_5[0] [%19] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>>)
    }
    return
  }
}
```

### 约束

- **可并行性**​：Auto Blockify算法仅适用于完全可并行的代码。这意味着各逻辑块的计算与访问必须可以安全地并行执行，块间不存在依赖关系。

- **使用场景**​：若逻辑块数量非常小，则此Pass不会带来任何优势。

## SIMT 模式

SIMT模式下的自动块化与SIMD模式的功能与用法基本相同，本文主要介绍SIMT路径上特有的功能。

### 算法原理

与SIMD模式基本相同，此调度自动添加一层循环，并将SIMT版本的逻辑核ID指令（`tt.get_program_id x/y/z`）替换为用新循环IV和物理核ID计算得到的逻辑核ID。计算逻辑如下：

```mlir
原 kernel：
   pid_x = tt.get_program_id x
   pid_y = tt.get_program_id y
   pid_z = tt.get_program_id z
   <kernel body>

重写后：
   logical = grid_x * grid_y * grid_z // 计算 logical_block_num 
   chunk   = ceildiv(logical, physical_block_dim)
   hw_idx  = gpu.linear_block_id // 内层循环（锁定到物理块）
   start   = hw_idx * chunk // 此物理块处理的第一个逻辑块
   upper   = min(start + chunk, logical) // 此物理块处理的最后一个逻辑块
   for iv in [start, upper) step 1:
       pid_x = iv % grid_x // 计算三维逻辑核ID
       pid_y = (iv / grid_x) % grid_y
       pid_z = iv / (grid_x * grid_y)
       <kernel body, 原始 tt.get_program_id 指令替换为上面解算出的 pid_*>
```

**接口说明**：

该功能在SIMT模式下通过bishengir-compile中的`--enable-auto-blockify-loop`标志控制，也可通过bishengir-opt的`--simt-auto-blockify`标志直接调用。

### 可选性能扩展Super-blocking

GPU SIMT 编程中常常会为单个内核函数配置较小的工作量，并启动较多逻辑核。此场景会导致 Vector 核算力未被充分运用，无法达到最高性能。该扩展在自动块化的基础上，在一个物理核上并行处理若干个相邻逻辑核，提高算力利用率。

**接口说明**：

该功能通过bishengir-compile中的`--super-block-factor=N`标志控制，其中`N`代表需要并行的逻辑核数量，默认为1，即关闭super-blocking，只做常规自动块化。也可通过bishengir-opt的pass选项`-simt-auto-blockify="superblock-factor=N`直接调用。

**逻辑说明**：

将常规自动块化引入的循环的步长改为`N`，表示同时处理`N`个相邻的逻辑块。若原始函数每个逻辑块启动`W`个线程束，则调度后的函数启动`NxW`个线程束，并根据线程束ID计算新逻辑核ID。计算逻辑如下：

```mlir
for iv in [start, upper) step N:
    warp_id = thread_id_x / 32 // 计算线程束ID
    local   = warp_id % N // 计算核内并行的逻辑核的ID
    linear  = iv + local // 计算一维逻辑核ID
    if linear < upper: // 确保逻辑核ID不出界
      <kernel body, 用linear计算三维逻辑核ID并替换原始 tt.get_program_id>
```

### 约束

- **线程束数量限制**​：昇腾支持的最大线程束为64，所以`NxW`的值必须小于等于64。

- **共享内存限制**：开启super-blocking后，`N`个并行的逻辑核会将物理核的共享内存等分为`N`份，需要注意单个逻辑核的内存用量，避免溢出。

- **使用场景**​：若逻辑块数量非常小，则此Pass不会带来任何优势；若每个逻辑核工作量较大，则super-blocking功能性能收益有限。
