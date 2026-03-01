# 编程模型

本节介绍 AscendNPU IR 的编程模型，并让读者熟悉其核心概念和抽象。我们将通过一系列实际程序逐步展开，最终构建一个动态、高性能的 GEMM（矩阵乘法）实现，该实现利用了 AscendNPU IR 的主要特性。

AscendNPU IR 基于Tile的编程模型能够保留算子的高层语义，从而指导编译器更好得进行流水并行、内存管理等一系列优化。我们将逐步调整示例程序，以更好地利用 Tile IR 的特性，这些特性有助于简化程序并使编译器能够提供性能可移植性，从而向用户介绍表示张量计算的多种方法。我们首先介绍基本概念。

## 异构编程模型

```
┌──────────────────────────────────────┐
│            Host（主机）               |
│  • CPU及其内存（主机内存）             │
│  • 执行顺序代码                       │
│  • 管理整个程序的流程                  │
└──────────────────────┬───────────────┘
                       │ NPU API桥接
┌──────────────────────┴───────────────┐
│            Device (设备)              │
│  • NPU 及其内存（设备内存）            │
│  • 执行大规模并行代码                  │
│  • 专注于数据并行计算                  │
└──────────────────────────────────────┘
```

AscendNPU IR支持Host/Device异构编程模型表达，Host函数运行在主机CPU上，Device核函数运行在设备侧，设备核函数可以并发执行。

hacc.function_kind属性定义该函数是Host侧还是Device侧函数，如果是Device核函数，hacc.entry表明函数是Device核函数的入口函数。

hacc.launch表明host侧launch调用Device核函数，launch时需要指定核函数的并发执行逻辑核数（即示例中的block_dim），逻辑核数可以大于实际物理核数，NPU硬件会自行调度运行。


```
module {
  func.func @device_kernel() 
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // 整改点：hivm hir debug需要修改支持打印纯字符串，hex提供默认值，当前必须指定value和hex，不友好
    hivm.hir.debug {debugtype = "print", prefix = "hello world"}
    return
  }

  func.func @host_func(%block_dim: i32) 
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // 整改点：ascendnpu ir需要新增hacc.launch，支持blockdim传递，当前是通过attribute    
    hacc.launch [%block_dim] @device_kernel() : () -> ()   
    return
  }
}
```

## 核函数编程模型


```
// 整改点：
// 1. memref.memref_as_ptr当前是打在module上，需整改到func上，表明func的memref入参作为ptr
// 2. hacc.target变成hacc硬件信息，该pass应该在每一层对接的pipeline上都加上，需确认是否是这样
module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  func.func @add_kernel(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>) 
                        attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>，memref.memref_as_ptr} {
    hivm.hir.set_mask_norm
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16], strides: [1] : memref<?xf16> to memref<16xf16, strided<[1]>>
    %alloc = memref.alloc() : memref<16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<16xf16, strided<[1]>>) outs(%alloc : memref<16xf16>)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16xf16>
    
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16], strides: [1] : memref<?xf16> to memref<16xf16, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16xf16, strided<[1]>>) outs(%alloc_1 : memref<16xf16>)
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16xf16>
    
    %4 = tensor.empty() : tensor<16xf16>
    %5 = hivm.hir.vadd ins(%2, %3 : tensor<16xf16>, tensor<16xf16>) outs(%4 : tensor<16xf16>) -> tensor<16xf16>
    
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16], strides: [1] : memref<?xf16> to memref<16xf16, strided<[1]>>
    hivm.hir.store ins(%5 : tensor<16xf16>) outs(%reinterpret_cast_2 : memref<16xf16, strided<[1]>>)
    return
  }
}
```

```
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>} {
  ...
}
```
