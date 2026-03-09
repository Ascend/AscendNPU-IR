# 调试模块

## device_print

### 硬件背景：

支持 Triton 语言规范中的 [triton.language.device_print](https://triton-lang.org/main/python-api/generated/triton.language.device_print.html#triton.language.device_print) 特性

### 算法原理：

实现原理涉及Triton Ascend, AscendNPU IR, 毕昇编译器三部分配合实现该部分主要以AscendNPU IR为重点展开说明

#### Triton Ascend

生成初始.ttadapter ir. 过程中会将triton侧tl.device_print转换成func.call @triton_print_接口

#### AscendNPU IR

接收到.ttadapter ir之后在AscendNPU IR阶段主要会经历如下变换

##### AdaptTritonKernel

将func.call @triton_print_接口转换成hfusion.print接口

```
// Before AdaptTritonKernel
%reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
%alloc = memref.alloc() : memref<8xi64>
memref.copy %reinterpret_cast, %alloc : memref<8xi64, strided<[1]>> to memref<8xi64>
%0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
call @triton_print_0(%0) : (tensor<8xi64>) -> ()

// After AdaptTritonKernel
%reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
%alloc = memref.alloc() : memref<8xi64>
memref.copy %reinterpret_cast, %alloc : memref<8xi64, strided<[1]>> to memref<8xi64>
%0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
hfusion.print " x: " {hex = false} %0 : tensor<8xi64>
```

##### HFusionToHIVM

将hfusion.print接口转换成hivm.hir.debug接口

```mlir
// Before ConvertHFusionToHIVM
%reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
%alloc = memref.alloc() : memref<8xi64>
memref.copy %reinterpret_cast, %alloc : memref<8xi64, strided<[1]>> to memref<8xi64>
%0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
hfusion.print " x: " {hex = false} %0 : tensor<8xi64>

// After ConvertHFusionToHIVM
%reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
%alloc = memref.alloc() : memref<8xi64>
memref.copy %reinterpret_cast, %alloc : memref<8xi64, strided<[1]>> to memref<8xi64>
%0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
hivm.hir.debug {debugtype = "print", hex = false, prefix = " x: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %0 : tensor<8xi64>
```

##### InlineFixpipe

插入 fixpipe 以用于 hivm.print，该 hivm.print 会打印 mmad 结果，而 mmad 结果是 scf.for 中的 yield。

```mlir
// Before InlineFixpipe
%init = tensor.empty()
%res = scf.for iter_arg(%arg = %init) {
    %t = hivm.mmadL1 ins() outs(%arg)
    hivm.print %t
    scf.yield %t
}

// After InlineFixpipe
%init = tensor.empty()
%res = scf.for iter_arg(%arg = %init) {
    %t = hivm.mmadL1 ins() outs(%arg)
    %fixpipe = hivm.fixpipe int(%t)
    hivm.print %fixpipe
    scf.yield %t
}
```

##### InsertNZ2NDForDebug

device_print仅支持ub/gm上的数据打印因此当打印L1的数据时需要将数据先从L1搬至gm，该pass的作用就是当识别到`hivm::MmadL1Op`时检查该op的输入若输入被`hivm::DebugOp`用到则需要申请一块workspace的大小然后插入NZ2ND的op确保搬至gm打印

```mlir
// Before InsertNZ2NDForDebug
%12 = bufferization.to_tensor %alloc restrict writable : memref<1x4xf32>
%13 = arith.index_cast %arg8 : i32 to index
%14 = arith.index_cast %5 : i32 to index
%reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [4, 1], strides: [%13, 1] : memref<?xf32> to memref<4x1xf32, strided<[?, 1], offset: ?>>
%alloc_1 = memref.alloc() : memref<4x1xf32>
hivm.hir.load ins(%reinterpret_cast_0 : memref<4x1xf32, strided<[?, 1], offset: ?>>) outs(%alloc_1 : memref<4x1xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
%15 = bufferization.to_tensor %alloc_1 restrict writable : memref<4x1xf32>
%16 = arith.muli %8, %arg8 : i32
%17 = arith.index_cast %16 : i32 to index
%18 = arith.addi %17, %14 : index
%19 = tensor.empty() : tensor<1x1xf32>
%c1 = arith.constant 1 : index
%c4 = arith.constant 4 : index
%c1_2 = arith.constant 1 : index
%20 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%12, %15, %true, %c1, %c4, %c1_2 : tensor<1x4xf32>, tensor<4x1xf32>, i1, index, index, index) outs(%19 : tensor<1x1xf32>) -> tensor<1x1xf32>
hivm.hir.debug {debugtype = "print", hex = false, prefix = " a_vals: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %12 : tensor<1x4xf32>

// After InsertNZ2NDForDebug
%12 = bufferization.to_tensor %alloc restrict writable : memref<1x4xf32>
%13 = memref_ext.alloc_workspace() : memref<1x4xf32>
%14 = bufferization.to_tensor %13 restrict writable : memref<1x4xf32>
%15 = hivm.hir.nz2nd ins(%12 : tensor<1x4xf32>) outs(%14 : tensor<1x4xf32>) -> tensor<1x4xf32>
%16 = arith.index_cast %arg8 : i32 to index
%17 = arith.index_cast %5 : i32 to index
%reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [4, 1], strides: [%16, 1] : memref<?xf32> to memref<4x1xf32, strided<[?, 1], offset: ?>>
%alloc_1 = memref.alloc() : memref<4x1xf32>
hivm.hir.load ins(%reinterpret_cast_0 : memref<4x1xf32, strided<[?, 1], offset: ?>>) outs(%alloc_1 : memref<4x1xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
%18 = bufferization.to_tensor %alloc_1 restrict writable : memref<4x1xf32>
%19 = arith.muli %8, %arg8 : i32
%20 = arith.index_cast %19 : i32 to index
%21 = arith.addi %20, %17 : index
%22 = tensor.empty() : tensor<1x1xf32>
%23 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%12, %18, %true, %c1, %c4, %c1 : tensor<1x4xf32>, tensor<4x1xf32>, i1, index, index, index) outs(%22 : tensor<1x1xf32>) -> tensor<1x1xf32>
hivm.hir.debug {debugtype = "print", hex = false, prefix = " a_vals: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %15 : tensor<1x4xf32>
```

##### SplitMixKernel

mix类用例 Debug op会在该pass内先进行InferCoreType推断出精确的coretype(VECTOR/CUBE)，默认是CUBE_OR_VECTOR, 然后对mix函数进行拆分后生成纯cube函数和纯vector函数，这将决定Debug op最终在cube核上运行还是vector核上运行

##### InsertInitAndFinishForDebug

若存在Debug op则将hivm.hir.init_print 调用添加到每个函数开头，将 hivm.hir.finish_print 添加到每个 hivm.hir.print 之后。hivm.hir.init_print 用于打印之前的准备工作，hivm.hir.finish_print 用于打印之后的工作，目前没有特别具体作用，为将来扩展device_print预留了接口

```mlir
// Before InsertInitAndFinishForDebug
hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast, %cast_1, %true, %c1, %c4, %c1 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_2 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c1_i64, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
%16 = arith.index_cast %2 : i64 to index
%17 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%16]
%view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<1x1xf32, #hivm.address_space<gm>>
hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
hivm.hir.fixpipe {enable_nz2nd} ins(%cast_2 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view : memref<1x1xf32, #hivm.address_space<gm>>)
hivm.hir.pipe_barrier[<PIPE_ALL>]
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
hivm.hir.debug {debugtype = "print", hex = false, prefix = " acc_11: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %view : memref<1x1xf32, #hivm.address_space<gm>>

// After InsertInitAndFinishForDebug
hivm.hir.init_debug
hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast, %cast_1, %true, %c1, %c4, %c1 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_2 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c1_i64, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
%14 = arith.index_cast %0 : i64 to index
%15 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%14]
%view = memref.view %arg2[%15][] : memref<?xi8, #hivm.address_space<gm>> to memref<1x1xf32, #hivm.address_space<gm>>
hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
hivm.hir.fixpipe {enable_nz2nd} ins(%cast_2 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%view : memref<1x1xf32, #hivm.address_space<gm>>)
hivm.hir.pipe_barrier[<PIPE_ALL>]
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
hivm.hir.debug {debugtype = "print", finishInserted = 0 : i32, hex = false, prefix = " acc_11: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %view : memref<1x1xf32, #hivm.address_space<gm>>
hivm.hir.finish_debug
```

##### ConvertHIVMToStandard

将 hivm.hir.init_print/hivm.hir.print/hivm.hir.finish_print 转换为库函数调用

##### ConvertHIVMToLLVM

ConvertHIVMToLLVM 引入真正的库函数，并设置 print 相关函数的链接为 ExternWeak（允许在多个 llvm modules 中重复定义）

##### Debug op库实现

op库当前实现是通过scalar打印去实现的，通过for循环外抛的方式调用毕昇编译器提供的`cce::printf`接口进行scalar打印

#### 毕昇编译器

triton-ascend 产生的 host 侧 launcher调用 bisheng 编译器编好的 kernel 并将打印缓冲区传给 kernel，待 kernel 返回后在 host launcher 侧读取缓冲区并进行真正的打印。此部分代码在 bisheng 编译器自带的头文件中实现，并由 triton-ascend 自动从 bisheng 编译器的路径中抽取。

### 接口说明：

通过设置环境变量 `TRITON_DEVICE_PRINT=1`来开启该功能，开启后triton ascend侧会设置相关宏信息__CCE_ENABLE_PRINT__，该宏信息在毕昇编译器侧会影响是否开启打印，其次编译meta op库的时候需要开启--cce-enable-print(当前默认一直开启)已确保开启打印

```mlir
// hfusion op接口
// dtype - 待打印tensor/scalar对应的数据类型
hfusion.print " prefix = xxx " {hex = xxx} %args : dtype

// hivm op接口
// tcoretype - 指明是在core核上运行还是vector核上运行（默认初始值为: CUBE_OR_VECTOR）
hivm.hir.debug {debugtype = "print", hex = xxx, prefix = " xxx: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %args : dtype
```

### 约束能力：

1. 仅支持tensor和scalar的打印
2. 当前device_print打印的大小固定为16k
3. 目前triton侧sanitizer和device_print不支持同时开启
4. 打印支持如下数据类型:bool/int8/uint8/int16/uint16/int32/uint32/int64/bfloat16/half/float32
