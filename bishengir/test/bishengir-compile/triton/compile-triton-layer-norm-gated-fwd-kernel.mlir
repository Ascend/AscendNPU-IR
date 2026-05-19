// REQUIRES: enable-lir-compile
// RUN: bishengir-compile %s --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @layer_norm_gated_fwd_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: f32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 5.120000e+02 : f32
    %0 = tensor.empty() : tensor<32xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    %2 = tensor.empty() : tensor<32x512xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %6 = arith.muli %arg16, %c32_i32 : i32
    %7 = arith.maxsi %6, %c0_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.muli %8, %c512 : index
    %10 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 512], strides: [512, 1] : memref<?xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<32x512xf32>
    %11 = arith.divsi %9, %c512 : index
    %12 = arith.subi %10, %11 : index
    %13 = arith.maxsi %12, %c0 : index
    %14 = arith.minsi %13, %c32 : index
    %15 = arith.remsi %9, %c512 : index
    %16 = arith.subi %c512, %15 : index
    %17 = arith.maxsi %16, %c0 : index
    %18 = arith.minsi %17, %c512 : index
    %19 = arith.subi %c0_i32, %6 : i32
    %20 = arith.maxsi %19, %c0_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.minsi %21, %14 : index
    %23 = arith.subi %14, %22 : index
    %24 = arith.minsi %18, %c0 : index
    %25 = arith.subi %18, %24 : index
    %26 = arith.cmpi slt, %23, %c32 : index
    %27 = arith.cmpi slt, %25, %c512 : index
    %28 = arith.ori %26, %27 : i1
    scf.if %28 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<32x512xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%23, %25] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[%22, %24] [%23, %25] [1, 1] : memref<32x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    memref.copy %subview, %subview_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    %29 = bufferization.to_tensor %alloc restrict writable : memref<32x512xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%9], sizes: [32, 512], strides: [512, 1] : memref<?xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %alloc_4 = memref.alloc() : memref<32x512xf32>
    scf.if %28 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc_4 : memref<32x512xf32>)
    } {hivm.unlikely_condition}
    %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%23, %25] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[%22, %24] [%23, %25] [1, 1] : memref<32x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    memref.copy %subview_5, %subview_6 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<32x512xf32>
    %31 = arith.addf %29, %30 : tensor<32x512xf32>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%9], sizes: [32, 512], strides: [512, 1] : memref<?xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %extracted_slice = tensor.extract_slice %31[%22, %24] [%23, %25] [1, 1] : tensor<32x512xf32> to tensor<?x?xf32>
    %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [%23, %25] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_8 : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
    %32 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    %reduced = linalg.reduce ins(%31 : tensor<32x512xf32>) outs(%32 : tensor<32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %58 = arith.addf %in, %init : f32
        linalg.yield %58 : f32
      }
    %33 = arith.divf %reduced, %1 : tensor<32xf32>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg9 to offset: [%8], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
    %34 = arith.subi %10, %8 : index
    %35 = arith.maxsi %34, %c0 : index
    %36 = arith.minsi %35, %c32 : index
    %37 = arith.minsi %21, %36 : index
    %38 = arith.subi %36, %37 : index
    %extracted_slice_10 = tensor.extract_slice %33[%37] [%38] [1] : tensor<32xf32> to tensor<?xf32>
    %subview_11 = memref.subview %reinterpret_cast_9[0] [%38] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_10 in writable %subview_11 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    %broadcasted = linalg.broadcast ins(%33 : tensor<32xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [1] 
    %39 = arith.subf %31, %broadcasted : tensor<32x512xf32>
    %40 = arith.mulf %39, %39 : tensor<32x512xf32>
    %reduced_12 = linalg.reduce ins(%40 : tensor<32x512xf32>) outs(%32 : tensor<32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %58 = arith.addf %in, %init : f32
        linalg.yield %58 : f32
      }
    %41 = arith.divf %reduced_12, %1 : tensor<32xf32>
    %42 = linalg.fill ins(%arg11 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    %43 = arith.addf %41, %42 : tensor<32xf32>
    %44 = math.sqrt %43 : tensor<32xf32>
    %45 = arith.divf %4, %44 : tensor<32xf32>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg10 to offset: [%8], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
    %extracted_slice_14 = tensor.extract_slice %45[%37] [%38] [1] : tensor<32xf32> to tensor<?xf32>
    %subview_15 = memref.subview %reinterpret_cast_13[0] [%38] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_14 in writable %subview_15 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_16 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [512], strides: [1] : memref<?xf32> to memref<512xf32, strided<[1]>>
    %alloc_17 = memref.alloc() : memref<512xf32>
    memref.copy %reinterpret_cast_16, %alloc_17 : memref<512xf32, strided<[1]>> to memref<512xf32>
    %46 = bufferization.to_tensor %alloc_17 restrict writable : memref<512xf32>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [512], strides: [1] : memref<?xf32> to memref<512xf32, strided<[1]>>
    %alloc_19 = memref.alloc() : memref<512xf32>
    memref.copy %reinterpret_cast_18, %alloc_19 : memref<512xf32, strided<[1]>> to memref<512xf32>
    %47 = bufferization.to_tensor %alloc_19 restrict writable : memref<512xf32>
    %broadcasted_20 = linalg.broadcast ins(%45 : tensor<32xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [1] 
    %48 = arith.mulf %39, %broadcasted_20 : tensor<32x512xf32>
    %broadcasted_21 = linalg.broadcast ins(%46 : tensor<512xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [0] 
    %49 = arith.mulf %48, %broadcasted_21 : tensor<32x512xf32>
    %broadcasted_22 = linalg.broadcast ins(%47 : tensor<512xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [0] 
    %50 = arith.addf %49, %broadcasted_22 : tensor<32x512xf32>
    %reinterpret_cast_23 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 512], strides: [512, 1] : memref<?xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %alloc_24 = memref.alloc() : memref<32x512xf32>
    scf.if %28 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc_24 : memref<32x512xf32>)
    } {hivm.unlikely_condition}
    %subview_25 = memref.subview %reinterpret_cast_23[0, 0] [%23, %25] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    %subview_26 = memref.subview %alloc_24[%22, %24] [%23, %25] [1, 1] : memref<32x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    memref.copy %subview_25, %subview_26 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    %51 = bufferization.to_tensor %alloc_24 restrict writable : memref<32x512xf32>
    %52 = arith.mulf %50, %51 : tensor<32x512xf32>
    %53 = arith.subf %3, %51 : tensor<32x512xf32>
    %54 = math.exp %53 : tensor<32x512xf32>
    %55 = arith.addf %54, %5 : tensor<32x512xf32>
    %56 = arith.divf %5, %55 : tensor<32x512xf32>
    %57 = arith.mulf %52, %56 : tensor<32x512xf32>
    %reinterpret_cast_27 = memref.reinterpret_cast %arg4 to offset: [%9], sizes: [32, 512], strides: [512, 1] : memref<?xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %extracted_slice_28 = tensor.extract_slice %57[%22, %24] [%23, %25] [1, 1] : tensor<32x512xf32> to tensor<?x?xf32>
    %subview_29 = memref.subview %reinterpret_cast_27[0, 0] [%23, %25] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_28 in writable %subview_29 : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
    return
  }
}