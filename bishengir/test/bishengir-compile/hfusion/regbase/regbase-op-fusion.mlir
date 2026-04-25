// RUN: bishengir-compile -target=Ascend310B4 -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=1 -enable-triton-kernel-compile=true %s

module {
  func.func @triton_unk_fused_add_addmm_native_layer_norm_7(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.tensor_kind = 2 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %cst = arith.constant dense<[16, 1]> : tensor<2xi64>
    %c357_i32 = arith.constant 357 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 6.400000e+01 : f32
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c23_i32 = arith.constant 23 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %0 = tensor.empty() : tensor<16x1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<16x1xf32>) -> tensor<16x1xf32>
    %2 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x1xf32>) -> tensor<16x1xf32>
    %3 = tensor.empty() : tensor<16x64xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %5 = arith.muli %arg11, %c357_i32 : i32
    %6 = arith.addi %5, %c357_i32 : i32
    %7 = arith.minsi %6, %arg6 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1]>>
    scf.for %arg14 = %c0_i32 to %c23_i32 step %c1_i32  : i32 {
      %8 = arith.muli %arg14, %c16_i32 : i32
      %9 = arith.addi %5, %8 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.muli %10, %c64 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [16, 64], strides: [64, 1] : memref<?xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x64xf32>
      %12 = arith.index_cast %arg7 : i32 to index
      %13 = arith.maxsi %12, %c0 : index
      %14 = arith.minsi %13, %c64 : index
      %15 = arith.addi %10, %c16 : index
      %16 = arith.index_cast %7 : i32 to index
      %17 = arith.maxsi %10, %16 : index
      %18 = arith.minsi %15, %17 : index
      %19 = arith.subi %18, %10 : index
      %20 = arith.minsi %19, %c16 : index
      %21 = arith.minsi %14, %c64 : index
      %22 = arith.cmpi slt, %20, %c16 : index
      %23 = arith.cmpi slt, %21, %c64 : index
      %24 = arith.ori %22, %23 : i1
      scf.if %24 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<16x64xf32>)
      }
      %subview = memref.subview %reinterpret_cast_3[0, 0] [%20, %21] [1, 1] : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[0, 0] [%20, %21] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview, %subview_4 : memref<?x?xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %25 = bufferization.to_tensor %alloc restrict writable : memref<16x64xf32>
      %alloc_5 = memref.alloc() : memref<1x64xf32>
      %26 = arith.cmpi slt, %14, %c64 : index
      scf.if %26 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<1x64xf32>)
      }
      %subview_6 = memref.subview %reinterpret_cast[0, 0] [1, %14] [1, 1] : memref<1x64xf32, strided<[64, 1]>> to memref<1x?xf32, strided<[64, 1]>>
      %subview_7 = memref.subview %alloc_5[0, 0] [1, %14] [1, 1] : memref<1x64xf32> to memref<1x?xf32, strided<[64, 1]>>
      memref.copy %subview_6, %subview_7 : memref<1x?xf32, strided<[64, 1]>> to memref<1x?xf32, strided<[64, 1]>>
      %27 = bufferization.to_tensor %alloc_5 restrict writable : memref<1x64xf32>
      %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%11], sizes: [16, 64], strides: [64, 1] : memref<?xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      %alloc_9 = memref.alloc() : memref<16x64xf32>
      scf.if %24 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc_9 : memref<16x64xf32>)
      }
      %subview_10 = memref.subview %reinterpret_cast_8[0, 0] [%20, %21] [1, 1] : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      %subview_11 = memref.subview %alloc_9[0, 0] [%20, %21] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_10, %subview_11 : memref<?x?xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %28 = bufferization.to_tensor %alloc_9 restrict writable : memref<16x64xf32>
      %collapsed = tensor.collapse_shape %27 [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
      %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%3 : tensor<16x64xf32>) dimensions = [0]
      %29 = arith.addf %25, %broadcasted : tensor<16x64xf32>
      %30 = arith.addf %29, %28 : tensor<16x64xf32>
      %extracted_slice = tensor.extract_slice %30[0, 0] [%20, %21] [1, 1] : tensor<16x64xf32> to tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %extracted_slice into %4[0, 0] [%20, %21] [1, 1] : tensor<?x?xf32> into tensor<16x64xf32>
      %31 = tensor.empty() : tensor<16xf32>
      %32 = linalg.fill ins(%cst_0 : f32) outs(%31 : tensor<16xf32>) -> tensor<16xf32>
      %reduced = linalg.reduce ins(%inserted_slice : tensor<16x64xf32>) outs(%32 : tensor<16xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %40 = arith.addf %in, %init : f32
          linalg.yield %40 : f32
        }
      %reshape = tensor.reshape %reduced(%cst) : (tensor<16xf32>, tensor<2xi64>) -> tensor<16x1xf32>
      %33 = arith.divf %reshape, %2 : tensor<16x1xf32>
      %collapsed_12 = tensor.collapse_shape %33 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
      %broadcasted_13 = linalg.broadcast ins(%collapsed_12 : tensor<16xf32>) outs(%3 : tensor<16x64xf32>) dimensions = [1]
      %34 = arith.subf %30, %broadcasted_13 : tensor<16x64xf32>
      %35 = arith.mulf %34, %34 : tensor<16x64xf32>
      %extracted_slice_14 = tensor.extract_slice %35[0, 0] [%20, %21] [1, 1] : tensor<16x64xf32> to tensor<?x?xf32>
      %inserted_slice_15 = tensor.insert_slice %extracted_slice_14 into %4[0, 0] [%20, %21] [1, 1] : tensor<?x?xf32> into tensor<16x64xf32>
      %reduced_16 = linalg.reduce ins(%inserted_slice_15 : tensor<16x64xf32>) outs(%32 : tensor<16xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %40 = arith.addf %in, %init : f32
          linalg.yield %40 : f32
        }
      %reshape_17 = tensor.reshape %reduced_16(%cst) : (tensor<16xf32>, tensor<2xi64>) -> tensor<16x1xf32>
      %36 = arith.divf %reshape_17, %2 : tensor<16x1xf32>
      %37 = arith.addf %36, %1 : tensor<16x1xf32>
      %38 = math.rsqrt %37 : tensor<16x1xf32>
      %collapsed_18 = tensor.collapse_shape %38 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
      %broadcasted_19 = linalg.broadcast ins(%collapsed_18 : tensor<16xf32>) outs(%3 : tensor<16x64xf32>) dimensions = [1]
      %39 = arith.mulf %34, %broadcasted_19 : tensor<16x64xf32>
      bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[64, 1], offset: ?>>) -> ()
      %reinterpret_cast_20 = memref.reinterpret_cast %arg5 to offset: [%11], sizes: [16, 64], strides: [64, 1] : memref<?xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      %extracted_slice_21 = tensor.extract_slice %39[0, 0] [%20, %21] [1, 1] : tensor<16x64xf32> to tensor<?x?xf32>
      %subview_22 = memref.subview %reinterpret_cast_20[0, 0] [%20, %21] [1, 1] : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_21 in writable %subview_22 : (tensor<?x?xf32>, memref<?x?xf32, strided<[64, 1], offset: ?>>) -> ()
    }
    return
  }
}

