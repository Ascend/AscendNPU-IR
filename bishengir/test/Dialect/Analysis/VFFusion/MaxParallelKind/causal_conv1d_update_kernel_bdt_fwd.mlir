

// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @causal_conv1d_update_kernel_bdt_fwd_fused_0
// CHECK: tensor.extract_slice
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: return

// CHECK-LABEL: func.func private @causal_conv1d_update_kernel_bdt_fwd_fused_1
// CHECK: linalg.transpose
// CHECK: tensor.extract_slice
// CHECK: linalg.transpose
// CHECK: return

// CHECK-LABEL: func.func private @causal_conv1d_update_kernel_bdt_fwd_fused_2
// CHECK: linalg.transpose
// CHECK: linalg.transpose
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: hfusion.cast
// CHECK: hfusion.cast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: hfusion.cast
// CHECK: return

// CHECK-LABEL: func.func @causal_conv1d_update_kernel_bdt_fwd
// CHECK: scf.for
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: scf.if
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: scf.if
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: tensor.extract_slice
// CHECK: tensor.concat
// CHECK: linalg.transpose
// CHECK: func.call @causal_conv1d_update_kernel_bdt_fwd_fused_1
// CHECK: memref.reinterpret_cast
// CHECK: tensor.extract_slice
// CHECK: memref.subview
// CHECK: bufferization.materialize_in_destination
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK: tensor.expand_shape
// CHECK: func.call @causal_conv1d_update_kernel_bdt_fwd_fused_0
// CHECK: scf.yield
// CHECK: func.call @causal_conv1d_update_kernel_bdt_fwd_fused_2
// CHECK: tensor.extract_slice
// CHECK: memref.subview
// CHECK: bufferization.materialize_in_destination
// CHECK: return


module {
  func.func @causal_conv1d_update_kernel_bdt_fwd(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant -1.000000e+00 : f16
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16384 = arith.constant 16384 : index
    %c259 = arith.constant 259 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 1.000000e+00 : f16
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x16x256xf16>
    %1 = tensor.empty() : tensor<256x1x16xf16>
    %2 = tensor.empty() : tensor<1x16x256xf32>
    scf.for %arg13 = %arg10 to %c16_i32 step %arg7  : i32 {
      %3 = arith.remsi %arg13, %c8_i32 : i32
      %4 = arith.divsi %arg13, %c8_i32 : i32
      %5 = arith.muli %3, %c16_i32 : i32
      %6 = arith.maxsi %5, %c0_i32 : i32
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.muli %7, %c4 : index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%8], sizes: [16, 4], strides: [4, 1] : memref<?xf16> to memref<16x4xf16, strided<[4, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x4xf16>
      %9 = arith.divsi %8, %c4 : index
      %10 = arith.subi %c128, %9 : index
      %11 = arith.maxsi %10, %c0 : index
      %12 = arith.minsi %11, %c16 : index
      %13 = arith.remsi %8, %c4 : index
      %14 = arith.subi %c4, %13 : index
      %15 = arith.maxsi %14, %c0 : index
      %16 = arith.minsi %15, %c4 : index
      %17 = arith.subi %c0_i32, %5 : i32
      %18 = arith.maxsi %17, %c0_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = arith.minsi %19, %12 : index
      %21 = arith.subi %12, %20 : index
      %22 = arith.minsi %16, %c0 : index
      %23 = arith.subi %16, %22 : index
      %24 = arith.cmpi slt, %21, %c16 : index
      %25 = arith.cmpi slt, %23, %c4 : index
      %26 = arith.ori %24, %25 : i1
      scf.if %26 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc : memref<16x4xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0, 0] [%21, %23] [1, 1] : memref<16x4xf16, strided<[4, 1], offset: ?>> to memref<?x?xf16, strided<[4, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%20, %22] [%21, %23] [1, 1] : memref<16x4xf16> to memref<?x?xf16, strided<[4, 1], offset: ?>>
      memref.copy %subview, %subview_2 : memref<?x?xf16, strided<[4, 1], offset: ?>> to memref<?x?xf16, strided<[4, 1], offset: ?>>
      %27 = bufferization.to_tensor %alloc restrict writable : memref<16x4xf16>
      %28 = arith.muli %4, %c512_i32 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = arith.addi %8, %29 : index
      %31 = arith.addi %30, %c1 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [16, 259], strides: [4, 1] : memref<?xf16> to memref<16x259xf16, strided<[4, 1], offset: ?>>
      %alloc_4 = memref.alloc() : memref<16x259xf16>
      %32 = arith.subi %31, %29 : index
      %33 = arith.divsi %32, %c4 : index
      %34 = arith.subi %c128, %33 : index
      %35 = arith.maxsi %34, %c0 : index
      %36 = arith.minsi %35, %c16 : index
      %37 = arith.remsi %32, %c4 : index
      %38 = arith.subi %c4, %37 : index
      %39 = arith.maxsi %38, %c0 : index
      %40 = arith.minsi %39, %c259 : index
      %41 = arith.minsi %19, %36 : index
      %42 = arith.subi %36, %41 : index
      %43 = arith.minsi %40, %c0 : index
      %44 = arith.subi %40, %43 : index
      %45 = arith.cmpi slt, %42, %c16 : index
      %46 = arith.cmpi slt, %44, %c259 : index
      %47 = arith.ori %45, %46 : i1
      scf.if %47 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc_4 : memref<16x259xf16>)
      } {hivm.unlikely_condition}
      %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%42, %44] [1, 1] : memref<16x259xf16, strided<[4, 1], offset: ?>> to memref<?x?xf16, strided<[4, 1], offset: ?>>
      %subview_6 = memref.subview %alloc_4[%41, %43] [%42, %44] [1, 1] : memref<16x259xf16> to memref<?x?xf16, strided<[259, 1], offset: ?>>
      memref.copy %subview_5, %subview_6 : memref<?x?xf16, strided<[4, 1], offset: ?>> to memref<?x?xf16, strided<[259, 1], offset: ?>>
      %48 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x259xf16>
      %49 = arith.muli %4, %c16384_i32 : i32
      %50 = arith.index_cast %49 : i32 to index
      %51 = arith.index_cast %5 : i32 to index
      %52 = arith.muli %51, %c128 : index
      %53 = arith.addi %50, %52 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%53], sizes: [16, 256], strides: [128, 1] : memref<?xf16> to memref<16x256xf16, strided<[128, 1], offset: ?>>
      %alloc_8 = memref.alloc() : memref<16x256xf16>
      %54 = arith.addi %51, %c16 : index
      %55 = arith.maxsi %51, %c128 : index
      %56 = arith.minsi %54, %55 : index
      %57 = arith.subi %56, %51 : index
      %58 = arith.minsi %57, %c16 : index
      linalg.fill ins(%cst_1 : f16) outs(%alloc_8 : memref<16x256xf16>)
      %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [%58, 128] [1, 1] : memref<16x256xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16, strided<[128, 1], offset: ?>>
      %subview_10 = memref.subview %alloc_8[0, 0] [%58, 128] [1, 1] : memref<16x256xf16> to memref<?x128xf16, strided<[256, 1]>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16, strided<[256, 1]>>
      %59 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x256xf16>
      %extracted_slice = tensor.extract_slice %48[0, 0] [16, 3] [1, 1] : tensor<16x259xf16> to tensor<16x3xf16>
      %concat = tensor.concat dim(1) %extracted_slice, %59 : (tensor<16x3xf16>, tensor<16x256xf16>) -> tensor<16x259xf16>
      %60 = tensor.empty() : tensor<259x16xf16>
      %transposed = linalg.transpose ins(%concat : tensor<16x259xf16>) outs(%60 : tensor<259x16xf16>) permutation = [1, 0] 
      %61 = tensor.empty() : tensor<4x16xf16>
      %transposed_11 = linalg.transpose ins(%27 : tensor<16x4xf16>) outs(%61 : tensor<4x16xf16>) permutation = [1, 0] 
      %extracted_slice_12 = tensor.extract_slice %transposed[127, 0] [4, 16] [1, 1] : tensor<259x16xf16> to tensor<4x16xf16>
      %62 = tensor.empty() : tensor<16x4xf16>
      %transposed_13 = linalg.transpose ins(%extracted_slice_12 : tensor<4x16xf16>) outs(%62 : tensor<16x4xf16>) permutation = [1, 0] 
      %63 = arith.muli %51, %c4 : index
      %64 = arith.addi %29, %63 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [16, 4], strides: [4, 1] : memref<?xf16> to memref<16x4xf16, strided<[4, 1], offset: ?>>
      %65 = arith.minsi %58, %c16 : index
      %extracted_slice_15 = tensor.extract_slice %transposed_13[0, 0] [%65, 4] [1, 1] : tensor<16x4xf16> to tensor<?x4xf16>
      %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%65, 4] [1, 1] : memref<16x4xf16, strided<[4, 1], offset: ?>> to memref<?x4xf16, strided<[4, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 : (tensor<?x4xf16>, memref<?x4xf16, strided<[4, 1], offset: ?>>) -> ()
      %66 = linalg.fill ins(%cst_1 : f16) outs(%1 : tensor<256x1x16xf16>) -> tensor<256x1x16xf16>
      %67 = scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg15 = %66) -> (tensor<256x1x16xf16>)  : i32 {
        %103 = arith.index_cast %arg14 : i32 to index
        %extracted_slice_22 = tensor.extract_slice %transposed[%103, 0] [256, 16] [1, 1] : tensor<259x16xf16> to tensor<256x16xf16>
        %expanded = tensor.expand_shape %extracted_slice_22 [[0], [1, 2]] output_shape [256, 1, 16] : tensor<256x16xf16> into tensor<256x1x16xf16>
        %extracted_slice_23 = tensor.extract_slice %transposed_11[%103, 0] [1, 16] [1, 1] : tensor<4x16xf16> to tensor<1x16xf16>
        %broadcasted = linalg.broadcast ins(%extracted_slice_23 : tensor<1x16xf16>) outs(%1 : tensor<256x1x16xf16>) dimensions = [0] 
        %104 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %broadcasted : tensor<256x1x16xf16>, tensor<256x1x16xf16>) outs(%1 : tensor<256x1x16xf16>) -> tensor<256x1x16xf16>
        %105 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg15, %104 : tensor<256x1x16xf16>, tensor<256x1x16xf16>) outs(%1 : tensor<256x1x16xf16>) -> tensor<256x1x16xf16>
        scf.yield %105 : tensor<256x1x16xf16>
      }
      %68 = tensor.empty() : tensor<16x1x256xf16>
      %transposed_17 = linalg.transpose ins(%67 : tensor<256x1x16xf16>) outs(%68 : tensor<16x1x256xf16>) permutation = [2, 1, 0] 
      %transposed_18 = linalg.transpose ins(%transposed_17 : tensor<16x1x256xf16>) outs(%0 : tensor<1x16x256xf16>) permutation = [1, 0, 2] 
      %69 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%transposed_18, %cst : tensor<1x16x256xf16>, f16) outs(%0 : tensor<1x16x256xf16>) -> tensor<1x16x256xf16>
      %70 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%69 : tensor<1x16x256xf16>) outs(%0 : tensor<1x16x256xf16>) -> tensor<1x16x256xf16>
      %71 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%70, %cst_0 : tensor<1x16x256xf16>, f16) outs(%0 : tensor<1x16x256xf16>) -> tensor<1x16x256xf16>
      %72 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%71 : tensor<1x16x256xf16>) outs(%2 : tensor<1x16x256xf32>) -> tensor<1x16x256xf32>
      %73 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%transposed_18 : tensor<1x16x256xf16>) outs(%2 : tensor<1x16x256xf32>) -> tensor<1x16x256xf32>
      %74 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%73, %72 : tensor<1x16x256xf32>, tensor<1x16x256xf32>) outs(%2 : tensor<1x16x256xf32>) -> tensor<1x16x256xf32>
      %75 = arith.maxsi %4, %c0_i32 : i32
      %76 = arith.index_cast %75 : i32 to index
      %77 = arith.muli %76, %c16384 : index
      %78 = arith.muli %7, %c128 : index
      %79 = arith.addi %77, %78 : index
      %reinterpret_cast_19 = memref.reinterpret_cast %arg6 to offset: [%79], sizes: [1, 16, 256], strides: [16384, 128, 1] : memref<?xf16> to memref<1x16x256xf16, strided<[16384, 128, 1], offset: ?>>
      %80 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%74 : tensor<1x16x256xf32>) outs(%0 : tensor<1x16x256xf16>) -> tensor<1x16x256xf16>
      %81 = arith.divsi %79, %c16384 : index
      %82 = arith.subi %c2, %81 : index
      %83 = arith.maxsi %82, %c0 : index
      %84 = arith.minsi %83, %c1 : index
      %85 = arith.remsi %79, %c16384 : index
      %86 = arith.divsi %85, %c128 : index
      %87 = arith.subi %c128, %86 : index
      %88 = arith.maxsi %87, %c0 : index
      %89 = arith.minsi %88, %c16 : index
      %90 = arith.remsi %85, %c128 : index
      %91 = arith.subi %c128, %90 : index
      %92 = arith.maxsi %91, %c0 : index
      %93 = arith.minsi %92, %c256 : index
      %94 = arith.subi %c0_i32, %4 : i32
      %95 = arith.maxsi %94, %c0_i32 : i32
      %96 = arith.index_cast %95 : i32 to index
      %97 = arith.minsi %96, %84 : index
      %98 = arith.subi %84, %97 : index
      %99 = arith.minsi %19, %89 : index
      %100 = arith.subi %89, %99 : index
      %101 = arith.minsi %93, %c0 : index
      %102 = arith.subi %93, %101 : index
      %extracted_slice_20 = tensor.extract_slice %80[%97, %99, %101] [%98, %100, %102] [1, 1, 1] : tensor<1x16x256xf16> to tensor<?x?x?xf16>
      %subview_21 = memref.subview %reinterpret_cast_19[0, 0, 0] [%98, %100, %102] [1, 1, 1] : memref<1x16x256xf16, strided<[16384, 128, 1], offset: ?>> to memref<?x?x?xf16, strided<[16384, 128, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_20 in writable %subview_21 : (tensor<?x?x?xf16>, memref<?x?x?xf16, strided<[16384, 128, 1], offset: ?>>) -> ()
    }
    return
  }
}