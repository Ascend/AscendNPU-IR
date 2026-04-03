// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s
// CHECK-LABEL: func.func private @triton_unk_fused_native_layer_norm_0_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: tensor.extract_slice

// CHECK-LABEL: func.func private @triton_unk_fused_native_layer_norm_0_fused_1(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: tensor.extract_slice

// CHECK-LABEL: func.func @triton_unk_fused_native_layer_norm_0(
// CHECK: arith.constant
// CHECK: tensor.empty
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.minsi
// CHECK: memref.reinterpret_cast
// CHECK: scf.for
// CHECK: memref.alloc
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: linalg.reduce
// CHECK: func.call @triton_unk_fused_native_layer_norm_0_fused_1
// CHECK: bufferization.materialize_in_destination
// CHECK: func.call @triton_unk_fused_native_layer_norm_0_fused_0
// CHECK: tensor.extract_slice
// CHECK: return

module {
    func.func @triton_unk_fused_native_layer_norm_0(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c768 = arith.constant 768 : index
    %c55_i32 = arith.constant 55 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 9.99999996E-13 : f32
    %0 = tensor.empty() : tensor<16xf32>
    %1 = arith.muli %arg13, %c55_i32 : i32
    %2 = arith.addi %1, %c55_i32 : i32
    %3 = arith.minsi %2, %arg8 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [768], strides: [1] : memref<?xf32> to memref<768xf32, strided<[1]>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [768], strides: [1] : memref<?xf32> to memref<768xf32, strided<[1]>>
    scf.for %arg16 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %4 = arith.muli %arg16, %c16_i32 : i32
      %5 = arith.addi %1, %4 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7 = arith.muli %6, %c768 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%7], sizes: [16, 768], strides: [768, 1] : memref<?xf32> to memref<16x768xf32, strided<[768, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x768xf32>
      %8 = arith.addi %6, %c16 : index
      %9 = arith.index_cast %3 : i32 to index
      %10 = arith.maxsi %6, %9 : index
      %11 = arith.minsi %8, %10 : index
      %12 = arith.subi %11, %6 : index
      %13 = arith.cmpi slt, %12, %c16 : index
      scf.if %13 {
        linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<16x768xf32>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_4[0, 0] [%12, 768] [1, 1] : memref<16x768xf32, strided<[768, 1], offset: ?>> to memref<?x768xf32, strided<[768, 1], offset: ?>>
      %subview_5 = memref.subview %alloc[0, 0] [%12, 768] [1, 1] : memref<16x768xf32> to memref<?x768xf32, strided<[768, 1]>>
      memref.copy %subview, %subview_5 : memref<?x768xf32, strided<[768, 1], offset: ?>> to memref<?x768xf32, strided<[768, 1]>>
      %14 = bufferization.to_tensor %alloc restrict writable : memref<16x768xf32>
      %alloc_6 = memref.alloc() : memref<768xf32>
      memref.copy %reinterpret_cast, %alloc_6 : memref<768xf32, strided<[1]>> to memref<768xf32>
      %15 = bufferization.to_tensor %alloc_6 restrict writable : memref<768xf32>
      %alloc_7 = memref.alloc() : memref<768xf32>
      memref.copy %reinterpret_cast_3, %alloc_7 : memref<768xf32, strided<[1]>> to memref<768xf32>
      %16 = bufferization.to_tensor %alloc_7 restrict writable : memref<768xf32>
      %17 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
      %reduced = linalg.reduce ins(%14 : tensor<16x768xf32>) outs(%17 : tensor<16xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %29 = arith.addf %in, %init : f32
          linalg.yield %29 : f32
        }
      %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %cst_1 : tensor<16xf32>, f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
      %19 = tensor.empty() : tensor<16x768xf32>
      %broadcasted = linalg.broadcast ins(%18 : tensor<16xf32>) outs(%19 : tensor<16x768xf32>) dimensions = [1] 
      %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%14, %broadcasted : tensor<16x768xf32>, tensor<16x768xf32>) outs(%19 : tensor<16x768xf32>) -> tensor<16x768xf32>
      %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%20, %20 : tensor<16x768xf32>, tensor<16x768xf32>) outs(%19 : tensor<16x768xf32>) -> tensor<16x768xf32>
      %reduced_8 = linalg.reduce ins(%21 : tensor<16x768xf32>) outs(%17 : tensor<16xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %29 = arith.addf %in, %init : f32
          linalg.yield %29 : f32
        }
      %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced_8, %cst_1 : tensor<16xf32>, f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
      %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%22, %cst_2 : tensor<16xf32>, f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
      %24 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%23 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
      %25 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%24 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
      %broadcasted_9 = linalg.broadcast ins(%25 : tensor<16xf32>) outs(%19 : tensor<16x768xf32>) dimensions = [1] 
      %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%20, %broadcasted_9 : tensor<16x768xf32>, tensor<16x768xf32>) outs(%19 : tensor<16x768xf32>) -> tensor<16x768xf32>
      %broadcasted_10 = linalg.broadcast ins(%15 : tensor<768xf32>) outs(%19 : tensor<16x768xf32>) dimensions = [0] 
      %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %broadcasted_10 : tensor<16x768xf32>, tensor<16x768xf32>) outs(%19 : tensor<16x768xf32>) -> tensor<16x768xf32>
      %broadcasted_11 = linalg.broadcast ins(%16 : tensor<768xf32>) outs(%19 : tensor<16x768xf32>) dimensions = [0] 
      %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%27, %broadcasted_11 : tensor<16x768xf32>, tensor<16x768xf32>) outs(%19 : tensor<16x768xf32>) -> tensor<16x768xf32>
      %reinterpret_cast_12 = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %18[0] [%12] [1] : tensor<16xf32> to tensor<?xf32>
      %subview_13 = memref.subview %reinterpret_cast_12[0] [%12] [1] : memref<16xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_13 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
      %reinterpret_cast_14 = memref.reinterpret_cast %arg6 to offset: [%7], sizes: [16, 768], strides: [768, 1] : memref<?xf32> to memref<16x768xf32, strided<[768, 1], offset: ?>>
      %extracted_slice_15 = tensor.extract_slice %28[0, 0] [%12, 768] [1, 1] : tensor<16x768xf32> to tensor<?x768xf32>
      %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%12, 768] [1, 1] : memref<16x768xf32, strided<[768, 1], offset: ?>> to memref<?x768xf32, strided<[768, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 : (tensor<?x768xf32>, memref<?x768xf32, strided<[768, 1], offset: ?>>) -> ()
      %reinterpret_cast_17 = memref.reinterpret_cast %arg7 to offset: [%6], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %extracted_slice_18 = tensor.extract_slice %25[0] [%12] [1] : tensor<16xf32> to tensor<?xf32>
      %subview_19 = memref.subview %reinterpret_cast_17[0] [%12] [1] : memref<16xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_18 in writable %subview_19 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}