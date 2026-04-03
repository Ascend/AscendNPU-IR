// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s
// CHECK-LABEL: func.func private @triton_unk_fused_silu_0_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: tensor.extract_slice
// CHECK: return

// CHECK-LABEL: func.func @triton_unk_fused_silu_0(
// CHECK: arith.constant
// CHECK: tensor.empty
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.minsi
// CHECK: scf.for
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: func.call @triton_unk_fused_silu_0_fused_0
// CHECK: bufferization.materialize_in_destination
// CHECK: return

module{
   func.func @triton_unk_fused_silu_0(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant -1.000000e+00 : f32
    %c8192 = arith.constant 8192 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1048576_i32 = arith.constant 1048576 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8192xf32>
    %1 = arith.muli %arg8, %c1048576_i32 : i32
    %2 = arith.addi %1, %c1048576_i32 : i32
    %3 = arith.minsi %2, %arg4 : i32
    scf.for %arg11 = %c0_i32 to %c128_i32 step %c1_i32  : i32 {
      %4 = arith.muli %arg11, %c8192_i32 : i32
      %5 = arith.addi %1, %4 : i32
      %6 = arith.index_cast %5 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [8192], strides: [1] : memref<?xf32> to memref<8192xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<8192xf32>
      %7 = arith.addi %6, %c8192 : index
      %8 = arith.index_cast %3 : i32 to index
      %9 = arith.maxsi %6, %8 : index
      %10 = arith.minsi %7, %9 : index
      %11 = arith.subi %10, %6 : index
      %12 = arith.cmpi slt, %11, %c8192 : index
      scf.if %12 {
        linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<8192xf32>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0] [%11] [1] : memref<8192xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_2 = memref.subview %alloc[0] [%11] [1] : memref<8192xf32> to memref<?xf32>
      memref.copy %subview, %subview_2 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32>
      %13 = bufferization.to_tensor %alloc restrict writable : memref<8192xf32>
      %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %cst : tensor<8192xf32>, f32) outs(%0 : tensor<8192xf32>) -> tensor<8192xf32>
      %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%14 : tensor<8192xf32>) outs(%0 : tensor<8192xf32>) -> tensor<8192xf32>
      %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %cst_0 : tensor<8192xf32>, f32) outs(%0 : tensor<8192xf32>) -> tensor<8192xf32>
      %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%13, %16 : tensor<8192xf32>, tensor<8192xf32>) outs(%0 : tensor<8192xf32>) -> tensor<8192xf32>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%6], sizes: [8192], strides: [1] : memref<?xf32> to memref<8192xf32, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %17[0] [%11] [1] : tensor<8192xf32> to tensor<?xf32>
      %subview_4 = memref.subview %reinterpret_cast_3[0] [%11] [1] : memref<8192xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_4 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}