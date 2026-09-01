// RUN: bishengir-opt %s -hfusion-auto-vectorize="tree-reduce" | FileCheck %s

func.func @triton_sum_3D_dim0_dim2(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %c0_i32 = arith.constant 0 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [5, 8, 7], strides: [56, 7, 1] : memref<?xi32> to memref<5x8x7xi32, strided<[56, 7, 1]>>
  %alloc = memref.alloc() : memref<5x8x7xi32>
  memref.copy %reinterpret_cast, %alloc : memref<5x8x7xi32, strided<[56, 7, 1]>> to memref<5x8x7xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<5x8x7xi32>
  %1 = tensor.empty() : tensor<8xi32>
  annotation.mark %1 keys = ["pad_const"] values = [%c0_i32 : i32] : tensor<8xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1 : tensor<8xi32>) {
  ^bb0(%out: i32):
    linalg.yield %c0_i32 : i32
  } -> tensor<8xi32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction"]} ins(%0 : tensor<5x8x7xi32>) outs(%2 : tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    %4 = arith.addi %in, %out : i32
    linalg.yield %4 : i32
  } -> tensor<8xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xi32> to memref<8xi32, strided<[1]>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<8xi32>, memref<8xi32, strided<[1]>>) -> ()
  return
}

// CHECK-LABEL:   func.func @triton_sum_3D_dim0_dim2_outlined_vf_0(
// CHECK-LABEL:   func.func @triton_sum_3D_dim0_dim2_outlined_vf_1(
// CHECK-LABEL:   func.func @triton_sum_3D_dim0_dim2_outlined_vf_2(
// CHECK-LABEL:   func.func @triton_sum_3D_dim0_dim2(
