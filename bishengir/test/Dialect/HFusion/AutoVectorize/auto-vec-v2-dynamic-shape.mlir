// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 --outline-vector-function -split-input-file | FileCheck %s
// CHECK-LABEL: func.func
// CHECK: scf.for {{%.*}} = {{%.*}} to {{%.*}} step {{%.*}} iter_args({{%.*}} = [[ARG1:%.+]], {{%.*}} = [[ARG2:%.+]]) -> (tensor<?x8xi8>, tensor<?x8xi8>) {
// CHECK-NOT: [[ARG1]] = [[ARG2]]
// CHECK:   scf.yield {{%.*}}, {{%.*}} : tensor<?x8xi8>, tensor<?x8xi8>
// CHECK: }

func.func @triton_dot_sub_2D_acc_None_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [7, 8], strides: [8, 1] : memref<?xi8> to memref<7x8xi8, strided<[8, 1]>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [7, 8], strides: [8, 1] : memref<?xi8> to memref<7x8xi8, strided<[8, 1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [7, 8], strides: [8, 1] : memref<?xi8> to memref<7x8xi8, strided<[8, 1]>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [7, 8], strides: [8, 1] : memref<?xi8> to memref<7x8xi8, strided<[8, 1]>>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [7, 8], strides: [8, 1] : memref<?xi8> to memref<7x8xi8, strided<[8, 1]>>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [7, 8], strides: [8, 1] : memref<?xi8> to memref<7x8xi8, strided<[8, 1]>>
  scf.for %arg13 = %c0 to %c2 step %c1 {
    %2 = affine.apply affine_map<()[s0] -> (s0 * (-s0 + 4))>()[%arg13]
    %3 = affine.apply affine_map<()[s0] -> (-s0 + 4)>()[%arg13]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    annotation.mark %1 {logical_block_num} : i32
    %4 = tensor.empty(%3) : tensor<?x8xi8>
    annotation.mark %4 {buffer_size_in_byte = 32 : i64} : tensor<?x8xi8>
    annotation.mark %4 {buffer_size_in_byte = 32 : i64} : tensor<?x8xi8>
    %alloc = memref.alloc(%3) : memref<?x8xi8>
    annotation.mark %alloc {buffer_size_in_byte = 32 : i64} : memref<?x8xi8>
    %subview = memref.subview %reinterpret_cast[%2, 0] [%3, 8] [1, 1] : memref<7x8xi8, strided<[8, 1]>> to memref<?x8xi8, strided<[8, 1], offset: ?>>
    hivm.hir.load ins(%subview : memref<?x8xi8, strided<[8, 1], offset: ?>>) outs(%alloc : memref<?x8xi8>) eviction_policy = <EvictFirst>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<?x8xi8>
    %alloc_5 = memref.alloc(%3) : memref<?x8xi8>
    annotation.mark %alloc_5 {buffer_size_in_byte = 32 : i64} : memref<?x8xi8>
    %subview_6 = memref.subview %reinterpret_cast_0[%2, 0] [%3, 8] [1, 1] : memref<7x8xi8, strided<[8, 1]>> to memref<?x8xi8, strided<[8, 1], offset: ?>>
    hivm.hir.load ins(%subview_6 : memref<?x8xi8, strided<[8, 1], offset: ?>>) outs(%alloc_5 : memref<?x8xi8>) eviction_policy = <EvictFirst>
    %6 = bufferization.to_tensor %alloc_5 restrict writable : memref<?x8xi8>
    annotation.mark %4 {buffer_size_in_byte = 32 : i64} : tensor<?x8xi8>
    annotation.mark %4 {buffer_size_in_byte = 32 : i64} : tensor<?x8xi8>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : tensor<?x8xi8>, tensor<?x8xi8>) outs(%4 : tensor<?x8xi8>) {
    ^bb0(%in: i8, %in_13: i8, %out: i8):
      %12 = arith.subi %in, %in_13 : i8
      linalg.yield %12 : i8
    } -> tensor<?x8xi8>
    %subview_7 = memref.subview %reinterpret_cast_1[%2, 0] [%3, 8] [1, 1] {to_be_bubbled_slice} : memref<7x8xi8, strided<[8, 1]>> to memref<?x8xi8, strided<[8, 1], offset: ?>>
    hivm.hir.store ins(%7 : tensor<?x8xi8>) outs(%subview_7 : memref<?x8xi8, strided<[8, 1], offset: ?>>) {tiled_op}
    %alloc_8 = memref.alloc(%3) : memref<?x8xi8>
    annotation.mark %alloc_8 {buffer_size_in_byte = 32 : i64} : memref<?x8xi8>
    %subview_9 = memref.subview %reinterpret_cast_2[%2, 0] [%3, 8] [1, 1] : memref<7x8xi8, strided<[8, 1]>> to memref<?x8xi8, strided<[8, 1], offset: ?>>
    hivm.hir.load ins(%subview_9 : memref<?x8xi8, strided<[8, 1], offset: ?>>) outs(%alloc_8 : memref<?x8xi8>) eviction_policy = <EvictFirst>
    %8 = bufferization.to_tensor %alloc_8 restrict writable : memref<?x8xi8>
    %alloc_10 = memref.alloc(%3) : memref<?x8xi8>
    annotation.mark %alloc_10 {buffer_size_in_byte = 32 : i64} : memref<?x8xi8>
    %subview_11 = memref.subview %reinterpret_cast_3[%2, 0] [%3, 8] [1, 1] : memref<7x8xi8, strided<[8, 1]>> to memref<?x8xi8, strided<[8, 1], offset: ?>>
    hivm.hir.load ins(%subview_11 : memref<?x8xi8, strided<[8, 1], offset: ?>>) outs(%alloc_10 : memref<?x8xi8>) eviction_policy = <EvictFirst>
    %9 = bufferization.to_tensor %alloc_10 restrict writable : memref<?x8xi8>
    %10 = tensor.empty(%3) : tensor<?x8xi1>
    annotation.mark %10 {buffer_size_in_byte = 32 : i64} : tensor<?x8xi1>
    annotation.mark %10 {buffer_size_in_byte = 32 : i64} : tensor<?x8xi1>
    annotation.mark %4 {buffer_size_in_byte = 32 : i64} : tensor<?x8xi8>
    annotation.mark %4 {buffer_size_in_byte = 32 : i64} : tensor<?x8xi8>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %5, %9 : tensor<?x8xi8>, tensor<?x8xi8>, tensor<?x8xi8>) outs(%4 : tensor<?x8xi8>) {
    ^bb0(%in: i8, %in_13: i8, %in_14: i8, %out: i8):
      %12 = arith.cmpi ne, %in, %c0_i8 : i8
      %13 = arith.select %12, %in_13, %in_14 : i8
      linalg.yield %13 : i8
    } -> tensor<?x8xi8>
    %subview_12 = memref.subview %reinterpret_cast_4[%2, 0] [%3, 8] [1, 1] {to_be_bubbled_slice} : memref<7x8xi8, strided<[8, 1]>> to memref<?x8xi8, strided<[8, 1], offset: ?>>
    hivm.hir.store ins(%11 : tensor<?x8xi8>) outs(%subview_12 : memref<?x8xi8, strided<[8, 1], offset: ?>>) {tiled_op}
    hivm.hir.set_ctrl true at ctrl[60]
  } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  return
}


