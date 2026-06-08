// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @inline_fixpipe_for_iter_arg_mmad
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true}
// CHECK: scf.yield
func.func @inline_fixpipe_for_iter_arg_mmad(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
    %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
    %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
    %arg5: i32,
    %arg6: i32,
    %arg7: i32) {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  %alloc = memref.alloc() : memref<16x16xf32>
  hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc : memref<16x16xf32>) eviction_policy = <EvictFirst>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  %alloc_1 = memref.alloc() : memref<16x16xf32>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xf32>) eviction_policy = <EvictFirst>
  %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf32>
  %4 = scf.for %arg8 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %2) -> (tensor<16x16xf32>)  : i32 {
    %5 = tensor.empty() : tensor<16x16xf32>
    %6 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%arg9, %3, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %6 : tensor<16x16xf32>
  }
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  hivm.hir.store ins(%4 : tensor<16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16xf32, strided<[16, 1]>>)
  hivm.hir.set_ctrl true at ctrl[60]
  return
}
