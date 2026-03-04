// RUN: bishengir-opt --hivm-insert-infer-sync-block-lock-num-and-init-func -split-input-file %s | FileCheck %s -check-prefixes=CHECK
// -----

// CHECK: func.func @insert_infer_sync_block_lock_num_and_size_func_infer_sync_block_lock_num_function() -> i64
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 3 : i64
// CHECK: return %[[BYTE_SIZE]]
// CHECK: func.func @insert_infer_sync_block_lock_num_and_size_func_infer_sync_block_lock_init_function() -> i64
// CHECK: %[[INIT:.*]] = arith.constant 0 : i64
// CHECK: return %[[INIT]]
func.func @insert_infer_sync_block_lock_num_and_size_func(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}){
  %cst_0 = arith.constant 0 : index
  %cst_8 = arith.constant 8 : index
  %cst_16 = arith.constant 16 : index
  hivm.hir.create_sync_block_lock from %arg1 : from memref<?xi8> to memref<1xi64>
  hivm.hir.create_sync_block_lock from %arg1 : from memref<?xi8> to memref<1xi64>
  hivm.hir.create_sync_block_lock from %arg1 : from memref<?xi8> to memref<1xi64>
  return
}