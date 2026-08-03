// RUN: bishengir-opt --hivm-insert-infer-sync-block-lock-num-and-init-func -split-input-file %s | FileCheck %s

// CHECK: func.func @unordered_single_lock_infer_sync_block_lock_num_function() -> i64
// CHECK: %[[NUM:.*]] = arith.constant 16392 : i64
// CHECK: return %[[NUM]]
// CHECK: func.func @unordered_single_lock_infer_sync_block_lock_init_function() -> i64
// CHECK: %[[INIT:.*]] = arith.constant 0 : i64
// CHECK: return %[[INIT]]
// CHECK-LABEL: func.func @unordered_single_lock(
func.func @unordered_single_lock(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}) {
  hivm.hir.create_sync_block_lock from %arg1 {hivm.sync_block_lock_unordered} : from memref<?xi8> to memref<1xi64>
  return
}

// -----

// CHECK: func.func @unordered_two_locks_infer_sync_block_lock_num_function() -> i64
// CHECK: %[[NUM:.*]] = arith.constant 32784 : i64
// CHECK: return %[[NUM]]
// CHECK: func.func @unordered_two_locks_infer_sync_block_lock_init_function() -> i64
// CHECK: %[[INIT:.*]] = arith.constant 0 : i64
// CHECK: return %[[INIT]]
// CHECK-LABEL: func.func @unordered_two_locks(
func.func @unordered_two_locks(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}) {
  hivm.hir.create_sync_block_lock from %arg1 {hivm.sync_block_lock_unordered} : from memref<?xi8> to memref<1xi64>
  hivm.hir.create_sync_block_lock from %arg1 {hivm.sync_block_lock_unordered} : from memref<?xi8> to memref<1xi64>
  return
}

