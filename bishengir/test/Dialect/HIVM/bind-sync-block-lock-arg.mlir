// RUN: bishengir-opt --hivm-bind-sync-block-lock-arg -split-input-file -verify-diagnostics %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @bind_sync_block_lock_arg
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}
func.func @bind_sync_block_lock_arg(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}){
  // CHECK: hivm.hir.create_sync_block_lock from %[[ARG]]
  hivm.hir.create_sync_block_lock : memref<1xi64>
  return
}