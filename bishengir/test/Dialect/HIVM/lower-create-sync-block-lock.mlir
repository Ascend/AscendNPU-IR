// RUN: bishengir-opt --hivm-lower-create-sync-block-lock %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK: module
module {
  func.func @test_alloc_sync_block_lock_infer_sync_block_lock_num_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_sync_block_lock_num_function>} {
    %c1_i64 = arith.constant 1 : i64
    return %c1_i64 : i64
  }
  func.func @test_alloc_sync_block_lock_infer_sync_block_lock_init_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_sync_block_lock_init_function>} {
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
  // CHECK-LABEL: func.func @test_alloc_sync_block_lock
  func.func @test_alloc_sync_block_lock(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> tensor<1xi64>{
    // %[[view:.*]] = memref.view %[[arg0:.*]][%[[OFFSET:.*]]][] : memref<?xi8> to memref<1xi64>
    %lock_var = hivm.hir.create_sync_block_lock from %arg0 : from memref<?xi8> to memref<1xi64>
    %res = bufferization.to_tensor %lock_var restrict writable : memref<1xi64>
    return %res : tensor<1xi64>
  }
}

// -----
// CHECK-LABEL: func.func @test_unordered_lock_mix_subblock_stride
// CHECK: %[[BLOCK_NUM:.*]] = hivm.hir.get_block_num
// CHECK: %[[SUBBLOCK_NUM:.*]] = hivm.hir.get_sub_block_num
// CHECK: %[[PARTICIPANT_NUM:.*]] = arith.muli %[[BLOCK_NUM]], %[[SUBBLOCK_NUM]]
// CHECK: memref.view
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @test_unordered_lock_mix_subblock_stride(
      %arg0: memref<?xi8>) -> (tensor<1xi64>, tensor<1xi64>) {
    %lock0 = hivm.hir.create_sync_block_lock from %arg0
        {hivm.sync_block_lock_unordered} :
        from memref<?xi8> to memref<1xi64>
    %lock1 = hivm.hir.create_sync_block_lock from %arg0
        {hivm.sync_block_lock_unordered} :
        from memref<?xi8> to memref<1xi64>
    %res0 = bufferization.to_tensor %lock0 restrict writable : memref<1xi64>
    %res1 = bufferization.to_tensor %lock1 restrict writable : memref<1xi64>
    return %res0, %res1 : tensor<1xi64>, tensor<1xi64>
  }
}
