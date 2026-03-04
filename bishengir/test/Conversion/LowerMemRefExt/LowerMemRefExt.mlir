// RUN: bishengir-opt -lower-memref-ext %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK: #map = affine_map<(d0)[s0] -> (d0 * 32768 + s0)>
// CHECK: module
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @test_alloc_workspace_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c32768 = arith.constant 32768 : index
    return %c32768: index
  }
  // CHECK-LABEL: func.func @test_alloc_workspace
  func.func @test_alloc_workspace(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> tensor<128x64xf32>{
    %c0 = arith.constant 0 : index
    // CHECK: %[[BLOCK_ID:.*]] = hivm.hir.get_block_idx -> i64
    // CHECK: %[[BLOCK_ID_INDEX:.*]] = arith.index_cast %[[BLOCK_ID]] : i64 to index
    // CHECK: %[[OFFSET:.*]] = affine.apply
    // %[[view:.*]] = memref.view %[[arg0:.*]][%[[OFFSET]]][] : memref<?xi8> to memref<128x64xf32>
    %alloc_workspace = memref_ext.alloc_workspace() from %arg0 offset = [%c0] : from memref<?xi8> to memref<128x64xf32>
    %res = bufferization.to_tensor %alloc_workspace restrict writable : memref<128x64xf32>
    return %res : tensor<128x64xf32>
  }
}
