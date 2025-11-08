// RUN: bishengir-opt --hivm-insert-infer-workspace-size-func -split-input-file %s | FileCheck %s -check-prefixes=CHECK
// -----

// CHECK: func.func @insert_infer_workspace_size_func_infer_workspace_shape_function() -> index
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 3400 : index
// CHECK: return %[[BYTE_SIZE]]
func.func @insert_infer_workspace_size_func(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}){
  %cst_0 = arith.constant 0 : index
  %cst_100 = arith.constant 100 : index
  %cst_200 = arith.constant 200 : index
  memref_ext.alloc_workspace() from %arg1 offset = [%cst_0] : from memref<?xi8> to memref<100xi8>
  memref_ext.alloc_workspace() from %arg1 offset = [%cst_100] : from memref<?xi8> to memref<800xi8>
  memref_ext.alloc_workspace() from %arg1 offset = [%cst_200] : from memref<?xi8> to memref<800xi32>
  return
}