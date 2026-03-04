// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG

// CHECK: @add_mul_fusion

func.func @nested_caller(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %ret = func.call @caller(%arg0, %arg1, %arg2) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
  return %ret : tensor<?xf32>
}

func.func @caller(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %ret = func.call @add_mul_fusion(%arg0, %arg1, %arg2) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
  return %ret : tensor<?xf32>
}

func.func @add_mul_fusion(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32>
 attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>

  // CHECK: scf.for
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  
  %1 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %arg2 : tensor<?xf32>, tensor<?xf32>) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

// CHECK-DEBUG: transform.structured.match ops{["func.func"]}
// CHECK-DEBUG: transform.func.get_func_argument
// CHECK-DEBUG: transform.structured.tile_using_for
// CHECK-DEBUG: transform.loop.coalesce
// CHECK-DEBUG: transform.structured.extended_fuse_into_containing_op
// CHECK-DEBUG: transform.structured.set_buffer_size
