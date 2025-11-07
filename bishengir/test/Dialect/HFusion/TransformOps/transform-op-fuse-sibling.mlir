// RUN: bishengir-opt -transform-interpreter -canonicalize-ext -cse --split-input-file  --verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_parallel_loops_from_tiled_reductions
// CHECK: scf.for
// CHECK: some_use
// CHECK-NOT: scf.for
// CHECK: some_use
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__loop_0__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes {__loop_1__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.loop.fuse_sibling %0 into %1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

func.func @fuse_parallel_loops_from_tiled_reductions(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>, %arg2: tensor<1x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %arg0, %arg5 = %arg1) -> (tensor<1x16xf32>, tensor<1x16xf32>) {
    %4 = linalg.fill ins(%cst : f32) outs(%arg4 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%arg5 : tensor<1x16xf32>) -> tensor<1x16xf32>
      scf.yield %4, %5 : tensor<1x16xf32>, tensor<1x16xf32>
  } {__loop_0__}
  %1:2 = "some_use"(%0#0, %0#1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> (tensor<1xf32>, tensor<1xf32>)
  %2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x16xf32>) {
    %4 = linalg.fill ins(%cst : f32) outs(%arg4 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%4 : tensor<1x16xf32>) outs(%arg4 : tensor<1x16xf32>) -> tensor<1x16xf32>
    scf.yield %5 : tensor<1x16xf32>
  } {__loop_1__}
  %3 = "some_use"(%2) : (tensor<1x16xf32>) -> tensor<1xf32>
  return %1#0, %1#1, %3 : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
}