// RUN: bishengir-opt -split-input-file -transform-interpreter -verify-diagnostics %s | FileCheck %s

module attributes { transform.with_named_sequence } {
// CHECK-LABEL: @cache_write
// CHECK: %[[RES:.*]] = linalg.elemwise_unary
// CHECK: %[[CACHE_INIT:.*]] = tensor.empty({{.*}}) : tensor<?xf32>
// CHECK: %[[CACHE_RESULT:.*]] = hfusion.store ins(%[[RES]] : tensor<?xf32>) 
// CHECK-SAME:                               outs(%[[CACHE_INIT]] : tensor<?xf32>)
// CHECK: return %[[CACHE_RESULT]] : tensor<?xf32>
func.func @cache_write(%arg0 : tensor<?xf32>, %dim : index) -> tensor<?xf32> {
  %empty = tensor.empty(%dim) : tensor<?xf32>
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?xf32>) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.get_operand %0 [0] : (!transform.any_op) -> !transform.any_value
  %2 = transform.structured.cache_write %1 : (!transform.any_value) -> !transform.any_op
  transform.yield 
}
}

// -----

module attributes { transform.with_named_sequence } {
// CHECK-LABEL: @cache_write_intermediate_output
// CHECK: %[[RES:.*]] = linalg.elemwise_unary
// CHECK: %[[CACHE_INIT:.*]] = tensor.empty({{.*}}) : tensor<?xf32>
// CHECK: %[[CACHE_RESULT:.*]] = hfusion.store ins(%[[RES]] : tensor<?xf32>) 
// CHECK-SAME:                               outs(%[[CACHE_INIT]] : tensor<?xf32>)
// CHECK: linalg.elemwise_unary ins(%[[CACHE_RESULT]]
func.func @cache_write_intermediate_output(%arg0 : tensor<?xf32>, %dim : index) -> (tensor<?xf32>, tensor<?xf32>) {
  %empty = tensor.empty(%dim) : tensor<?xf32>
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?xf32>) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %1 = linalg.elemwise_unary ins(%0 : tensor<?xf32>) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  func.return %0, %1 : tensor<?xf32>, tensor<?xf32>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.get_operand %0 [0] : (!transform.any_op) -> !transform.any_value
  %2 = transform.structured.cache_write %1 : (!transform.any_value) -> !transform.any_op
  %3 = transform.get_operand %0 [1] : (!transform.any_op) -> !transform.any_value
  %4 = transform.structured.cache_write %3 : (!transform.any_value) -> !transform.any_op
  transform.yield 
}
}

// -----

module attributes { transform.with_named_sequence } {
// CHECK-LABEL: @cache_write_output_only
// CHECK: %[[RES:.*]] = linalg.elemwise_unary
// CHECK: %[[CACHE_INIT:.*]] = tensor.empty({{.*}}) : tensor<?xf32>
// CHECK: %[[CACHE_RESULT:.*]] = hfusion.store ins(%[[RES]] : tensor<?xf32>) 
// CHECK-SAME:                               outs(%[[CACHE_INIT]] : tensor<?xf32>)
// CHECK: linalg.elemwise_unary ins(%[[RES]]
func.func @cache_write_output_only(%arg0 : tensor<?xf32>, %dim : index) -> (tensor<?xf32>, tensor<?xf32>) {
  %empty = tensor.empty(%dim) : tensor<?xf32>
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?xf32>) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %1 = linalg.elemwise_unary ins(%0 : tensor<?xf32>) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  func.return %0, %1 : tensor<?xf32>, tensor<?xf32>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.get_operand %0 [0] : (!transform.any_op) -> !transform.any_value
  %2 = transform.structured.cache_write %1 {output_only = true} : (!transform.any_value) -> !transform.any_op
  %3 = transform.get_operand %0 [1] : (!transform.any_op) -> !transform.any_value
  %4 = transform.structured.cache_write %3 {output_only = true} : (!transform.any_value) -> !transform.any_op
  transform.yield 
}
}

// -----

module attributes { transform.with_named_sequence } {
// CHECK: @cache_write_to_args
// CHECK-SAME:  ({{.*}}: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>)
func.func @cache_write_to_args(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> (tensor<?xf32>) {
  // CHECK: %[[EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?xf32>
  // CHECK: %[[RES:.*]] = linalg.elemwise_unary ins({{.*}}) outs(%[[EMPTY]]
  // CHECK: %[[CACHE_RESULT:.*]] = hfusion.store ins(%[[RES]] : tensor<?xf32>) outs(%[[ARG1]] : tensor<?xf32>)
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.get_operand %0 [0] : (!transform.any_op) -> !transform.any_value
  %2 = transform.structured.cache_write %1
    {cache_write_to_output_init = true} : (!transform.any_value) -> !transform.any_op
  transform.yield
}
}

// -----

module attributes { transform.with_named_sequence } {
// CHECK: @cache_write_to_args_output_only
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: tensor<?xf32>)
func.func @cache_write_to_args_output_only(%arg0 : tensor<?xf32>,
                                           %arg1 : tensor<?xf32>,
                                           %arg2 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  // CHECK: %[[CONST0_0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM0_0:.*]] = tensor.dim %[[ARG0]], %[[CONST0_0]] : tensor<?xf32>
  // CHECK: %[[EMPTY0:.*]] = tensor.empty(%[[DIM0_0]]
  // CHECK: %[[RES0:.*]] = linalg.elemwise_unary ins({{.*}}) outs(%[[EMPTY0]]
  // CHECK: hfusion.store ins(%[[RES0]] : tensor<?xf32>) outs(%[[ARG1]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[CONST0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM0_1:.*]] = tensor.dim %[[RES0]], %[[CONST0_1]] : tensor<?xf32>
  // CHECK: %[[EMPTY1:.*]] = tensor.empty(%[[DIM0_1]]
  // CHECK: %[[RES1:.*]] = linalg.elemwise_unary ins(%[[RES0]] : tensor<?xf32>) outs(%[[EMPTY1]]
  // CHECK: hfusion.store ins(%[[RES1]] : tensor<?xf32>) outs(%[[ARG2]]
  %1 = linalg.elemwise_unary ins(%0 : tensor<?xf32>) outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  func.return %0, %1 : tensor<?xf32>, tensor<?xf32>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.get_operand %0 [0] : (!transform.any_op) -> !transform.any_value
  %2 = transform.structured.cache_write %1
    {output_only = true, cache_write_to_output_init = true} : (!transform.any_value) -> !transform.any_op
  %3 = transform.get_operand %0 [1] : (!transform.any_op) -> !transform.any_value
  %4 = transform.structured.cache_write %3
    {output_only = true, cache_write_to_output_init = true} : (!transform.any_value) -> !transform.any_op
  transform.yield
}
}

// -----

module attributes { transform.with_named_sequence } {
// CHECK: @cache_write_to_args_output_only
// CHECK-SAME: %[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16xf32>
func.func @cache_write_to_args_output_only(%arg0 : tensor<16xf32>,
                                           %arg1 : tensor<16xf32>) -> (tensor<16x1xf32>) {
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<16xf32>) outs(%arg1 : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: %[[CACHED:.*]] = hfusion.store ins({{.*}}) outs(%[[ARG1]] : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: tensor.expand_shape %[[CACHED]]
  %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
  func.return %expanded : tensor<16x1xf32>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.get_result %0[0] : (!transform.any_op) -> !transform.any_value
  %2 = transform.structured.cache_write %1
    {output_only = true, cache_write_to_output_init = true} : (!transform.any_value) -> !transform.any_op
  transform.yield
 }
}

// -----

module attributes { transform.with_named_sequence } {
// CHECK: @cache_write_to_args_output_only
// CHECK-SAME: %[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16xf32>, %[[ARG2:.*]]: tensor<16xf32>
func.func @cache_write_to_args_output_only(%arg0 : tensor<16xf32>,
                                           %arg1 : tensor<16xf32>,
                                           %arg2 : tensor<16xf32>) -> (tensor<16xf32>, tensor<16x1xf32>) {
  // CHECK: %[[EMPTY0:.*]] = tensor.empty
  // CHECK: %[[RES0:.*]] = linalg.elemwise_unary ins({{.*}}) outs(%[[EMPTY0]]
  // CHECK: %[[CACHED0:.*]] = hfusion.store ins(%[[RES0]] : tensor<16xf32>) outs(%[[ARG1]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<16xf32>) outs(%arg1 : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: %[[EMPTY1:.*]] = tensor.empty
  // CHECK: %[[RES1:.*]] = linalg.elemwise_unary ins(%[[RES0]] : tensor<16xf32>) outs(%[[EMPTY1]]
  // CHECK: %[[CACHED1:.*]] = hfusion.store ins(%[[RES1]] : tensor<16xf32>) outs(%[[ARG2]]
  %1 = linalg.elemwise_unary ins(%0 : tensor<16xf32>) outs(%arg2 : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: tensor.expand_shape %[[CACHED0]]
  %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
  func.return %1, %expanded : tensor<16xf32>, tensor<16x1xf32>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.get_result %0[0] : (!transform.any_op) -> !transform.any_value
  %2 = transform.structured.cache_write %1
    {output_only = true, cache_write_to_output_init = true} : (!transform.any_value) -> !transform.any_op
  transform.yield
 }
}
