// RUN: bishengir-opt -split-input-file -transform-interpreter -verify-diagnostics %s

module attributes { transform.with_named_sequence } {
func.func @get_func_result(%arg0: index, %arg1: index) -> index {
  // expected-remark @+2 {{func result}}
  // expected-note @+1 {{value handle points to an op result #0}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.func.get_func_result %0[all] : (!transform.any_op) -> !transform.any_value
  transform.debug.emit_remark_at %1, "func result" : !transform.any_value
  transform.yield
 }
}

// -----

module attributes { transform.with_named_sequence } {
func.func @get_func_result(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @+1 {{target handle does not point to `func.func` op}}
  %1 = transform.func.get_func_result %0[all] : (!transform.any_op) -> !transform.any_value
  transform.yield
 }
}

// -----

module attributes { transform.with_named_sequence } {
func.func @get_func_result(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.func.get_func_result %0[except(0)] : (!transform.any_op) -> !transform.any_value
  transform.debug.emit_remark_at %1, "func arg" : !transform.any_value
  transform.yield
 }
}

// -----

module attributes { transform.with_named_sequence } {
// expected-note @+1 {{while considering positions of this payload operation}}
func.func @get_func_result(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @+1 {{position overflow 2 (updated from 2) for maximum 1}}
  %1 = transform.func.get_func_result %0[2] : (!transform.any_op) -> !transform.any_value
  transform.yield
 }
}

// -----

module attributes { transform.with_named_sequence } {
// expected-remark @+3 {{func arg}}
// expected-note @+2 {{value handle points to a block argument #0 in block #0 in region #0}}
// expected-note @+1 {{value handle points to a block argument #1 in block #0 in region #0}}
func.func @get_func_result(%tensor1: tensor<16x1xf16>, %tensor2: tensor<16xf16>) -> (tensor<16xf16>, tensor<16x1xf16>) {
  %collapsed = tensor.collapse_shape %tensor1 [[0, 1]] : tensor<16x1xf16> into tensor<16xf16>
  %expanded = tensor.expand_shape %tensor2 [[0, 1]] output_shape [16, 1] : tensor<16xf16> into tensor<16x1xf16>
  return %collapsed, %expanded : tensor<16xf16>, tensor<16x1xf16>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.func.get_func_result %0[all] {find_reshape_producer} : (!transform.any_op) -> !transform.any_value
  transform.debug.emit_remark_at %1, "func arg" : !transform.any_value
  transform.yield
 }
}

