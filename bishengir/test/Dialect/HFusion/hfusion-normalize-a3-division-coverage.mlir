// RUN: bishengir-opt %s --hfusion-normalize-ops -split-input-file -verify-diagnostics | FileCheck %s --enable-var-scope

// CHECK-LABEL: func.func @zero_constant_trunc(
// CHECK-DAG: arith.constant 2147483647 : i32
// CHECK-DAG: arith.constant -2147483648 : i32
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: hfusion.select
// CHECK: return
func.func @zero_constant_trunc(%a: tensor<16xi32>) -> tensor<16xi32> {
  %b = arith.constant 0 : i32
  %out = tensor.empty() : tensor<16xi32>
    %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @zero_constant_floor(
// CHECK-DAG: arith.constant 2147483647 : i32
// CHECK-DAG: arith.constant -2147483648 : i32
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: hfusion.select
// CHECK: return
func.func @zero_constant_floor(%a: tensor<16xi32>) -> tensor<16xi32> {
  %b = arith.constant 0 : i32
  %out = tensor.empty() : tensor<16xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @zero_constant_ceil(
// CHECK-DAG: arith.constant 2147483647 : i32
// CHECK-DAG: arith.constant -2147483648 : i32
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: hfusion.select
// CHECK: return
func.func @zero_constant_ceil(%a: tensor<16xi32>) -> tensor<16xi32> {
  %b = arith.constant 0 : i32
  %out = tensor.empty() : tensor<16xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @dynamic_scalar_divisor_floor(
// CHECK: arith.ori
// CHECK: arith.select
// CHECK: arith.sitofp
// CHECK-COUNT-2: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: round_mode = #hfusion.round_mode<rint>
// CHECK: return
func.func @dynamic_scalar_divisor_floor(%a: tensor<16xi32>, %b: i32) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @dynamic_scalar_divisor_ceil(
// CHECK: arith.ori
// CHECK: arith.select
// CHECK: arith.sitofp
// CHECK-COUNT-2: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: round_mode = #hfusion.round_mode<rint>
// CHECK: return
func.func @dynamic_scalar_divisor_ceil(%a: tensor<16xi32>, %b: i32) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// An identity division must materialize the declared static result shape.
// CHECK-LABEL: func.func @dynamic_identity_result(
// CHECK-NOT: f32
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: return {{.*}} : tensor<16xi32>
func.func @dynamic_identity_result(%a: tensor<?xi32>) -> tensor<16xi32> {
  %b = arith.constant 1 : i32
  %out = tensor.empty() : tensor<16xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<?xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}
