// RUN: bishengir-opt %s --hfusion-normalize-ops -split-input-file | FileCheck %s --enable-var-scope

// Small types deliberately retain the pre-compensation floating algorithm.

// -----

// CHECK-LABEL: func.func @legacy_i8_trunc(
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: return
func.func @legacy_i8_trunc(%a: tensor<16xi8>, %b: tensor<16xi8>) -> tensor<16xi8> {
  %out = tensor.empty() : tensor<16xi8>
  %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi8>, tensor<16xi8>) outs(%out : tensor<16xi8>) -> tensor<16xi8>
  return %q : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @legacy_i8_floor(
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: return
func.func @legacy_i8_floor(%a: tensor<16xi8>, %b: tensor<16xi8>) -> tensor<16xi8> {
  %out = tensor.empty() : tensor<16xi8>
  %q = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi8>, tensor<16xi8>) outs(%out : tensor<16xi8>) -> tensor<16xi8>
  return %q : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @legacy_i8_ceil(
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: return
func.func @legacy_i8_ceil(%a: tensor<16xi8>, %b: tensor<16xi8>) -> tensor<16xi8> {
  %out = tensor.empty() : tensor<16xi8>
  %q = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi8>, tensor<16xi8>) outs(%out : tensor<16xi8>) -> tensor<16xi8>
  return %q : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @legacy_i16_trunc(
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: return
func.func @legacy_i16_trunc(%a: tensor<16xi16>, %b: tensor<16xi16>) -> tensor<16xi16> {
  %out = tensor.empty() : tensor<16xi16>
  %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi16>, tensor<16xi16>) outs(%out : tensor<16xi16>) -> tensor<16xi16>
  return %q : tensor<16xi16>
}

// -----

// CHECK-LABEL: func.func @legacy_i16_floor(
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: return
func.func @legacy_i16_floor(%a: tensor<16xi16>, %b: tensor<16xi16>) -> tensor<16xi16> {
  %out = tensor.empty() : tensor<16xi16>
  %q = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi16>, tensor<16xi16>) outs(%out : tensor<16xi16>) -> tensor<16xi16>
  return %q : tensor<16xi16>
}

// -----

// CHECK-LABEL: func.func @legacy_i16_ceil(
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: return
func.func @legacy_i16_ceil(%a: tensor<16xi16>, %b: tensor<16xi16>) -> tensor<16xi16> {
  %out = tensor.empty() : tensor<16xi16>
  %q = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi16>, tensor<16xi16>) outs(%out : tensor<16xi16>) -> tensor<16xi16>
  return %q : tensor<16xi16>
}

// -----

// CHECK-LABEL: func.func @legacy_u8_trunc(
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: return
func.func @legacy_u8_trunc(%a: tensor<16xi8>, %b: tensor<16xi8>) -> tensor<16xi8> {
  %out = tensor.empty() : tensor<16xi8>
  %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%a, %b : tensor<16xi8>, tensor<16xi8>) outs(%out : tensor<16xi8>) -> tensor<16xi8>
  return %q : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @legacy_u8_ceil(
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: #linalg.binary_fn<mul>
// CHECK-NOT: #linalg.binary_fn<sub>
// CHECK-NOT: #linalg.binary_fn<div>
// CHECK: return
func.func @legacy_u8_ceil(%a: tensor<16xi8>, %b: tensor<16xi8>) -> tensor<16xi8> {
  %out = tensor.empty() : tensor<16xi8>
  %q = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivui>} ins(%a, %b : tensor<16xi8>, tensor<16xi8>) outs(%out : tensor<16xi8>) -> tensor<16xi8>
  return %q : tensor<16xi8>
}
