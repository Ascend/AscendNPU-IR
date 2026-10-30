// RUN: bishengir-opt %s --hfusion-normalize-ops -split-input-file -verify-diagnostics | FileCheck %s --enable-var-scope

// The first estimate sanitizes only the divisor. Both residual products stay
// in i32; native nearest-even must precede the second residual.
// CHECK-LABEL: func.func @signed_div(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i32
// CHECK: %[[SPECIAL:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
// CHECK-NOT: hfusion.select
// CHECK: %[[ISZERO:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg1, %[[ZERO]]
// CHECK: %[[GUARD:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[SPECIAL]], %[[ISZERO]]
// CHECK: %[[B:.*]] = hfusion.select ins(%[[GUARD]], %[[TWO]], %arg1
// CHECK: %[[AF:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<16xi32>) {{.*}} -> tensor<16xf32>
// CHECK: %[[BF:.*]] = hfusion.cast {{.*}} ins(%[[B]] : tensor<16xi32>) {{.*}} -> tensor<16xf32>
// CHECK: %[[EST:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[AF]], %[[BF]]
// CHECK: %[[M0:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<trunc>{{.*}} ins(%[[EST]]
// CHECK: %[[P0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[M0]], %[[B]] : tensor<16xi32>, tensor<16xi32>)
// CHECK: %[[R0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %[[P0]]
// CHECK: %[[RF:.*]] = hfusion.cast {{.*}} ins(%[[R0]] : tensor<16xi32>) {{.*}} -> tensor<16xf32>
// CHECK: %[[Z:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[RF]], %[[BF]]
// CHECK-NOT: linalg.elemwise_unary
// CHECK-NOT: hfusion.compare
// CHECK: %[[C:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<rint>{{.*}} ins(%[[Z]] : tensor<16xf32>) {{.*}} -> tensor<16xi32>
// CHECK: %[[M1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[M0]], %[[C]]
// CHECK: %[[P1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[M1]], %[[B]]
// CHECK: %[[R1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %[[P1]]
// CHECK: %[[R1F:.*]] = hfusion.cast {{.*}} ins(%[[R1]] : tensor<16xi32>) {{.*}} -> tensor<16xf32>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[R1F]],
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[R1]], %[[ZERO]]
// CHECK: %[[Q:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[M1]],
// CHECK: %[[SPECIAL_Q:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1
// CHECK: %[[RESULT:.*]] = hfusion.select ins(%[[SPECIAL]], %[[SPECIAL_Q]], %[[Q]] :
// CHECK: %[[FINAL:.*]] = hfusion.select ins(%[[ISZERO]], {{.*}}, %[[RESULT]] :
// CHECK-NEXT: return %[[FINAL]] : tensor<16xi32>
func.func @signed_div(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @signed_floor(
// CHECK-COUNT-2: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[C:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<rint>{{.*}} -> tensor<16xi32>
// CHECK: %[[Q:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[C]]
// CHECK: %[[P:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: %[[R:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins({{.*}}, %[[P]]
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[R]],
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: %[[MASK:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[DELTA:.*]] = hfusion.select ins(%[[MASK]],
// CHECK: %[[ROUNDED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[Q]], %[[DELTA]]
// CHECK: %[[SPECIAL_Q:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1
// CHECK: %[[RESULT:.*]] = hfusion.select {{.*}}, %[[SPECIAL_Q]], %[[ROUNDED]] :
// CHECK: %[[FINAL:.*]] = hfusion.select {{.*}}, %[[RESULT]] :
// CHECK-NEXT: return %[[FINAL]]
func.func @signed_floor(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @signed_ceil(
// CHECK-COUNT-2: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[C:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<rint>{{.*}} -> tensor<16xi32>
// CHECK: %[[Q:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[C]]
// CHECK: %[[P:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: %[[R:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins({{.*}}, %[[P]]
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[R]],
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[MASK:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[DELTA:.*]] = hfusion.select ins(%[[MASK]],
// CHECK: %[[ROUNDED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[Q]], %[[DELTA]]
// CHECK: %[[SPECIAL_Q:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1
// CHECK: %[[RESULT:.*]] = hfusion.select {{.*}}, %[[SPECIAL_Q]], %[[ROUNDED]] :
// CHECK: %[[FINAL:.*]] = hfusion.select {{.*}}, %[[RESULT]] :
// CHECK-NEXT: return %[[FINAL]]
func.func @signed_ceil(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @dynamic_scalar_divisor(
// CHECK: arith.cmpi eq, %arg1,
// CHECK: arith.cmpi eq, %arg1,
// CHECK: %[[SPECIAL:.*]] = arith.ori
// CHECK: %[[GUARD:.*]] = arith.ori %[[SPECIAL]],
// CHECK: %[[B:.*]] = arith.select %[[GUARD]],
// CHECK: arith.sitofp %[[B]] : i32 to f32
// CHECK-COUNT-2: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: hfusion.select ins(%[[SPECIAL]],
// CHECK: return {{.*}} : tensor<?xi32>
func.func @dynamic_scalar_divisor(%arg0: tensor<?xi32>, %arg1: i32) -> tensor<?xi32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %arg0, %c0 : tensor<?xi32>
  %out = tensor.empty(%n) : tensor<?xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<?xi32>, i32) outs(%out : tensor<?xi32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}

// -----

// Out-of-scope types must not enter the new compensation builder.
// CHECK-LABEL: func.func @unsigned_u32_unchanged(
// CHECK: hfusion.cast {{.*}}cast_unsigned
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: linalg.elemwise_binary
// CHECK: return
func.func @unsigned_u32_unchanged(%a: tensor<16xi32>, %b: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%a, %b : tensor<16xi32>, tensor<16xi32>) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @signed_i64_unchanged(
// CHECK-NOT: hfusion.cast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins({{.*}} : tensor<16xi64>, tensor<16xi64>)
// CHECK: return
func.func @signed_i64_unchanged(%a: tensor<16xi64>, %b: tensor<16xi64>) -> tensor<16xi64> {
  %out = tensor.empty() : tensor<16xi64>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi64>, tensor<16xi64>) outs(%out : tensor<16xi64>) -> tensor<16xi64>
  return %r : tensor<16xi64>
}

// -----

// CHECK-LABEL: func.func @dynamic_scalar_numerator_floor(
// CHECK: hfusion.select
// CHECK-COUNT-2: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: return {{.*}} : tensor<?xi32>
func.func @dynamic_scalar_numerator_floor(%a: i32, %b: tensor<?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> {
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : i32, tensor<?xi32>) outs(%out : tensor<?xi32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}

// -----

// CHECK-LABEL: func.func @non_power_constant(
// CHECK-COUNT-2: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: return
func.func @non_power_constant(%a: tensor<4xi32>) -> tensor<4xi32> {
  %b = arith.constant 3 : i32
  %out = tensor.empty() : tensor<4xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<4xi32>, i32) outs(%out : tensor<4xi32>) -> tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// CHECK-LABEL: func.func @nonuniform_constant(
// CHECK-COUNT-2: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: return
func.func @nonuniform_constant(%a: tensor<4xi32>) -> tensor<4xi32> {
  %b = arith.constant dense<[2, 4, -8, 16]> : tensor<4xi32>
  %out = tensor.empty() : tensor<4xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<4xi32>, tensor<4xi32>) outs(%out : tensor<4xi32>) -> tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// Feed b = [-1, 1, 4, -4, 4, -4, 1, -1] at runtime.
// CHECK-LABEL: func.func @i32_trunc_boundary_lanes(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: return {{.*}} : tensor<8xi32>
func.func @i32_trunc_boundary_lanes(%b: tensor<8xi32>) -> tensor<8xi32> {
  %a = arith.constant dense<[-2147483648, -2147483648, 7, 7, -7, -7, 1, 1]> : tensor<8xi32>
  %out = tensor.empty() : tensor<8xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<8xi32>, tensor<8xi32>) outs(%out : tensor<8xi32>) -> tensor<8xi32>
  return %r : tensor<8xi32>
}

// -----

// Feed b = [-1, 1, 4, -4, 4, -4, 1, -1] at runtime.
// CHECK-LABEL: func.func @i32_floor_boundary_lanes(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: return {{.*}} : tensor<8xi32>
func.func @i32_floor_boundary_lanes(%b: tensor<8xi32>) -> tensor<8xi32> {
  %a = arith.constant dense<[-2147483648, -2147483648, 7, 7, -7, -7, 1, 1]> : tensor<8xi32>
  %out = tensor.empty() : tensor<8xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<8xi32>, tensor<8xi32>) outs(%out : tensor<8xi32>) -> tensor<8xi32>
  return %r : tensor<8xi32>
}

// -----

// Feed b = [-1, 1, 4, -4, 4, -4, 1, -1] at runtime.
// CHECK-LABEL: func.func @i32_ceil_boundary_lanes(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: return {{.*}} : tensor<8xi32>
func.func @i32_ceil_boundary_lanes(%b: tensor<8xi32>) -> tensor<8xi32> {
  %a = arith.constant dense<[-2147483648, -2147483648, 7, 7, -7, -7, 1, 1]> : tensor<8xi32>
  %out = tensor.empty() : tensor<8xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<8xi32>, tensor<8xi32>) outs(%out : tensor<8xi32>) -> tensor<8xi32>
  return %r : tensor<8xi32>
}
