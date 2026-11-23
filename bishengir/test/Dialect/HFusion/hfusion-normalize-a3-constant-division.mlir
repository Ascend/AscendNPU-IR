// RUN: bishengir-opt %s --hfusion-normalize-ops -split-input-file -verify-diagnostics | FileCheck %s --enable-var-scope --implicit-check-not=f32 --implicit-check-not=i64 --implicit-check-not=shrsi
// RUN: bishengir-opt %s --hfusion-normalize-ops --convert-hfusion-to-hivm -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=HIVM --implicit-check-not=hivm.hir.vdiv

// Constant signed i32 divisors bypass both floating-point divisions.
// Magnitudes use logical shifts, including the INT_MIN bit pattern.

// CHECK-LABEL: func.func @trunc_one(
// CHECK-NEXT: return %arg0
func.func @trunc_one(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant 1 : i32
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @floor_minus_one(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: return {{.*}} : tensor<16xi32>
func.func @floor_minus_one(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant dense<-1> : tensor<16xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, tensor<16xi32>) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @ceil_one_scalar_numerator(
// CHECK: linalg.fill ins(%arg0
// CHECK: return {{.*}} : tensor<?xi32>
func.func @ceil_one_scalar_numerator(%a: i32, %out: tensor<?xi32>) -> tensor<?xi32> {
  %c = arith.constant 1 : i32
  %b = linalg.fill ins(%c : i32) outs(%out : tensor<?xi32>) -> tensor<?xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : i32, tensor<?xi32>) outs(%out : tensor<?xi32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}

// -----

// CHECK-LABEL: func.func @trunc_min(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[MIN:.*]] = arith.constant -2147483648 : i32
// CHECK: %[[IS_MIN:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %[[MIN]]
// CHECK: %[[RESULT:.*]] = hfusion.select ins(%[[IS_MIN]], %[[ONE]], %[[ZERO]]
// CHECK: return %[[RESULT]] : tensor<16xi32>
func.func @trunc_min(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant -2147483648 : i32
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @floor_min(
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: hfusion.select
// CHECK: return {{.*}} : tensor<16xi32>
func.func @floor_min(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant dense<-2147483648> : tensor<16xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, tensor<16xi32>) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @ceil_min(
// CHECK: hfusion.select
// CHECK: return {{.*}} : tensor<16xi32>
func.func @ceil_min(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant -2147483648 : i32
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// HIVM-LABEL: func.func @trunc_two(
// HIVM: hivm.hir.vshr {{.*}}i32
// CHECK-LABEL: func.func @trunc_two(
// CHECK: %[[MAG:.*]] = hfusion.select
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MAG]],
// CHECK: %[[NEG_PRODUCT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]],
// CHECK: %[[NEGATED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[NEG_PRODUCT]]
// CHECK: %[[Q:.*]] = hfusion.select {{.*}}, %[[NEGATED]], %[[SHIFT]]
// CHECK-NOT: #hfusion.binary_fn<vand>
// CHECK: return {{.*}} : tensor<16xi32>
func.func @trunc_two(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant 2 : i32
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// HIVM-LABEL: func.func @trunc_negative_power(
// HIVM: hivm.hir.vshr {{.*}}i32
// CHECK-LABEL: func.func @trunc_negative_power(
// CHECK: %[[MAG:.*]] = hfusion.select
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MAG]],
// CHECK: %[[NEG_PRODUCT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]],
// CHECK: %[[NEGATED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[NEG_PRODUCT]]
// CHECK: %[[Q:.*]] = hfusion.select {{.*}}, %[[NEGATED]], %[[SHIFT]]
// CHECK-NOT: #hfusion.binary_fn<vand>
// CHECK: return {{.*}} : tensor<16xi32>
func.func @trunc_negative_power(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant dense<-512> : tensor<16xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, tensor<16xi32>) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// HIVM-LABEL: func.func @floor_positive_power(
// HIVM: hivm.hir.vshr {{.*}}i32
// CHECK-LABEL: func.func @floor_positive_power(
// CHECK: %[[MAG:.*]] = hfusion.select
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MAG]],
// CHECK: %[[NEG_PRODUCT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]],
// CHECK: %[[NEGATED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[NEG_PRODUCT]]
// CHECK: %[[Q:.*]] = hfusion.select {{.*}}, %[[NEGATED]], %[[SHIFT]]
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG]],
// CHECK: %[[DELTA:.*]] = hfusion.select
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[Q]], %[[DELTA]]
// CHECK: return {{.*}} : tensor<16xi32>
func.func @floor_positive_power(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant 512 : i32
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// HIVM-LABEL: func.func @floor_negative_power(
// HIVM: hivm.hir.vshr {{.*}}i32
// CHECK-LABEL: func.func @floor_negative_power(
// CHECK: %[[MAG:.*]] = hfusion.select
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MAG]],
// CHECK: %[[NEG_PRODUCT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]],
// CHECK: %[[NEGATED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[NEG_PRODUCT]]
// CHECK: %[[Q:.*]] = hfusion.select {{.*}}, %[[NEGATED]], %[[SHIFT]]
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG]],
// CHECK: %[[DELTA:.*]] = hfusion.select
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[Q]], %[[DELTA]]
// CHECK: return {{.*}} : tensor<16xi32>
func.func @floor_negative_power(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant -8 : i32
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// HIVM-LABEL: func.func @ceil_positive_power(
// HIVM: hivm.hir.vshr {{.*}}i32
// CHECK-LABEL: func.func @ceil_positive_power(
// CHECK: %[[MAG:.*]] = hfusion.select
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MAG]],
// CHECK: %[[NEG_PRODUCT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]],
// CHECK: %[[NEGATED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[NEG_PRODUCT]]
// CHECK: %[[Q:.*]] = hfusion.select {{.*}}, %[[NEGATED]], %[[SHIFT]]
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG]],
// CHECK: %[[DELTA:.*]] = hfusion.select
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[Q]], %[[DELTA]]
// CHECK: return {{.*}} : tensor<16xi32>
func.func @ceil_positive_power(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant 8 : i32
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// HIVM-LABEL: func.func @ceil_negative_power(
// HIVM: hivm.hir.vshr {{.*}}i32
// CHECK-LABEL: func.func @ceil_negative_power(
// CHECK: %[[MAG:.*]] = hfusion.select
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MAG]],
// CHECK: %[[NEG_PRODUCT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]],
// CHECK: %[[NEGATED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[NEG_PRODUCT]]
// CHECK: %[[Q:.*]] = hfusion.select {{.*}}, %[[NEGATED]], %[[SHIFT]]
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG]],
// CHECK: %[[DELTA:.*]] = hfusion.select
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[Q]], %[[DELTA]]
// CHECK: return {{.*}} : tensor<16xi32>
func.func @ceil_negative_power(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %b = arith.constant -1073741824 : i32
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// HIVM-LABEL: func.func @dynamic_fill_power(
// HIVM: hivm.hir.vshr {{.*}}i32
// CHECK-LABEL: func.func @dynamic_fill_power(
// CHECK: %[[MAG:.*]] = hfusion.select
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MAG]],
// CHECK: %[[NEG_PRODUCT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]],
// CHECK: %[[NEGATED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[NEG_PRODUCT]]
// CHECK: %[[Q:.*]] = hfusion.select {{.*}}, %[[NEGATED]], %[[SHIFT]]
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG]],
// CHECK: %[[DELTA:.*]] = hfusion.select
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[Q]], %[[DELTA]]
// CHECK: return {{.*}} : tensor<?xi32>
func.func @dynamic_fill_power(%a: tensor<?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> {
  %c = arith.constant 4 : i32
  %b = linalg.fill ins(%c : i32) outs(%out : tensor<?xi32>) -> tensor<?xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<?xi32>, tensor<?xi32>) outs(%out : tensor<?xi32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}

// -----

// HIVM-LABEL: func.func @broadcast_power(
// HIVM: hivm.hir.vshr {{.*}}i32
// CHECK-LABEL: func.func @broadcast_power(
// CHECK: %[[MAG:.*]] = hfusion.select
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MAG]],
// CHECK: %[[NEG_PRODUCT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]],
// CHECK: %[[NEGATED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[NEG_PRODUCT]]
// CHECK: %[[Q:.*]] = hfusion.select {{.*}}, %[[NEGATED]], %[[SHIFT]]
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG]],
// CHECK: %[[DELTA:.*]] = hfusion.select
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[Q]], %[[DELTA]]
// CHECK: return {{.*}} : tensor<16xi32>
func.func @broadcast_power(%a: tensor<16xi32>) -> tensor<16xi32> {
  %out = tensor.empty() : tensor<16xi32>
  %c = arith.constant dense<-4> : tensor<i32>
  %b = linalg.broadcast ins(%c : tensor<i32>) outs(%out : tensor<16xi32>) dimensions = [0]
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, tensor<16xi32>) outs(%out : tensor<16xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}

// -----

// MIN/-1 must use wrapping integer arithmetic in every rounding mode.
// CHECK-LABEL: func.func @min_minus_one_trunc(
// CHECK-NOT: overflow<
// CHECK: return {{.*}} : tensor<4xi32>
func.func @min_minus_one_trunc() -> tensor<4xi32> {
  %a = arith.constant dense<[-2147483648, -2147483647, 0, 2147483647]> : tensor<4xi32>
  %b = arith.constant -1 : i32
  %out = tensor.empty() : tensor<4xi32>
  %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<4xi32>, i32) outs(%out : tensor<4xi32>) -> tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// MIN/-1 must use wrapping integer arithmetic in every rounding mode.
// CHECK-LABEL: func.func @min_minus_one_floor(
// CHECK-NOT: overflow<
// CHECK: return {{.*}} : tensor<4xi32>
func.func @min_minus_one_floor() -> tensor<4xi32> {
  %a = arith.constant dense<[-2147483648, -2147483647, 0, 2147483647]> : tensor<4xi32>
  %b = arith.constant -1 : i32
  %out = tensor.empty() : tensor<4xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<4xi32>, i32) outs(%out : tensor<4xi32>) -> tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// MIN/-1 must use wrapping integer arithmetic in every rounding mode.
// CHECK-LABEL: func.func @min_minus_one_ceil(
// CHECK-NOT: overflow<
// CHECK: return {{.*}} : tensor<4xi32>
func.func @min_minus_one_ceil() -> tensor<4xi32> {
  %a = arith.constant dense<[-2147483648, -2147483647, 0, 2147483647]> : tensor<4xi32>
  %b = arith.constant -1 : i32
  %out = tensor.empty() : tensor<4xi32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<4xi32>, i32) outs(%out : tensor<4xi32>) -> tensor<4xi32>
  return %r : tensor<4xi32>
}
