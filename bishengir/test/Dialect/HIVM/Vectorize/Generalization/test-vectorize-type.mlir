// RUN: bishengir-opt %s --split-input-file \
// RUN:   --hfusion-pre-vectorization-fusion --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Binary arithmetic.

// CHECK-HFUSION-LABEL: func.func @hfusion_add(
// CHECK-HFUSION: %[[LHS:.*]] = vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK-HFUSION: %[[RHS:.*]] = vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK-HFUSION: %[[RESULT:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<8x8xf32>
// CHECK-HFUSION: vector.transfer_write %[[RESULT]]
func.func @hfusion_add(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_add(
// CHECK-HIVM: %[[LHS:.*]] = vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK-HIVM: %[[RHS:.*]] = vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK-HIVM: %[[RESULT:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<8x8xf32>
// CHECK-HIVM: vector.transfer_write %[[RESULT]]
func.func @hivm_add(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vadd ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// -----

// Floating-point maximum.

// CHECK-HFUSION-LABEL: func.func @hfusion_max(
// CHECK-HFUSION: arith.maximumf {{.*}} : vector<8x8xf32>
func.func @hfusion_max(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_max(
// CHECK-HIVM: arith.maximumf {{.*}} : vector<8x8xf32>
func.func @hivm_max(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vmax ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// -----

// Unary math.

// CHECK-HFUSION-LABEL: func.func @hfusion_exp(
// CHECK-HFUSION: math.exp {{.*}} : vector<8x8xf32>
func.func @hfusion_exp(%input: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
      ins(%input : tensor<8x8xf32>) outs(%empty : tensor<8x8xf32>)
      -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_exp(
// CHECK-HIVM: math.exp {{.*}} : vector<8x8xf32>
func.func @hivm_exp(%input: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vexp ins(%input : tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// -----

// High-precision division.

// CHECK-HFUSION-LABEL: func.func @hfusion_divhp(
// CHECK-HFUSION: mathExt.divfhp {{.*}} : vector<8x8xf32>
func.func @hfusion_divhp(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_divhp(
// CHECK-HIVM: mathExt.divfhp {{.*}} : vector<8x8xf32>
func.func @hivm_divhp(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vdiv ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) isHP = true -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// -----

// Bitwise arithmetic.

// CHECK-HFUSION-LABEL: func.func @hfusion_xor(
// CHECK-HFUSION: arith.xori {{.*}} : vector<8x8xi32>
func.func @hfusion_xor(%lhs: tensor<8x8xi32>, %rhs: tensor<8x8xi32>)
    -> tensor<8x8xi32> {
  %empty = tensor.empty() : tensor<8x8xi32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>}
      ins(%lhs, %rhs : tensor<8x8xi32>, tensor<8x8xi32>)
      outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %result : tensor<8x8xi32>
}

// CHECK-HIVM-LABEL: func.func @hivm_xor(
// CHECK-HIVM: arith.xori {{.*}} : vector<8x8xi32>
func.func @hivm_xor(%lhs: tensor<8x8xi32>, %rhs: tensor<8x8xi32>)
    -> tensor<8x8xi32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xi32>
  %result = hivm.hir.vxor ins(%lhs, %rhs : tensor<8x8xi32>, tensor<8x8xi32>)
      outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %result : tensor<8x8xi32>
}

// -----

// Comparison.

// CHECK-HFUSION-LABEL: func.func @hfusion_cmp(
// CHECK-HFUSION: arith.cmpf olt, {{.*}} : vector<8x8xf32>
func.func @hfusion_cmp(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xi1> {
  %empty = tensor.empty() : tensor<8x8xi1>
  %result = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  return %result : tensor<8x8xi1>
}

// CHECK-HIVM-LABEL: func.func @hivm_cmp(
// CHECK-HIVM: arith.cmpf olt, {{.*}} : vector<8x8xf32>
func.func @hivm_cmp(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xi1> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xi1>
  %result = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <lt> -> tensor<8x8xi1>
  return %result : tensor<8x8xi1>
}

// -----

// Selection.

// CHECK-HFUSION-LABEL: func.func @hfusion_select(
// CHECK-HFUSION: arith.select {{.*}} : vector<8x8xi1>, vector<8x8xf32>
func.func @hfusion_select(%cond: tensor<8x8xi1>, %lhs: tensor<8x8xf32>,
    %rhs: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hfusion.select
      ins(%cond, %lhs, %rhs : tensor<8x8xi1>, tensor<8x8xf32>,
          tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_select(
// CHECK-HIVM: arith.select {{.*}} : vector<8x8xi1>, vector<8x8xf32>
func.func @hivm_select(%cond: tensor<8x8xi1>, %lhs: tensor<8x8xf32>,
    %rhs: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vsel
      ins(%cond, %lhs, %rhs : tensor<8x8xi1>, tensor<8x8xf32>,
          tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// -----

// Type conversion.

// CHECK-HFUSION-LABEL: func.func @hfusion_cast(
// CHECK-HFUSION: arith.truncf {{.*}} : vector<8x8xf32> to vector<8x8xf16>
func.func @hfusion_cast(%input: tensor<8x8xf32>) -> tensor<8x8xf16> {
  %empty = tensor.empty() : tensor<8x8xf16>
  %result = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
      ins(%input : tensor<8x8xf32>) outs(%empty : tensor<8x8xf16>)
      -> tensor<8x8xf16>
  return %result : tensor<8x8xf16>
}

// CHECK-HIVM-LABEL: func.func @hivm_cast(
// CHECK-HIVM: arith.truncf {{.*}} : vector<8x8xf32> to vector<8x8xf16>
func.func @hivm_cast(%input: tensor<8x8xf32>) -> tensor<8x8xf16>
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf16>
  %result = hivm.hir.vcast ins(%input : tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf16>) round_mode = <rint>
      cast = <cast_signed> -> tensor<8x8xf16>
  return %result : tensor<8x8xf16>
}

// -----

// Diff: reduce shape mismatch
// Linalg drops the reduced axis; HIVM retains it as a unit dimension.
// Reduction.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce(
// CHECK-HFUSION: vector.multi_reduction <add>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<8x8xf32> to vector<8xf32>
func.func @hfusion_reduce(%input: tensor<8x8xf32>, %init: tensor<8xf32>)
    -> tensor<8xf32> {
  %result = linalg.reduce ins(%input : tensor<8x8xf32>)
      outs(%init : tensor<8xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  return %result : tensor<8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_reduce(
// CHECK-HIVM: vector.multi_reduction <add>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<8x8xf32> to vector<8xf32>
func.func @hivm_reduce(%input: tensor<8x8xf32>, %init: tensor<8x1xf32>)
    -> tensor<8x1xf32> attributes {hivm.vector_function} {
  %result = hivm.hir.vreduce <sum> ins(%input : tensor<8x8xf32>)
      outs(%init : tensor<8x1xf32>) unsigned_src = false reduce_dims = [1]
      -> tensor<8x1xf32>
  return %result : tensor<8x1xf32>
}

// -----

// Transpose.

// CHECK-HFUSION-LABEL: func.func @hfusion_transpose(
// CHECK-HFUSION: vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK-HFUSION: vector.transfer_write {{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @hfusion_transpose(%input: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = linalg.transpose ins(%input : tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) permutation = [1, 0]
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_transpose(
// CHECK-HIVM: vector.transfer_read {{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK-HIVM: vector.transfer_write {{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @hivm_transpose(%input: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vtranspose ins(%input : tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) permutation = [1, 0]
      -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}
