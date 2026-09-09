// RUN: bishengir-opt %s --split-input-file \
// RUN:   --hfusion-pre-vectorization-fusion --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Unary producer feeding a binary consumer. Canonicalization should forward
// the produced vector instead of materializing an intermediate tensor read.

// CHECK-HFUSION-LABEL: func.func @hfusion_exp_add(
// CHECK-HFUSION: %[[INPUT:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[EXP:.*]] = math.exp %[[INPUT]] : vector<8x8xf32>
// CHECK-HFUSION-NOT: vector.transfer_read
// CHECK-HFUSION: %[[ADD:.*]] = arith.addf %[[EXP]], %[[INPUT]] : vector<8x8xf32>
// CHECK-HFUSION: vector.transfer_write %[[ADD]]
func.func @hfusion_exp_add(%input: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %tmp = tensor.empty() : tensor<8x8xf32>
  %output = tensor.empty() : tensor<8x8xf32>
  %exp = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
      ins(%input : tensor<8x8xf32>) outs(%tmp : tensor<8x8xf32>)
      -> tensor<8x8xf32>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%exp, %input : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%output : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_exp_add(
// CHECK-HIVM: %[[INPUT:.*]] = vector.transfer_read
// CHECK-HIVM: %[[EXP:.*]] = math.exp %[[INPUT]] : vector<8x8xf32>
// CHECK-HIVM-NOT: vector.transfer_read
// CHECK-HIVM: %[[ADD:.*]] = arith.addf %[[EXP]], %[[INPUT]] : vector<8x8xf32>
// CHECK-HIVM: vector.transfer_write %[[ADD]]
func.func @hivm_exp_add(%input: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %tmp = tensor.empty() : tensor<8x8xf32>
  %output = tensor.empty() : tensor<8x8xf32>
  %exp = hivm.hir.vexp ins(%input : tensor<8x8xf32>)
      outs(%tmp : tensor<8x8xf32>) -> tensor<8x8xf32>
  %result = hivm.hir.vadd
      ins(%exp, %input : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%output : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// -----

// Compare producer feeding select. The i1 vector must remain the select
// condition and both data operands must stay ordered.

// CHECK-HFUSION-LABEL: func.func @hfusion_cmp_select(
// CHECK-HFUSION: %[[LHS:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[RHS:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[COND:.*]] = arith.cmpf olt, %[[LHS]], %[[RHS]] : vector<8x8xf32>
// CHECK-HFUSION-NOT: vector.transfer_read
// CHECK-HFUSION: %[[SELECT:.*]] = arith.select %[[COND]], %[[LHS]], %[[RHS]]
// CHECK-HFUSION: vector.transfer_write %[[SELECT]]
func.func @hfusion_cmp_select(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %cond_empty = tensor.empty() : tensor<8x8xi1>
  %output = tensor.empty() : tensor<8x8xf32>
  %cond = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%cond_empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  %result = hfusion.select
      ins(%cond, %lhs, %rhs : tensor<8x8xi1>, tensor<8x8xf32>,
          tensor<8x8xf32>)
      outs(%output : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_cmp_select(
// CHECK-HIVM: %[[LHS:.*]] = vector.transfer_read
// CHECK-HIVM: %[[RHS:.*]] = vector.transfer_read
// CHECK-HIVM: %[[COND:.*]] = arith.cmpf olt, %[[LHS]], %[[RHS]] : vector<8x8xf32>
// CHECK-HIVM-NOT: vector.transfer_read
// CHECK-HIVM: %[[SELECT:.*]] = arith.select %[[COND]], %[[LHS]], %[[RHS]]
// CHECK-HIVM: vector.transfer_write %[[SELECT]]
func.func @hivm_cmp_select(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> attributes {hivm.vector_function} {
  %cond_empty = tensor.empty() : tensor<8x8xi1>
  %output = tensor.empty() : tensor<8x8xf32>
  %cond = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%cond_empty : tensor<8x8xi1>) compare_mode = <lt>
      -> tensor<8x8xi1>
  %result = hivm.hir.vsel
      ins(%cond, %lhs, %rhs : tensor<8x8xi1>, tensor<8x8xf32>,
          tensor<8x8xf32>)
      outs(%output : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}
