// RUN: bishengir-opt %s -canonicalize-ext -split-input-file | FileCheck %s

// CHECK-LABEL: func @fold_consecutive_scalar_mul_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant 6.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xf32>, f32)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xf32>)
// CHECK: return %[[RESULT]]
func.func @fold_consecutive_scalar_mul_f32(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst2 = arith.constant dense<2.0> : tensor<4x8xf32>
  %cst3 = arith.constant dense<3.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%arg0, %cst2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%mul1, %cst3 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul2 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @fold_consecutive_scalar_mul_i32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xi32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant 15 : i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xi32>
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xi32>, i32)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xi32>)
// CHECK: return %[[RESULT]]
func.func @fold_consecutive_scalar_mul_i32(%arg0: tensor<4x8xi32>) -> tensor<4x8xi32> {
  %cst5 = arith.constant dense<5> : tensor<4x8xi32>
  %cst3 = arith.constant dense<3> : tensor<4x8xi32>
  %empty = tensor.empty() : tensor<4x8xi32>
  %mul1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%arg0, %cst5 : tensor<4x8xi32>, tensor<4x8xi32>)
    outs(%empty : tensor<4x8xi32>) -> tensor<4x8xi32>
  %mul2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%mul1, %cst3 : tensor<4x8xi32>, tensor<4x8xi32>)
    outs(%empty : tensor<4x8xi32>) -> tensor<4x8xi32>
  return %mul2 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: func @fold_scalar_mul_lhs
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant 1.200000e+01 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xf32>, f32)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xf32>)
// CHECK: return %[[RESULT]]
func.func @fold_scalar_mul_lhs(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst4 = arith.constant dense<4.0> : tensor<4x8xf32>
  %cst3 = arith.constant dense<3.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%cst4, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%cst3, %mul1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul2 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @no_fold_multi_use
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
func.func @no_fold_multi_use(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst2 = arith.constant dense<2.0> : tensor<4x8xf32>
  %cst3 = arith.constant dense<3.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%arg0, %cst2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%mul1, %cst3 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    ins(%mul2, %mul1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %add : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @no_fold_non_const
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
func.func @no_fold_non_const(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%mul1, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul2 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @fold_sub_as_neg_mul_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xf32>, f32)
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @fold_sub_as_neg_mul_f32(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %zero = arith.constant dense<0.0> : tensor<4x8xf32>
  %cst2 = arith.constant dense<2.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%neg, %cst2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @fold_sub_neg_mul_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant -4.000000e+00 : f32
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xf32>, f32)
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @fold_sub_neg_mul_f32(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %zero = arith.constant dense<0.0> : tensor<4x8xf32>
  %cst4 = arith.constant dense<4.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%arg0, %cst4 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %mul : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %neg : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @fold_sub_as_neg_mul_i32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xi32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant -3 : i32
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xi32>, i32)
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @fold_sub_as_neg_mul_i32(%arg0: tensor<4x8xi32>) -> tensor<4x8xi32> {
  %zero = arith.constant dense<0> : tensor<4x8xi32>
  %cst3 = arith.constant dense<3> : tensor<4x8xi32>
  %empty = tensor.empty() : tensor<4x8xi32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %arg0 : tensor<4x8xi32>, tensor<4x8xi32>)
    outs(%empty : tensor<4x8xi32>) -> tensor<4x8xi32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%neg, %cst3 : tensor<4x8xi32>, tensor<4x8xi32>)
    outs(%empty : tensor<4x8xi32>) -> tensor<4x8xi32>
  return %mul : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: func @fold_sub_as_neg_mul_const_lhs
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant -3.000000e+00 : f32
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xf32>, f32)
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @fold_sub_as_neg_mul_const_lhs(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %zero = arith.constant dense<0.0> : tensor<4x8xf32>
  %cst3 = arith.constant dense<3.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%cst3, %neg : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @no_fold_sub_nonzero_lhs
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
func.func @no_fold_sub_nonzero_lhs(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %one = arith.constant dense<1.0> : tensor<4x8xf32>
  %cst2 = arith.constant dense<2.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %sub = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%one, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    ins(%sub, %cst2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @fold_double_negation_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[ONE]] : tensor<4x8xf32>, f32)
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @fold_double_negation_f32(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %zero = arith.constant dense<0.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %neg1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %neg2 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %neg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %neg2 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @fold_double_negation_i32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xi32>)
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[RESULT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-SAME: ins(%[[ARG]], %[[ONE]] : tensor<4x8xi32>, i32)
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
func.func @fold_double_negation_i32(%arg0: tensor<4x8xi32>) -> tensor<4x8xi32> {
  %zero = arith.constant dense<0> : tensor<4x8xi32>
  %empty = tensor.empty() : tensor<4x8xi32>
  %neg1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %arg0 : tensor<4x8xi32>, tensor<4x8xi32>)
    outs(%empty : tensor<4x8xi32>) -> tensor<4x8xi32>
  %neg2 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %neg1 : tensor<4x8xi32>, tensor<4x8xi32>)
    outs(%empty : tensor<4x8xi32>) -> tensor<4x8xi32>
  return %neg2 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: func @no_fold_sub_sub_multi_use
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
func.func @no_fold_sub_sub_multi_use(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %zero = arith.constant dense<0.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %neg1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %neg2 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    ins(%zero, %neg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    ins(%neg2, %neg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %add : tensor<4x8xf32>
}
