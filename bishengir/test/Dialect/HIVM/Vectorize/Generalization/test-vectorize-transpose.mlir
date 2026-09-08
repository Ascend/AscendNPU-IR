// RUN: bishengir-opt %s --split-input-file --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Diff: transpose identity canonicalization
// The HIVM identity transpose folds before vectorization. The Linalg path
// retains an explicit transfer pair with the same SSA value.

// CHECK-HFUSION-LABEL: func.func @hfusion_identity(
// CHECK-HFUSION: %[[READ:.*]] = vector.transfer_read
// CHECK-HFUSION-SAME: vector<2x4x8xf32>
// CHECK-HFUSION: vector.transfer_write %[[READ]]
func.func @hfusion_identity(%input: tensor<2x4x8xf32>)
    -> tensor<2x4x8xf32> {
  %empty = tensor.empty() : tensor<2x4x8xf32>
  %result = linalg.transpose ins(%input : tensor<2x4x8xf32>)
      outs(%empty : tensor<2x4x8xf32>) permutation = [0, 1, 2]
  return %result : tensor<2x4x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_identity(
// CHECK-HIVM-SAME: %[[ARG0:.*]]: tensor<2x4x8xf32>)
// CHECK-HIVM-NEXT: return %[[ARG0]] : tensor<2x4x8xf32>
func.func @hivm_identity(%input: tensor<2x4x8xf32>)
    -> tensor<2x4x8xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<2x4x8xf32>
  %result = hivm.hir.vtranspose ins(%input : tensor<2x4x8xf32>)
      outs(%empty : tensor<2x4x8xf32>) permutation = [0, 1, 2]
      -> tensor<2x4x8xf32>
  return %result : tensor<2x4x8xf32>
}

// -----

// A non-square 2-D transpose checks the inverse transfer permutation and tail
// masks, not only the output vector type.

// CHECK-HFUSION: affine_map<(d0, d1) -> (d1, d0)>
// CHECK-HFUSION-LABEL: func.func @hfusion_swap_tail(
// CHECK-HFUSION-DAG: %[[READ_MASK:.*]] = vector.constant_mask [3, 7] : vector<3x21xi1>
// CHECK-HFUSION-DAG: %[[WRITE_MASK:.*]] = vector.constant_mask [7, 3] : vector<21x3xi1>
// CHECK-HFUSION-DAG: %[[READ:.*]] = vector.transfer_read {{.*}}, %[[READ_MASK]] {{.*}}permutation_map = #{{.*}}{{.*}}vector<21x3xf32>
// CHECK-HFUSION: vector.transfer_write %[[READ]], {{.*}}, %[[WRITE_MASK]]
func.func @hfusion_swap_tail(%input: tensor<3x7xf32>) -> tensor<7x3xf32> {
  %empty = tensor.empty() : tensor<7x3xf32>
  %result = linalg.transpose ins(%input : tensor<3x7xf32>)
      outs(%empty : tensor<7x3xf32>) permutation = [1, 0]
  return %result : tensor<7x3xf32>
}

// CHECK-HIVM: affine_map<(d0, d1) -> (d1, d0)>
// CHECK-HIVM-LABEL: func.func @hivm_swap_tail(
// CHECK-HIVM-DAG: %[[READ_MASK:.*]] = vector.constant_mask [3, 7] : vector<3x21xi1>
// CHECK-HIVM-DAG: %[[WRITE_MASK:.*]] = vector.constant_mask [7, 3] : vector<21x3xi1>
// CHECK-HIVM-DAG: %[[READ:.*]] = vector.transfer_read {{.*}}, %[[READ_MASK]] {{.*}}permutation_map = #{{.*}}{{.*}}vector<21x3xf32>
// CHECK-HIVM: vector.transfer_write %[[READ]], {{.*}}, %[[WRITE_MASK]]
func.func @hivm_swap_tail(%input: tensor<3x7xf32>) -> tensor<7x3xf32>
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<7x3xf32>
  %result = hivm.hir.vtranspose ins(%input : tensor<3x7xf32>)
      outs(%empty : tensor<7x3xf32>) permutation = [1, 0]
      -> tensor<7x3xf32>
  return %result : tensor<7x3xf32>
}

// -----

// A non-self-inverse 4-D cycle catches accidental use of the HIVM indexing
// map in the transfer direction. The trailing unit dimension is intentional.

// CHECK-HFUSION: affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>
// CHECK-HFUSION-LABEL: func.func @hfusion_cycle_unit(
// CHECK-HFUSION-DAG: %[[READ_MASK:.*]] = vector.constant_mask [2, 3, 4, 1] : vector<2x3x10x1xi1>
// CHECK-HFUSION-DAG: %[[WRITE_MASK:.*]] = vector.constant_mask [4, 2, 3, 1] : vector<10x2x3x1xi1>
// CHECK-HFUSION-DAG: %[[READ:.*]] = vector.transfer_read {{.*}}, %[[READ_MASK]] {{.*}}permutation_map = #{{.*}}{{.*}}vector<10x2x3x1xf32>
// CHECK-HFUSION: vector.transfer_write %[[READ]], {{.*}}, %[[WRITE_MASK]]
func.func @hfusion_cycle_unit(%input: tensor<2x3x4x1xf32>)
    -> tensor<4x2x3x1xf32> {
  %empty = tensor.empty() : tensor<4x2x3x1xf32>
  %result = linalg.transpose ins(%input : tensor<2x3x4x1xf32>)
      outs(%empty : tensor<4x2x3x1xf32>) permutation = [2, 0, 1, 3]
  return %result : tensor<4x2x3x1xf32>
}

// CHECK-HIVM: affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>
// CHECK-HIVM-LABEL: func.func @hivm_cycle_unit(
// CHECK-HIVM-DAG: %[[READ_MASK:.*]] = vector.constant_mask [2, 3, 4, 1] : vector<2x3x10x1xi1>
// CHECK-HIVM-DAG: %[[WRITE_MASK:.*]] = vector.constant_mask [4, 2, 3, 1] : vector<10x2x3x1xi1>
// CHECK-HIVM-DAG: %[[READ:.*]] = vector.transfer_read {{.*}}, %[[READ_MASK]] {{.*}}permutation_map = #{{.*}}{{.*}}vector<10x2x3x1xf32>
// CHECK-HIVM: vector.transfer_write %[[READ]], {{.*}}, %[[WRITE_MASK]]
func.func @hivm_cycle_unit(%input: tensor<2x3x4x1xf32>)
    -> tensor<4x2x3x1xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<4x2x3x1xf32>
  %result = hivm.hir.vtranspose ins(%input : tensor<2x3x4x1xf32>)
      outs(%empty : tensor<4x2x3x1xf32>) permutation = [2, 0, 1, 3]
      -> tensor<4x2x3x1xf32>
  return %result : tensor<4x2x3x1xf32>
}
