// RUN: split-file %s %t
// RUN: bishengir-opt %t/positive.mlir --split-input-file \
// RUN:   --hfusion-vectorize-ops --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %t/positive.mlir --split-input-file \
// RUN:   --hivm-vectorize-ops --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM
// RUN: not bishengir-opt %t/multi-dynamic.mlir --hfusion-vectorize-ops \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-HFUSION-MULTI
// RUN: not bishengir-opt %t/multi-dynamic.mlir --hivm-vectorize-ops \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-HIVM-MULTI
// RUN: not bishengir-opt %t/over-capacity.mlir --hfusion-vectorize-ops \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-HFUSION-CAPACITY
// RUN: not bishengir-opt %t/over-capacity.mlir --hivm-vectorize-ops \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-HIVM-CAPACITY

//--- positive.mlir

// One dynamic axis uses the full f32 register capacity.

// CHECK-HFUSION-LABEL: func.func @hfusion_dynamic_1d(
// CHECK-HFUSION: vector.transfer_read {{.*}} : tensor<?xf32>, vector<64xf32>
// CHECK-HFUSION: arith.addf {{.*}} : vector<64xf32>
// CHECK-HFUSION: vector.transfer_write {{.*}} : vector<64xf32>, tensor<?xf32>
func.func @hfusion_dynamic_1d(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>)
    -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %size = tensor.dim %lhs, %c0 : tensor<?xf32>
  %empty = tensor.empty(%size) : tensor<?xf32>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_dynamic_1d(
// CHECK-HIVM: vector.transfer_read {{.*}} : tensor<?xf32>, vector<64xf32>
// CHECK-HIVM: arith.addf {{.*}} : vector<64xf32>
// CHECK-HIVM: vector.transfer_write {{.*}} : vector<64xf32>, tensor<?xf32>
func.func @hivm_dynamic_1d(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>)
    -> tensor<?xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %size = tensor.dim %lhs, %c0 : tensor<?xf32>
  %empty = tensor.empty(%size) : tensor<?xf32>
  %result = hivm.hir.vadd ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// Unit dimensions may accompany one dynamic axis.

// CHECK-HFUSION-LABEL: func.func @hfusion_dynamic_unit(
// CHECK-HFUSION: vector.transfer_read {{.*}} : tensor<1x?xf16>, vector<1x128xf16>
// CHECK-HFUSION: arith.mulf {{.*}} : vector<1x128xf16>
func.func @hfusion_dynamic_unit(%lhs: tensor<1x?xf16>,
    %rhs: tensor<1x?xf16>) -> tensor<1x?xf16> {
  %c1 = arith.constant 1 : index
  %size = tensor.dim %lhs, %c1 : tensor<1x?xf16>
  %empty = tensor.empty(%size) : tensor<1x?xf16>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
      ins(%lhs, %rhs : tensor<1x?xf16>, tensor<1x?xf16>)
      outs(%empty : tensor<1x?xf16>) -> tensor<1x?xf16>
  return %result : tensor<1x?xf16>
}

// CHECK-HIVM-LABEL: func.func @hivm_dynamic_unit(
// CHECK-HIVM: vector.transfer_read {{.*}} : tensor<1x?xf16>, vector<1x128xf16>
// CHECK-HIVM: arith.mulf {{.*}} : vector<1x128xf16>
func.func @hivm_dynamic_unit(%lhs: tensor<1x?xf16>,
    %rhs: tensor<1x?xf16>) -> tensor<1x?xf16>
    attributes {hivm.vector_function} {
  %c1 = arith.constant 1 : index
  %size = tensor.dim %lhs, %c1 : tensor<1x?xf16>
  %empty = tensor.empty(%size) : tensor<1x?xf16>
  %result = hivm.hir.vmul
      ins(%lhs, %rhs : tensor<1x?xf16>, tensor<1x?xf16>)
      outs(%empty : tensor<1x?xf16>) -> tensor<1x?xf16>
  return %result : tensor<1x?xf16>
}

//--- multi-dynamic.mlir

// Both paths reject more than one non-unit or dynamic axis.
// CHECK-HFUSION-MULTI: error: Failed to compute dynamic vector sizes
// CHECK-HIVM-MULTI: error: Failed to compute dynamic vector sizes
func.func @hfusion_multi_dynamic(%lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>, %empty: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

func.func @hivm_multi_dynamic(%lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>, %empty: tensor<?x?xf32>) -> tensor<?x?xf32>
    attributes {hivm.vector_function} {
  %result = hivm.hir.vadd
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

//--- over-capacity.mlir

// Both paths reject an innermost static dimension larger than one register.
// CHECK-HFUSION-CAPACITY: error: Exceeds vector capacity
// CHECK-HIVM-CAPACITY: error: Exceeds vector capacity
func.func @hfusion_over_capacity(%lhs: tensor<65xf32>,
    %rhs: tensor<65xf32>) -> tensor<65xf32> {
  %empty = tensor.empty() : tensor<65xf32>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%lhs, %rhs : tensor<65xf32>, tensor<65xf32>)
      outs(%empty : tensor<65xf32>) -> tensor<65xf32>
  return %result : tensor<65xf32>
}

func.func @hivm_over_capacity(%lhs: tensor<65xf32>, %rhs: tensor<65xf32>)
    -> tensor<65xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<65xf32>
  %result = hivm.hir.vadd
      ins(%lhs, %rhs : tensor<65xf32>, tensor<65xf32>)
      outs(%empty : tensor<65xf32>) -> tensor<65xf32>
  return %result : tensor<65xf32>
}
