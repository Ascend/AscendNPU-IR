// RUN: bishengir-opt %s -transform-interpreter -canonicalize -cse \
// RUN:     -split-input-file -verify-diagnostics | FileCheck %s

module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @vadd_auto_sizes(
  // CHECK-NOT: hivm.hir.vadd
  // CHECK: arith.addf %{{.*}}, %{{.*}} : vector<4x16xf32>
  func.func @vadd_auto_sizes(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %0 = tensor.empty() : tensor<4x16xf32>
    %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%0 : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %1 : tensor<4x16xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["hivm.hir.vadd"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.hivm.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @vexp_explicit_sizes(
  // CHECK-NOT: hivm.hir.vexp
  // CHECK: math.exp %{{.*}} : vector<8x8xf32>
  func.func @vexp_explicit_sizes(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = tensor.empty() : tensor<8x8xf32>
    %1 = hivm.hir.vexp ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["hivm.hir.vexp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.hivm.vectorize %0 vector_sizes = [8, 8] : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @vand_auto_sizes(
  // CHECK-NOT: hivm.hir.vand
  // CHECK: arith.andi %{{.*}}, %{{.*}} : vector<8x8xi32>
  func.func @vand_auto_sizes(%arg0: tensor<8x8xi32>, %arg1: tensor<8x8xi32>) -> tensor<8x8xi32> {
    %0 = tensor.empty() : tensor<8x8xi32>
    %1 = hivm.hir.vand ins(%arg0, %arg1 : tensor<8x8xi32>, tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
    return %1 : tensor<8x8xi32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["hivm.hir.vand"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.hivm.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: func.func @vreduce_auto_sizes(
  // CHECK-NOT: hivm.hir.vreduce
  // CHECK: vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1] : vector<4x16xf32> to vector<4xf32>
  func.func @vreduce_auto_sizes(%arg0: tensor<4x16xf32>, %init: tensor<4x1xf32>) -> tensor<4x1xf32> {
    %0 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<4x16xf32>) outs(%init : tensor<4x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["hivm.hir.vreduce"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.hivm.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  func.func @not_vectorizable() {
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{payload op does not implement VectorizableOpInterface}}
    transform.hivm.vectorize %0 : !transform.any_op
    transform.yield
  }
}
