// RUN: bishengir-opt %s --hivm-vectorize --verify-each -split-input-file \
// RUN:   | FileCheck %s

// The cleanup stage flattens vector computation to one-dimensional rows and
// removes linalg operations. Multidimensional arithmetic becomes shape casts
// around a single flat operation.

module {
  func.func @flatten(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>,
                     %init: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<4x8xf32>, tensor<4x8xf32>)
        outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %sum : tensor<4x8xf32>
  }
}

// CHECK-LABEL: func.func private @flatten_fused_0(
// CHECK: vector.transfer_read {{.*}} : tensor<4x8xf32>, vector<4x8xf32>
// CHECK: vector.shape_cast {{.*}} : vector<4x8xf32> to vector<32xf32>
// CHECK: arith.addf {{.*}} : vector<32xf32>

// -----

// A reduction keeps its loop but its unit lower-dimension arithmetic is
// flattened to a one-dimensional vector row.

module {
  func.func @row_reduce(%arg: tensor<4x8xf32>, %init: tensor<1x8xf32>)
      -> tensor<1x8xf32> {
    %0 = hivm.hir.vreduce<sum> ins(%arg : tensor<4x8xf32>)
        outs(%init : tensor<1x8xf32>) reduce_dims = [0] -> tensor<1x8xf32>
    return %0 : tensor<1x8xf32>
  }
}

// CHECK-LABEL: func.func private @row_reduce_fused_0(
// CHECK: vector.shape_cast {{.*}} : vector<1x8xf32> to vector<8xf32>
// CHECK: arith.addf {{.*}} : vector<8xf32>
// CHECK: scf.yield
// CHECK-NOT: linalg.
