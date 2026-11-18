// RUN: bishengir-opt %s --hivm-vectorize --verify-each -split-input-file \
// RUN:   | FileCheck %s

// Dynamic and non-row-aligned shapes. Dynamic extents materialize masks from
// the runtime dimension; static tails keep the 64-lane row plus a mask.

module {
  func.func @dyn1d(%a: tensor<?xf32>, %b: tensor<?xf32>,
                   %init: tensor<?xf32>) -> tensor<?xf32> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
        outs(%init : tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func private @dyn1d_fused_0
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: %[[DIM:.*]] = tensor.dim {{.*}} : tensor<?xf32>
// CHECK: scf.for {{.*}} = %c0 to %[[DIM]] step %c64
// CHECK: %[[MASK:.*]] = vector.create_mask
// CHECK: vector.transfer_read {{.*}}, %[[MASK]] {{.*}} : tensor<?xf32>, vector<64xf32>
// CHECK: vector.transfer_write {{.*}}, %[[MASK]] {{.*}} : vector<64xf32>, tensor<?xf32>
// CHECK-LABEL: func.func @dyn1d(
// CHECK: call @dyn1d_fused_0

// -----

// A dynamic leading dimension keeps a whole static row per iteration.

module {
  func.func @dyn2d(%a: tensor<?x64xf32>, %b: tensor<?x64xf32>,
                   %init: tensor<?x64xf32>) -> tensor<?x64xf32> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<?x64xf32>, tensor<?x64xf32>)
        outs(%init : tensor<?x64xf32>) -> tensor<?x64xf32>
    return %0 : tensor<?x64xf32>
  }
}

// CHECK-LABEL: func.func private @dyn2d_fused_0
// CHECK: %[[DIM:.*]] = tensor.dim {{.*}} : tensor<?x64xf32>
// CHECK: scf.for {{.*}} = %c0 to %[[DIM]] step %c1
// CHECK: tensor.extract_slice {{.*}} [1, 64] [1, 1] : tensor<?x64xf32> to tensor<1x64xf32>
// CHECK: vector.shape_cast {{.*}} : vector<1x64xf32> to vector<64xf32>

// -----

module {
  func.func @tail(%a: tensor<67xf32>, %b: tensor<67xf32>,
                  %init: tensor<67xf32>) -> tensor<67xf32> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<67xf32>, tensor<67xf32>)
        outs(%init : tensor<67xf32>) -> tensor<67xf32>
    return %0 : tensor<67xf32>
  }
}

// CHECK-LABEL: func.func private @tail_fused_0
// CHECK: scf.for {{.*}} = %c0 to %c67 step %c64
// CHECK: vector.create_mask
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<?xf32>
