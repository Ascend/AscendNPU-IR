// RUN: bishengir-opt %s --hivm-vectorize --verify-each | FileCheck %s

// The partition stage uses the SameBlock and UnseparatedByBarrier models.
// A call separates contiguous tensor-compute groups, so the two vectorizable
// operations become independent vector functions.

module {
  func.func private @opaque()

  func.func @twogroups(%a: tensor<64xf32>, %b: tensor<64xf32>,
                       %init: tensor<64xf32>) -> tensor<64xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    func.call @opaque() : () -> ()
    %product = hivm.hir.vmul ins(%sum, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %product : tensor<64xf32>
  }
}

// CHECK-LABEL: func.func private @twogroups_fused_0(
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: arith.mulf {{.*}} : vector<64xf32>
// CHECK-LABEL: func.func private @twogroups_fused_1(
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK-LABEL: func.func @twogroups(
// CHECK: call @twogroups_fused_1
// CHECK-NEXT: call @opaque() : () -> ()
// CHECK-NEXT: call @twogroups_fused_0
