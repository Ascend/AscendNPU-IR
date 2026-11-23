// RUN: bishengir-opt %s --hivm-vectorize -split-input-file | FileCheck %s

module {
  func.func @empty() {
    return
  }


  func.func @vectorizable(%a: memref<8xf32>, %b: memref<8xf32>,
                          %c: memref<8xf32>) {
    hivm.hir.vadd ins(%a, %b : memref<8xf32>, memref<8xf32>)
        outs(%c : memref<8xf32>)
    return
  }
}

// CHECK-LABEL: func.func @empty()
// CHECK-NEXT: return

// CHECK-LABEL: func.func @vectorizable
// CHECK: hivm.hir.vadd
// CHECK-NEXT: return

// -----


module {
  func.func @tensor(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>,
                    %c: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = hivm.hir.vadd
        ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
        outs(%c : tensor<8x16xf32>) -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}

// CHECK-LABEL: func.func private @tensor_fused_0
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: arith.addf
// CHECK: vector.transfer_write
// CHECK-NOT: linalg.
// CHECK: func.func @tensor
// CHECK: call @tensor_fused_0
// CHECK-NOT: hivm.hir.vadd

// -----


module {
  func.func @concat_chain(%a: tensor<4x2xf32>, %b: tensor<4x2xf32>,
                          %init: tensor<4x2xf32>, %rhs: tensor<4x2xf32>,
                          %concatInit: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<4x2xf32>, tensor<4x2xf32>)
        outs(%init : tensor<4x2xf32>) -> tensor<4x2xf32>
    %1 = hivm.hir.vconcat dim(1)
        ins(%0, %rhs : tensor<4x2xf32>, tensor<4x2xf32>)
        outs(%concatInit : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}

// CHECK-LABEL: func.func private @concat_chain_fused_0
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK-NOT: vconcat
// CHECK: func.func @concat_chain
// CHECK: call @concat_chain_fused_0
// CHECK: hivm.hir.vconcat
// CHECK-NOT: hivm.hir.vadd
