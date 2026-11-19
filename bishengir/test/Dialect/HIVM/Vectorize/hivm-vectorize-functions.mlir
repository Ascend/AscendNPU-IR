// RUN: bishengir-opt %s --hivm-vectorize --verify-each | FileCheck %s --implicit-check-not=hivm.hir.vadd --implicit-check-not=hivm.hir.vmul

module {
  func.func @first(%a: tensor<64xf32>, %b: tensor<64xf32>,
                   %init: tensor<64xf32>) -> tensor<64xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %sum : tensor<64xf32>
  }
  func.func @second(%a: tensor<64xf32>, %b: tensor<64xf32>,
                    %init: tensor<64xf32>) -> tensor<64xf32> {
    %product = hivm.hir.vmul ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %product : tensor<64xf32>
  }
}

// CHECK-DAG: func.func private @first_fused_0
// CHECK-DAG: func.func private @second_fused_0
// CHECK-DAG: vector.transfer_read
// CHECK-DAG: arith.addf
// CHECK-DAG: arith.mulf
// CHECK-DAG: call @first_fused_0
// CHECK-DAG: call @second_fused_0
