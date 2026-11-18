// RUN: bishengir-opt %s --hivm-vectorize --verify-each | FileCheck %s

// The vectorize stage chooses row widths from the register budget: f32
// elementwise computation runs on 64-lane rows, and a wider tensor is split
// into rows of that width.

module {
  func.func @wide(%a: tensor<128xf32>, %b: tensor<128xf32>,
                  %init: tensor<128xf32>) -> tensor<128xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<128xf32>, tensor<128xf32>)
        outs(%init : tensor<128xf32>) -> tensor<128xf32>
    return %sum : tensor<128xf32>
  }
}

// CHECK-LABEL: func.func private @wide_fused_0(
// CHECK: %[[STEP:.*]] = arith.constant 64 : index
// CHECK: scf.for {{.*}} step %[[STEP]]
// CHECK: vector.transfer_read {{.*}} : tensor<64xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<64xf32>
// CHECK-NOT: hivm.hir.vadd
// CHECK-LABEL: func.func @wide(
// CHECK: call @wide_fused_0
