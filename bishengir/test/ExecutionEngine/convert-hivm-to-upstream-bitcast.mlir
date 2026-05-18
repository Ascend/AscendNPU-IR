// RUN: bishengir-opt --execution-engine-convert-hivm-to-upstream %s | FileCheck %s

module {
  func.func @convert_hivm_bitcast(%arg0: tensor<4x8xf32>) -> tensor<4x8xi32> {
    %0 = hivm.hir.bitcast %arg0 : tensor<4x8xf32> -> tensor<4x8xi32>
    return %0 : tensor<4x8xi32>
  }
}

// CHECK-LABEL: func.func @convert_hivm_bitcast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32>) -> tensor<4x8xi32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xi32>
// CHECK: %[[BITCAST:.*]] = hfusion.bitcast
// CHECK-SAME: %[[ARG0]]
// CHECK-SAME: %[[EMPTY]]
// CHECK: return %[[BITCAST]] : tensor<4x8xi32>
// CHECK-NOT: hivm.hir.bitcast