// RUN: bishengir-opt --hfusion-decompose="hfusion-decompose-phase=before-lower-to-loops" %s | FileCheck %s

module {
  func.func @interleave_1d(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<8xf32> {
    %0 = hfusion.interleave %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @interleave_2d(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x6xi32> {
    %0 = hfusion.interleave %arg0, %arg1 : tensor<2x3xi32>, tensor<2x3xi32> -> tensor<2x6xi32>
    return %0 : tensor<2x6xi32>
  }
}

// CHECK-LABEL: func.func @interleave_1d(
// CHECK-NOT: hfusion.interleave
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] = %[[EMPTY]]) -> (tensor<8xf32>)
// CHECK: %[[MUL:.*]] = arith.muli
// CHECK: %[[EXTRACT0:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<4xf32>
// CHECK: %[[INSERT0:.*]] = tensor.insert %[[EXTRACT0]] into %[[ARG]][%[[MUL]]] : tensor<8xf32>
// CHECK: %[[ADD:.*]] = arith.addi
// CHECK: %[[EXTRACT1:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<4xf32>
// CHECK: %[[INSERT1:.*]] = tensor.insert %[[EXTRACT1]] into %[[INSERT0]][%[[ADD]]] : tensor<8xf32>
// CHECK: scf.yield %[[INSERT1]]
// CHECK: return %[[FOR]]

// CHECK-LABEL: func.func @interleave_2d(
// CHECK-NOT: hfusion.interleave
// CHECK: tensor.empty() : tensor<2x6xi32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.muli
// CHECK: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3xi32>
// CHECK: tensor.insert %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x6xi32>
// CHECK: arith.addi
// CHECK: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3xi32>
// CHECK: tensor.insert %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x6xi32>
// CHECK: return
