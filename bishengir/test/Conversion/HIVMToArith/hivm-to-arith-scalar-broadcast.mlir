// RUN: bishengir-opt %s -convert-hivm-to-tritongpu="allow-return-value=true" | FileCheck %s

// CHECK-LABEL: func.func @vmul_scalar_broadcast
// CHECK-SAME: %[[INPUT:.*]]: tensor<2x1xi32>, %[[SCALE:.*]]: i32
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[INPUT]] : tensor<2x1xi32> -> tensor<2x2xi32>
// CHECK: %[[SPLAT:.*]] = tt.splat %[[SCALE]] : i32 -> tensor<2x2xi32>
// CHECK: %[[RESULT:.*]] = arith.muli %[[BROADCAST]], %[[SPLAT]] : tensor<2x2xi32>
// CHECK: return %[[RESULT]] : tensor<2x2xi32>
func.func @vmul_scalar_broadcast(%input: tensor<2x1xi32>, %scale: i32) -> tensor<2x2xi32> {
  %empty = tensor.empty() : tensor<2x2xi32>
  %result = hivm.hir.vmul ins(%input, %scale : tensor<2x1xi32>, i32)
      outs(%empty : tensor<2x2xi32>) broadcast = [1] -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
