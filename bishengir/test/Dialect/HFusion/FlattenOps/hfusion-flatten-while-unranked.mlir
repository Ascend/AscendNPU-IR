// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=false multi-dynamic-shape=false" | FileCheck %s
// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=false multi-dynamic-shape=false use-preorder-walk-traversal=false" | FileCheck %s

// These cases previously passed in memory-based mode. Keep the producers
// outside the tracked graph and skip unranked while-boundary values.
// CHECK-LABEL: func.func @while_unranked
// CHECK: scf.while {{.*}} : (tensor<*xf32>) -> tensor<*xf32>
func.func @while_unranked(%arg: tensor<*xf32>, %cond: i1) -> tensor<*xf32> {
  %r = scf.while (%x = %arg) : (tensor<*xf32>) -> tensor<*xf32> {
    scf.condition(%cond) %x : tensor<*xf32>
  } do {
  ^bb0(%x: tensor<*xf32>):
    scf.yield %x : tensor<*xf32>
  }
  return %r : tensor<*xf32>
}

// CHECK-LABEL: func.func @while_untracked_cast
// CHECK: tensor.splat
// CHECK: tensor.cast {{.*}} : tensor<4x8xf32> to tensor<?x8xf32>
// CHECK: scf.while
func.func @while_untracked_cast(%scalar: f32, %cond: i1) -> tensor<?x8xf32> {
  %s = tensor.splat %scalar : tensor<4x8xf32>
  %v = tensor.cast %s : tensor<4x8xf32> to tensor<?x8xf32>
  %r = scf.while (%x = %v) : (tensor<?x8xf32>) -> tensor<?x8xf32> {
    scf.condition(%cond) %x : tensor<?x8xf32>
  } do {
  ^bb0(%x: tensor<?x8xf32>):
    scf.yield %x : tensor<?x8xf32>
  }
  return %r : tensor<?x8xf32>
}
