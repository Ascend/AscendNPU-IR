// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false" | FileCheck %s
// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false use-preorder-walk-traversal=false" | FileCheck %s

// CHECK-LABEL: func.func @while_slice
// CHECK: scf.while
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
// CHECK: scf.yield
// CHECK: tensor.extract_slice
// CHECK: return
func.func @while_slice(%arg: tensor<32xf32>, %n: index, %cond: i1) -> tensor<?xf32> {
  %r = scf.while (%x = %arg) : (tensor<32xf32>) -> tensor<32xf32> {
    scf.condition(%cond) %x : tensor<32xf32>
  } do {
  ^bb0(%x: tensor<32xf32>):
    %s = tensor.extract_slice %x[0] [%n] [1] : tensor<32xf32> to tensor<?xf32>
    %i = tensor.insert_slice %s into %arg[0] [%n] [1] : tensor<?xf32> into tensor<32xf32>
    scf.yield %i : tensor<32xf32>
  }
  %s = tensor.extract_slice %r[0] [%n] [1] : tensor<32xf32> to tensor<?xf32>
  return %s : tensor<?xf32>
}

// The before and after regions deliberately have different tensor ranks.
// Slicing in both regions must preserve each boundary's original type.
// CHECK-LABEL: func.func @while_distinct_signatures
// CHECK: scf.while {{.*}} : (tensor<4x8xf32>) -> tensor<?x8xf32> {
// CHECK: tensor.extract_slice {{.*}} : tensor<4x8xf32> to tensor<?x8xf32>
// CHECK: scf.condition({{.*}}) {{.*}} : tensor<?x8xf32>
// CHECK: ^bb0({{.*}}: tensor<?x8xf32>):
// CHECK: tensor.extract_slice {{.*}} : tensor<?x8xf32> to tensor<?x8xf32>
// CHECK: tensor.insert_slice {{.*}} : tensor<?x8xf32> into tensor<4x8xf32>
// CHECK: scf.yield {{.*}} : tensor<4x8xf32>
// CHECK: return {{.*}} : tensor<?x8xf32>
func.func @while_distinct_signatures(%arg: tensor<4x8xf32>, %n: index, %cond: i1) -> tensor<?x8xf32> {
  %r = scf.while (%x = %arg) : (tensor<4x8xf32>) -> tensor<?x8xf32> {
    %s = tensor.extract_slice %x[0, 0] [%n, 8] [1, 1] : tensor<4x8xf32> to tensor<?x8xf32>
    scf.condition(%cond) %s : tensor<?x8xf32>
  } do {
  ^bb0(%x: tensor<?x8xf32>):
    %s = tensor.extract_slice %x[0, 0] [%n, 8] [1, 1] : tensor<?x8xf32> to tensor<?x8xf32>
    %i = tensor.insert_slice %s into %arg[0, 0] [%n, 8] [1, 1] : tensor<?x8xf32> into tensor<4x8xf32>
    scf.yield %i : tensor<4x8xf32>
  }
  return %r : tensor<?x8xf32>
}

// Independent work inside a loop can still flatten to one dimension.
// CHECK-LABEL: func.func @while_internal_flatten
// CHECK: scf.while {{.*}} : (tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK: scf.condition({{.*}}) {{.*}} : tensor<4x8xf32>
// CHECK: ^bb0({{.*}}: tensor<4x8xf32>):
// CHECK: linalg.fill {{.*}} outs({{.*}} : memref<32xf32>)
// CHECK: scf.yield {{.*}} : tensor<4x8xf32>
// CHECK: return {{.*}} : tensor<4x8xf32>
func.func @while_internal_flatten(%arg: tensor<4x8xf32>, %work: memref<4x8xf32>, %value: f32, %cond: i1) -> tensor<4x8xf32> {
  %r = scf.while (%x = %arg) : (tensor<4x8xf32>) -> tensor<4x8xf32> {
    scf.condition(%cond) %x : tensor<4x8xf32>
  } do {
  ^bb0(%x: tensor<4x8xf32>):
    linalg.fill ins(%value : f32) outs(%work : memref<4x8xf32>)
    scf.yield %x : tensor<4x8xf32>
  }
  return %r : tensor<4x8xf32>
}
