// RUN: bishengir-opt %s -hivm-decompose-alloc-tensor-copy -split-input-file | FileCheck %s

// All discardable attributes of the `alloc_tensor` (e.g., the `to_be_replaced`
// marker inserted by `normalize-to-tensor`) are carried over to the
// `hivm.hir.copy`. A dynamically shaped source gets a `tensor.dim` and a
// sized `alloc_tensor`.
// CHECK-LABEL: func.func @test_copy_attrs_dynamic_shape
func.func @test_copy_attrs_dynamic_shape(%arg0: tensor<?x20x16x16xf16>) -> tensor<?x20x16x16xf16> {
  %0 = bufferization.alloc_tensor() copy(%arg0) {to_be_replaced} : tensor<?x20x16x16xf16>
  // CHECK: %[[DIM:.*]] = tensor.dim %arg0, %{{.*}} : tensor<?x20x16x16xf16>
  // CHECK-NEXT: %[[ALLOC_ATTRS:.*]] = bufferization.alloc_tensor(%[[DIM]]) : tensor<?x20x16x16xf16>
  // CHECK-NEXT: %[[CP_ATTRS:.*]] = hivm.hir.copy ins(%arg0 : tensor<?x20x16x16xf16>) outs(%[[ALLOC_ATTRS]] : tensor<?x20x16x16xf16>) {to_be_replaced} -> tensor<?x20x16x16xf16>
  // CHECK-NEXT: return %[[CP_ATTRS]] : tensor<?x20x16x16xf16>
  return %0 : tensor<?x20x16x16xf16>
}

// -----

// A statically shaped source keeps a plain `alloc_tensor` with no dynamic
// sizes and no `tensor.dim`.
// CHECK-LABEL: func.func @test_copy_attrs_static_shape
// CHECK-NOT: tensor.dim
func.func @test_copy_attrs_static_shape(%arg0: tensor<5x20x16x16xf16>) -> tensor<5x20x16x16xf16> {
  %0 = bufferization.alloc_tensor() copy(%arg0) {to_be_replaced} : tensor<5x20x16x16xf16>
  // CHECK: %[[ALLOC_STATIC:.*]] = bufferization.alloc_tensor() : tensor<5x20x16x16xf16>
  // CHECK-NEXT: %[[CP_STATIC:.*]] = hivm.hir.copy ins(%arg0 : tensor<5x20x16x16xf16>) outs(%[[ALLOC_STATIC]] : tensor<5x20x16x16xf16>) {to_be_replaced} -> tensor<5x20x16x16xf16>
  // CHECK-NEXT: return %[[CP_STATIC]] : tensor<5x20x16x16xf16>
  return %0 : tensor<5x20x16x16xf16>
}

// -----

// Prefer recomputing a scalar broadcast over the generic copy pattern.
// CHECK-LABEL: func.func @copy_scalar_broadcast
// CHECK-SAME: %[[SCALAR:.*]]: f32
// CHECK: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<128xf32>
// CHECK-NEXT: %[[BRC:.*]] = hivm.hir.vbrc {broadcast_tag} ins(%[[SCALAR]] : f32) outs(%[[ALLOC]] : tensor<128xf32>) -> tensor<128xf32>
// CHECK-NOT: hivm.hir.copy
// CHECK: return %[[BRC]]
func.func @copy_scalar_broadcast(%scalar: f32) -> tensor<128xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %empty = tensor.empty() : tensor<128xf32>
  %brc = hivm.hir.vbrc {broadcast_tag} ins(%scalar : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %copy = bufferization.alloc_tensor() copy(%brc) : tensor<128xf32>
  return %copy : tensor<128xf32>
}

// -----

// Preserve broadcast dimensions and derive dynamic allocation sizes.
// CHECK-LABEL: func.func @copy_dynamic_tensor_broadcast
// CHECK: %[[ALLOC:.*]] = bufferization.alloc_tensor(%{{.*}}) : tensor<?x16xf32>
// CHECK-NEXT: %[[BRC:.*]] = hivm.hir.vbrc ins(%arg0 : tensor<1x16xf32>) outs(%[[ALLOC]] : tensor<?x16xf32>) broadcast_dims = [0] -> tensor<?x16xf32>
// CHECK-NOT: hivm.hir.copy
// CHECK: return %[[BRC]]
func.func @copy_dynamic_tensor_broadcast(%src: tensor<1x16xf32>, %n: index) -> tensor<?x16xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %empty = tensor.empty(%n) : tensor<?x16xf32>
  %brc = hivm.hir.vbrc ins(%src : tensor<1x16xf32>) outs(%empty : tensor<?x16xf32>) broadcast_dims = [0] -> tensor<?x16xf32>
  %copy = bufferization.alloc_tensor() copy(%brc) : tensor<?x16xf32>
  return %copy : tensor<?x16xf32>
}

// -----

// Vector functions retain the generic copy decomposition.
// CHECK-LABEL: func.func @copy_broadcast_aiv
// CHECK: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<128xf32>
// CHECK-NEXT: %[[COPY:.*]] = hivm.hir.copy
// CHECK: return %[[COPY]]
func.func @copy_broadcast_aiv(%scalar: f32) -> tensor<128xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %empty = tensor.empty() : tensor<128xf32>
  %brc = hivm.hir.vbrc ins(%scalar : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %copy = bufferization.alloc_tensor() copy(%brc) : tensor<128xf32>
  return %copy : tensor<128xf32>
}

// -----

// Any discardable attribute disables broadcast recomputation, including
// arbitrary attributes that are not known transformation markers.
// CHECK-LABEL: func.func @copy_marked_broadcast
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc
// CHECK: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<128xf32>
// CHECK-NEXT: %[[COPY:.*]] = hivm.hir.copy ins(%[[BRC]] : tensor<128xf32>) outs(%[[ALLOC]] : tensor<128xf32>) {custom_marker = 7 : i64} -> tensor<128xf32>
// CHECK-NEXT: return %[[COPY]]
func.func @copy_marked_broadcast(%scalar: f32) -> tensor<128xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %empty = tensor.empty() : tensor<128xf32>
  %brc = hivm.hir.vbrc ins(%scalar : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %copy = bufferization.alloc_tensor() copy(%brc) {custom_marker = 7 : i64} : tensor<128xf32>
  return %copy : tensor<128xf32>
}
