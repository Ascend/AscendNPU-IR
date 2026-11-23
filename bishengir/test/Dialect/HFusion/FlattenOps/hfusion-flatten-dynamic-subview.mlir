// RUN: bishengir-opt %s                              \
// RUN:   -pass-pipeline="builtin.module(func.func(   \
// RUN:      hfusion-flatten-ops{flatten-mode=tidy register-based=true}), \
// RUN:      canonicalize-ext)"                       \
// RUN:   -split-input-file | FileCheck %s

// Dynamic-shape flatten tests reduced from an index_select simd kernel.
//
// The slice views a <4x1x128> region of a fully dynamic <?x?x?xf32> buffer,
// so the unit second dimension bridges the first two dimensions into one
// collapse group [[0, 1], [2]]. A static offset (4) sits before a dynamic
// dimension inside the group: the flattened offset must multiply it by the
// runtime dim (memref.dim) of the dynamic dimension instead of a constant
// poisoned by ShapedType::kDynamic. processSubviewOp must also keep the
// subview's dynamic source dims unbound (joinCollapser) so the runtime dims
// stay available for materialization.

// CHECK-LABEL: func.func @copy_dyn_src(
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?x?xf32>
// CHECK-SAME: %[[ROW:.*]]: index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// The runtime multiplier is the dynamic dim 1, not a constant.
// CHECK: %[[DIM1:.*]] = memref.dim %[[ARG0]], %[[C1]] : memref<?x?x?xf32>
// CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %[[ARG0]] {{\[\[}}0, 1], {{\[}}2]] : memref<?x?x?xf32> into memref<?x?xf32>
// Flattened offset = 4 * dim1 + %row (missing the 4 * dim1 term is the regression).
// CHECK: %[[MUL:.*]] = arith.muli %[[DIM1]], %[[C4]] : index
// CHECK: %[[OFF:.*]] = arith.addi %[[MUL]], %[[ROW]] : index
// CHECK: memref.subview %[[COLLAPSED]]{{\[}}%[[OFF]], 0] [4, 128] [%{{.*}}, 1] : memref<?x?xf32> to memref<4x128xf32
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<4x128xf32
func.func @copy_dyn_src(%src: memref<?x?x?xf32>, %row: index)
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %sv = memref.subview %src[4, %row, 0] [4, 1, 128] [1, 1, 1] : memref<?x?x?xf32> to memref<4x1x128xf32, strided<[?, ?, 1], offset: ?>>
  %buf = memref.alloc() : memref<4x1x128xf32>
  memref.copy %sv, %buf : memref<4x1x128xf32, strided<[?, ?, 1], offset: ?>> to memref<4x1x128xf32>
  return
}

// -----

// Same pattern through the tensor path: a <1x1x128> slice of <?x?x128>
// collapses fully to 1D and expands back at the return. The flattened offset
// must be dim1 * (2 * 128) + %row * 128.
// CHECK-LABEL: func.func @subview_dyn_tensor_tail(
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[DIM1:.*]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?x128xf32>
// CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %{{.*}} {{\[\[}}0, 1, 2]] : memref<?x?x128xf32> into memref<?xf32>
// CHECK: %[[MUL0:.*]] = arith.muli %[[DIM1]], %[[C256]] : index
// CHECK: %[[MUL1:.*]] = arith.muli %{{.*}}, %[[C128]] : index
// CHECK: %[[OFF:.*]] = arith.addi %[[MUL0]], %[[MUL1]] : index
// CHECK: memref.subview %[[COLLAPSED]]{{\[}}%[[OFF]]] [128] [1] : memref<?xf32> to memref<128xf32
// CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0, 1, 2]] output_shape [1, 1, 128]
func.func @subview_dyn_tensor_tail(%src: memref<?x?x128xf32>, %row: index) -> tensor<1x1x128xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %sv = memref.subview %src[2, %row, 0] [1, 1, 128] [1, 1, 1] : memref<?x?x128xf32> to memref<1x1x128xf32, strided<[?, 128, 1], offset: ?>>
  %t = bufferization.to_tensor %sv restrict : memref<1x1x128xf32, strided<[?, 128, 1], offset: ?>>
  return %t : tensor<1x1x128xf32>
}
