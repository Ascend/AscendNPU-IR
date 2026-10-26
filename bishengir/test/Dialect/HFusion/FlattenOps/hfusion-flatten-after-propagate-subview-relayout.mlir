// RUN: bishengir-opt %s --propagate-reshape="for-regbased=true" --hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false" | FileCheck %s

// PropagateReshape (RegBase) rebuilds `expand_shape(subview)` with a
// reinterpret_cast whose dynamic offset comes from extract_strided_metadata.
// FlattenOps then rewrites the propagated rank-2 subview. The metadata op must
// keep an operand whose rank matches its results, so the IR stays valid.
// CHECK-LABEL: func.func @flatten_after_subview_relayout(
// CHECK: %[[SUBVIEW:.*]] = memref.subview %{{.*}}[%{{.*}}, 0] [2, 5] [1, 1]
// CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]] : memref<2x1x5xf32, strided<[8, 8, 1], offset: ?>> into memref<2x5xf32, strided<[8, 1], offset: ?>>
// CHECK: %{{.*}}, %[[OFFSET:.*]], %{{.*}}:2, %{{.*}}:2 = memref.extract_strided_metadata %[[COLLAPSED]] : memref<2x5xf32, strided<[8, 1], offset: ?>>
// CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[SUBVIEW]] to offset: [%[[OFFSET]]], sizes: [2, 1, 5], strides: [8, 5, 1]
// CHECK: %[[INPUT:.*]] = memref.collapse_shape %[[CAST]] {{\[\[}}0], [1, 2]]
// CHECK: linalg.exp ins(%[[INPUT]] : memref<2x5xf32, strided<[8, 1], offset: ?>>)
func.func @flatten_after_subview_relayout(
    %src: memref<?x8xf32, strided<[8, 1], offset: ?>>, %i: index,
    %dst: memref<2x1x5xf32>) {
  %subview = memref.subview %src[%i, 0] [2, 5] [1, 1] :
      memref<?x8xf32, strided<[8, 1], offset: ?>> to
      memref<2x5xf32, strided<[8, 1], offset: ?>>
  %expanded = memref.expand_shape %subview [[0], [1, 2]]
      output_shape [2, 1, 5] :
      memref<2x5xf32, strided<[8, 1], offset: ?>> into
      memref<2x1x5xf32, strided<[8, 5, 1], offset: ?>>
  linalg.exp ins(%expanded : memref<2x1x5xf32, strided<[8, 5, 1], offset: ?>>)
      outs(%dst : memref<2x1x5xf32>)
  return
}
