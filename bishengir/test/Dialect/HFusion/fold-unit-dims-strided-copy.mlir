// RUN: bishengir-opt -hfusion-fold-unit-dims -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @fold_strided_mid_loop_copy
// CHECK-SAME:  (%[[ARG0:.*]]: memref<?xf16>, %[[ALLOC:.*]]: memref<1x1x2048xf16>, %[[S0:.*]]: index, %[[S1:.*]]: index, %[[O0:.*]]: index, %[[O1:.*]]: index) {
func.func @fold_strided_mid_loop_copy(
    %arg0: memref<?xf16>, 
    %alloc: memref<1x1x2048xf16>,
    %s0: index, %s1: index, 
    %o0: index, %o1: index) {
  
  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%o0], sizes: [1, %s0, %s1], strides: [%s0, %s1, 1] 
    : memref<?xf16> to memref<1x?x?xf16, strided<[?, ?, 1], offset: ?>>

  %subview_2 = memref.subview %reinterpret_cast_1 [0, 0, 0] [%s0, 1, %s1] [1, 1, 1]
    : memref<1x?x?xf16, strided<[?, ?, 1], offset: ?>> to memref<?x1x?xf16, strided<[?, ?, 1], offset: ?>>
  
  %subview_3 = memref.subview %alloc[%o0, 0, %o1] [%s0, 1, %s1] [1, 1, 1]
    : memref<1x1x2048xf16> to memref<?x1x?xf16, strided<[2048, 2048, 1], offset: ?>>
  
  memref.copy %subview_2, %subview_3 
    : memref<?x1x?xf16, strided<[?, ?, 1], offset: ?>> to memref<?x1x?xf16, strided<[2048, 2048, 1], offset: ?>>

  return
}

// CHECK:        %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [%[[O0]]], sizes: [%[[S0]], %[[S1]]], strides: [%[[S1]], 1] : memref<?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
// CHECK:        %[[SRC_COLLAPSED:.*]] = memref.subview %[[REINTERPRET]][0, 0] [%[[S0]], %[[S1]]] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
// CHECK:        %[[ALLOC_COLLAPSED:.*]] = memref.collapse_shape %[[ALLOC]] {{\[\[}}0, 1, 2{{\]\]}} : memref<1x1x2048xf16> into memref<2048xf16>
// CHECK:        %[[MUL:.*]] = arith.muli %[[S0]], %[[S1]] : index
// CHECK:        %[[DST_SUBVIEW:.*]] = memref.subview %[[ALLOC_COLLAPSED]][%[[O1]]] [%[[MUL]]] [1] : memref<2048xf16> to memref<?xf16, strided<[1], offset: ?>>
// CHECK:        %[[DST_EXPANDED:.*]] = memref.expand_shape %[[DST_SUBVIEW]] {{\[\[}}0, 1, 2{{\]\]}} output_shape [%[[S0]], 1, %[[S1]]] : memref<?xf16, strided<[1], offset: ?>> into memref<?x1x?xf16, strided<[?, ?, 1], offset: ?>>
// CHECK:        %[[DST_FINAL_COLLAPSED:.*]] = memref.collapse_shape %[[DST_EXPANDED]] {{\[\[}}0, 1{{\]}}, {{\[}}2{{\]\]}} : memref<?x1x?xf16, strided<[?, ?, 1], offset: ?>> into memref<?x?xf16, strided<[?, 1], offset: ?>>
// CHECK:        memref.copy %[[SRC_COLLAPSED]], %[[DST_FINAL_COLLAPSED]] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
