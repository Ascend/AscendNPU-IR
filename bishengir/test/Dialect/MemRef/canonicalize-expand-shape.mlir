// RUN: bishengir-opt %s -canonicalize-ext -cse -split-input-file | FileCheck %s

// -----
module {
// CHECK-LABEL: func @memref.expand_shape_constant_fold_arg
// CHECK: %[[cast_from_src:.*]] = memref.cast %[[arg:.*]] : memref<?xf32> to memref<1xf32>
// CHECK-NEXT: %[[expanded:.*]] = memref.expand_shape %[[cast_from_src]] {{\[\[}}0, 1]] output_shape [1, 1] : memref<1xf32> into memref<1x1xf32>
// CHECK-NEXT: %[[cast_to_dest:.*]] = memref.cast %[[expanded]] : memref<1x1xf32> to memref<?x1xf32>
// CHECK-NEXT: return %[[cast_to_dest]]
func.func @memref.expand_shape_constant_fold_arg(%src : memref<?xf32>) -> memref<?x1xf32> {
    %c1 = arith.constant 1 : index
    %expanded = memref.expand_shape %src [[0, 1]] output_shape [%c1, 1] : memref<?xf32> into memref<?x1xf32>
    return %expanded : memref<?x1xf32>
}
}

// -----
module {
// CHECK-LABEL: func @memref.expand_shape_constant_fold_after_collapsed
// CHECK-NOT: arith.constant
// CHECK: %[[collapsed:.*]] = memref.collapse_shape %[[arg:.*]] {{\[\[}}0, 1], {{\[}}2, 3]] : memref<?x?x3x4xf32> into memref<?x12xf32>
// CHECK-NEXT: %[[cast0:.*]] = memref.cast %[[arg:.*]] : memref<?x12xf32> to memref<1x12xf32>
// CHECK-NEXT: %[[expanded:.*]] = memref.expand_shape %[[cast0]] {{\[\[}}0, 1],  {{\[}}2, 3]] output_shape [1, 1, 2, 6] : memref<1x12xf32> into memref<1x1x2x6xf32>
// CHECK-NEXT: %[[cast1:.*]] = memref.cast %[[expanded]] : memref<1x1x2x6xf32> to memref<?x?x2x6xf32>
// CHECK-NEXT: return %[[cast1]]
func.func @memref.expand_shape_constant_fold_after_collapsed(%arg0 : memref<?x?x3x4xf32>) -> memref<?x?x2x6xf32>{
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %collapsed = memref.collapse_shape %arg0 [[0, 1], [2, 3]] : memref<?x?x3x4xf32> into memref<?x12xf32>
    %expanded = memref.expand_shape %collapsed [[0, 1], [2, 3]] output_shape [%c1, %c1, 2, 6] : memref<?x12xf32> into memref<?x?x2x6xf32>
    return %expanded : memref<?x?x2x6xf32>
}
}

// -----

// CHECK-LABEL: @fold_collapse_of_one_size_expand_success
func.func @fold_collapse_of_one_size_expand_success(%arg0: index, %arg1: index, %arg2: memref<1024xf32>) -> (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>)
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %subview = memref.subview %arg2[%arg0] [%arg1] [1] : memref<1024xf32> to memref<?xf32, strided<[1], offset: ?>>
    // CHECK-NOT: memref.expand_shape
    // CHECK-NOT: memref.collapse_shape
    %expand_shape = memref.expand_shape %subview [[0, 1, 2]] output_shape [1, 1, %arg1] : memref<?xf32, strided<[1], offset: ?>> into memref<1x1x?xf32, strided<[?, ?, 1], offset: ?>>
    %collapse_shape_0 = memref.collapse_shape %expand_shape [[0, 1, 2]] : memref<1x1x?xf32, strided<[?, ?, 1], offset: ?>> into memref<?xf32, strided<[1], offset: ?>>
    // CHECK-NOT: memref.expand_shape
    // CHECK-NOT: memref.collapse_shape
    %expand_shape_1 = memref.expand_shape %subview [[0, 1, 2]] output_shape [1, %arg1, 1] : memref<?xf32, strided<[1], offset: ?>> into memref<1x?x1xf32, strided<[?, 1, 1], offset: ?>>
    %collapse_shape_1 = memref.collapse_shape %expand_shape_1 [[0, 1, 2]] : memref<1x?x1xf32, strided<[?, 1, 1], offset: ?>> into memref<?xf32, strided<[1], offset: ?>>
    return %collapse_shape_0, %collapse_shape_1 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>
}

// -----

// CHECK-LABEL: @fold_collapse_of_one_size_expand_failure
func.func @fold_collapse_of_one_size_expand_failure(%arg0: index, %arg1: index, %arg2: memref<1024xf32>) -> (memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>)
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %subview = memref.subview %arg2[%arg0] [%arg1] [1] : memref<1024xf32> to memref<?xf32, strided<[1], offset: ?>>
    // CHECK: memref.expand_shape
    // CHECK: memref.collapse_shape
    %expand_shape = memref.expand_shape %subview [[0, 1, 2]] output_shape [1, 2, %arg1] : memref<?xf32, strided<[1], offset: ?>> into memref<1x2x?xf32, strided<[?, ?, 1], offset: ?>>
    %collapse_shape_0 = memref.collapse_shape %expand_shape [[0, 1, 2]] : memref<1x2x?xf32, strided<[?, ?, 1], offset: ?>> into memref<?xf32, strided<[?], offset: ?>>
    // CHECK: memref.expand_shape
    // CHECK: memref.collapse_shape
    %expand_shape_1 = memref.expand_shape %subview [[0, 1, 2]] output_shape [1, %arg0, %arg1] : memref<?xf32, strided<[1], offset: ?>> into memref<1x?x?xf32, strided<[?, ?, 1], offset: ?>>
    %collapse_shape_1 = memref.collapse_shape %expand_shape_1 [[0, 1, 2]] : memref<1x?x?xf32, strided<[?, ?, 1], offset: ?>> into memref<?xf32, strided<[?], offset: ?>>
    return %collapse_shape_0, %collapse_shape_1 : memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>
}