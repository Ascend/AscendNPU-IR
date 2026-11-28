// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false" | FileCheck %s
// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false use-preorder-walk-traversal=false" | FileCheck %s

// Dropped dimensions must not hide a source collapse boundary between retained
// dimensions. Otherwise the result is collapsed to tensor<4xi32> while the
// rebuilt slice still has sizes [2, 2].
// CHECK-LABEL: func.func @rank_reducing_slice_boundary
// CHECK-SAME: %[[ARG:.*]]: tensor<2x1x2x2xi32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG]] {{\[\[0\], \[1\], \[2, 3\]\]}}
// CHECK-SAME: into tensor<2x1x4xi32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[COLLAPSED]][0, 0, 0] [2, 1, 2] [1, 1, 1]
// CHECK-SAME: to tensor<2x2xi32>
// CHECK: return %[[SLICE]] : tensor<2x2xi32>
func.func @rank_reducing_slice_boundary(%arg: tensor<2x1x2x2xi32>) -> tensor<2x2xi32> {
  %slice = tensor.extract_slice %arg[0, 0, 0, 0] [2, 1, 1, 2] [1, 1, 1, 1] : tensor<2x1x2x2xi32> to tensor<2x2xi32>
  return %slice : tensor<2x2xi32>
}

// A boundary before a dropped axis must not split the contiguous suffix.
// CHECK-LABEL: func.func @insert_contiguous_suffix
// CHECK: tensor.insert_slice {{.*}}[0, 0] [12, 450] [1, 1] : tensor<12x450xi32> into tensor<12x900xi32>
func.func @insert_contiguous_suffix(%src: tensor<12x450xi32>, %dst: tensor<12x2x450xi32>) -> tensor<12x2x450xi32> {
  %slice = tensor.insert_slice %src into %dst[0, 0, 0] [12, 1, 450] [1, 1, 1] : tensor<12x450xi32> into tensor<12x2x450xi32>
  return %slice : tensor<12x2x450xi32>
}

// The same constraint applies to extraction.
// CHECK-LABEL: func.func @extract_contiguous_suffix
// CHECK: tensor.extract_slice {{.*}}[0, 0] [12, 450] [1, 1] : tensor<12x900xi32> to tensor<12x450xi32>
func.func @extract_contiguous_suffix(%src: tensor<12x2x450xi32>) -> tensor<12x450xi32> {
  %slice = tensor.extract_slice %src[0, 0, 0] [12, 1, 450] [1, 1, 1] : tensor<12x2x450xi32> to tensor<12x450xi32>
  return %slice : tensor<12x450xi32>
}

// A non-unit partial middle axis can merge with its full suffix.
// CHECK-LABEL: func.func @partial_middle
// CHECK: tensor.insert_slice {{.*}}[0, 5] [3, 10] [1, 1] : tensor<3x10xi32> into tensor<3x20xi32>
func.func @partial_middle(%src: tensor<3x2x5xi32>, %dst: tensor<3x4x5xi32>) -> tensor<3x4x5xi32> {
  %r = tensor.insert_slice %src into %dst[0, 1, 0] [3, 2, 5] [1, 1, 1] : tensor<3x2x5xi32> into tensor<3x4x5xi32>
  return %r : tensor<3x4x5xi32>
}

// Partial, strided suffixes must retain their indexing boundaries.
// CHECK-LABEL: func.func @partial_two_axes
// CHECK: tensor.insert_slice {{.*}}[0, 1, 1] [3, 2, 2] [1, 1, 2] : tensor<3x2x2xi32> into tensor<3x4x5xi32>
func.func @partial_two_axes(%src: tensor<3x2x2xi32>, %dst: tensor<3x4x5xi32>) -> tensor<3x4x5xi32> {
  %r = tensor.insert_slice %src into %dst[0, 1, 1] [3, 2, 2] [1, 1, 2] : tensor<3x2x2xi32> into tensor<3x4x5xi32>
  return %r : tensor<3x4x5xi32>
}

// Multiple dropped axes preserve the boundary and the suffix offset.
// CHECK-LABEL: func.func @multiple_dropped
// CHECK: tensor.insert_slice {{.*}}[0, 1, 5] [3, 1, 5] [1, 1, 1] : tensor<3x5xi32> into tensor<3x4x10xi32>
func.func @multiple_dropped(%src: tensor<3x5xi32>, %dst: tensor<3x4x2x5xi32>) -> tensor<3x4x2x5xi32> {
  %r = tensor.insert_slice %src into %dst[0, 1, 1, 0] [3, 1, 1, 5] [1, 1, 1, 1] : tensor<3x5xi32> into tensor<3x4x2x5xi32>
  return %r : tensor<3x4x2x5xi32>
}

// Collapsing axes [0, 1] multiplies the pivot stride 2 by extent 4.
// CHECK-LABEL: func.func @extract_nonunit_pivot_stride
// CHECK: tensor.extract_slice {{.*}}[0, 0] [3, 2] [8, 2] : tensor<20x5xi32> to tensor<3x2xi32>
func.func @extract_nonunit_pivot_stride(%arg: tensor<5x4x5xi32>) -> tensor<3x2xi32> {
  %slice = tensor.extract_slice %arg[0, 0, 0] [3, 1, 2] [2, 1, 2] : tensor<5x4x5xi32> to tensor<3x2xi32>
  return %slice : tensor<3x2xi32>
}

// CHECK-LABEL: func.func @insert_nonunit_pivot_stride
// CHECK: tensor.insert_slice {{.*}}[0, 0] [3, 2] [8, 2] : tensor<3x2xi32> into tensor<20x5xi32>
func.func @insert_nonunit_pivot_stride(%source: tensor<3x2xi32>, %dest: tensor<5x4x5xi32>) -> tensor<5x4x5xi32> {
  %result = tensor.insert_slice %source into %dest[0, 0, 0] [3, 1, 2] [2, 1, 2] : tensor<3x2xi32> into tensor<5x4x5xi32>
  return %result : tensor<5x4x5xi32>
}

// A dropped unit axis stride must not replace the retained pivot stride.
// CHECK-LABEL: func.func @subview_wrong_stride
// CHECK: memref.subview {{.*}}[0, 0, 0] [2, 5, 6] [1, 2, 1]
// CHECK-SAME: to memref<2x5x6xf32, strided<[60, 12, 1]>>
func.func @subview_wrong_stride(%m: memref<3x1x9x6xf32, strided<[60, 60, 6, 1]>>) -> tensor<2x5x6xf32> {
  %v = memref.subview %m[0, 0, 0, 0] [2, 1, 5, 6] [1, 7, 2, 1] : memref<3x1x9x6xf32, strided<[60, 60, 6, 1]>> to memref<2x5x6xf32, strided<[60, 12, 1]>>
  %t = bufferization.to_tensor %v : memref<2x5x6xf32, strided<[60, 12, 1]>>
  return %t : tensor<2x5x6xf32>
}

// Source-axis indices must not index a rank-reduced result layout mask.
// CHECK-LABEL: func.func @subview_oob_mask
// CHECK: memref.subview {{.*}}[0, 0] [2, 1] [1, 1]
// CHECK-SAME: to memref<2x1xf32, strided<[7, 1]>>
func.func @subview_oob_mask(%m: memref<2x2x3xf32, strided<[7, 3, 1]>>) -> f32 {
  %c0 = arith.constant 0 : index
  %v = memref.subview %m[0, 0, 0] [2, 1, 1] [1, 1, 1] : memref<2x2x3xf32, strided<[7, 3, 1]>> to memref<2x1xf32, strided<[7, 1]>>
  %x = memref.load %v[%c0, %c0] : memref<2x1xf32, strided<[7, 1]>>
  return %x : f32
}

// An unknown pivot stride prevents collapsing this group; preserve it intact.
// CHECK-LABEL: func.func @subview_dynamic_pivot
// CHECK: memref.subview {{.*}}[0, 0, 0, 0] [2, 1, 5, 6] [%{{.*}}, 7, 2, 1]
// CHECK-SAME: to memref<2x5x6xf32, strided<[?, 12, 1]>>
func.func @subview_dynamic_pivot(%m: memref<3x1x9x6xf32, strided<[60, 60, 6, 1]>>, %step: index) -> tensor<2x5x6xf32> {
  %v = memref.subview %m[0, 0, 0, 0] [2, 1, 5, 6] [%step, 7, 2, 1] : memref<3x1x9x6xf32, strided<[60, 60, 6, 1]>> to memref<2x5x6xf32, strided<[?, 12, 1]>>
  %t = bufferization.to_tensor %v : memref<2x5x6xf32, strided<[?, 12, 1]>>
  return %t : tensor<2x5x6xf32>
}

// Retain the outer singleton's layout when the inner singleton is dropped.
// CHECK-LABEL: func.func @subview_retained_outer_unit
// CHECK: memref.subview {{.*}}[0, 0] [2, 1] [1, 3]
// CHECK-SAME: to memref<2x1xf32, strided<[7, 3]>>
func.func @subview_retained_outer_unit(%m: memref<2x2x3xf32, strided<[7, 3, 1]>>) -> f32 {
  %c0 = arith.constant 0 : index
  %v = memref.subview %m[0, 0, 0] [2, 1, 1] [1, 1, 1] : memref<2x2x3xf32, strided<[7, 3, 1]>> to memref<2x1xf32, strided<[7, 3]>>
  %x = memref.load %v[%c0, %c0] : memref<2x1xf32, strided<[7, 3]>>
  return %x : f32
}

// Dynamic trailing source extents participate in the collapsed stride.
// CHECK-LABEL: func.func @dyn_unit_bridge
// CHECK: tensor.dim
// CHECK: %[[STEP:.*]] = arith.muli
// CHECK: tensor.extract_slice {{.*}}[0, 0] [2, 2] [%[[STEP]], 1]
// CHECK-SAME: to tensor<2x2xf32>
func.func @dyn_unit_bridge(%arg: tensor<4x?x4xf32>) -> tensor<2x2xf32> {
  %r = tensor.extract_slice %arg[0, 0, 0] [2, 1, 2] [1, 1, 1] : tensor<4x?x4xf32> to tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// Recover original-axis sizes after a producer slice has already collapsed.
// CHECK-LABEL: func.func @two_slices
// CHECK: tensor.extract_slice {{.*}}[0, 0] [12, %{{.*}}] [1, 1]
// CHECK: tensor.extract_slice {{.*}}[0, 0] [3, %{{.*}}] [4, 1]
// CHECK-SAME: to tensor<3x?xi32>
func.func @two_slices(%t: tensor<3x4x?xi32>, %n: index) -> tensor<3x?xi32> {
  %a = tensor.extract_slice %t[0, 0, 0] [3, 4, %n] [1, 1, 1] : tensor<3x4x?xi32> to tensor<3x4x?xi32>
  %b = tensor.extract_slice %a[0, 0, 0] [3, 1, %n] [1, 1, 1] : tensor<3x4x?xi32> to tensor<3x?xi32>
  return %b : tensor<3x?xi32>
}

// Runtime stride arithmetic must follow its operands and precede the slice.
// CHECK-LABEL: func.func @extract_dynamic_stride_dominance
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK: %[[STRIDE:.*]] = arith.muli %[[SELECT]], %[[FOUR]] : index
// CHECK: tensor.extract_slice {{.*}}[0] [3] [%[[STRIDE]]]
func.func @extract_dynamic_stride_dominance(%t: tensor<5x4xi32>, %pick: i1) -> tensor<3x1xi32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %s = arith.select %pick, %c2, %c1 : index
  %r = tensor.extract_slice %t[0, 0] [3, 1] [%s, 1] : tensor<5x4xi32> to tensor<3x1xi32>
  return %r : tensor<3x1xi32>
}

// A later collapse group must also materialize its stride before insertion.
// CHECK-LABEL: func.func @insert_dynamic_stride_dominance
// CHECK: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK: %[[STRIDE:.*]] = arith.muli %{{.*}}, %[[FOUR]] : index
// CHECK: tensor.insert_slice {{.*}}[0, 0] [2, 3] [1, %[[STRIDE]]]
func.func @insert_dynamic_stride_dominance(%src: tensor<2x3x1xi32>, %dst: tensor<2x5x4xi32>, %s: index) -> tensor<2x5x4xi32> {
  %r = tensor.insert_slice %src into %dst[0, 0, 0] [2, 3, 1] [1, %s, 1] : tensor<2x3x1xi32> into tensor<2x5x4xi32>
  return %r : tensor<2x5x4xi32>
}

// The producer drops its leading unit axis. Its consumer needs [3, 4, %n]
// in original result-axis order to rebuild the stride across the middle axis.
// CHECK-LABEL: func.func @stacked_rank_reducing_slice(
// CHECK-SAME: %[[N:[^:]+]]: index)
// CHECK: %[[FIRST:.*]] = tensor.extract_slice {{.*}} to tensor<12x?xi32>
// CHECK: %[[SIZE:.*]] = arith.muli %{{.*}}, %[[N]] : index
// CHECK: %[[SECOND:.*]] = tensor.extract_slice %[[FIRST]][0, 0] [3, %[[SIZE]]] [4, 1] : tensor<12x?xi32> to tensor<3x?xi32>
// CHECK: return %[[SECOND]]
func.func @stacked_rank_reducing_slice(%t: tensor<1x3x4x?xi32>, %n: index) -> tensor<3x?xi32> {
  %a = tensor.extract_slice %t[0, 0, 0, 0] [1, 3, 4, %n] [1, 1, 1, 1] : tensor<1x3x4x?xi32> to tensor<3x4x?xi32>
  %b = tensor.extract_slice %a[0, 0, 0] [3, 1, %n] [1, 1, 1] : tensor<3x4x?xi32> to tensor<3x?xi32>
  return %b : tensor<3x?xi32>
}

// The result collapses two retained unit axes with strides 12 and 4.
// Its layout uses the first one's stride, even though both select one element.
// CHECK-LABEL: func.func @subview_two_retained_units(
// CHECK: %[[VIEW:.*]] = memref.subview {{.*}}[0, 0] [2, 1] [1, 12] : memref<2x24xf32, strided<[31, 1]>> to memref<2x1xf32, strided<[31, 12]>>
// CHECK: memref.load %[[VIEW]]
func.func @subview_two_retained_units(%t: memref<2x2x3x4xf32, strided<[31, 12, 4, 1]>>) -> f32 {
  %z = arith.constant 0 : index
  %a = memref.subview %t[0, 0, 0, 0] [2, 1, 1, 1] [1, 1, 1, 1] : memref<2x2x3x4xf32, strided<[31, 12, 4, 1]>> to memref<2x1x1xf32, strided<[31, 12, 4]>>
  %v = memref.load %a[%z, %z, %z] : memref<2x1x1xf32, strided<[31, 12, 4]>>
  return %v : f32
}
