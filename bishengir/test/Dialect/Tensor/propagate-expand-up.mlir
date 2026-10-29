// RUN: bishengir-opt -propagate-reshape="for-regbased=false" -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// `PropagateExpandUp` deliberately does not lift `tensor.expand_shape` across
// `linalg.fill` (non-termination with collapse-down / concat). The expand must
// stay on the fill result.
//
// CHECK-LABEL: func.func @no_expand_through_fill
func.func @no_expand_through_fill() {
  %cst = arith.constant 1.000000e+00 : f32
  %empty = tensor.empty() : tensor<6xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<6xf32>) -> tensor<6xf32>
  %expanded = tensor.expand_shape %fill [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  "some_use"(%expanded) : (tensor<2x3xf32>) -> ()
  return
}
// CHECK: linalg.fill
// CHECK-NEXT: tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>

// -----

// Rank-reducing slices need a reassociation adjusted for dropped dimensions.
// Even this unit-only expand must therefore remain below the extract.
// CHECK-LABEL: func.func @no_unit_expand_through_rank_reducing_extract
func.func @no_unit_expand_through_rank_reducing_extract(
    %src: tensor<2xi32>) -> tensor<1xi32> {
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %arg0[0] [1] [1]
  %slice = tensor.extract_slice %src[0] [1] [1] :
      tensor<2xi32> to tensor<i32>
  // CHECK: tensor.expand_shape %[[SLICE]] [] output_shape [1]
  %expanded = tensor.expand_shape %slice [] output_shape [1] :
      tensor<i32> into tensor<1xi32>
  return %expanded : tensor<1xi32>
}

// -----

// The insert drops the destination dimension selected by its unit-size slice.
// Keep the expand below it rather than reusing an invalid reassociation.
// CHECK-LABEL: func.func @no_expand_through_rank_reducing_insert
func.func @no_expand_through_rank_reducing_insert(
    %src: tensor<i32>, %dest: tensor<2xi32>) -> tensor<2x1xi32> {
  // CHECK: %[[INSERTED:.*]] = tensor.insert_slice %arg0 into %arg1[0] [1] [1]
  %inserted = tensor.insert_slice %src into %dest[0] [1] [1] :
      tensor<i32> into tensor<2xi32>
  // CHECK: tensor.expand_shape %[[INSERTED]] {{\[\[}}0, 1]] output_shape [2, 1]
  %expanded = tensor.expand_shape %inserted [[0, 1]] output_shape [2, 1] :
      tensor<2xi32> into tensor<2x1xi32>
  return %expanded : tensor<2x1xi32>
}

// -----

// A load -> expand -> collapse -> elementwise -> store chain. The expand's only
// user is a collapse, but the expand must still lift across the to_tensor
// boundary so it lands as a memref.expand_shape on the load side.
// CHECK-LABEL: func.func @expand_collapse_load_store
// CHECK: %[[EXPANDED:.*]] = memref.expand_shape %arg0 {{\[\[}}0], [1, 2]]
// CHECK-SAME: output_shape [1, 2, 2] : memref<1x4xf32> into memref<1x2x2xf32>
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[EXPANDED]] : memref<1x2x2xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[TENSOR]] {{\[\[}}0, 1], [2]] : tensor<1x2x2xf32> into tensor<2x2xf32>
// CHECK-NOT: tensor.expand_shape
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %arg1 : (tensor<2x2xf32>, memref<2x2xf32>) -> ()
func.func @expand_collapse_load_store(%src: memref<1x4xf32>, %dst: memref<2x2xf32>) {
  %t = bufferization.to_tensor %src : memref<1x4xf32>
  %expanded = tensor.expand_shape %t [[0], [1, 2]] output_shape [1, 2, 2] : tensor<1x4xf32> into tensor<1x2x2xf32>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2]] : tensor<1x2x2xf32> into tensor<2x2xf32>
  %empty = tensor.empty() : tensor<2x2xf32>
  %res = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%collapsed : tensor<2x2xf32>) outs(%empty : tensor<2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x2xf32>
  bufferization.materialize_in_destination %res in writable %dst : (tensor<2x2xf32>, memref<2x2xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @no_unit_expand_through_to_tensor
// CHECK: %[[T:.*]] = bufferization.to_tensor %arg0 : memref<64x1xf32>
// CHECK-NEXT: %[[E:.*]] = tensor.expand_shape %[[T]] {{\[\[}}0, 1], [2]]
// CHECK-SAME: output_shape [1, 64, 1] : tensor<64x1xf32> into tensor<1x64x1xf32>
// CHECK-NEXT: %[[C:.*]] = tensor.collapse_shape %[[E]] {{\[\[}}0], [1, 2]] : tensor<1x64x1xf32> into tensor<1x64xf32>
// CHECK-NOT: memref.expand_shape
// CHECK: return %[[C]] : tensor<1x64xf32>
func.func @no_unit_expand_through_to_tensor(%src: memref<64x1xf32>) -> tensor<1x64xf32> {
  %t = bufferization.to_tensor %src : memref<64x1xf32>
  %expanded = tensor.expand_shape %t [[0, 1], [2]] output_shape [1, 64, 1] : tensor<64x1xf32> into tensor<1x64x1xf32>
  %collapsed = tensor.collapse_shape %expanded [[0], [1, 2]] : tensor<1x64x1xf32> into tensor<1x64xf32>
  return %collapsed : tensor<1x64xf32>
}
