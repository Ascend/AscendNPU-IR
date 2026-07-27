// RUN: bishengir-opt -propagate-reshape="for-regbased=false" -allow-unregistered-dialect %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_collapsed_reduce
func.func @test_collapsed_reduce(%src: tensor<7x3x5x13xi32>) {
    %collapsed = tensor.collapse_shape %src [[0, 1, 2, 3]] : tensor<7x3x5x13xi32> into tensor<1365xi32>
    %1 = tensor.empty() : tensor<i32>
    // CHECK: %[[RED:.*]]:2 = hfusion.reduce_with_index {tie_break_left = true} <min> ins(%[[COLLAPSED:.*]]: tensor<1365xi32>) outs(%[[VAR:.*]], %[[VAR:.*]] : tensor<i32>, tensor<i32>) dimensions = {{\[}}0] -> tensor<i32>, tensor<i32>
    %2:2 = hfusion.reduce_with_index {tie_break_left = true} <min> ins(%collapsed : tensor<1365xi32>) outs(%1, %1 : tensor<i32>, tensor<i32>) dimensions = [0] -> tensor<i32>, tensor<i32>
    "some_use"(%2#1) : (tensor<i32>) -> ()
    return
}

// CHECK-LABEL: func.func @test_propagated_collapsed_reduce
func.func @test_propagated_collapsed_reduce(%src: memref<7x3x5x13xi32>) {
    %0 = bufferization.to_tensor %src restrict writable : memref<7x3x5x13xi32>
    %collapsed = tensor.collapse_shape %0 [[0, 1, 2, 3]] : tensor<7x3x5x13xi32> into tensor<1365xi32>
    %1 = tensor.empty() : tensor<i32>
    // CHECK: %[[RED:.*]]:2 = hfusion.reduce_with_index {tie_break_left = true} <min> ins(%[[TENSOR:.*]] : tensor<7x3x5x13xi32>) outs(%[[VAR:.*]], %[[VAR:.*]] : tensor<i32>, tensor<i32>) dimensions = {{\[}}0, 1, 2, 3] -> tensor<i32>, tensor<i32>
    %2:2 = hfusion.reduce_with_index {tie_break_left = true} <min> ins(%collapsed : tensor<1365xi32>) outs(%1, %1 : tensor<i32>, tensor<i32>) dimensions = [0] -> tensor<i32>, tensor<i32>
    "some_use"(%2#0) : (tensor<i32>) -> ()
    return
}

// -----

// CHECK-LABEL: func.func @skip_unit_dim_collapse_insert_slice
func.func @skip_unit_dim_collapse_insert_slice(
    %src: tensor<2xi32>, %dest: tensor<2xi32>) -> tensor<2xi32> {
  // Match the crash reproducer: deinterleave produces tensor<1xi32>, which is
  // collapsed to a scalar and inserted into a rank-1 destination.
  // CHECK: %[[DEINTERLEAVED:.*]] = hfusion.deinterleave
  %deinterleaved = hfusion.deinterleave %src channel<0> :
      tensor<2xi32> -> tensor<1xi32>
  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[DEINTERLEAVED]] []
  %collapsed = tensor.collapse_shape %deinterleaved [] :
      tensor<1xi32> into tensor<i32>
  // CHECK: tensor.insert_slice %[[COLLAPSED]] into %arg1[0] [1] [1]
  %inserted = tensor.insert_slice %collapsed into %dest[0] [1] [1] :
      tensor<i32> into tensor<2xi32>
  return %inserted : tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @skip_collapse_through_rank_reducing_extract
func.func @skip_collapse_through_rank_reducing_extract(
    %src: tensor<2x3xi32>, %init: tensor<2x3xi32>) -> tensor<i32> {
  %filled = linalg.copy ins(%src : tensor<2x3xi32>)
      outs(%init : tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1]]
  %collapsed = tensor.collapse_shape %filled [[0, 1]] :
      tensor<2x3xi32> into tensor<6xi32>
  // CHECK: tensor.extract_slice %[[COLLAPSED]][0] [1] [1]
  %slice = tensor.extract_slice %collapsed[0] [1] [1] :
      tensor<6xi32> to tensor<i32>
  return %slice : tensor<i32>
}