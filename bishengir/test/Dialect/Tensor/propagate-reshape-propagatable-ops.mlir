// RUN: bishengir-opt -propagate-reshape -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// RUN: bishengir-opt -propagate-reshape="for-regbased=true" -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// Both collapsed operands must be replaced by their exact high-rank sources.
// CHECK-LABEL: func.func @mulext_through_collapse
// CHECK: %[[SOURCE:.*]] = hfusion.elemwise_binary
// CHECK-NOT: tensor.collapse_shape {{.*}} into tensor<6x4xi32>
// CHECK: %[[LOW:.*]], %[[HIGH:.*]] = hfusion.mulext %[[SOURCE]], %arg1 : tensor<2x3x4xi32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[LOW]]
// CHECK-SAME: : tensor<2x3x4xi32> into tensor<6x4xi32>
// CHECK: return %[[COLLAPSED]]
func.func @mulext_through_collapse(%arg0: tensor<2x3x4xi32>,
                                   %arg1: tensor<2x3x4xi32>) -> tensor<6x4xi32> {
  %0 = tensor.empty() : tensor<2x3x4xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>}
      ins(%arg0, %arg0 : tensor<2x3x4xi32>, tensor<2x3x4xi32>)
      outs(%0 : tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2]]
      : tensor<2x3x4xi32> into tensor<6x4xi32>
  %collapsed_arg = tensor.collapse_shape %arg1 [[0, 1], [2]]
      : tensor<2x3x4xi32> into tensor<6x4xi32>
  %low, %high = hfusion.mulext %collapsed, %collapsed_arg
      : tensor<6x4xi32>
  return %low : tensor<6x4xi32>
}

// -----

// Collapse with an extra user must stay for that user; MulExt still gets the
// high-rank source and emits a new collapse on its result.
// CHECK-LABEL: func.func @mulext_collapse_shared_user
// CHECK: %[[SOURCE:.*]] = hfusion.elemwise_binary
// CHECK-DAG: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[SOURCE]]
// CHECK-DAG: "test.keep"(%[[COLLAPSE]])
// CHECK-DAG: %[[LOW:.*]], %[[HIGH:.*]] = hfusion.mulext %[[SOURCE]], %arg1 : tensor<2x3x4xi32>
// CHECK-DAG: tensor.collapse_shape %[[LOW]]
// CHECK: return
func.func @mulext_collapse_shared_user(%arg0: tensor<2x3x4xi32>,
                                       %arg1: tensor<2x3x4xi32>)
    -> (tensor<6x4xi32>, tensor<6x4xi32>) {
  %0 = tensor.empty() : tensor<2x3x4xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>}
      ins(%arg0, %arg0 : tensor<2x3x4xi32>, tensor<2x3x4xi32>)
      outs(%0 : tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2]]
      : tensor<2x3x4xi32> into tensor<6x4xi32>
  %collapsed_arg = tensor.collapse_shape %arg1 [[0, 1], [2]]
      : tensor<2x3x4xi32> into tensor<6x4xi32>
  "test.keep"(%collapsed) : (tensor<6x4xi32>) -> ()
  %low, %high = hfusion.mulext %collapsed, %collapsed_arg
      : tensor<6x4xi32>
  return %low, %collapsed : tensor<6x4xi32>, tensor<6x4xi32>
}

// -----

// MarkOp has no results; collapse must be erased when mark was its only user.
// CHECK-LABEL: func.func @mark_through_collapse
// CHECK: %[[SOURCE:.*]] = math.absf %arg0 : tensor<2x3xf32>
// CHECK-NOT: tensor.collapse_shape
// CHECK: annotation.mark %[[SOURCE]] {overflow_mode = "saturate"} : tensor<2x3xf32>
func.func @mark_through_collapse(%arg0: tensor<2x3xf32>) {
  %source = math.absf %arg0 : tensor<2x3xf32>
  %collapsed = tensor.collapse_shape %source [[0, 1]]
      : tensor<2x3xf32> into tensor<6xf32>
  annotation.mark %collapsed {overflow_mode = "saturate"} : tensor<6xf32>
  return
}

// -----

// A shared collapse remains, but the mark must consume the exact source.
// CHECK-LABEL: func.func @mark_collapse_shared_user
// CHECK: %[[SOURCE:.*]] = math.absf %arg0 : tensor<2x3xf32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[SOURCE]]
// CHECK: "test.keep"(%[[COLLAPSE]])
// CHECK: annotation.mark %[[SOURCE]] {overflow_mode = "saturate"} : tensor<2x3xf32>
func.func @mark_collapse_shared_user(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %source = math.absf %arg0 : tensor<2x3xf32>
  %collapsed = tensor.collapse_shape %source [[0, 1]]
      : tensor<2x3xf32> into tensor<6xf32>
  "test.keep"(%collapsed) : (tensor<6xf32>) -> ()
  annotation.mark %collapsed {overflow_mode = "saturate"} : tensor<6xf32>
  return %collapsed : tensor<6xf32>
}

// -----

// The dedicated i1 cast pattern moves the cast above a single-use collapse and
// restores the requested result shape with one collapse after the cast.
// CHECK-LABEL: func.func @i1_cast_through_collapse
// CHECK: %[[SOURCE:.*]] = arith.cmpi slt, %arg0, %arg1 : tensor<2x3xi8>
// CHECK-NOT: tensor.collapse_shape
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}} ins(%[[SOURCE]] : tensor<2x3xi1>)
// CHECK-SAME: outs({{.*}} : tensor<2x3xi8>) -> tensor<2x3xi8>
// CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[CAST]] {{\[\[}}0, 1]]
// CHECK: return %[[RESULT]]
func.func @i1_cast_through_collapse(
    %arg0: tensor<2x3xi8>, %arg1: tensor<2x3xi8>) -> tensor<6xi8> {
  %source = arith.cmpi slt, %arg0, %arg1 : tensor<2x3xi8>
  %collapsed = tensor.collapse_shape %source [[0, 1]]
      : tensor<2x3xi1> into tensor<6xi1>
  %out = tensor.empty() : tensor<6xi8>
  %cast = hfusion.cast {enable_overflow = true,
      round_mode = #hfusion.round_mode<trunc>}
      ins(%collapsed : tensor<6xi1>) outs(%out : tensor<6xi8>)
      -> tensor<6xi8>
  return %cast : tensor<6xi8>
}
