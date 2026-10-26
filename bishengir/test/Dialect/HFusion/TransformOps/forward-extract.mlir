// RUN: bishengir-opt %s -transform-interpreter | FileCheck %s --enable-var-scope

module attributes {transform.with_named_sequence} {

// Forward through multiple non-overlapping inserts.
// CHECK-LABEL: @forward_through_multiple_inserts(
// CHECK-SAME: %[[BASE:[^:]+]]:
// CHECK-NEXT: %[[E:.*]] = tensor.extract_slice %[[BASE]][6] [2] [1]
// CHECK-NEXT: return %[[E]]
func.func @forward_through_multiple_inserts(%base: tensor<8xf32>, %part: tensor<2xf32>) -> tensor<2xf32> {
  %i0 = tensor.insert_slice %part into %base[0] [2] [1] : tensor<2xf32> into tensor<8xf32>
  %i1 = tensor.insert_slice %part into %i0[2] [2] [1] : tensor<2xf32> into tensor<8xf32>
  %i2 = tensor.insert_slice %part into %i1[4] [2] [1] : tensor<2xf32> into tensor<8xf32>
  %e = tensor.extract_slice %i2[6] [2] [1] : tensor<8xf32> to tensor<2xf32>
  return %e : tensor<2xf32>
}

// Forward through a non-overlapping insert with a dynamic slice size.
// CHECK-LABEL: @skip_non_overlapping_insert(
// CHECK-SAME: %[[BASE:[^:]+]]:
// CHECK: %[[I:.*]] = tensor.insert_slice
// CHECK: %[[E:.*]] = tensor.extract_slice %[[BASE]][1, 0, 0]
// CHECK: return %[[E]], %[[I]]
func.func @skip_non_overlapping_insert(%base: tensor<2x1x?xf32>, %row0: tensor<1x1x?xf32>, %n: index) -> (tensor<1x1x?xf32>, tensor<2x1x?xf32>) {
  %i = tensor.insert_slice %row0 into %base[0, 0, 0] [1, 1, %n] [1, 1, 1] : tensor<1x1x?xf32> into tensor<2x1x?xf32>
  %e = tensor.extract_slice %i[1, 0, 0] [1, 1, %n] [1, 1, 1] : tensor<2x1x?xf32> to tensor<1x1x?xf32>
  return %e, %i : tensor<1x1x?xf32>, tensor<2x1x?xf32>
}

// Skip forwarding past an overlapping write.
// CHECK-LABEL: @partial_overlap(
// CHECK: %[[I:.*]] = tensor.insert_slice
// CHECK: tensor.extract_slice %[[I]][1] [2] [1]
func.func @partial_overlap(%base: tensor<8xf32>, %part: tensor<3xf32>) -> tensor<2xf32> {
  %i = tensor.insert_slice %part into %base[0] [3] [1] : tensor<3xf32> into tensor<8xf32>
  %e = tensor.extract_slice %i[1] [2] [1] : tensor<8xf32> to tensor<2xf32>
  return %e : tensor<2xf32>
}

// Skip forwarding when overlap is unknown.
// CHECK-LABEL: @unknown_overlap(
// CHECK: %[[I:.*]] = tensor.insert_slice
// CHECK: tensor.extract_slice %[[I]][0] [2] [1]
func.func @unknown_overlap(%base: tensor<8xf32>, %part: tensor<2xf32>, %offset: index) -> tensor<2xf32> {
  %i = tensor.insert_slice %part into %base[%offset] [2] [1] : tensor<2xf32> into tensor<8xf32>
  %e = tensor.extract_slice %i[0] [2] [1] : tensor<8xf32> to tensor<2xf32>
  return %e : tensor<2xf32>
}

// Check that the two patterns work together.
// First, bubble the extract through the expand. Then apply ForwardExtract.
// CHECK-LABEL: @expand_then_forward(
// CHECK-SAME: %[[BASE:[^:]+]]:
// CHECK-NOT: tensor.insert_slice
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[BASE]][5] [1] [1]
// CHECK: %[[EXPAND:.*]] = tensor.expand_shape %[[SLICE]]
func.func @expand_then_forward(%base: tensor<6xf32>, %part: tensor<1xf32>) -> tensor<1x1xf32> {
  %i = tensor.insert_slice %part into %base[0] [1] [1] : tensor<1xf32> into tensor<6xf32>
  %expanded = tensor.expand_shape %i [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  %e = tensor.extract_slice %expanded[1, 2] [1, 1] [1, 1] : tensor<2x3xf32> to tensor<1x1xf32>
  return %e : tensor<1x1xf32>
}

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.hfusion.forward_extract_slice
    } : !transform.any_op
    transform.yield
  }
}
