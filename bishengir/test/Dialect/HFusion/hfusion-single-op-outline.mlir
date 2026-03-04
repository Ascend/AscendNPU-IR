// RUN: bishengir-opt %s -hfusion-outline-single-op -split-input-file | FileCheck %s

// CHECK-LABEL: @mlir_fused_clone_0
// CHECK: #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>
// CHECK: func.func @mlir_fused_clone_0_single_outlined
// CHECK-SAME: hacc.entry
// CHECK-SAME: #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>
func.func @mlir_fused_clone_0(%arg0: tensor<1x2047x4x512x1xf32>) -> tensor<1x2047x4x512xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2], [3, 4]] : tensor<1x2047x4x512x1xf32> into tensor<1x2047x4x512xf32>
  return %collapsed : tensor<1x2047x4x512xf32>
}

// -----
// CHECK-LABEL: @test_concat_align_single_outlined_0_0(
// CHECK-SAME: DEVICE
// CHECK-SAME: PURE_ELEMWISE
// CHECK-LABEL: @test_concat_align(
// CHECK: #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>
// CHECK: call
// CHECK: return
module {
  func.func @test_concat_align(%arg0: tensor<136x2048xf32>, %arg1: tensor<136x2048xf32>, %arg2: tensor<136x1024xf32>, %arg3: tensor<136x2048xf32>) -> tensor<136x7168xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<136x2048xf32>
    %concat = tensor.concat dim(1) %0, %arg0, %arg1, %arg2 : (tensor<136x2048xf32>, tensor<136x2048xf32>, tensor<136x2048xf32>, tensor<136x1024xf32>) -> tensor<136x7168xf32>
    return %concat : tensor<136x7168xf32>
  }
}

// -----

// CHECK-LABEL: func.func @fuse_shared_aux_ops_after_fused_function_single_outlined_0_0(
// CHECK-NEXT: hfusion.cast
// CHECK-NEXT: linalg.elemwise_unary
// CHECK-NEXT: hfusion.cast
// CHECK-NEXT: linalg.elemwise_unary
// CHECK-NEXT: linalg.broadcast
// CHECK-NEXT: return
// CHECK-LABEL: func.func @fuse_shared_aux_ops_after_fused_function_single_outlined_1_0(
// CHECK-NEXT: linalg.reduce
// CHECK-NEXT: hfusion.cast
// CHECK-NEXT: return
// CHECK-LABEL: func.func @fuse_shared_aux_ops_after_fused_function_single_outlined_2_0(
// CHECK-NEXT: hfusion.cast
// CHECK-LABEL: func.func @fuse_shared_aux_ops_after_fused_function_single_outlined_3_0(
// CHECK: hfusion.cast
// CHECK-NEXT: return
// CHECK-LABEL: func.func @fuse_shared_aux_ops_after_fused_function(
// CHECK: call @fuse_shared_aux_ops_after_fused_function_single_outlined_0_0
// CHECK-NEXT: call @fuse_shared_aux_ops_after_fused_function_single_outlined_1_0
// CHECK: call @fuse_shared_aux_ops_after_fused_function_single_outlined_2_0
// CHECK-NEXT: call @fuse_shared_aux_ops_after_fused_function_single_outlined_3_0
// CHECK-NEXT: linalg.elemwise_binary
// CHECK-NEXT: return

func.func @fuse_shared_aux_ops_after_fused_function() -> (tensor<f32>, tensor<f32>, tensor<f32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %empty = tensor.empty() : tensor<f32>
  %empty_1 = tensor.empty() : tensor<f16>
  %empty_2 = tensor.empty() : tensor<f16>
  %empty_3 = tensor.empty() : tensor<f32>
  %empty_4 = tensor.empty() : tensor<f32>
  %empty_5 = tensor.empty() : tensor<1xf32>
  %empty_6 = tensor.empty() : tensor<f32>
  %empty_7 = tensor.empty() : tensor<f32>
  %empty_8 = tensor.empty() : tensor<f32>
  %0 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%empty : tensor<f32>) outs(%empty_1 : tensor<f16>) -> tensor<f16>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%0 : tensor<f16>) outs(%empty_2 : tensor<f16>) -> tensor<f16>
  %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<f16>) outs(%empty_3 : tensor<f32>) -> tensor<f32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%2 : tensor<f32>) outs(%empty_4 : tensor<f32>) -> tensor<f32>
  %4 = linalg.broadcast ins(%3 : tensor<f32>) outs(%empty_5: tensor<1xf32>) dimensions = [0]
  %sum = linalg.reduce {arith.addf} ins(%4 : tensor<1xf32>) outs(%empty_6 : tensor<f32>) dimensions = [0]
  %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%sum : tensor<f32>) outs(%empty_7 : tensor<f32>) -> tensor<f32>

  %empty_41 = tensor.empty() : tensor<f32>
  %empty_51 = tensor.empty() : tensor<1xf32>
  %empty_61 = tensor.empty() : tensor<f32>
  %empty_71 = tensor.empty() : tensor<f32>
  %31 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%2 : tensor<f32>) outs(%empty_41 : tensor<f32>) -> tensor<f32>
  %41 = linalg.broadcast ins(%31 : tensor<f32>) outs(%empty_51: tensor<1xf32>) dimensions = [0]
  %sum1 = linalg.reduce {arith.addf} ins(%41 : tensor<1xf32>) outs(%empty_61 : tensor<f32>) dimensions = [0]
  %51 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%sum1 : tensor<f32>) outs(%empty_71 : tensor<f32>) -> tensor<f32>
  // test case to not fuse the linalg.elemwise_binary because, it will cause dependency issue.
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %51 : tensor<f32>, tensor<f32>) outs(%empty_8 : tensor<f32>) -> tensor<f32>
  return %5, %51, %6 : tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: func.func @aux_connects_with_arg_single_outlined_0_0(
// CHECK-NEXT: linalg.reduce
// CHECK-NEXT: linalg.elemwise_binary
// CHECK-NEXT: return
// CHECK-LABEL: func.func @aux_connects_with_arg(
// CHECK: call @aux_connects_with_arg_single_outlined_0_0(
// CHECK-NEXT: return
// test case for null definingOp of an operand (arg0)
func.func @aux_connects_with_arg(%arg0 : tensor<f32>) -> (tensor<f32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %empty = tensor.empty() : tensor<1xf32>
  %empty_1 = tensor.empty() : tensor<f32>
  %5 = linalg.reduce {arith.addf} ins(%empty : tensor<1xf32>) outs(%empty_1 : tensor<f32>) dimensions = [0]
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %arg0 : tensor<f32>, tensor<f32>) outs(%empty_1 : tensor<f32>) -> tensor<f32>
  return %6 : tensor<f32>
}

// -----
// CHECK-LABEL: func.func @mlir_fused_clone_13_single_outlined_0_0(
// CHECK: tensor.extract_slice
// CHECK: return
// CHECK-LABEL: func.func @mlir_fused_clone_13(
// CHECK: call @mlir_fused_clone_13_single_outlined_0_0(
// CHECK: return
func.func @mlir_fused_clone_13(%arg0: tensor<?x1x9216xbf16>) -> tensor<?x1x3072xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %c0 = arith.constant 0 : index
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<?x1x9216xbf16> into tensor<?x9216xbf16>
  %dim = tensor.dim %arg0, %c0 : tensor<?x1x9216xbf16>
  %extracted_slice = tensor.extract_slice %collapsed[0, 6144] [%dim, 3072] [1, 1] : tensor<?x9216xbf16> to tensor<?x3072xbf16>
  %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%dim, 1, 3072] : tensor<?x3072xbf16> into tensor<?x1x3072xbf16>
  return %expanded : tensor<?x1x3072xbf16>
}