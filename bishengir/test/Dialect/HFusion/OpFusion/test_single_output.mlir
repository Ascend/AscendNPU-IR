// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=single' %s | FileCheck %s --check-prefix=SINGLE
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=single-aggr' %s | FileCheck %s --check-prefix=SINGLEAGGR
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=multi' %s | FileCheck %s --check-prefix=MULTI
// RUN: bishengir-opt -hfusion-outline-single-op %s | FileCheck %s --check-prefix=SINGLEFUSE


// MULTI-LABEL: func.func @testSingle_0(
// MULTI: floor
// MULTI: min_signed
// MULTI: max_signed
// MULTI: floor
// MULTI: return
// MULTI-NOT: func.func @testSingle_1(
// MULTI-LABEL: func.func @testSingle(
// MULTI-NOT: linalg.elemwise
// MULTI: return


// SINGLE-LABEL: func.func @testSingle_0(
// SINGLE: max_signed
// SINGLE: floor
// SINGLE: return
// SINGLE: transpose
// SINGLE-NOT: func.func @testSingle_1(
// SINGLE-LABEL: func.func @testSingle(
// SINGLE: floor
// SINGLE: min_signed
// SINGLE: transpose
// SINGLE-NOT: linalg.elemwise
// SINGLE: return

// SINGLEAGGR-LABEL: func.func @testSingle_0(
// SINGLEAGGR: floor
// SINGLEAGGR: min_signed
// SINGLEAGGR: max_signed
// SINGLEAGGR: floor
// SINGLEAGGR: transpose
// SINGLEAGGR: return
// SINGLEAGGR-LABEL: func.func @testSingle_1(
// SINGLEAGGR: floor
// SINGLEAGGR: min_signed
// SINGLEAGGR: transpose
// SINGLEAGGR: return
// SINGLEAGGR-LABEL: func.func @testSingle(
// SINGLE-NOT: linalg.elemwise
// SINGLEAGGR: return
func.func @testSingle(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %0 = tensor.empty() : tensor<3x3xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%arg0 : tensor<3x3xf32>) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %2 = tensor.empty() : tensor<3x3xf32>
  %3 = linalg.elemwise_binary {min_signed, fun = #linalg.binary_fn<min_signed>} ins(%1, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%2 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %4 = tensor.empty() : tensor<3x3xf32>
  %5 = linalg.elemwise_binary {max_signed, fun = #linalg.binary_fn<max_signed>} ins(%1, %3 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%4 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %6 = tensor.empty() : tensor<3x3xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%5 : tensor<3x3xf32>) outs(%6 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %8 = tensor.empty() : tensor<3x3xf32>
  %9 = linalg.transpose ins(%7 : tensor<3x3xf32>) outs(%8 : tensor<3x3xf32>) permutation = [0, 1] 
  %10 = tensor.empty() : tensor<3x3xf32>
  %11 = linalg.transpose ins(%3 : tensor<3x3xf32>) outs(%10 : tensor<3x3xf32>) permutation = [0, 1]
  return %9, %11 : tensor<3x3xf32>, tensor<3x3xf32>
}

// SINGLEFUSE-LABEL: func.func @tileOtherBroadcast_single_outlined
// SINGLEFUSE-SAME: ANY_PB
// SINGLEFUSE: linalg.broadcast
// SINGLEFUSE: return
// SINGLEFUSE-LABEL: func.func @tileOtherBroadcast(
func.func @tileOtherBroadcast(%arg0: tensor<1x1x2xf32>, %arg1: tensor<1x2x2xf32>) -> tensor<1x2x2xf32> attributes {OperatorType = "Default", compute_capability = "", frontend_symbol = {input_0 = ["1", "1", "2"], output_0 = ["1", "2", "2"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x2xf32> into tensor<2xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<2xf32>) outs(%arg1 : tensor<1x2x2xf32>) dimensions = [0, 2] 
  return %broadcasted : tensor<1x2x2xf32>
}
// SINGLEFUSE-LABEL: func.func @mlir_fused_clone_0(
// SINGLEFUSE-NEXT: collapse_shape
// SINGLEFUSE-NEXT: call
// SINGLEFUSE-NEXT: return
// SINGLEFUSE-LABEL: func.func @mlir_fused_clone_0_single_outlined(
// SINGLEFUSE-SAME: DEVICE
// SINGLEFUSE-NEXT: return
func.func @mlir_fused_clone_0(%arg0: tensor<1x2047x4x512x1xf32>) -> tensor<1x2047x4x512xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2], [3, 4]] : tensor<1x2047x4x512x1xf32> into tensor<1x2047x4x512xf32>
  return %collapsed : tensor<1x2047x4x512xf32>
}