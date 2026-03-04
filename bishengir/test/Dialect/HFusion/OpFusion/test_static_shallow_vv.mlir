// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="SHALLOW_VV" -hfusion-fuse-ops -split-input-file %s | FileCheck %s --check-prefix=SHALLOW-VV
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="SHALLOW_VV" -hfusion-fuse-ops -split-input-file %s | FileCheck %s --check-prefix=SHALLOW-VV-ASSERT-MATMUL

// SHALLOW-VV-ASSERT-MATMUL-LABEL: func.func @testA_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV-ASSERT-MATMUL-NOT: linalg.matmul
// SHALLOW-VV-ASSERT-MATMUL-LABEL: func.func @testA(

// SHALLOW-VV-LABEL: func.func @testA_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV: linalg.elemwise_unary
// SHALLOW-VV: linalg.elemwise_binary
// SHALLOW-VV: linalg.elemwise_unary
// SHALLOW-VV: linalg.elemwise_unary
// SHALLOW-VV: linalg.broadcast
// SHALLOW-VV-LABEL: func.func @testA(
// SHALLOW-VV: call @testA_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV: linalg.matmul
func.func @testA(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty() : tensor<7x7xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%arg2 : tensor<7x7xf32>) outs(%2 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %4 = tensor.empty() : tensor<7x7xf32>
  %5 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %3 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%4 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %6 = tensor.empty() : tensor<7x7xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<7x7xf32>) outs(%6 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %8 = tensor.empty() : tensor<7x7xf32>
  %9 = linalg.matmul ins(%arg2, %7 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%8 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %10 = tensor.empty() : tensor<7x7xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%7 : tensor<7x7xf32>) outs(%10 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %12 = tensor.empty() : tensor<7x7x7xf32>
  %13 = linalg.broadcast ins(%arg2 : tensor<7x7xf32>) outs(%12: tensor<7x7x7xf32>) dimensions = [0]
  %14 = tensor.empty() : tensor<7x7xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<7x7xf32>) outs(%14 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %16 = tensor.empty() : tensor<7x7x7x7xf32>
  %17 = linalg.broadcast ins(%13 : tensor<7x7x7xf32>) outs(%16: tensor<7x7x7x7xf32>) dimensions = [3]
  %18 = tensor.empty() : tensor<7x7xf32>
  %19 = linalg.transpose ins(%15 : tensor<7x7xf32>) outs(%18 : tensor<7x7xf32>) permutation = [0, 1]
  return %arg1, %9, %11, %13, %15, %17, %19 : tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7xf32>
}

// -----

// SHALLOW-VV-ASSERT-MATMUL-LABEL: func.func @testB_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV-ASSERT-MATMUL-NOT: linalg.matmul
// SHALLOW-VV-ASSERT-MATMUL-LABEL: func.func @testB(

// SHALLOW-VV-LABEL: func.func @testB_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV: linalg.broadcast
// SHALLOW-VV: linalg.elemwise_unary
// SHALLOW-VV: linalg.elemwise_binary
// SHALLOW-VV-LABEL: func.func @testB(
// SHALLOW-VV: call @testB_multi_LAST_AXIS_PBR_0
// SHALLOW-VV: linalg.matmul
func.func @testB(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty() : tensor<7x7xf32>
  %3 = linalg.transpose ins(%arg2 : tensor<7x7xf32>) outs(%2 : tensor<7x7xf32>) permutation = [0, 1]
  %4 = tensor.empty() : tensor<7x7xf32>
  %5 = linalg.transpose ins(%arg2 : tensor<7x7xf32>) outs(%4 : tensor<7x7xf32>) permutation = [0, 1]
  %6 = tensor.empty() : tensor<7x7xf32>
  %7 = linalg.matmul ins(%5, %5 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%6 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %8 = tensor.empty() : tensor<7x7x7xf32>
  %9 = linalg.broadcast ins(%arg2 : tensor<7x7xf32>) outs(%8: tensor<7x7x7xf32>) dimensions = [2]
  %10 = tensor.empty() : tensor<7x7xf32>
  %11 = linalg.transpose ins(%7 : tensor<7x7xf32>) outs(%10 : tensor<7x7xf32>) permutation = [0, 1]
  %12 = tensor.empty() : tensor<7x7x7xf32>
  %13 = linalg.broadcast ins(%7 : tensor<7x7xf32>) outs(%12: tensor<7x7x7xf32>) dimensions = [2]
  %14 = tensor.empty() : tensor<7x7xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg2 : tensor<7x7xf32>) outs(%14 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %16 = tensor.empty() : tensor<7x7xf32>
  %17 = linalg.matmul ins(%15, %7 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%16 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %18 = tensor.empty() : tensor<7x7xf32>
  %19 = linalg.elemwise_binary {min_signed, fun = #linalg.binary_fn<min_signed>} ins(%5, %15 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%18 : tensor<7x7xf32>) -> tensor<7x7xf32>
  return %arg0, %arg1, %3, %9, %11, %13, %17, %19 : tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>
}

// -----

// SHALLOW-VV-ASSERT-MATMUL-LABEL: func.func @testC_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV-ASSERT-MATMUL-NOT: linalg.matmul
// SHALLOW-VV-ASSERT-MATMUL-LABEL: func.func @testC(

// SHALLOW-VV-LABEL: func.func @testC_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV: linalg.broadcast
// SHALLOW-VV: linalg.broadcast
// SHALLOW-VV: linalg.elemwise_binary
// SHALLOW-VV: linalg.elemwise_unary
// SHALLOW-VV: return
// SHALLOW-VV-LABEL: func.func @testC(
// SHALLOW-VV: linalg.matmul
// SHALLOW-VV: call @testC_multi_LAST_AXIS_PBR_0
// SHALLOW-VV: linalg.reduce
// SHALLOW-VV: return
func.func @testC(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty() : tensor<7x7xf32>
  %3 = linalg.matmul ins(%arg1, %arg2 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%2 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %4 = tensor.empty() : tensor<7x7xf32>
  %5 = linalg.transpose ins(%arg2 : tensor<7x7xf32>) outs(%4 : tensor<7x7xf32>) permutation = [0, 1]
  %6 = tensor.empty() : tensor<7x7x7xf32>
  %7 = linalg.broadcast ins(%3 : tensor<7x7xf32>) outs(%6: tensor<7x7x7xf32>) dimensions = [2]
  %8 = tensor.empty() : tensor<7x7x7xf32>
  %9 = linalg.broadcast ins(%5 : tensor<7x7xf32>) outs(%8: tensor<7x7x7xf32>) dimensions = [2]
  %10 = tensor.empty() : tensor<7x7xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<7x7xf32>) outs(%10 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %12 = tensor.empty() : tensor<7x7x7xf32>
  %13 = linalg.elemwise_binary {min_signed, fun = #linalg.binary_fn<min_signed>} ins(%9, %7 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%12 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %14 = tensor.empty() : tensor<7x7xf32>
  %15 = linalg.reduce {arith.addf} ins(%9 : tensor<7x7x7xf32>) outs(%14: tensor<7x7xf32>) dimensions = [0]
  %16 = tensor.empty() : tensor<7x7x7x7xf32>
  %17 = linalg.broadcast ins(%9 : tensor<7x7x7xf32>) outs(%16: tensor<7x7x7x7xf32>) dimensions = [0]
  %18 = tensor.empty() : tensor<7x7x7xf32>
  %19 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%9 : tensor<7x7x7xf32>) outs(%18 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  return %arg0, %3, %11, %13, %15, %17, %19 : tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>
}

// -----
// SHALLOW-VV-LABEL: testD_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV: linalg.reduce
// SHALLOW-VV: linalg.broadcast
// SHALLOW-VV: linalg.elemwise_unary
// SHALLOW-VV: return
// SHALLOW-VV-LABEL: testD(
// SHALLOW-VV: call @testD_multi_LAST_AXIS_PBR_0
// SHALLOW-VV: return
module {
  func.func @testD(%arg0: tensor<7x4096xf16>, %arg1: tensor<7xf16>) -> (tensor<7xf16>, tensor<7x1xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = tensor.empty() : tensor<7x1xf16>
    %reduced = linalg.reduce { arith.addf } ins(%arg0 : tensor<7x4096xf16>) outs(%arg1 : tensor<7xf16>) dimensions = [1]
    %broadcasted = linalg.broadcast ins(%reduced : tensor<7xf16>) outs(%0 : tensor<7x1xf16>) dimensions = [1]
    %1 = tensor.empty() : tensor<7x1xf16>
    %logged = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%broadcasted : tensor<7x1xf16>) outs(%1 : tensor<7x1xf16>) -> tensor<7x1xf16>
    return %reduced, %logged : tensor<7xf16>, tensor<7x1xf16>
  }
}

// -----
// SHALLOW-VV-LABEL: testE_multi_LAST_AXIS_PBR_0(
// SHALLOW-VV: linalg.reduce
// SHALLOW-VV: linalg.broadcast
// SHALLOW-VV: linalg.elemwise_unary
// SHALLOW-VV: return
// SHALLOW-VV-LABEL: testE(
// SHALLOW-VV: call @testE_multi_LAST_AXIS_PBR
// SHALLOW-VV: return
module {
  func.func @testE(%arg0: tensor<7x4096xf16>, %arg1: tensor<7xf16>) -> (tensor<7xf16>, tensor<7x1xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_VV>} {
    %0 = tensor.empty() : tensor<7x1xf16>
    %reduced = linalg.reduce { arith.addf } ins(%arg0 : tensor<7x4096xf16>) outs(%arg1 : tensor<7xf16>) dimensions = [1]
    %broadcasted = linalg.broadcast ins(%reduced : tensor<7xf16>) outs(%0 : tensor<7x1xf16>) dimensions = [1]
    %1 = tensor.empty() : tensor<7x1xf16>
    %logged = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%broadcasted : tensor<7x1xf16>) outs(%1 : tensor<7x1xf16>) -> tensor<7x1xf16>
    return %reduced, %logged : tensor<7xf16>, tensor<7x1xf16>
  }
}