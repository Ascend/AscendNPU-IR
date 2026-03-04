// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" --hfusion-fuse-ops -split-input-file %s | FileCheck %s --check-prefix=ELEMWISE
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="LAST_AXIS_PBR" --hfusion-fuse-ops -split-input-file %s | FileCheck %s --check-prefix=LASTAXIS
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="SHALLOW_CV" --hfusion-fuse-ops -split-input-file %s | FileCheck %s --check-prefix=SHALLOW-CV
// RUN: bishengir-opt %s -test-buffer-utils -test-buffer-utils-var="enable-dma-opt" -split-input-file | FileCheck %s --check-prefix=MAXBUFF

// MAXBUFF: testCaseChainA: 10
// LASTAXIS-LABEL: func.func @testCaseChainA_0(
// LASTAXIS: elemwise_unary
// LASTAXIS: elemwise_unary
// LASTAXIS: broadcast
// LASTAXIS: elemwise_unary
// LASTAXIS: elemwise_unary
// LASTAXIS: reduce
// LASTAXIS: elemwise_unary
// ELEMWISE-LABEL: func.func @testCaseChainA_0(
// ELEMWISE: elemwise_unary
// ELEMWISE: elemwise_unary
// ELEMWISE-LABEL: func.func @testCaseChainA_1(
// ELEMWISE: elemwise_unary
// ELEMWISE: elemwise_unary
// SHALLOW-CV-LABEL: func.func @testCaseChainA(
// SHALLOW-CV-NOT: func.call
func.func @testCaseChainA(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%arg0 : tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%4 : tensor<?x?x?xf32>) outs(%5 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %7 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %8 = linalg.broadcast ins(%6 : tensor<?x?x?xf32>) outs(%7: tensor<?x?x?x?xf32>) dimensions = [3]
  %9 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %10 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%8 : tensor<?x?x?x?xf32>) outs(%9 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %11 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%10 : tensor<?x?x?x?xf32>) outs(%11 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %13 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %14 = linalg.reduce {arith.addf} ins(%12 : tensor<?x?x?x?xf32>) outs(%13: tensor<?x?x?xf32>) dimensions = [3]
  %15 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %16 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%14 : tensor<?x?x?xf32>) outs(%15 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %16 : tensor<?x?x?xf32>
}

// -----

// MAXBUFF: testCaseChainB: 3
// LASTAXIS-LABEL: func.func @testCaseChainB_0(
// LASTAXIS: reduce
// LASTAXIS: elemwise_unary
// LASTAXIS: elemwise_unary
// LASTAXIS-LABEL: func.func @testCaseChainB_1(
// LASTAXIS: reduce
// LASTAXIS: elemwise_unary
// ELEMWISE-LABEL: func.func @testCaseChainB_0(
// ELEMWISE: elemwise_unary
// ELEMWISE: elemwise_unary
// SHALLOW-CV-LABEL: func.func @testCaseChainB(
// SHALLOW-CV-NOT: func.call
func.func @testCaseChainB(%arg0: tensor<?x?x?xf32>) -> (tensor<?xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<?x?x?xf32>) outs(%3: tensor<?x?xf32>) dimensions = [2]
  %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%4 : tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%6 : tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = tensor.empty(%0) : tensor<?xf32>
  %10 = linalg.reduce {arith.addf} ins(%8 : tensor<?x?xf32>) outs(%9: tensor<?xf32>) dimensions = [1]
  %11 = tensor.empty(%0) : tensor<?xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%10 : tensor<?xf32>) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
  return %12 : tensor<?xf32>
}

// -----

// MAXBUFF: testCaseChainC: 10
// LASTAXIS-LABEL: func.func @testCaseChainC_0(
// LASTAXIS: broadcast
// LASTAXIS: elemwise_unary
// LASTAXIS: elemwise_unary
// LASTAXIS: broadcast
// LASTAXIS: elemwise_unary
// ELEMWISE-LABEL: func.func @testCaseChainC_0(
// ELEMWISE: elemwise_unary
// ELEMWISE: elemwise_unary
// SHALLOW-CV-LABEL: func.func @testCaseChainC(
// SHALLOW-CV-NOT: func.call
func.func @testCaseChainC(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?x?x?xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %4 = linalg.broadcast ins(%arg0 : tensor<?x?x?xf32>) outs(%3: tensor<?x?x?x?xf32>) dimensions = [3]
  %5 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%4 : tensor<?x?x?x?xf32>) outs(%5 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %7 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%6 : tensor<?x?x?x?xf32>) outs(%7 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %9 = tensor.empty(%0, %1, %2, %c3, %c3) : tensor<?x?x?x?x?xf32>
  %10 = linalg.broadcast ins(%8 : tensor<?x?x?x?xf32>) outs(%9: tensor<?x?x?x?x?xf32>) dimensions = [4]
  %11 = tensor.empty(%0, %1, %2, %c3, %c3) : tensor<?x?x?x?x?xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%10 : tensor<?x?x?x?x?xf32>) outs(%11 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %12 : tensor<?x?x?x?x?xf32>
}

// -----

// MAXBUFF: testCaseChainD: 10
// LASTAXIS-LABEL: func.func @testCaseChainD_0(
// LASTAXIS: reduce
// LASTAXIS: elemwise_unary
// LASTAXIS: elemwise_unary
// LASTAXIS: broadcast
// LASTAXIS: elemwise_unary
// LASTAXIS: elemwise_unary
// LASTAXIS: reduce
// LASTAXIS: broadcast
// LASTAXIS: elemwise_unary
// LASTAXIS: elemwise_unary
// LASTAXIS: reduce
// ELEMWISE-LABEL: func.func @testCaseChainD_0(
// ELEMWISE: elemwise_unary
// ELEMWISE: elemwise_unary
// ELEMWISE-LABEL: func.func @testCaseChainD_1(
// ELEMWISE: elemwise_unary
// ELEMWISE: elemwise_unary
// ELEMWISE-LABEL: func.func @testCaseChainD_2(
// ELEMWISE: elemwise_unary
// ELEMWISE: elemwise_unary
// SHALLOW-CV-LABEL: func.func @testCaseChainD(
// SHALLOW-CV-NOT: func.call
func.func @testCaseChainD(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<?x?x?xf32>) outs(%3: tensor<?x?xf32>) dimensions = [2]
  %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%4 : tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%6 : tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %10 = linalg.broadcast ins(%8 : tensor<?x?xf32>) outs(%9: tensor<?x?x?xf32>) dimensions = [2]
  %11 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%10 : tensor<?x?x?xf32>) outs(%11 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %13 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %14 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%12 : tensor<?x?x?xf32>) outs(%13 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %15 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %16 = linalg.reduce {arith.addf} ins(%14 : tensor<?x?x?xf32>) outs(%15: tensor<?x?xf32>) dimensions = [2]
  %17 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %18 = linalg.broadcast ins(%16 : tensor<?x?xf32>) outs(%17: tensor<?x?x?xf32>) dimensions = [2]
  %19 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %20 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%18 : tensor<?x?x?xf32>) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %21 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %22 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%20 : tensor<?x?x?xf32>) outs(%21 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %23 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %24 = linalg.reduce {arith.addf} ins(%22 : tensor<?x?x?xf32>) outs(%23: tensor<?x?xf32>) dimensions = [2]
  return %24 : tensor<?x?xf32>
}

// -----

// LASTAXIS-LABEL: @fuse_reduce_with_index_0(
// LASTAXIS: linalg.elemwise_unary
// LASTAXIS: hfusion.reduce_with_index
// LASTAXIS: linalg.elemwise_unary
// LASTAXIS-LABEL: @fuse_reduce_with_index(
// LASTAXIS: call @fuse_reduce_with_index_0
module {
  func.func @fuse_reduce_with_index(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = tensor.empty() : tensor<i32>
    %2 = tensor.empty() : tensor<4xf32>
    %3 = tensor.empty() : tensor<f32>
    %4 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %5 = linalg.fill ins(%cst : f32) outs(%1 : tensor<i32>) -> tensor<i32>
    %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<4xf32>) outs(%2 : tensor<4xf32>) -> tensor<4xf32>
    %7:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%6, %arg1 : tensor<4xf32>, tensor<4xi32>) outs(%0, %1 : tensor<f32>, tensor<i32>) dimensions = [0]  -> tensor<f32>, tensor<i32>
    %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%7#0 : tensor<f32>) outs(%3 : tensor<f32>) -> tensor<f32>
    return %8 : tensor<f32>
  }
}

