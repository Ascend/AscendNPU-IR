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
// SHALLOW-CV-NOT: func.func @testCaseChainA_0(
// SHALLOW-CV-LABEL: func.func @testCaseChainA(
// SHALLOW-CV: linalg
func.func @testCaseChainA(%arg0: tensor<7x7x7xf32>) -> (tensor<7x7x7xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%arg0 : tensor<7x7x7xf32>) outs(%3 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %5 = tensor.empty() : tensor<7x7x7xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%4 : tensor<7x7x7xf32>) outs(%5 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %7 = tensor.empty() : tensor<7x7x7x7xf32>
  %8 = linalg.broadcast ins(%6 : tensor<7x7x7xf32>) outs(%7: tensor<7x7x7x7xf32>) dimensions = [3]
  %9 = tensor.empty() : tensor<7x7x7x7xf32>
  %10 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%8 : tensor<7x7x7x7xf32>) outs(%9 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
  %11 = tensor.empty() : tensor<7x7x7x7xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%10 : tensor<7x7x7x7xf32>) outs(%11 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
  %13 = tensor.empty() : tensor<7x7x7xf32>
  %14 = linalg.reduce {arith.addf} ins(%12 : tensor<7x7x7x7xf32>) outs(%13: tensor<7x7x7xf32>) dimensions = [3]
  %15 = tensor.empty() : tensor<7x7x7xf32>
  %16 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%14 : tensor<7x7x7xf32>) outs(%15 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  return %16 : tensor<7x7x7xf32>
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
// SHALLOW-CV-NOT: func.func @testCaseChainB_0(
// SHALLOW-CV-LABEL: func.func @testCaseChainB(
// SHALLOW-CV: linalg
func.func @testCaseChainB(%arg0: tensor<7x7x7xf32>) -> (tensor<7xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<7x7x7xf32>) outs(%3: tensor<7x7xf32>) dimensions = [2]
  %5 = tensor.empty() : tensor<7x7xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%4 : tensor<7x7xf32>) outs(%5 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %7 = tensor.empty() : tensor<7x7xf32>
  %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%6 : tensor<7x7xf32>) outs(%7 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %9 = tensor.empty() : tensor<7xf32>
  %10 = linalg.reduce {arith.addf} ins(%8 : tensor<7x7xf32>) outs(%9: tensor<7xf32>) dimensions = [1]
  %11 = tensor.empty() : tensor<7xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%10 : tensor<7xf32>) outs(%11 : tensor<7xf32>) -> tensor<7xf32>
  return %12 : tensor<7xf32>
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
// SHALLOW-CV-NOT: func.func @testCaseChainC_0(
// SHALLOW-CV-LABEL: func.func @testCaseChainC(
// SHALLOW-CV: linalg
func.func @testCaseChainC(%arg0: tensor<7x7x7xf32>) -> (tensor<7x7x7x7x7xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7x7xf32>
  %4 = linalg.broadcast ins(%arg0 : tensor<7x7x7xf32>) outs(%3: tensor<7x7x7x7xf32>) dimensions = [3]
  %5 = tensor.empty() : tensor<7x7x7x7xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%4 : tensor<7x7x7x7xf32>) outs(%5 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
  %7 = tensor.empty() : tensor<7x7x7x7xf32>
  %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%6 : tensor<7x7x7x7xf32>) outs(%7 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
  %9 = tensor.empty() : tensor<7x7x7x7x7xf32>
  %10 = linalg.broadcast ins(%8 : tensor<7x7x7x7xf32>) outs(%9: tensor<7x7x7x7x7xf32>) dimensions = [4]
  %11 = tensor.empty() : tensor<7x7x7x7x7xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%10 : tensor<7x7x7x7x7xf32>) outs(%11 : tensor<7x7x7x7x7xf32>) -> tensor<7x7x7x7x7xf32>
  return %12 : tensor<7x7x7x7x7xf32>
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
// SHALLOW-CV-NOT: func.func @testCaseChainD_0(
// SHALLOW-CV-LABEL: func.func @testCaseChainD(
// SHALLOW-CV: linalg
func.func @testCaseChainD(%arg0: tensor<7x7x7xf32>) -> (tensor<7x7xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<7x7x7xf32>) outs(%3: tensor<7x7xf32>) dimensions = [2]
  %5 = tensor.empty() : tensor<7x7xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%4 : tensor<7x7xf32>) outs(%5 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %7 = tensor.empty() : tensor<7x7xf32>
  %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%6 : tensor<7x7xf32>) outs(%7 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %9 = tensor.empty() : tensor<7x7x7xf32>
  %10 = linalg.broadcast ins(%8 : tensor<7x7xf32>) outs(%9: tensor<7x7x7xf32>) dimensions = [2]
  %11 = tensor.empty() : tensor<7x7x7xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%10 : tensor<7x7x7xf32>) outs(%11 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %13 = tensor.empty() : tensor<7x7x7xf32>
  %14 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%12 : tensor<7x7x7xf32>) outs(%13 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %15 = tensor.empty() : tensor<7x7xf32>
  %16 = linalg.reduce {arith.addf} ins(%14 : tensor<7x7x7xf32>) outs(%15: tensor<7x7xf32>) dimensions = [2]
  %17 = tensor.empty() : tensor<7x7x7xf32>
  %18 = linalg.broadcast ins(%16 : tensor<7x7xf32>) outs(%17: tensor<7x7x7xf32>) dimensions = [2]
  %19 = tensor.empty() : tensor<7x7x7xf32>
  %20 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%18 : tensor<7x7x7xf32>) outs(%19 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %21 = tensor.empty() : tensor<7x7x7xf32>
  %22 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%20 : tensor<7x7x7xf32>) outs(%21 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %23 = tensor.empty() : tensor<7x7xf32>
  %24 = linalg.reduce {arith.addf} ins(%22 : tensor<7x7x7xf32>) outs(%23: tensor<7x7xf32>) dimensions = [2]
  return %24 : tensor<7x7xf32>
}
