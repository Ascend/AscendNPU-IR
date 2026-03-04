// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="LAST_AXIS_PBR" --hfusion-fuse-ops="max-horizontal-fusion-size=-1" --split-input-file %s | FileCheck %s
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="SHALLOW_CV" --hfusion-fuse-ops="max-horizontal-fusion-size=-1" --split-input-file %s | FileCheck %s --check-prefix=SHALLOW-CV
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="ANY_PB" --hfusion-fuse-ops="max-horizontal-fusion-size=-1" --split-input-file %s | FileCheck %s --check-prefix=ANY-PB
// RUN: bishengir-opt -test-buffer-utils -test-buffer-utils-var="enable-dma-opt" -split-input-file %s | FileCheck %s --check-prefix=MAXBUFF

// CHECK-LABEL: func.func @testCaseSimple_0(
func.func @testCaseSimple(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCaseSimple(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %6 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%4, %arg2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%5 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %7 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %8 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%4, %6 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%7 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %9 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %10 = linalg.reduce {arith.addf} ins(%8 : tensor<?x?x?xf32>) outs(%9: tensor<?x?xf32>) dimensions = [2]
  %11 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %12 = linalg.broadcast ins(%10 : tensor<?x?xf32>) outs(%11: tensor<?x?x?xf32>) dimensions = [2]
// CHECK: %[[CALL1:.*]]:2 = call @testCaseSimple_0(
// CHECK-SAME: -> (tensor<?x?xf32>, tensor<?x?x?xf32>)
// CHECK: return %[[CALL1]]#0, %[[CALL1]]#1 : tensor<?x?xf32>, tensor<?x?x?xf32>
  return %10, %12 : tensor<?x?xf32>, tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePPtoPR_0(
func.func @testCasePPtoPR(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoPR(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<?x?x?xf32>) outs(%5: tensor<?x?xf32>) dimensions = [2]
// CHECK: %[[CALL1:.*]] = call @testCasePPtoPR_0(
// CHECK-SAME: -> tensor<?x?xf32>
// CHECK: return %[[CALL1]] : tensor<?x?xf32>
  return %6 : tensor<?x?xf32>
}

// -----

// PP to RP is not allowed
// CHECK-NOT: func.func @testCasePPtoRP_0(
func.func @testCasePPtoRP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoRP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = tensor.empty(%1, %2) : tensor<?x?xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<?x?x?xf32>) outs(%5: tensor<?x?xf32>) dimensions = [0]
  return %6 : tensor<?x?xf32>
}

// -----

// CHECK-NOT: func.func @testCasePPtoPRP_0(
func.func @testCasePPtoPRP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoPRP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = tensor.empty(%0, %2) : tensor<?x?xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<?x?x?xf32>) outs(%5: tensor<?x?xf32>) dimensions = [1]
  return %6 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePPtoBP_0(
func.func @testCasePPtoBP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoBP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = tensor.empty(%c3, %0, %1, %2) : tensor<?x?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?x?xf32>) outs(%5: tensor<?x?x?x?xf32>) dimensions = [0]
  return %6 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePPtoBPBP_0(
func.func @testCasePPtoBPBP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoBPBP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = tensor.empty(%c3, %0, %c3, %1, %2) : tensor<?x?x?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?x?xf32>) outs(%5: tensor<?x?x?x?x?xf32>) dimensions = [0, 2]
  return %6 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePPtoPBP_0(
func.func @testCasePPtoPBP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoPBP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = tensor.empty(%0, %c3, %1, %2) : tensor<?x?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?x?xf32>) outs(%5: tensor<?x?x?x?xf32>) dimensions = [1]
// CHECK-NOT: linalg.broadcast
  return %6 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePRtoPB_0(
func.func @testCasePRtoPB(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePRtoPB(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<?x?x?xf32>) outs(%3: tensor<?x?xf32>) dimensions = [2]
  %5 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?xf32>) outs(%5: tensor<?x?x?xf32>) dimensions = [2]
// CHECK: %[[CALL1:.*]] = call @testCasePRtoPB_0(
// CHECK-SAME: -> tensor<?x?x?xf32>
// CHECK: return %[[CALL1]] : tensor<?x?x?xf32>
  return %6 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePRtoBP_0(
func.func @testCasePRtoBP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePRtoBP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<?x?x?xf32>) outs(%3: tensor<?x?xf32>) dimensions = [2]
  %5 = tensor.empty(%c3, %0, %1) : tensor<?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?xf32>) outs(%5: tensor<?x?x?xf32>) dimensions = [0]
// CHECK: call
  return %6 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePRtoPP_0(
func.func @testCasePRtoPP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePRtoPP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<?x?x?xf32>) outs(%3: tensor<?x?xf32>) dimensions = [2]
  %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %6 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%4, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[CALL1:.*]] = call @testCasePRtoPP_0(
// CHECK-SAME: -> tensor<?x?xf32>
// CHECK: return %[[CALL1]] : tensor<?x?xf32>
  return %6 : tensor<?x?xf32>
}

// -----

// CHECK-NOT: func.func @testCaseRPtoBP_0(
// SHALLOW-CV-NOT: func.func @testCaseRPtoBP_0(
func.func @testCaseRPtoBP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%1, %2) : tensor<?x?xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<?x?x?xf32>) outs(%3: tensor<?x?xf32>) dimensions = [0]
  %5 = tensor.empty(%c3, %1, %2) : tensor<?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?xf32>) outs(%5: tensor<?x?x?xf32>) dimensions = [0]
  return %6 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePBtoPP_0(
func.func @testCasePBtoPP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePBtoPP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<?x?x?xf32>) outs(%3: tensor<?x?x?x?xf32>) dimensions = [3]
  %5 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %6 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%4, %arg2 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%5 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK: %[[CALL1:.*]] = call @testCasePBtoPP_0(%
// CHECK-SAME: -> tensor<?x?x?x?xf32>
// CHECK: return %[[CALL1]] : tensor<?x?x?x?xf32>
  return %6 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-NOT: func.func @testCasePBtoPR_0(
// SHALLOW-CV-NOT: func.func @testCasePBtoPR_0(
func.func @testCasePBtoPR(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<?x?x?xf32>) outs(%3: tensor<?x?x?x?xf32>) dimensions = [3]
  %5 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<?x?x?x?xf32>) outs(%5: tensor<?x?x?xf32>) dimensions = [3]
  return %6 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePBtoBP_0(
func.func @testCasePBtoBP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2, %c3) : tensor<?x?x?x?xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<?x?x?xf32>) outs(%3: tensor<?x?x?x?xf32>) dimensions = [3]
  %5 = tensor.empty(%c3, %0, %1, %2, %c3) : tensor<?x?x?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?x?x?xf32>) outs(%5: tensor<?x?x?x?x?xf32>) dimensions = [0]
  return %6 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCaseBPtoPP_0(
func.func @testCaseBPtoPP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCaseBPtoPP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%c3, %0, %1, %2) : tensor<?x?x?x?xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<?x?x?xf32>) outs(%3: tensor<?x?x?x?xf32>) dimensions = [0]
  %5 = tensor.empty(%c3, %0, %1, %2) : tensor<?x?x?x?xf32>
  %6 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%4, %arg2 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%5 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK: call
  return %6 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-NOT: func.func @testCaseBPtoPR_0(
func.func @testCaseBPtoPR(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%c3, %0, %1, %2) : tensor<?x?x?x?xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<?x?x?xf32>) outs(%3: tensor<?x?x?x?xf32>) dimensions = [0]
  %5 = tensor.empty(%c3, %0, %1) : tensor<?x?x?xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<?x?x?x?xf32>) outs(%5: tensor<?x?x?xf32>) dimensions = [3]
  return %6 : tensor<?x?x?xf32>
}

// -----

// ANY-PB-LABEL: func.func @testCaseBPtoBP_0(
// ANY-PB: linalg.broadcast
// ANY-PB: linalg.broadcast
// ANY-PB: return
// ANY-PB-LABEL: func.func @testCaseBPtoBP(
// ANY-PB: call
// ANY-PB: return
func.func @testCaseBPtoBP(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%c3, %0, %1, %2) : tensor<?x?x?x?xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<?x?x?xf32>) outs(%3: tensor<?x?x?x?xf32>) dimensions = [0]
  %5 = tensor.empty(%c3, %c3, %0, %1, %2) : tensor<?x?x?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?x?x?xf32>) outs(%5: tensor<?x?x?x?x?xf32>) dimensions = [0]
  return %6 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testCaseBPtoPB_0(
func.func @testCaseBPtoPB(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%c3, %0, %1, %2) : tensor<?x?x?x?xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<?x?x?xf32>) outs(%3: tensor<?x?x?x?xf32>) dimensions = [0]
  %5 = tensor.empty(%c3, %0, %1, %2, %c3) : tensor<?x?x?x?x?xf32>
  %6 = linalg.broadcast ins(%4 : tensor<?x?x?x?xf32>) outs(%5: tensor<?x?x?x?x?xf32>) dimensions = [4]
  return %6 : tensor<?x?x?x?x?xf32>
}

// -----

// There are 15 linalg.reduce and/or broadcast
// MAXBUFF: Considering 22 and 15 extra Live Range:
// MAXBUFF: testComplex1: 20


// CHECK-LABEL: func.func @testComplex1_0(
// CHECK-SAME: hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>
// CHECK: return %[[RET0:.*]] : tensor<?x?x?xf32>
// CHECK-LABEL: func.func @testComplex1_1(
// CHECK-SAME: hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>
// CHECK: return %[[RET1:.*]], %[[RET2:.*]] : tensor<?x?xf32>, tensor<?x?xf32>
func.func @testComplex1(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %3 = linalg.broadcast ins(%arg2 : tensor<?x?xf32>) outs(%2: tensor<?x?x?xf32>) dimensions = [0]
  %4 = tensor.empty(%1) : tensor<?xf32>
  %5 = linalg.reduce {arith.addf} ins(%arg1 : tensor<?x?xf32>) outs(%4: tensor<?xf32>) dimensions = [0]
  %6 = tensor.empty(%0, %0, %0, %1) : tensor<?x?x?x?xf32>
  %7 = linalg.broadcast ins(%3 : tensor<?x?x?xf32>) outs(%6: tensor<?x?x?x?xf32>) dimensions = [0]
  %8 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%3 : tensor<?x?x?xf32>) outs(%8 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %10 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %11 = linalg.broadcast ins(%arg2 : tensor<?x?xf32>) outs(%10: tensor<?x?x?xf32>) dimensions = [0]
  %12 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%9 : tensor<?x?x?xf32>) outs(%12 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.reduce {arith.addf} ins(%3 : tensor<?x?x?xf32>) outs(%14: tensor<?x?xf32>) dimensions = [0]
  %16 = tensor.empty(%0, %1, %0) : tensor<?x?x?xf32>
  %17 = linalg.broadcast ins(%arg2 : tensor<?x?xf32>) outs(%16: tensor<?x?x?xf32>) dimensions = [2]
  %18 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %19 = linalg.reduce {arith.addf} ins(%11 : tensor<?x?x?xf32>) outs(%18: tensor<?x?xf32>) dimensions = [0]
  %20 = tensor.empty(%0, %1, %0, %0) : tensor<?x?x?x?xf32>
  %21 = linalg.broadcast ins(%17 : tensor<?x?x?xf32>) outs(%20: tensor<?x?x?x?xf32>) dimensions = [3]
  %22 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %23 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%15 : tensor<?x?xf32>) outs(%22 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %24 = tensor.empty(%1, %0) : tensor<?x?xf32>
  %25 = linalg.broadcast ins(%5 : tensor<?xf32>) outs(%24: tensor<?x?xf32>) dimensions = [1]
  %26 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %27 = linalg.broadcast ins(%3 : tensor<?x?x?xf32>) outs(%26: tensor<?x?x?x?xf32>) dimensions = [3]
  %28 = tensor.empty(%0, %1, %0) : tensor<?x?x?xf32>
  %29 = linalg.broadcast ins(%19 : tensor<?x?xf32>) outs(%28: tensor<?x?x?xf32>) dimensions = [2]
  %30 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %31 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%23, %19 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %32 = tensor.empty(%0) : tensor<?xf32>
  %33 = linalg.reduce {arith.addf} ins(%25 : tensor<?x?xf32>) outs(%32: tensor<?xf32>) dimensions = [0]
  %34 = tensor.empty(%1, %0) : tensor<?x?xf32>
  %35 = linalg.reduce {arith.addf} ins(%17 : tensor<?x?x?xf32>) outs(%34: tensor<?x?xf32>) dimensions = [0]
  %36 = tensor.empty(%0, %0, %0) : tensor<?x?x?xf32>
  %37 = linalg.reduce {arith.addf} ins(%7 : tensor<?x?x?x?xf32>) outs(%36: tensor<?x?x?xf32>) dimensions = [3]
  %38 = tensor.empty(%0, %0) : tensor<?x?xf32>
  %39 = linalg.reduce {arith.addf} ins(%11 : tensor<?x?x?xf32>) outs(%38: tensor<?x?xf32>) dimensions = [2]
  return %arg0, %arg1, %arg2, %5, %7, %11, %13, %15, %17, %21, %23, %25, %27, %29, %31, %33, %35, %37, %39 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>
}

// -----

// MAXBUFF: Considering 22 and 13 extra Live Range:
// MAXBUFF: testComplex2: 25


// CHECK-LABEL: func.func @testComplex2_0(
// CHECK: linalg.reduce
// CHECK-LABEL: func.func @testComplex2_1(
// CHECK: linalg.reduce
// CHECK-NOT:  func.func @testComplex2_2(
// CHECK-LABEL: func.func @testComplex2(
// SHALLOW-CV-LABEL: func.func @testComplex2
// SHALLOW-CV: linalg.unary
// SHALLOW-CV: linalg.reduce
// SHALLOW-CV: linalg.broadcast
// SHALLOW-CV: linalg.broadcast
// SHALLOW-CV: linalg.reduce
// SHALLOW-CV: linalg.unary
// SHALLOW-CV: linalg.broadcast
// SHALLOW-CV: linalg.broadcast
// SHALLOW-CV-LABEL-NOT: func.func
func.func @testComplex2(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg1 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0) : tensor<?xf32>
  %5 = linalg.reduce {arith.addf} ins(%3 : tensor<?x?xf32>) outs(%4: tensor<?xf32>) dimensions = [1]
  %6 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %7 = linalg.broadcast ins(%3 : tensor<?x?xf32>) outs(%6: tensor<?x?x?xf32>) dimensions = [0]
  %8 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %9 = linalg.broadcast ins(%7 : tensor<?x?x?xf32>) outs(%8: tensor<?x?x?x?xf32>) dimensions = [3]
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.reduce {arith.addf} ins(%7 : tensor<?x?x?xf32>) outs(%10: tensor<?x?xf32>) dimensions = [0]
  %12 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%11 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%9 : tensor<?x?x?x?xf32>) outs(%14 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %16 = tensor.empty(%1) : tensor<?xf32>
  %17 = linalg.reduce {arith.addf} ins(%13 : tensor<?x?xf32>) outs(%16: tensor<?xf32>) dimensions = [0]
  %18 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %19 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%11 : tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %20 = tensor.empty(%0) : tensor<?xf32>
  %21 = linalg.reduce {arith.addf} ins(%19 : tensor<?x?xf32>) outs(%20: tensor<?xf32>) dimensions = [1]
  %22 = tensor.empty(%0, %1, %0) : tensor<?x?x?xf32>
  %23 = linalg.broadcast ins(%19 : tensor<?x?xf32>) outs(%22: tensor<?x?x?xf32>) dimensions = [2]
  %24 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %25 = linalg.broadcast ins(%7 : tensor<?x?x?xf32>) outs(%24: tensor<?x?x?x?xf32>) dimensions = [3]
  %26 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %27 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%15 : tensor<?x?x?x?xf32>) outs(%26 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %28 = tensor.empty(%0, %1, %0) : tensor<?x?x?xf32>
  %29 = linalg.broadcast ins(%arg1 : tensor<?x?xf32>) outs(%28: tensor<?x?x?xf32>) dimensions = [2]
  %30 = tensor.empty(%0) : tensor<?xf32>
  %31 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%5 : tensor<?xf32>) outs(%30 : tensor<?xf32>) -> tensor<?xf32>
  %32 = tensor.empty(%1) : tensor<?xf32>
  %33 = linalg.reduce {arith.addf} ins(%19 : tensor<?x?xf32>) outs(%32: tensor<?xf32>) dimensions = [0]
  %34 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %35 = linalg.broadcast ins(%19 : tensor<?x?xf32>) outs(%34: tensor<?x?x?xf32>) dimensions = [0]
  %36 = tensor.empty(%0, %0, %1, %0, %0) : tensor<?x?x?x?x?xf32>
  %37 = linalg.broadcast ins(%25 : tensor<?x?x?x?xf32>) outs(%36: tensor<?x?x?x?x?xf32>) dimensions = [4]
  %38 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %39 = linalg.broadcast ins(%35 : tensor<?x?x?xf32>) outs(%38: tensor<?x?x?x?xf32>) dimensions = [3]
  return %arg0, %arg2, %3, %7, %9, %13, %15, %17, %19, %21, %23, %25, %27, %29, %31, %33, %35, %37, %39 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>
}

// -----

// ANY-PB-LABEL: func.func @shallow_connected_with_scalar_hfusion_0
// ANY-PB: arith.constant
// ANY-PB: tensor.empty
// ANY-PB: linalg.fill
// ANY-PB: hfusion.elemwise_unary
// ANY-PB: linalg.elemwise_binary
// ANY-PB: linalg.elemwise_binary
// ANY-PB: linalg.broadcast
// ANY-PB: linalg.elemwise_binary
// ANY-PB: tensor.empty
// ANY-PB: linalg.broadcast
// ANY-PB: linalg.elemwise_binary
// ANY-PB: linalg.elemwise_binary
// ANY-PB-LABEL: func.func @shallow_connected_with_scalar_hfusion
// ANY-PB: arith.constant
// ANY-PB: tensor.empty
// ANY-PB: arith.truncf
// ANY-PB: tensor.empty
// ANY-PB: %[[RET:.*]]:2 = call @shallow_connected_with_scalar_hfusion_0
// ANY-PB: return [[_:.*]] %[[RET]]#0, %[[RET]]#1
module {
  func.func @shallow_connected_with_scalar_hfusion(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg5: tensor<16xf32>, %arg6: tensor<16xf32>) -> (tensor<16x16xf32>, tensor<16xf32>) attributes {debug_instruction_number = 190 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %cst = arith.constant {debug_instruction_number = 6 : i32} 0.099999999999999978 : f64
    %cst_0 = arith.constant {debug_instruction_number = 8 : i32} 1.000000e+02 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = tensor.empty() : tensor<f32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<f32>) -> tensor<f32>
    %3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%2 : tensor<f32>) outs(%1 : tensor<f32>) -> tensor<f32>
    %4 = linalg.elemwise_binary {debug_instruction_number = 14 : i32, fun = #linalg.binary_fn<mul>} ins(%arg0, %0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = linalg.elemwise_binary {debug_instruction_number = 17 : i32, fun = #linalg.binary_fn<sub>} ins(%arg1, %4 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %6 = arith.truncf %cst {debug_instruction_number = 18 : i32} : f64 to f32
    %7 = linalg.broadcast ins(%3 : tensor<f32>) outs(%0 : tensor<16x16xf32>) dimensions = [0, 1]
    %8 = linalg.elemwise_binary {debug_instruction_number = 23 : i32, fun = #linalg.binary_fn<mul>} ins(%5, %7 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %9 = tensor.empty() {debug_instruction_number = 107 : i32} : tensor<16xf32>
    %10 = tensor.empty() : tensor<16xf32>
    %broadcasted = linalg.broadcast ins(%3 : tensor<f32>) outs(%10 : tensor<16xf32>) dimensions = [0]
    %11 = linalg.elemwise_binary {debug_instruction_number = 112 : i32, fun = #linalg.binary_fn<mul>} ins(%arg5, %broadcasted : tensor<16xf32>, tensor<16xf32>) outs(%9 : tensor<16xf32>) -> tensor<16xf32>
    %12 = linalg.elemwise_binary {debug_instruction_number = 115 : i32, fun = #linalg.binary_fn<sub>} ins(%arg6, %11 : tensor<16xf32>, tensor<16xf32>) outs(%9 : tensor<16xf32>) -> tensor<16xf32>
    return {debug_instruction_number = 189 : i32} %8, %12 : tensor<16x16xf32>, tensor<16xf32>
  }
}