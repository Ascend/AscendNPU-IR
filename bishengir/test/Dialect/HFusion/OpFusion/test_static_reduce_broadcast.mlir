// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="LAST_AXIS_PBR" --hfusion-fuse-ops="max-horizontal-fusion-size=-1" --split-input-file %s | FileCheck %s
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="SHALLOW_CV" --hfusion-fuse-ops="max-horizontal-fusion-size=-1" --split-input-file %s | FileCheck %s --check-prefix=SHALLOW-CV
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="ANY_PB" --hfusion-fuse-ops="max-horizontal-fusion-size=-1" --split-input-file %s | FileCheck %s --check-prefix=ANY-PB

// CHECK-LABEL: func.func @testCaseSimple_0(
func.func @testCaseSimple(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCaseSimple(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%3 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %5 = tensor.empty() : tensor<7x7x7xf32>
  %6 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%4, %arg2 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%5 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %7 = tensor.empty() : tensor<7x7x7xf32>
  %8 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%4, %6 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%7 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %9 = tensor.empty() : tensor<7x7xf32>
  %10 = linalg.reduce {arith.addf} ins(%8 : tensor<7x7x7xf32>) outs(%9: tensor<7x7xf32>) dimensions = [2]
  %11 = tensor.empty() : tensor<7x7x7xf32>
  %12 = linalg.broadcast ins(%10 : tensor<7x7xf32>) outs(%11: tensor<7x7x7xf32>) dimensions = [2]
// CHECK: %[[CALL1:.*]]:2 = call @testCaseSimple_0(
// CHECK-SAME: -> (tensor<7x7xf32>, tensor<7x7x7xf32>)
// CHECK: return %[[CALL1]]#0, %[[CALL1]]#1 : tensor<7x7xf32>, tensor<7x7x7xf32>
  return %10, %12 : tensor<7x7xf32>, tensor<7x7x7xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePPtoPR_0(
func.func @testCasePPtoPR(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoPR(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%3 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %5 = tensor.empty() : tensor<7x7xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<7x7x7xf32>) outs(%5: tensor<7x7xf32>) dimensions = [2]
// CHECK: %[[CALL1:.*]] = call @testCasePPtoPR_0(
// CHECK-SAME: -> tensor<7x7xf32>
// CHECK: return %[[CALL1]] : tensor<7x7xf32>
  return %6 : tensor<7x7xf32>
}


// -----

// PP to RP is not allowed
// CHECK-NOT: func.func @testCasePPtoRP_0(
func.func @testCasePPtoRP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoRP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%3 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %5 = tensor.empty() : tensor<7x7xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<7x7x7xf32>) outs(%5: tensor<7x7xf32>) dimensions = [0]
  return %6 : tensor<7x7xf32>
}

// -----

// CHECK-NOT: func.func @testCasePPtoPRP_0(
func.func @testCasePPtoPRP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoPRP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%3 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %5 = tensor.empty() : tensor<7x7xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<7x7x7xf32>) outs(%5: tensor<7x7xf32>) dimensions = [1]
  return %6 : tensor<7x7xf32>
}

// -----

// CHECK-LABEL: func.func @testCasePPtoBP_0(
func.func @testCasePPtoBP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoBP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%3 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %5 = tensor.empty() : tensor<7x7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7x7xf32>) outs(%5: tensor<7x7x7x7xf32>) dimensions = [0]
// CHECK: call
  return %6 : tensor<7x7x7x7xf32>
}

// -----
// CHECK-LABEL: func.func @testCasePPtoBPBP_0(
func.func @testCasePPtoBPBP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoBPBP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%3 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %5 = tensor.empty() : tensor<7x7x7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7x7xf32>) outs(%5: tensor<7x7x7x7x7xf32>) dimensions = [0, 2]
  return %6 : tensor<7x7x7x7x7xf32>
}

// -----
// CHECK-LABEL: func.func @testCasePPtoPBP_0(
func.func @testCasePPtoPBP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePPtoPBP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%3 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %5 = tensor.empty() : tensor<7x7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7x7xf32>) outs(%5: tensor<7x7x7x7xf32>) dimensions = [1]
// CHECK-NOT: linalg.broadcast
  return %6 : tensor<7x7x7x7xf32>
}

// -----
// CHECK-LABEL: func.func @testCasePRtoPB_0(
func.func @testCasePRtoPB(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePRtoPB(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<7x7x7xf32>) outs(%3: tensor<7x7xf32>) dimensions = [2]
  %5 = tensor.empty() : tensor<7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7xf32>) outs(%5: tensor<7x7x7xf32>) dimensions = [2]
// CHECK: %[[CALL1:.*]] = call @testCasePRtoPB_0(
// CHECK-SAME: -> tensor<7x7x7xf32>
// CHECK: return %[[CALL1]] : tensor<7x7x7xf32>
  return %6 : tensor<7x7x7xf32>
}

// -----
// CHECK-LABEL: func.func @testCasePRtoBP_0(
func.func @testCasePRtoBP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePRtoBP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<7x7x7xf32>) outs(%3: tensor<7x7xf32>) dimensions = [2]
  %5 = tensor.empty() : tensor<7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7xf32>) outs(%5: tensor<7x7x7xf32>) dimensions = [0]
// CHECK: call
  return %6 : tensor<7x7x7xf32>
}

// -----
// CHECK-LABEL: func.func @testCasePRtoPP_0(
func.func @testCasePRtoPP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePRtoPP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<7x7x7xf32>) outs(%3: tensor<7x7xf32>) dimensions = [2]
  %5 = tensor.empty() : tensor<7x7xf32>
  %6 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%4, %arg2 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%5 : tensor<7x7xf32>) -> tensor<7x7xf32>
// CHECK: %[[CALL1:.*]] = call @testCasePRtoPP_0(
// CHECK-SAME: -> tensor<7x7xf32>
// CHECK: return %[[CALL1]] : tensor<7x7xf32>
  return %6 : tensor<7x7xf32>
}

// -----

// CHECK-NOT: func.func @testCaseRPtoBP_0(
// SHALLOW-CV-NOT: func.func @testCaseRPtoBP_0(
func.func @testCaseRPtoBP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<7x7x7xf32>) outs(%3: tensor<7x7xf32>) dimensions = [0]
  %5 = tensor.empty() : tensor<7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7xf32>) outs(%5: tensor<7x7x7xf32>) dimensions = [0]
  return %6 : tensor<7x7x7xf32>
}

// -----
// CHECK-LABEL: func.func @testCasePBtoPP_0(
func.func @testCasePBtoPP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7x7xf32>) -> (tensor<7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCasePBtoPP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7x7xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<7x7x7xf32>) outs(%3: tensor<7x7x7x7xf32>) dimensions = [3]
  %5 = tensor.empty() : tensor<7x7x7x7xf32>
  %6 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%4, %arg2 : tensor<7x7x7x7xf32>, tensor<7x7x7x7xf32>) outs(%5 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
// CHECK: %[[CALL1:.*]] = call @testCasePBtoPP_0(%
// CHECK-SAME: -> tensor<7x7x7x7xf32>
// CHECK: return %[[CALL1]] : tensor<7x7x7x7xf32>
  return %6 : tensor<7x7x7x7xf32>
}

// -----
// CHECK-NOT: func.func @testCasePBtoPR_0(
// SHALLOW-CV: func.func @testCasePBtoPR_0(
func.func @testCasePBtoPR(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<7x7xf32>) outs(%3: tensor<7x7x7xf32>) dimensions = [2]
  %5 = tensor.empty() : tensor<7x7xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<7x7x7xf32>) outs(%5: tensor<7x7xf32>) dimensions = [2]
  %7 = linalg.matmul ins(%5, %5 : tensor<7x7xf32>, tensor<7x7xf32>)
               outs(%5 : tensor<7x7xf32>) -> tensor<7x7xf32>
  return %7 : tensor<7x7xf32>
}




// -----
// CHECK-NOT: func.func @testCasePBtoPR2_0(
// SHALLOW-CV-NOT: func.func @testCasePBtoPR2_0(
func.func @testCasePBtoPR2(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<7x7xf32>) outs(%3: tensor<7x7x7xf32>) dimensions = [2]
  %5 = tensor.empty() : tensor<7x7xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<7x7x7xf32>) outs(%5: tensor<7x7xf32>) dimensions = [2]
  return %6 : tensor<7x7xf32>
}



// -----
// CHECK-LABEL: func.func @testCasePBtoBP_0(
func.func @testCasePBtoBP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7x7xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<7x7x7xf32>) outs(%3: tensor<7x7x7x7xf32>) dimensions = [3]
  %5 = tensor.empty() : tensor<7x7x7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7x7x7xf32>) outs(%5: tensor<7x7x7x7x7xf32>) dimensions = [0]
  return %6 : tensor<7x7x7x7x7xf32>
}


// -----

// CHECK-LABEL: func.func @testCaseBPtoPP_0(
func.func @testCaseBPtoPP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7x7xf32>) -> (tensor<7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @testCaseBPtoPP(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7x7x7xf32>, %[[ARG1:.*]]: tensor<7x7x7xf32>, %[[ARG2:.*]]: tensor<7x7x7x7xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7x7xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<7x7x7xf32>) outs(%3: tensor<7x7x7x7xf32>) dimensions = [0]
  %5 = tensor.empty() : tensor<7x7x7x7xf32>
  %6 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%4, %arg2 : tensor<7x7x7x7xf32>, tensor<7x7x7x7xf32>) outs(%5 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
// CHECK: call
  return %6 : tensor<7x7x7x7xf32>
}

// -----

// CHECK-NOT: func.func @testCaseBPtoPR_0(
func.func @testCaseBPtoPR(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7x7xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<7x7x7xf32>) outs(%3: tensor<7x7x7x7xf32>) dimensions = [0]
  %5 = tensor.empty() : tensor<7x7x7xf32>
  %6 = linalg.reduce {arith.addf} ins(%4 : tensor<7x7x7x7xf32>) outs(%5: tensor<7x7x7xf32>) dimensions = [3]
  return %6 : tensor<7x7x7xf32>
}

// -----
// CHECK: func.func @testCaseBPtoBP_0(
func.func @testCaseBPtoBP(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7x7xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<7x7x7xf32>) outs(%3: tensor<7x7x7x7xf32>) dimensions = [0]
  %5 = tensor.empty() : tensor<7x7x7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7x7x7xf32>) outs(%5: tensor<7x7x7x7x7xf32>) dimensions = [0]
  return %6 : tensor<7x7x7x7x7xf32>
}

// -----
// CHECK: func.func @testCaseBPtoPB_0(
func.func @testCaseBPtoPB(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7xf32>, %arg2: tensor<7x7x7xf32>) -> (tensor<7x7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7x7xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<7x7x7xf32>
  %3 = tensor.empty() : tensor<7x7x7x7xf32>
  %4 = linalg.broadcast ins(%arg1 : tensor<7x7x7xf32>) outs(%3: tensor<7x7x7x7xf32>) dimensions = [0]
  %5 = tensor.empty() : tensor<7x7x7x7x7xf32>
  %6 = linalg.broadcast ins(%4 : tensor<7x7x7x7xf32>) outs(%5: tensor<7x7x7x7x7xf32>) dimensions = [4]
  return %6 : tensor<7x7x7x7x7xf32>
}

// -----
// CHECK-LABEL: func.func @testComplex1_0(
// CHECK-SAME: hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>
// CHECK: return %[[RET0:.*]] : tensor<7x7x7xf32>
// CHECK-LABEL: func.func @testComplex1_1(
// CHECK-SAME: hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>
// CHECK: return %[[RET1:.*]], %[[RET2:.*]] : tensor<7x7xf32>, tensor<7x7xf32>
func.func @testComplex1(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty() : tensor<7x7x7xf32>
  %3 = linalg.broadcast ins(%arg2 : tensor<7x7xf32>) outs(%2: tensor<7x7x7xf32>) dimensions = [0]
  %4 = tensor.empty() : tensor<7xf32>
  %5 = linalg.reduce {arith.addf} ins(%arg1 : tensor<7x7xf32>) outs(%4: tensor<7xf32>) dimensions = [0]
  %6 = tensor.empty() : tensor<7x7x7x7xf32>
  %7 = linalg.broadcast ins(%3 : tensor<7x7x7xf32>) outs(%6: tensor<7x7x7x7xf32>) dimensions = [0]
  %8 = tensor.empty() : tensor<7x7x7xf32>
  %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%3 : tensor<7x7x7xf32>) outs(%8 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %10 = tensor.empty() : tensor<7x7x7xf32>
  %11 = linalg.broadcast ins(%arg2 : tensor<7x7xf32>) outs(%10: tensor<7x7x7xf32>) dimensions = [0]
  %12 = tensor.empty() : tensor<7x7x7xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%9 : tensor<7x7x7xf32>) outs(%12 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %14 = tensor.empty() : tensor<7x7xf32>
  %15 = linalg.reduce {arith.addf} ins(%3 : tensor<7x7x7xf32>) outs(%14: tensor<7x7xf32>) dimensions = [0]
  %16 = tensor.empty() : tensor<7x7x7xf32>
  %17 = linalg.broadcast ins(%arg2 : tensor<7x7xf32>) outs(%16: tensor<7x7x7xf32>) dimensions = [2]
  %18 = tensor.empty() : tensor<7x7xf32>
  %19 = linalg.reduce {arith.addf} ins(%11 : tensor<7x7x7xf32>) outs(%18: tensor<7x7xf32>) dimensions = [0]
  %20 = tensor.empty() : tensor<7x7x7x7xf32>
  %21 = linalg.broadcast ins(%17 : tensor<7x7x7xf32>) outs(%20: tensor<7x7x7x7xf32>) dimensions = [3]
  %22 = tensor.empty() : tensor<7x7xf32>
  %23 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%15 : tensor<7x7xf32>) outs(%22 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %24 = tensor.empty() : tensor<7x7xf32>
  %25 = linalg.broadcast ins(%5 : tensor<7xf32>) outs(%24: tensor<7x7xf32>) dimensions = [1]
  %26 = tensor.empty() : tensor<7x7x7x7xf32>
  %27 = linalg.broadcast ins(%3 : tensor<7x7x7xf32>) outs(%26: tensor<7x7x7x7xf32>) dimensions = [3]
  %28 = tensor.empty() : tensor<7x7x7xf32>
  %29 = linalg.broadcast ins(%19 : tensor<7x7xf32>) outs(%28: tensor<7x7x7xf32>) dimensions = [2]
  %30 = tensor.empty() : tensor<7x7xf32>
  %31 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%23, %19 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%30 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %32 = tensor.empty() : tensor<7xf32>
  %33 = linalg.reduce {arith.addf} ins(%25 : tensor<7x7xf32>) outs(%32: tensor<7xf32>) dimensions = [0]
  %34 = tensor.empty() : tensor<7x7xf32>
  %35 = linalg.reduce {arith.addf} ins(%17 : tensor<7x7x7xf32>) outs(%34: tensor<7x7xf32>) dimensions = [0]
  %36 = tensor.empty() : tensor<7x7x7xf32>
  %37 = linalg.reduce {arith.addf} ins(%7 : tensor<7x7x7x7xf32>) outs(%36: tensor<7x7x7xf32>) dimensions = [3]
  %38 = tensor.empty() : tensor<7x7xf32>
  %39 = linalg.reduce {arith.addf} ins(%11 : tensor<7x7x7xf32>) outs(%38: tensor<7x7xf32>) dimensions = [2]
  return %arg0, %arg1, %arg2, %5, %7, %11, %13, %15, %17, %21, %23, %25, %27, %29, %31, %33, %35, %37, %39 : tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>
}


// -----
// CHECK-LABEL: func.func @testComplex2_0(
// CHECK: linalg.reduce
// CHECK-LABEL: func.func @testComplex2_1(
// CHECK: linalg.reduce
// SHALLOW-CV-NOT: func.func @testComplex2_0(
// ANY-PB-LABEL: func.func @testComplex2_0(
// ANY-PB: linalg.elemwise_unary
// ANY-PB: linalg.broadcast
// ANY-PB: linalg.broadcast
// ANY-PB: linalg.elemwise_unary
// ANY-PB: linalg.broadcast
// ANY-PB: linalg.elemwise_unary
// ANY-PB: linalg.broadcast
// ANY-PB: return
// ANY-PB-LABEL: func.func @testComplex2_1(
// ANY-PB: return
// ANY-PB-LABEL: func.func @testComplex2(
// ANY-PB: return
func.func @testComplex2(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7xf32>, tensor<7x7xf32>, tensor<7xf32>, tensor<7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>, tensor<7xf32>, tensor<7xf32>, tensor<7x7x7xf32>, tensor<7x7x7x7x7xf32>, tensor<7x7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty() : tensor<7x7xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg1 : tensor<7x7xf32>) outs(%2 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %4 = tensor.empty() : tensor<7xf32>
  %5 = linalg.reduce {arith.addf} ins(%3 : tensor<7x7xf32>) outs(%4: tensor<7xf32>) dimensions = [1]
  %6 = tensor.empty() : tensor<7x7x7xf32>
  %7 = linalg.broadcast ins(%3 : tensor<7x7xf32>) outs(%6: tensor<7x7x7xf32>) dimensions = [0]
  %8 = tensor.empty() : tensor<7x7x7x7xf32>
  %9 = linalg.broadcast ins(%7 : tensor<7x7x7xf32>) outs(%8: tensor<7x7x7x7xf32>) dimensions = [3]
  %10 = tensor.empty() : tensor<7x7xf32>
  %11 = linalg.reduce {arith.addf} ins(%7 : tensor<7x7x7xf32>) outs(%10: tensor<7x7xf32>) dimensions = [0]
  %12 = tensor.empty() : tensor<7x7xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%11 : tensor<7x7xf32>) outs(%12 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %14 = tensor.empty() : tensor<7x7x7x7xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%9 : tensor<7x7x7x7xf32>) outs(%14 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
  %16 = tensor.empty() : tensor<7xf32>
  %17 = linalg.reduce {arith.addf} ins(%13 : tensor<7x7xf32>) outs(%16: tensor<7xf32>) dimensions = [0]
  %18 = tensor.empty() : tensor<7x7xf32>
  %19 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%11 : tensor<7x7xf32>) outs(%18 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %20 = tensor.empty() : tensor<7xf32>
  %21 = linalg.reduce {arith.addf} ins(%19 : tensor<7x7xf32>) outs(%20: tensor<7xf32>) dimensions = [1]
  %22 = tensor.empty() : tensor<7x7x7xf32>
  %23 = linalg.broadcast ins(%19 : tensor<7x7xf32>) outs(%22: tensor<7x7x7xf32>) dimensions = [2]
  %24 = tensor.empty() : tensor<7x7x7x7xf32>
  %25 = linalg.broadcast ins(%7 : tensor<7x7x7xf32>) outs(%24: tensor<7x7x7x7xf32>) dimensions = [3]
  %26 = tensor.empty() : tensor<7x7x7x7xf32>
  %27 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%15 : tensor<7x7x7x7xf32>) outs(%26 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
  %28 = tensor.empty() : tensor<7x7x7xf32>
  %29 = linalg.broadcast ins(%arg1 : tensor<7x7xf32>) outs(%28: tensor<7x7x7xf32>) dimensions = [2]
  %30 = tensor.empty() : tensor<7xf32>
  %31 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%5 : tensor<7xf32>) outs(%30 : tensor<7xf32>) -> tensor<7xf32>
  %32 = tensor.empty() : tensor<7xf32>
  %33 = linalg.reduce {arith.addf} ins(%19 : tensor<7x7xf32>) outs(%32: tensor<7xf32>) dimensions = [0]
  %34 = tensor.empty() : tensor<7x7x7xf32>
  %35 = linalg.broadcast ins(%19 : tensor<7x7xf32>) outs(%34: tensor<7x7x7xf32>) dimensions = [0]
  %36 = tensor.empty() : tensor<7x7x7x7x7xf32>
  %37 = linalg.broadcast ins(%25 : tensor<7x7x7x7xf32>) outs(%36: tensor<7x7x7x7x7xf32>) dimensions = [4]
  %38 = tensor.empty() : tensor<7x7x7x7xf32>
  %39 = linalg.broadcast ins(%35 : tensor<7x7x7xf32>) outs(%38: tensor<7x7x7x7xf32>) dimensions = [3]
  return %arg0, %arg2, %3, %7, %9, %13, %15, %17, %19, %21, %23, %25, %27, %29, %31, %33, %35, %37, %39 : tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7xf32>, tensor<7x7xf32>, tensor<7xf32>, tensor<7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>, tensor<7xf32>, tensor<7xf32>, tensor<7x7x7xf32>, tensor<7x7x7x7x7xf32>, tensor<7x7x7x7xf32>
}


// -----
// SHALLOW-CV-LABEL: func.func @forward_0({{.*}}: tensor<2x10xf32>, {{.*}}: tensor<20x10xf32>, {{.*}}: tensor<20xf32>, {{.*}}: tensor<2x20xf32>, {{.*}}: tensor<20x20xf32>, {{.*}}: tensor<20xf32>, {{.*}}: tensor<10x20xf32>, {{.*}}: tensor<10xf32>, {{.*}}: tensor<2x10xf32>) -> tensor<2x10xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>}
func.func @forward(%arg0: tensor<2x10xf32>, %cst : tensor<2x20xf32> ,%cst_0 : tensor<20x10xf32>, %cst_1 : tensor<20xf32>, %cst_2: tensor<20x20xf32>, %cst_3: tensor<20xf32>, %cst_4: tensor<10x20xf32>, %cst_5: tensor<10xf32>) -> tensor<2x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>}{
  %0 = tensor.empty() : tensor<2x20xf32>
  %1 = linalg.matmul_transpose_b ins(%arg0, %cst_0 : tensor<2x10xf32>, tensor<20x10xf32>) outs(%0 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %2 = tensor.empty() : tensor<2x20xf32>
  %broadcasted = linalg.broadcast ins(%cst_1 : tensor<20xf32>) outs(%2 : tensor<2x20xf32>) dimensions = [0]
  %3 = tensor.empty() : tensor<2x20xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %broadcasted : tensor<2x20xf32>, tensor<2x20xf32>) outs(%3 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %5 = tensor.empty() : tensor<2x20xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%4, %cst : tensor<2x20xf32>, tensor<2x20xf32>) outs(%5 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %7 = tensor.empty() : tensor<2x20xf32>
  %8 = linalg.matmul_transpose_b ins(%6, %cst_2 : tensor<2x20xf32>, tensor<20x20xf32>) outs(%7 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %9 = tensor.empty() : tensor<2x20xf32>
  %broadcasted_6 = linalg.broadcast ins(%cst_3 : tensor<20xf32>) outs(%9 : tensor<2x20xf32>) dimensions = [0]
  %10 = tensor.empty() : tensor<2x20xf32>
  %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %broadcasted_6 : tensor<2x20xf32>, tensor<2x20xf32>) outs(%10 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %12 = tensor.empty() : tensor<2x20xf32>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%11, %cst : tensor<2x20xf32>, tensor<2x20xf32>) outs(%12 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %14 = tensor.empty() : tensor<2x10xf32>
  %15 = linalg.matmul_transpose_b ins(%13, %cst_4 : tensor<2x20xf32>, tensor<10x20xf32>) outs(%14 : tensor<2x10xf32>) -> tensor<2x10xf32>
  %16 = tensor.empty() : tensor<2x10xf32>
  %broadcasted_7 = linalg.broadcast ins(%cst_5 : tensor<10xf32>) outs(%16 : tensor<2x10xf32>) dimensions = [0]
  %17 = tensor.empty() : tensor<2x10xf32>
  %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %broadcasted_7 : tensor<2x10xf32>, tensor<2x10xf32>) outs(%17 : tensor<2x10xf32>) -> tensor<2x10xf32>
  return %18 : tensor<2x10xf32>
}

// -----

// CHECK-LABEL: @main_multi_LAST_AXIS_PBR_0_0(
// CHECK: broadcast
// CHECK: broadcast
// CHECK: binary
// CHECK: return
// CHECK-LABEL: @main_multi_LAST_AXIS_PBR_0(
// CHECK: fill
// CHECK: return

func.func @main_multi_LAST_AXIS_PBR_0(%arg0: tensor<8xi64>) -> (tensor<8x1152x16x16xf32>, tensor<8x128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x1152x16x16xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<8x1152x16x16xf32>) -> tensor<8x1152x16x16xf32>
    %2 = tensor.empty() : tensor<8xf32>
    %3 = hfusion.cast ins(%arg0 : tensor<8xi64>) outs(%2 : tensor<8xf32>) -> tensor<8xf32>
    %4 = tensor.empty() : tensor<8x128xf32>
    %broadcasted = linalg.broadcast ins(%3 : tensor<8xf32>) outs(%4 : tensor<8x128xf32>) dimensions = [1]
    %broadcasted_1 = linalg.broadcast ins(%cst : tensor<128xf32>) outs(%4 : tensor<8x128xf32>) dimensions = [0]
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_1 : tensor<8x128xf32>, tensor<8x128xf32>) outs(%4 : tensor<8x128xf32>) -> tensor<8x128xf32>
    return %1, %5 : tensor<8x1152x16x16xf32>, tensor<8x128xf32>
  }