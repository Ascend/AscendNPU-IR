// RUN: bishengir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: bishengir-opt --transform-interpreter --split-input-file --mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.func.get_func_argument %1[3]: (!transform.any_op) -> !transform.any_value
    %3, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [%2, %2, 4] : (!transform.any_op, !transform.any_value, !transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func @tile_linalg_matmul_dynamic(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[SZ:[0-9a-z]+]]: i64
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul_dynamic(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %sz : i64)
    -> tensor<128x128xf32> {
//      CHECK: %[[STEP0:.*]] = arith.index_cast %[[SZ]] : i64 to index
//      CHECK: %[[STEP1:.*]] = arith.index_cast %[[SZ]] : i64 to index
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step %[[STEP0]] iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step %[[STEP1]] iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<?x4xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<4x?xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<?x4xf32>, tensor<4x?xf32>)
// CHECK-SAME:                                   outs(%[[sTC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<128x128xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

//      CHECK: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["hfusion.arange"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.func.get_func_argument %1[4]: (!transform.any_op) -> !transform.any_value
    %3, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [%2, 4] : (!transform.any_op, !transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: tile_arange_with_offset_dynamic
func.func @tile_arange_with_offset_dynamic(
  %arg0: tensor<128x128xi32>, %offset:index, %stride0:index, %stride1:index, %sz:i64)
    -> tensor<128x128xi32> {
  // CHECK: , %[[ORIGOFF:[a-zA-Z0-9]*]]: index
  // CHECK: scf.for %[[I:[a-zA-Z0-9]*]]
  // CHECK:   scf.for %[[J:[a-zA-Z0-9]*]]
  // CHECK-DAG:      %[[ISTRIDE:[a-zA-Z0-9]*]] = arith.muli %[[I]]
  // CHECK-DAG:      %[[JSTRIDE:[a-zA-Z0-9]*]] = arith.muli %[[J]]
  // CHECK:          %[[NEWOFF:[a-zA-Z0-9]*]] = arith.addi
  // CHECK-SAME-DAG: %[[ISTRIDE]]
  // CHECK-SAME-DAG: %[[JSTRIDE]]
  // CHECK:          %[[OFF:[a-zA-Z0-9]*]] = arith.addi
  // CHECK-SAME-DAG: %[[NEWOFF]]
  // CHECK-SAME-DAG: %[[ORIGOFF]]
  // CHECK: hfusion.arange offset[%[[OFF]]]
  %0 = hfusion.arange offset[%offset] strides[%stride0, %stride1]
                      outs(%arg0: tensor<128x128xi32>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["hfusion.arange"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.func.get_func_argument %1[3]: (!transform.any_op) -> !transform.any_value
    %3, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [%2, 4] : (!transform.any_op, !transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @tile_arange_no_offset_dynamic(
  %arg0: tensor<128x128xi32>, %stride0:index, %stride1:index, %sz:i64) -> tensor<128x128xi32> {
  // GENERIC: hfusion.arange
  // GENERIC: ^bb0(%[[OFFSETARG:[a-zA-Z0-9]*]]
  // GENERIC: %[[PREVOFFSET:[a-zA-Z0-9]*]] = "arith.addi"
  // GENERIC: arith.addi
  // GENERIC-SAME-DAG: %[[OFFSETARG]]
  // GENERIC-SAME-DAG: %[[PREVOFFSET]]
  %0 = hfusion.arange strides[%stride0, %stride1]
                      outs(%arg0: tensor<128x128xi32>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}

