// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops -split-input-file %s | FileCheck %s
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=multi' -split-input-file %s | FileCheck %s
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=single' -split-input-file %s | FileCheck %s --check-prefix=Single
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=single-aggr' -split-input-file %s | FileCheck %s --check-prefix=Single-Aggr
// RUN: bishengir-opt %s -test-buffer-utils -test-buffer-utils-var="enable-dma-opt" -verify-diagnostics -split-input-file | FileCheck %s --check-prefix=MAXBUFF

// MAXBUFF: Considering 6 and 0
// MAXBUFF: test_graph_a: 3
// CHECK-LABEL: func.func @test_graph_a_0(
// CHECK-SAME: #hfusion.fusion_kind<PURE_ELEMWISE>
// Single-LABEL: func.func @test_graph_a(
// Single-Aggr-LABEL: func.func @test_graph_a
func.func @test_graph_a(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @test_graph_a(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
// CHECK: %[[MATMUL1:.*]] = linalg.matmul
// CHECK: %[[MATMUL2:.*]] = linalg.matmul
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.matmul ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
// Single-NOT: call
// Single-Aggr-NOT: call
// CHECK: %[[CALL1:.*]]:2 = call @test_graph_a_0(
// CHECK-SAME: -> (tensor<?x?xf32>, tensor<?x?xf32>)
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%3 : tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[MATMUL3:.*]] = linalg.matmul
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.matmul ins(%arg0, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.matmul ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
// Single-NOT: call
  %11 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%5, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: return %[[CALL1]]#1, %[[MATMUL3]] : tensor<?x?xf32>, tensor<?x?xf32>
  return %11, %7 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// MAXBUFF: Considering 10
// MAXBUFF: test_graph_b: 4

// Single-LABEL: func.func @test_graph_b_0(
// Single: log
// Single: mul
// Single: return {{.*}} : tensor<?x?xf32>
// Single-LABEL: func.func @test_graph_b(
// Single: call @test_graph_b_0
// Single-Aggr-LABEL: func.func @test_graph_b_0
// Single-Aggr: log
// Single-Aggr: mul
// Single-Aggr: return {{.*}} : tensor<?x?xf32>
// Single-Aggr: call @test_graph_b_0
// Single-Aggr: return
// CHECK-LABEL: func.func @test_graph_b(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
// CHECK: %[[CALL1:.*]]:5 = call @test_graph_b_0(
// CHECK: %[[MATMUL1:.*]] = linalg.matmul
// CHECK: %[[CALL2:.*]] = call @test_graph_b_1(
// CHECK: return %[[ARG0]], %[[CALL1]]#2, %[[CALL1]]#3, %[[CALL2]], %[[CALL1]]#4 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
func.func @test_graph_b(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg2 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg1 : tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%5, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.matmul ins(%arg1, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%9 : tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %13 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%3, %11 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %arg0, %5, %7, %13, %15 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// MAXBUFF: Considering 12
// MAXBUFF: test_graph_c: 7
// Single-LABEL: func.func @test_graph_c_0(
// Single: log
// Single: mul
// Single: return {{.*}} : tensor<?x?xf32>

// Single-Aggr-LABEL: func.func @test_graph_c_0(
// Single-Aggr: log
// Single-Aggr: log
// Single-Aggr: return {{.*}} : tensor<?x?xf32>
// Single-Aggr-LABEL: func.func @test_graph_c_1(
// Single-Aggr: log
// Single-Aggr: sub
// Single-Aggr: return {{.*}} : tensor<?x?xf32>
// Single-Aggr-LABEL: func.func @test_graph_c_2(
// Single-Aggr: log
// Single-Aggr: mul
// Single-Aggr: return {{.*}} : tensor<?x?xf32>
func.func @test_graph_c(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @test_graph_c(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
// Single: call @test_graph_c_0
// Single-Aggr: call @test_graph_c_0
// Single-Aggr: call @test_graph_c_1
// Single-Aggr: call @test_graph_c_2
// CHECK: %[[CALL1:.*]]:7 = call @test_graph_c_0(
// CHECK-SAME: -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg1 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%3 : tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%7, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[MATMUL1:.*]] = linalg.matmul
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.matmul ins(%7, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%11 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[CALL2:.*]] = call @test_graph_c_1(
// CHECK-SAME: -> tensor<?x?xf32>
  %16 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %17 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %19 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%13, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: return %[[ARG0]], %[[CALL1]]#3, %[[CALL1]]#4, %[[MATMUL1]], %[[CALL1]]#5, %[[CALL1]]#6, %[[CALL2]] : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  return %arg0, %7, %9, %11, %15, %17, %19 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// MAXBUFF: Considering 7
// MAXBUFF: test_graph_d: 2

// Single-LABEL: func.func @test_graph_d_0
// Single: log
// Single: mul
// Single: return {{.*}} : tensor<?x?xf32>
// Single-Aggr-LABEL: func.func @test_graph_d_0
// Single-Aggr: log
// Single-Aggr: mul
// Single-Aggr: return {{.*}} : tensor<?x?xf32>
func.func @test_graph_d(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @test_graph_d(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
// CHECK: %[[MATMUL1:.*]] = linalg.matmul
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.matmul ins(%arg2, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[MATMUL2:.*]] = linalg.matmul
  %12 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %13 = linalg.matmul ins(%arg2, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
// Single: call @test_graph_d_0
// Single-Aggr: call @test_graph_d_0
// CHECK: %[[CALL1:.*]] = call @test_graph_d_0(
// CHECK-SAME: -> tensor<?x?xf32>
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%13, %11 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: return %[[CALL1]] : tensor<?x?xf32>
  return %15 : tensor<?x?xf32>
}

// -----

// MAXBUFF: Considering 8
// MAXBUFF: test_graph_e: 4

// Single-LABEL: func.func @test_graph_e(
// Single-Aggr-LABEL: func.func @test_graph_e(
func.func @test_graph_e(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @test_graph_e(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
// Single-NOT: call
// Single-Aggr-NOT: call
// CHECK: %[[CALL1:.*]]:4 = call @test_graph_e_0(
// CHECK-SAME: -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%arg2, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[MATMUL1:.*]] = linalg.matmul
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.matmul ins(%7, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%arg2, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: return %[[ARG0]], %[[ARG1]], %[[CALL1]]#0, %[[MATMUL1]], %[[CALL1]]#3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  return %arg0, %arg1, %3, %9, %11 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// MAXBUFF: Considering 8
// MAXBUFF: test_graph_f: 4

// Single-LABEL: func.func @test_graph_f(
// Single-Aggr-LABEL: func.func @test_graph_f(
func.func @test_graph_f(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @test_graph_f(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg1 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
// Single-NOT: call
// Single-Aggr-NOT: call
// CHECK: %[[CALL1:.*]]:4 = call @test_graph_f_0(
// CHECK-SAME: -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
  %5 = linalg.matmul ins(%3, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[MATMUL1:.*]] = linalg.matmul
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%arg2, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%7, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[ELEMWISE1:.*]] = linalg.elemwise_binary
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%7, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: return %[[ARG1]], %[[CALL1]]#1, %[[CALL1]]#3, %[[ELEMWISE1]] : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  return %arg1, %3, %9, %11 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// MAXBUFF: Considering 6
// MAXBUFF: test_graph_g: 2

// Single-LABEL: func.func @test_graph_g_0(
// Single: log
// Single: sub
// Single: return {{.*}} : tensor<?x?xf32>
// Single-Aggr-LABEL: func.func @test_graph_g_0(
// Single-Aggr: min_signed
// Single-Aggr: exp
// Single-Aggr: return {{.*}} : tensor<?x?xf32>
// Single-Aggr-LABEL: func.func @test_graph_g_1(
// Single-Aggr: min_signed
// Single-Aggr: log
// Single-Aggr: sub
// Single-Aggr: return {{.*}} : tensor<?x?xf32>
func.func @test_graph_g(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @test_graph_g(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
// Single: call @test_graph_g_0
// Single-Aggr: call @test_graph_g_0
// Single-Aggr: call @test_graph_g_1
// CHECK: %[[CALL1:.*]]:2 = call @test_graph_g_0(
// CHECK-SAME: -> (tensor<?x?xf32>, tensor<?x?xf32>)
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_binary {min_signed, fun = #linalg.binary_fn<min_signed>} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%3 : tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%5, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: return %[[CALL1]]#0, %[[CALL1]]#1 : tensor<?x?xf32>, tensor<?x?xf32>
  return %7, %9 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// MAXBUFF: complex_elementwise: -1

// Single-LABEL: func.func @complex_elementwise_0(
// Single: sub
// Single: log
// Single: return {{.*}} : tensor<?xf32>
func.func @complex_elementwise(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<f32>  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @complex_elementwise(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: tensor<?xf32>)
  // Block 1: Entry block - Initial element-wise operations
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %initial_sum = tensor.empty(%dim0) : tensor<?xf32>
  %sum = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%initial_sum : tensor<?xf32>) -> tensor<?xf32>
  cf.br ^block2(%sum : tensor<?xf32>)

  // Block 2: More complex element-wise operations
^block2(%input: tensor<?xf32>):
  %mul_result = tensor.empty(%dim0) : tensor<?xf32>
  %product = linalg.elemwise_binary { mul, fun = #linalg.binary_fn<mul> } ins(%input, %arg2 : tensor<?xf32>, tensor<?xf32>) outs(%mul_result : tensor<?xf32>) -> tensor<?xf32>
  cf.br ^block3(%product : tensor<?xf32>)

  // Block 3: Combining results with additional element-wise ops

^block3(%intermediate: tensor<?xf32>):
// CHECK: ^bb2(%[[PARAM:.*]]: tensor<?xf32>):
  %sub_result = tensor.empty(%dim0) : tensor<?xf32>
  %difference = linalg.elemwise_binary { sub, fun = #linalg.binary_fn<sub> } ins(%intermediate, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%sub_result : tensor<?xf32>) -> tensor<?xf32>
  %log_result = tensor.empty(%dim0) : tensor<?xf32>
  %logged = linalg.elemwise_unary { log, fun = #linalg.unary_fn<log> } ins(%difference : tensor<?xf32>) outs(%log_result : tensor<?xf32>) -> tensor<?xf32>
  cf.br ^block4(%logged : tensor<?xf32>)
// Single: call @complex_elementwise_0
// CHECK: %[[CALLRES:.*]] = call @complex_elementwise_0(
// CHECK-SAME: -> tensor<?xf32>
  // Block 4: Final reduction to a scalar
^block4(%final_tensor: tensor<?xf32>):
  %reduced = tensor.empty() : tensor<f32>
  %final_reduce = linalg.reduce { arith.addf } ins(%final_tensor : tensor<?xf32>) outs(%reduced : tensor<f32>) dimensions = [0]
  return %final_reduce : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @mlir_fused__to_copy_npu_prompt_flash_attention_18_0(
// CHECK: %[[EXPAND:.*]] = tensor.expand_shape
// CHECK: return %[[EXPAND]], %[[EXPAND]], %[[EXPAND]]
// CHECK-LABEL: func.func @mlir_fused__to_copy_npu_prompt_flash_attention_18(
// CHECK: %[[CALL:.*]]:3 = call @mlir_fused__to_copy_npu_prompt_flash_attention_18_0(
// CHECK: return %[[CALL]]#0, %[[CALL]]#1, %[[CALL]]#2
#map = affine_map<()[s0] -> (s0 floordiv 2)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 2) * 1280)>
module {
  func.func @mlir_fused__to_copy_npu_prompt_flash_attention_18(%arg0: tensor<?x640xf16>, %arg1: i64) -> (tensor<2x?x640xbf16>, tensor<2x?x640xbf16>, tensor<2x?x640xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x640xf16> into tensor<?xf16>
    %dim = tensor.dim %arg0, %c0 : tensor<?x640xf16>
    %0 = affine.apply #map()[%dim]
    %1 = affine.apply #map1()[%dim]
    %2 = tensor.empty(%1) : tensor<?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<?xf16>) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
    %4 = affine.apply #map1()[%dim]
    %5 = tensor.empty(%4) : tensor<?xbf16>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<?xf32>) outs(%5 : tensor<?xbf16>) -> tensor<?xbf16>
    %expanded = tensor.expand_shape %6 [[0, 1, 2]] output_shape [2, %0, 640] : tensor<?xbf16> into tensor<2x?x640xbf16>
    return %expanded, %expanded, %expanded : tensor<2x?x640xbf16>, tensor<2x?x640xbf16>, tensor<2x?x640xbf16>
  }
}