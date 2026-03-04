// RUN: bishengir-opt -encoding-to-symbol %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: test_encoding_to_symbol_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<?x640x?xf16>
// CHECK: %[[S0:.*]] = symbol.symbolic_int
// CHECK: %[[S1:.*]] = symbol.symbolic_int
// CHECK: symbol.bind_symbolic_shape %[[arg0]], [%[[S0]], %[[S1]]]
// CHECK: %[[out0:.*]] = tensor.empty(%[[S0]], %[[S1]])
// CHECK: symbol.bind_symbolic_shape %[[out0]], [%[[S0]], %[[S1]]]
// CHECK: %[[add0:.*]] = linalg.elemwise_binary
// CHECK: symbol.bind_symbolic_shape %[[add0]], [%[[S0]], %[[S1]]]
// CHECK: %[[out1:.*]] = tensor.empty(%[[S0]], %[[S1]])
// CHECK: symbol.bind_symbolic_shape %[[out1]], [%[[S0]], %[[S1]]]
// CHECK: %[[add1:.*]] = linalg.elemwise_binary
// CHECK: symbol.bind_symbolic_shape %[[add1]], [%[[S0]], %[[S1]]]
func.func @test_encoding_to_symbol_0(%arg0: tensor<?x640x?xf16, [@S0, 640, @S1]>) -> (tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S0, 640, @S1]>) {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %0 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16, [@S0, 640, @S1]>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S0, 640, @S1]>) outs(%0 : tensor<?x640x?xf16, [@S0, 640, @S1]>) -> tensor<?x640x?xf16, [@S0, 640, @S1]>
  %2 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16, [@S0, 640, @S1]>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %1 : tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S0, 640, @S1]>) outs(%2 : tensor<?x640x?xf16, [@S0, 640, @S1]>) -> tensor<?x640x?xf16, [@S0, 640, @S1]>
  return %1, %3 : tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S0, 640, @S1]>
}

// -----

// CHECK-LABEL: test_encoding_to_symbol_1(
// CHECK-SAME: %[[arg0:.*]]: tensor<?x640x?xf16>
// CHECK: %[[S0:.*]] = symbol.symbolic_int
// CHECK: %[[S1:.*]] = symbol.symbolic_int
// CHECK: symbol.bind_symbolic_shape %[[arg0]], [%[[S0]], %[[S1]]]
// CHECK: %[[out0:.*]] = tensor.empty(%[[S0]], %[[S1]])
// CHECK: symbol.bind_symbolic_shape %[[out0]], [%[[S0]], %[[S1]]]
// CHECK: %[[add0:.*]] = linalg.elemwise_binary
// CHECK: symbol.bind_symbolic_shape %[[add0]], [%[[S0]], %[[S1]]]
// CHECK: %[[S2:.*]] = symbol.symbolic_int {{.*}} {{\[}}%[[S0]], %[[S0]]], affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK: %[[concat:.*]] = tensor.concat dim(0) %[[add0]], %[[add0]]
// CHECK: symbol.bind_symbolic_shape %[[concat]], [%[[S2]], %[[S1]]]
// CHECK: %[[out1:.*]] = tensor.empty(%[[S2]], %[[S1]])
// CHECK: symbol.bind_symbolic_shape %[[out1]], [%[[S2]], %[[S1]]]
// CHECK: %[[add1:.*]] = linalg.elemwise_binary
// CHECK: symbol.bind_symbolic_shape %[[add1]], [%[[S2]], %[[S1]]]
func.func @test_encoding_to_symbol_1(%arg0: tensor<?x640x?xf16, [@S0, 640, @S1]>) -> (tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S2, 640, @S1]>) {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %0 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16, [@S0, 640, @S1]>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S0, 640, @S1]>) outs(%0 : tensor<?x640x?xf16, [@S0, 640, @S1]>) -> tensor<?x640x?xf16, [@S0, 640, @S1]>
  %S2 = symbol.symbolic_int @S2 [%S0, %S0], affine_map<()[s0, s1] -> (s0 + s1)> {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %concat = tensor.concat dim(0) %1, %1 : (tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S0, 640, @S1]>) -> tensor<?x640x?xf16, [@S2, 640, @S1]>
  %2 = tensor.empty(%S2, %S1) : tensor<?x640x?xf16, [@S2, 640, @S1]>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%concat, %concat : tensor<?x640x?xf16, [@S2, 640, @S1]>, tensor<?x640x?xf16, [@S2, 640, @S1]>) outs(%2 : tensor<?x640x?xf16, [@S2, 640, @S1]>) -> tensor<?x640x?xf16, [@S2, 640, @S1]>
  return %1, %3 : tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S2, 640, @S1]>
}

// -----

func.func @test_encoding_to_symbol_callee_0(%arg0: tensor<?x640x?xf16, [@S0, 640, @S1]>) -> tensor<?x640x?xf16, [@S0, 640, @S1]> {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %0 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16, [@S0, 640, @S1]>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16, [@S0, 640, @S1]>, tensor<?x640x?xf16, [@S0, 640, @S1]>) outs(%0 : tensor<?x640x?xf16, [@S0, 640, @S1]>) -> tensor<?x640x?xf16, [@S0, 640, @S1]>
  return %1 : tensor<?x640x?xf16, [@S0, 640, @S1]>
}

func.func @test_encoding_to_symbol_caller_0(%arg0: tensor<?x640x?xf16>) -> tensor<?x640x?xf16> {
  %cast = tensor.cast %arg0 : tensor<?x640x?xf16> to tensor<?x640x?xf16, [@S0, 640, @S1]>
  %0 = call @test_encoding_to_symbol_callee_0(%cast) : (tensor<?x640x?xf16, [@S0, 640, @S1]>) -> tensor<?x640x?xf16, [@S0, 640, @S1]>
  %cast_0 = tensor.cast %0 : tensor<?x640x?xf16, [@S0, 640, @S1]> to tensor<?x640x?xf16>
  return %cast_0 : tensor<?x640x?xf16>
}