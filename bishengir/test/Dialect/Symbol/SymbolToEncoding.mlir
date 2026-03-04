// RUN: bishengir-opt -symbol-to-encoding %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: tensor<?x640x?xf16, [@S0, 640, @S1]>
func.func @test_symbol_to_encoding_0(%arg0: tensor<?x640x?xf16>) -> (tensor<?x640x?xf16>, tensor<?x640x?xf16>) {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  symbol.bind_symbolic_shape %arg0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %0 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%0 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %1, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %2 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %2, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %1 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%2 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %3, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  return %1, %3 : tensor<?x640x?xf16>, tensor<?x640x?xf16>
}

// -----

// CHECK: tensor<?x640x?xf16, [@S0, 640, @S1]>
// CHECK: tensor<?x640x?xf16, [@S2, 640, @S1]>
func.func @test_symbol_to_encoding_1(%arg0: tensor<?x640x?xf16>) -> (tensor<?x640x?xf16>, tensor<?x640x?xf16>) {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  symbol.bind_symbolic_shape %arg0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %0 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%0 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %1, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %S2 = symbol.symbolic_int @S2 [%S0, %S0], affine_map<()[s0, s1] -> (s0 + s1)> {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %concat = tensor.concat dim(0) %1, %1 : (tensor<?x640x?xf16>, tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %concat, [%S2, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %2 = tensor.empty(%S2, %S1) : tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %2, [%S2, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%concat, %concat : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%2 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %3, [%S2, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  return %1, %3 : tensor<?x640x?xf16>, tensor<?x640x?xf16>
}

// -----

func.func @test_symbol_to_encoding_callee_0(%arg0: tensor<?x640x?xf16>) -> tensor<?x640x?xf16> {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  symbol.bind_symbolic_shape %arg0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %0 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%0 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  symbol.bind_symbolic_shape %1, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  return %1 : tensor<?x640x?xf16>
}

// CHECK-LABEL: func.func @test_symbol_to_encoding_caller_0(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x640x?xf16>
// CHECK: %[[CAST1:.*]] = tensor.cast %[[ARG0]] : tensor<?x640x?xf16> to tensor<?x640x?xf16, [@S0, 640, @S1]>
// CHECK: %[[RET:.*]] = call @test_symbol_to_encoding_callee_0(%[[CAST1]]) : (tensor<?x640x?xf16, [@S0, 640, @S1]>) -> tensor<?x640x?xf16, [@S0, 640, @S1]>
// CHECK: %[[CAST2:.*]] = tensor.cast %[[RET]] : tensor<?x640x?xf16, [@S0, 640, @S1]> to tensor<?x640x?xf16>
// CHECK: return %[[CAST2]] : tensor<?x640x?xf16>
func.func @test_symbol_to_encoding_caller_0(%arg0: tensor<?x640x?xf16>) -> tensor<?x640x?xf16> {
  %0 = func.call @test_symbol_to_encoding_callee_0(%arg0) : (tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  return %0 : tensor<?x640x?xf16>
}

