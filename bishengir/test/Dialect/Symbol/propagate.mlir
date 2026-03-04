// RUN: bishengir-opt -propagate-symbol %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: test_build_and_propagate_symbol_0(
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
func.func @test_build_and_propagate_symbol_0(%arg0: tensor<?x640x?xf16>) -> (tensor<?x640x?xf16>, tensor<?x640x?xf16>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x640x?xf16>
  %dim1 = tensor.dim %arg0, %c2 : tensor<?x640x?xf16>
  %out0 = tensor.empty(%dim0, %dim1) : tensor<?x640x?xf16>
  %add0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
          ins(%arg0, %arg0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) 
          outs(%out0 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>

  %dim2 = tensor.dim %add0, %c0 : tensor<?x640x?xf16>
  %dim3 = tensor.dim %add0, %c2 : tensor<?x640x?xf16>
  %out1 = tensor.empty(%dim2, %dim3) : tensor<?x640x?xf16>
  %add1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
          ins(%add0, %add0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) 
          outs(%out1 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  
  return %add0, %add1 : tensor<?x640x?xf16>, tensor<?x640x?xf16>
}

// -----

// CHECK-LABEL: test_build_and_propagate_symbol_1(
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
func.func @test_build_and_propagate_symbol_1(%arg0: tensor<?x640x?xf16>) -> (tensor<?x640x?xf16>, tensor<?x640x?xf16>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x640x?xf16>
  %dim1 = tensor.dim %arg0, %c2 : tensor<?x640x?xf16>
  %out0 = tensor.empty(%dim0, %dim1) : tensor<?x640x?xf16>
  %add0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
          ins(%arg0, %arg0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) 
          outs(%out0 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>

  %concat = tensor.concat dim(0) %add0, %add0 : (tensor<?x640x?xf16>, tensor<?x640x?xf16>) -> tensor<?x640x?xf16>

  %dim2 = tensor.dim %concat, %c0 : tensor<?x640x?xf16>
  %dim3 = tensor.dim %concat, %c2 : tensor<?x640x?xf16>
  %out1 = tensor.empty(%dim2, %dim3) : tensor<?x640x?xf16>
  %add1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
          ins(%concat, %concat : tensor<?x640x?xf16>, tensor<?x640x?xf16>) 
          outs(%out1 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  
  return %add0, %add1 : tensor<?x640x?xf16>, tensor<?x640x?xf16>
}

// -----

// CHECK-LABEL: test_already_bind_symbol_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<?x640x?xf16>
// CHECK: %[[S0:.*]] = symbol.symbolic_int
// CHECK: %[[S1:.*]] = symbol.symbolic_int
// CHECK: symbol.bind_symbolic_shape %[[arg0]], [%[[S0]], %[[S1]]]
// CHECK-NOT: symbol.bind_symbolic_shape %[[arg0]]
func.func @test_already_bind_symbol_0(%arg0: tensor<?x640x?xf16>) -> tensor<?x640x?xf16> {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  symbol.bind_symbolic_shape %arg0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
  %0 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%0 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
  return %1 : tensor<?x640x?xf16>
}