// RUN: bishengir-opt --unfold-symbolic-int -split-input-file %s | FileCheck %s
// RUN: bishengir-opt --unfold-symbolic-int -split-input-file %s | FileCheck --check-prefix=NO-SYMBOLIC %s

// after unfold-symbolic-int shouldn't have any symbolic_int

// NO-SYMBOLIC-NOT: symbol.symbolic_int
// CHECK: @test_build_and_propagate_symbol_0(%[[ARG0:.*]]: tensor<?x640x?xf16>)
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[DIM2:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x640x?xf16>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x640x?xf16>
// CHECK: tensor.empty(%[[DIM0]], %[[DIM2]])
// CHECK: tensor.empty(%[[DIM0]], %[[DIM2]])
module {
  func.func @test_build_and_propagate_symbol_0(%arg0: tensor<?x640x?xf16>) -> (tensor<?x640x?xf16>, tensor<?x640x?xf16>) {
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
}

// -----

// NO-SYMBOLIC-NOT: symbol.symbolic_int
// CHECK: @test_build_and_propagate_symbol_1(%[[ARG0:.*]]: tensor<?x640x?xf16>)
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[DIM2:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x640x?xf16>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x640x?xf16>
// CHECK: tensor.empty(%[[DIM0]], %[[DIM2]])
// CHECK: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: %[[AFFINE:.*]] = affine.apply #map()[%[[DIM0]], %[[DIM0]]]
// CHECK: %[[CONCAT:.*]] = tensor.concat dim(0) %[[ADD]], %[[ADD]]
// CHECK: tensor.empty(%[[AFFINE]], %[[DIM2]])
module {
  func.func @test_build_and_propagate_symbol_1(%arg0: tensor<?x640x?xf16>) -> (tensor<?x640x?xf16>, tensor<?x640x?xf16>) {
    %S2 = symbol.symbolic_int @S2 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
    %S3 = symbol.symbolic_int @S3 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
    symbol.bind_symbolic_shape %arg0, [%S2, %S3], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    %0 = tensor.empty(%S2, %S3) : tensor<?x640x?xf16>
    symbol.bind_symbolic_shape %0, [%S2, %S3], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%0 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
    symbol.bind_symbolic_shape %1, [%S2, %S3], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    %S4 = symbol.symbolic_int @S4 [%S2, %S2], affine_map<()[s0, s1] -> (s0 + s1)> {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
    %concat = tensor.concat dim(0) %1, %1 : (tensor<?x640x?xf16>, tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
    symbol.bind_symbolic_shape %concat, [%S4, %S3], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    %2 = tensor.empty(%S4, %S3) : tensor<?x640x?xf16>
    symbol.bind_symbolic_shape %2, [%S4, %S3], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%concat, %concat : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%2 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
    symbol.bind_symbolic_shape %3, [%S4, %S3], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    return %1, %3 : tensor<?x640x?xf16>, tensor<?x640x?xf16>
  }
}

// -----

// NO-SYMBOLIC-NOT: symbol.symbolic_int
// CHECK: @test_already_bind_symbol_0(%[[ARG0:.*]]: tensor<?x640x?xf16>)
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[DIM2:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x640x?xf16>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x640x?xf16>
// CHECK: tensor.empty(%[[DIM0]], %[[DIM2]])
module {
  func.func @test_already_bind_symbol_0(%arg0: tensor<?x640x?xf16>) -> tensor<?x640x?xf16> {
    %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
    %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
    symbol.bind_symbolic_shape %arg0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    %0 = tensor.empty(%S0, %S1) : tensor<?x640x?xf16>
    symbol.bind_symbolic_shape %0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<?x640x?xf16>, tensor<?x640x?xf16>) outs(%0 : tensor<?x640x?xf16>) -> tensor<?x640x?xf16>
    symbol.bind_symbolic_shape %1, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
    return %1 : tensor<?x640x?xf16>
  }
}

// -----

// CHECK: @unfold_arg_index(%[[ARG0:.*]]: tensor<?x640xf16>, %[[ARG1:.*]]: i64)
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[ARG1]]
// CHECK: %[[AFFINE:.*]] = affine.apply #map()[%[[INDEX]]]
// CHECK: tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2]] output_shape [2, %[[AFFINE]], 640]
module {
  func.func @unfold_arg_index(%arg0: tensor<?x640xf16>, %arg1: i64) -> tensor<2x?x640xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = arith.index_cast %arg1 : i64 to index
    %S1 = symbol.symbolic_int @S1 [%0], affine_map<()[s0] -> (s0)> {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [2, %S1, 640] : tensor<?x640xf16> into tensor<2x?x640xf16>
    symbol.bind_symbolic_shape %expanded, [%S1], affine_map<()[s0] -> (2, s0, 640)> : tensor<2x?x640xf16>
    return %expanded : tensor<2x?x640xf16>
  }
}