// RUN: bishengir-opt -propagate-symbol %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: test_symbol_unify_by_binary_0(
// CHECK: %[[S0:.*]] = symbol.symbolic_int
// CHECK: symbol.symbolic_int
// CHECK: %[[slice:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0] [%[[S0]], 32, 128]
// CHECK: symbol.bind_symbolic_shape %[[slice]], [%[[S0]]], affine_map<()[s0] -> (s0, 32, 128)>
func.func @test_symbol_unify_by_binary_0(%arg0: tensor<?x32x128xf32>, %arg1: tensor<?x32x128xf32>, %arg2: i64) -> tensor<?x32x128xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x32x128xf32>
  %0 = arith.index_cast %dim : index to i64
  %1 = arith.addi %arg2, %0 : i64
  %2 = arith.index_cast %1 : i64 to index
  %extracted_slice = tensor.extract_slice %arg1[0, 0, 0] [%2, 32, 128] [1, 1, 1] : tensor<?x32x128xf32> to tensor<?x32x128xf32>
  %3 = tensor.empty(%dim) : tensor<?x32x128xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %extracted_slice : tensor<?x32x128xf32>, tensor<?x32x128xf32>) outs(%3 : tensor<?x32x128xf32>) -> tensor<?x32x128xf32>
  return %4 : tensor<?x32x128xf32>
}

// -----

// CHECK-LABEL: test_symbol_unify_by_binary_1(
// CHECK: %[[S0:.*]] = symbol.symbolic_int @[[S0]]
// CHECK: %[[cast0:.*]] = hfusion.cast
// CHECK: symbol.bind_symbolic_shape %[[cast0]], [%[[S0]]]
// CHECK: %[[cast1:.*]] = hfusion.cast
// CHECK: symbol.bind_symbolic_shape %[[cast1]], [%[[S0]]]
// CHECK: %[[binary:.*]] = hfusion.elemwise_binary
// CHECK: symbol.bind_symbolic_shape %[[binary]], [%[[S0]]]
func.func @test_symbol_unify_by_binary_1(%arg0: tensor<?x32x128xf32>, %arg1: tensor<?x32x128xf32>) -> (tensor<?x32x128xbf16>, tensor<?x32x128xbf16>, tensor<?x32x128xf32>) {
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?x32x128xf32>
    %dim1 = tensor.dim %arg1, %c0 : tensor<?x32x128xf32>
    %dim2 = tensor.dim %arg1, %c0 : tensor<?x32x128xf32>

    %0 = tensor.empty(%dim0) : tensor<?x32x128xbf16>
    %1 = tensor.empty(%dim1) : tensor<?x32x128xbf16>
    %2 = tensor.empty(%dim2) : tensor<?x32x128xf32>

    %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<?x32x128xf32>) outs(%0 : tensor<?x32x128xbf16>) -> tensor<?x32x128xbf16>
    %4 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<?x32x128xf32>) outs(%1 : tensor<?x32x128xbf16>) -> tensor<?x32x128xbf16>
    %5 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0,  %arg1: tensor<?x32x128xf32>, tensor<?x32x128xf32>) outs(%2: tensor<?x32x128xf32>) -> tensor<?x32x128xf32>
    return %3, %4, %5: tensor<?x32x128xbf16>, tensor<?x32x128xbf16>, tensor<?x32x128xf32>
}

// -----

// CHECK-LABEL: test_symbol_unify_operands_and_results_by_binary(
// CHECK: %[[S0:.*]] = symbol.symbolic_int @[[S0]]
// CHECK: symbol.bind_symbolic_shape {{.*}} [%[[S0]]]
// CHECK: symbol.bind_symbolic_shape {{.*}} [%[[S0]]]
// CHECK: symbol.bind_symbolic_shape {{.*}} [%[[S0]]]
// CHECK: %[[binary:.*]] = linalg.elemwise_binary
// CHECK: symbol.bind_symbolic_shape %[[binary]], [%[[S0]]]
func.func @test_symbol_unify_operands_and_results_by_binary(%arg0: tensor<1x?x4096xf32>, %arg1: tensor<1x?x4096xf32>, %arg2: tensor<1x?x4096xf32>) -> tensor<1x?x4096xf32> {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  symbol.bind_symbolic_shape %arg0, [%S0], affine_map<()[s0] -> (1, s0, 4096)> : tensor<1x?x4096xf32>
  symbol.bind_symbolic_shape %arg1, [%S0], affine_map<()[s0] -> (1, s0, 4096)> : tensor<1x?x4096xf32>
  %S2 = symbol.symbolic_int @S2 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  symbol.bind_symbolic_shape %arg2, [%S2], affine_map<()[s0] -> (1, s0, 4096)> : tensor<1x?x4096xf32>
  %0 = tensor.empty(%S0) : tensor<1x?x4096xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<1x?x4096xf32>, tensor<1x?x4096xf32>) outs(%0 : tensor<1x?x4096xf32>) -> tensor<1x?x4096xf32>
  symbol.bind_symbolic_shape %1, [%S2], affine_map<()[s0] -> (1, s0, 4096)> : tensor<1x?x4096xf32>
  return %1 : tensor<1x?x4096xf32>
}
