// RUN: bishengir-opt -propagate-symbol %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: test_not_bind_symbol_for_static_output_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<2x32x20x?xf32>)
// CHECK: symbol.bind_symbolic_shape %[[arg0]]
// CHECK: %[[reduced:.*]] = linalg.reduce
// CHECK: symbol.bind_symbolic_shape %[[reduced]], [], affine_map<() -> (2, 32)>
func.func @test_not_bind_symbol_for_static_output_0(%arg0: tensor<2x32x20x?xf32>) -> tensor<2x32xf32> {
  %0 = tensor.empty() : tensor<2x32xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<2x32x20x?xf32>) outs(%0 : tensor<2x32xf32>) dimensions = [2, 3]
    (%in: f32, %init: f32) {
      %1 = arith.addf %in, %init : f32
      linalg.yield %1 : f32
    }
  return %reduced : tensor<2x32xf32>
}

// -----

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

// -----

// CHECK-LABEL: test_symbol_from_arith_op_0(
// CHECK: symbol.symbolic_int
// CHECK: symbol.symbolic_int
// CHECK: %[[cast:.*]] = arith.index_cast {{.*}} : i64 to index
// CHECK: %[[S2:.*]] = symbol.symbolic_int @[[S2]] [%[[cast]]]
// CHECK: %[[slice:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0] [%[[S2]], 32, 128]
// CHECK: symbol.bind_symbolic_shape %[[slice]], [%[[S2]]], affine_map<()[s0] -> (s0, 32, 128)>
func.func @test_symbol_from_arith_op_0(%arg0: tensor<?x32x128xf32>, %arg1: tensor<?x32x128xf32>, %arg2: i64) -> tensor<?x32x128xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x32x128xf32>
  %0 = arith.index_cast %dim : index to i64
  %1 = arith.addi %arg2, %0 : i64
  %2 = arith.index_cast %1 : i64 to index
  %extracted_slice = tensor.extract_slice %arg1[0, 0, 0] [%2, 32, 128] [1, 1, 1] : tensor<?x32x128xf32> to tensor<?x32x128xf32>
  return %extracted_slice : tensor<?x32x128xf32>
}

// -----

// CHECK-LABEL: test_fold_symbol_to_tensor_empty(
// CHECK-DAG: %[[S0:.*]] = symbol.symbolic_int
// CHECK-DAG: %[[S1:.*]] = symbol.symbolic_int
// CHECK: tensor.empty(%[[S1]])
func.func @test_fold_symbol_to_tensor_empty(%arg0: tensor<?x1152xbf16>, %arg1: tensor<?x1024x1152xf32>) -> tensor<?x1024x1152xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x1152xbf16>
  %0 = affine.apply affine_map<()[s0] -> (s0 floordiv 1024)>()[%dim]
  %1 = tensor.empty(%0) : tensor<?x1024x1152xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
       ins(%arg1, %arg1 : tensor<?x1024x1152xf32>, tensor<?x1024x1152xf32>)
       outs(%1 : tensor<?x1024x1152xf32>) -> tensor<?x1024x1152xf32>
  return %2 : tensor<?x1024x1152xf32>
}

// -----

// CHECK-LABEL: test_unify_same_operands_and_result_shape_0(
// CHECK: %[[S0:.*]] = symbol.symbolic_int
// CHECK-NOT: symbol.symbolic_int
// CHECK: symbol.bind_symbolic_shape {{.*}} [%[[S0]]], affine_map<()[s0] -> (s0, 1024, 1152)>
func.func @test_unify_same_operands_and_result_shape_0(%arg0: tensor<?x1024x1152xf32>, %arg1: tensor<?x1024x1152xf32>,
          %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> tensor<?x1024x1152xf16> {

  %size = arith.index_cast %arg2 : i64 to index
  %dst = tensor.empty(%size) : tensor<?x1024x1152xf32>
  %load = hfusion.load ins(%arg0 : tensor<?x1024x1152xf32>)
          outs(%dst : tensor<?x1024x1152xf32>) -> tensor<?x1024x1152xf32>

  %0 = arith.index_cast %arg3 : i64 to index
  %1 = tensor.empty(%0) : tensor<?x1024x1152xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
       ins(%load, %arg1 : tensor<?x1024x1152xf32>, tensor<?x1024x1152xf32>)
       outs(%1 : tensor<?x1024x1152xf32>) -> tensor<?x1024x1152xf32>

  %3 = arith.index_cast %arg4 : i64 to index
  %4 = tensor.empty(%3) : tensor<?x1024x1152xf32>
  %5 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
       ins(%2, %arg1 : tensor<?x1024x1152xf32>, tensor<?x1024x1152xf32>)
       outs(%4 : tensor<?x1024x1152xf32>) -> tensor<?x1024x1152xf32>

  %6 = arith.index_cast %arg5 : i64 to index
  %7 = tensor.empty(%6) : tensor<?x1024x1152xf32>
  %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
       ins(%5 : tensor<?x1024x1152xf32>) outs(%7 : tensor<?x1024x1152xf32>) -> tensor<?x1024x1152xf32>

  %9 = arith.index_cast %arg6 : i64 to index
  %10 = tensor.empty(%9) : tensor<?x1024x1152xf32>
  %11 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
        ins(%8 : tensor<?x1024x1152xf32>) outs(%10 : tensor<?x1024x1152xf32>) -> tensor<?x1024x1152xf32>

  %12 = arith.index_cast %arg7 : i64 to index
  %13 = tensor.empty(%12) : tensor<?x1024x1152xf16>
  %cast = hfusion.cast {round_mode = #hfusion.round_mode<round>}
          ins(%11 : tensor<?x1024x1152xf32>)
          outs(%13 : tensor<?x1024x1152xf16>) -> tensor<?x1024x1152xf16>

  %14 = arith.index_cast %arg8 : i64 to index
  %15 = tensor.empty(%14) : tensor<?x1024x1152xf16>
  %store = hfusion.store ins(%cast : tensor<?x1024x1152xf16>)
           outs(%15 : tensor<?x1024x1152xf16>) -> tensor<?x1024x1152xf16>
  return %store : tensor<?x1024x1152xf16>
}
