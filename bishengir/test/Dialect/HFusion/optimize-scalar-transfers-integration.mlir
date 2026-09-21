// RUN: split-file %s %t
// RUN: bishengir-opt %t/inline-brc.mlir --hfusion-optimize-scalar-transfers --cse --symbol-dce --convert-arith-to-hfusion --hfusion-inline-brc | FileCheck %s --check-prefix=INLINE --implicit-check-not=linalg.fill --implicit-check-not=tensor.extract --implicit-check-not=tensor.insert
// RUN: bishengir-opt %t/extract-fill.mlir --hfusion-optimize-scalar-transfers --cse --symbol-dce --convert-arith-to-hfusion --hfusion-pre-vectorization-fusion --mlir-print-local-scope | FileCheck %s --check-prefix=FILL --implicit-check-not=linalg.fill --implicit-check-not=tensor.extract --implicit-check-not=tensor.insert
// RUN: bishengir-opt %t/combined.mlir --hfusion-optimize-scalar-transfers --cse --symbol-dce --convert-arith-to-hfusion --hfusion-inline-brc --hfusion-pre-vectorization-fusion --mlir-print-local-scope | FileCheck %s --check-prefix=CHAIN --implicit-check-not=linalg.fill --implicit-check-not=tensor.extract --implicit-check-not=tensor.insert --implicit-check-not=linalg.elemwise --implicit-check-not=linalg.generic

//--- inline-brc.mlir

// InlineBrc must inline the singleton fill created for the scalar constant
// operand of the lifted add.
// INLINE-LABEL: func.func @inline_generated_adaptor(
// INLINE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1xf32>
// INLINE: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// INLINE: %[[SUM:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// INLINE-SAME: ins(%[[INPUT]], %[[CST]] : tensor<1xf32>, f32)
// INLINE: return %[[SUM]] : tensor<1xf32>
func.func @inline_generated_adaptor(%input: tensor<1xf32>) -> tensor<1xf32> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<1xf32>
  %scalar = tensor.extract %input[%c0] : tensor<1xf32>
  %sum = arith.addf %scalar, %cst : f32
  %result = tensor.insert %sum into %empty[%c0] : tensor<1xf32>
  return %result : tensor<1xf32>
}

//--- extract-fill.mlir

// Lifting leaves an extract before the fill. PreVectorizationFusion must
// replace that boundary with an implicit singleton broadcast.
// FILL-LABEL: func.func @eliminate_extract_before_fill(
// FILL-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1xf32>
// FILL: %[[RESULT:.*]] = linalg.generic
// FILL-SAME: indexing_maps = [affine_map<(d0) -> (0)>, affine_map<(d0) -> (d0)>]
// FILL-SAME: iterator_types = ["parallel"]
// FILL-SAME: ins(%[[INPUT]] : tensor<1xf32>)
// FILL: ^bb0(%[[VALUE:[a-zA-Z0-9_]+]]: f32,
// FILL: %[[SUM:.*]] = arith.addf %[[VALUE]], %[[VALUE]] : f32
// FILL: linalg.yield %[[SUM]] : f32
// FILL: } -> tensor<8xf32>
// FILL: return %[[RESULT]] : tensor<8xf32>
func.func @eliminate_extract_before_fill(%input: tensor<1xf32>) -> tensor<8xf32> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<8xf32>
  %scalar = tensor.extract %input[%c0] : tensor<1xf32>
  %sum = arith.addf %scalar, %scalar : f32
  %result = linalg.fill ins(%sum : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

//--- combined.mlir

// The constant adaptor and extract/fill boundary must disappear, leaving the
// entire lifted arithmetic chain in one elementwise region without VFFusion.
// CHAIN-LABEL: func.func @fuse_lifted_chain(
// CHAIN-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1xf32>
// CHAIN: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHAIN: %[[RESULT:.*]] = linalg.generic
// CHAIN-SAME: indexing_maps = [affine_map<(d0) -> (0)>, affine_map<(d0) -> (d0)>]
// CHAIN-SAME: iterator_types = ["parallel"]
// CHAIN-SAME: ins(%[[INPUT]] : tensor<1xf32>)
// CHAIN: ^bb0(%[[VALUE:[a-zA-Z0-9_]+]]: f32,
// CHAIN: %[[SUM:.*]] = arith.addf %[[VALUE]], %[[CST]] : f32
// CHAIN: %[[PRODUCT:.*]] = arith.mulf %[[SUM]], %[[SUM]] : f32
// CHAIN: %[[ROOT:.*]] = arith.addf %[[PRODUCT]], %[[PRODUCT]] : f32
// CHAIN: linalg.yield %[[ROOT]] : f32
// CHAIN: } -> tensor<8xf32>
// CHAIN: return %[[RESULT]] : tensor<8xf32>
func.func @fuse_lifted_chain(%input: tensor<1xf32>) -> tensor<8xf32> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %scalar = tensor.extract %input[%c0] : tensor<1xf32>
  %sum = arith.addf %scalar, %cst : f32
  %product = arith.mulf %sum, %sum : f32
  %root = arith.addf %product, %product : f32
  %result = linalg.fill ins(%root : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}
