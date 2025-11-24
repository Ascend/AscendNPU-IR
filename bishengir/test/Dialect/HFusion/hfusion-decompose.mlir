// RUN: bishengir-opt --hfusion-decompose="hfusion-decompose-phase=after-hfusion-flatten" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_isfinite
func.func @test_isfinite() -> tensor<8192xi1> {
  // CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<8192xf32>
  %0 = tensor.empty() : tensor<8192xf32>
  // CHECK: %[[ISINF:.*]] = hfusion.isinf %[[ZERO:.*]] : tensor<8192xf32> -> tensor<8192xi1>
  // CHECK: %[[ISNAN:.*]] = hfusion.isnan %[[ZERO:.*]] : tensor<8192xf32> -> tensor<8192xi1>
  // CHECK: %[[VOROUTPUT:.*]] = tensor.empty() : tensor<8192xi1>
  // CHECK: %[[VOR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[ISINF:.*]], %[[ISNAN:.*]] : tensor<8192xi1>, tensor<8192xi1>) outs(%[[VOROUTPUT:.*]] : tensor<8192xi1>) -> tensor<8192xi1>
  // CHECK: %[[VNOTOUTPUT:.*]] = tensor.empty() : tensor<8192xi1>
  // CHECK: %[[VNOT:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VOR:.*]] : tensor<8192xi1>) outs(%[[VNOTOUTPUT:.*]] : tensor<8192xi1>) -> tensor<8192xi1>
  // CHECK: return %[[VNOT:.*]] : tensor<8192xi1>
  %2 = hfusion.isfinite %0 : tensor<8192xf32> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}

// -----

// CHECK-LABEL: func.func @test_linalg_decompose_multiaxis_transpose
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x16x8x4x3xf32>) -> tensor<2x3x4x8x16xf32>
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<2x3x8x4x16xf32>
// CHECK: %[[trans0:.*]] = linalg.transpose ins(%[[arg0]] : tensor<2x16x8x4x3xf32>) outs(%[[empty0]] : tensor<2x3x8x4x16xf32>) permutation = [0, 4, 2, 3, 1]
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<2x3x4x8x16xf32>
// CHECK: %[[trans1:.*]] = linalg.transpose ins(%[[trans0]] : tensor<2x3x8x4x16xf32>) outs(%[[empty1]] : tensor<2x3x4x8x16xf32>) permutation = [0, 1, 3, 2, 4]
func.func @test_linalg_decompose_multiaxis_transpose(%arg0: tensor<2x16x8x4x3xf32>) -> tensor<2x3x4x8x16xf32> {
  %0 = tensor.empty() : tensor<2x3x4x8x16xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<2x16x8x4x3xf32>) outs(%0 : tensor<2x3x4x8x16xf32>) permutation = [0, 4, 3, 2, 1]
  return %1 : tensor<2x3x4x8x16xf32>
}

// -----

// CHECK-LABEL: func.func @test_linalg_decompose_multiaxis_transpose_dyn
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x16x8x4x3xf32>) -> tensor<3x4x8x16x?xf32>
// CHECK: %[[c4:.*]] = arith.constant 4 : index
// CHECK: %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[dim0:.*]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x16x8x4x3xf32>
// CHECK: %[[empty0:.*]] = tensor.empty(%[[dim0]]) : tensor<3x16x8x4x?xf32>
// CHECK: %[[trans0:.*]] = linalg.transpose ins(%[[arg0]] : tensor<?x16x8x4x3xf32>) outs(%[[empty0]] : tensor<3x16x8x4x?xf32>) permutation = [4, 1, 2, 3, 0]
// CHECK: %[[dim1:.*]] = tensor.dim %[[trans0]], %[[c4]] : tensor<3x16x8x4x?xf32>
// CHECK: %[[empty1:.*]] = tensor.empty(%[[dim1]]) : tensor<3x4x8x16x?xf32>
// CHECK: %[[trans1:.*]] = linalg.transpose ins(%[[trans0]] : tensor<3x16x8x4x?xf32>) outs(%[[empty1]] : tensor<3x4x8x16x?xf32>) permutation = [0, 3, 2, 1, 4]
func.func @test_linalg_decompose_multiaxis_transpose_dyn(%arg0: tensor<?x16x8x4x3xf32>) -> tensor<3x4x8x16x?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x16x8x4x3xf32>
  %0 = tensor.empty(%dim) : tensor<3x4x8x16x?xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<?x16x8x4x3xf32>) outs(%0 : tensor<3x4x8x16x?xf32>) permutation = [4, 3, 2, 1, 0]
  return %1 : tensor<3x4x8x16x?xf32>
}

// -----

// CHECK-LABEL: test_decompose_gather
func.func @test_decompose_gather(%src:tensor<4x16x16x16x8xf16>, %idx:tensor<4x16x4x16x8xi32>) -> tensor<4x16x4x16x8xf16>{
  %init = tensor.empty() : tensor<4x16x4x16x8xf16>
  
  // CHECK-DAG: %[[C8:[0-9a-z]+]] = arith.constant 8 : index
  // CHECK-DAG: %[[C16:[0-9a-z]+]] = arith.constant 16 : index
  // CHECK-DAG: %[[C4:[0-9a-z]+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C1:[0-9a-z]+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:[0-9a-z]+]] = arith.constant 0 : index
  // CHECK-NOT: gather
  // CHECK: scf.for
  // CHECK-SAME: %[[C0]] to %[[C4]] step %[[C1]]
  // CHECK: scf.for
  // CHECK-SAME: %[[C0]] to %[[C16]] step %[[C1]]
  // CHECK: scf.for
  // CHECK-SAME: %[[C0]] to %[[C4]] step %[[C1]]
  // CHECK: scf.for
  // CHECK-SAME: %[[C0]] to %[[C16]] step %[[C1]]
  // CHECK: scf.for
  // CHECK-SAME: %[[C0]] to %[[C8]] step %[[C1]]
  // CHECK: tensor.extract
  // CHECK: tensor.extract
  // CHECK: tensor.insert
  %res = hfusion.gather ins(%src, %idx : tensor<4x16x16x16x8xf16>, tensor<4x16x4x16x8xi32>) outs(%init:tensor<4x16x4x16x8xf16>) axis = 2 -> tensor<4x16x4x16x8xf16>
  return %res : tensor<4x16x4x16x8xf16>
}
 
// -----

// CHECK-LABEL: test_decompose_gather_idx64
func.func @test_decompose_gather_idx64(%src: tensor<4x64xf32>, %idx: tensor<4x32xi64>) -> tensor<4x32xf32> {
  %init = tensor.empty() : tensor<4x32xf32>
  // CHECK:  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
  %res = hfusion.gather ins(%src, %idx : tensor<4x64xf32>, tensor<4x32xi64>) outs(%init : tensor<4x32xf32>) axis = 1 -> tensor<4x32xf32>
  return %res : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: test_decompose_gather_src64
func.func @test_decompose_gather_src64(%src: tensor<4x64xi64>, %idx: tensor<4x32xi32>) -> tensor<4x32xi64> {
  %init = tensor.empty() : tensor<4x32xi64>
  // CHECK-DAG: %[[C4:[0-9a-z]+]] = arith.constant 4 : index 
  // CHECK-DAG: %[[C32:[0-9a-z]+]] = arith.constant 32 : index 
  // CHECK-DAG: %[[C1:[0-9a-z]+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:[0-9a-z]+]] = arith.constant 0 : index
  // CHECK-NOT: gather
  // CHECK: scf.for
  // CHECK-SAME: %[[C0]] to %[[C4]] step %[[C1]]
  // CHECK: scf.for
  // CHECK-SAME: %[[C0]] to %[[C32]] step %[[C1]]
  // CHECK: tensor.extract
  // CHECK: tensor.extract
  // CHECK: tensor.insert
  %res = hfusion.gather ins(%src, %idx : tensor<4x64xi64>, tensor<4x32xi32>) outs(%init : tensor<4x32xi64>) axis = 1 -> tensor<4x32xi64>
  return %res : tensor<4x32xi64>
}

// -----

func.func @histogram_nomask(%arg0: tensor<8xi32>) -> tensor<4xi32> {
  // CHECK-LABEL: func.func @histogram_nomask
  // CHECK: scf.for
  // CHECK: tensor.extract
  // CHECK: arith.index_castui
  // CHECK: tensor.insert
  %res = hfusion.histogram %arg0, 4 : tensor<8xi32> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----

func.func @histogram_mask(%arg0: tensor<8xi32>, %mask: tensor<8xi1>)
    -> tensor<4xi32> {
  // CHECK-LABEL: func.func @histogram_mask
  // CHECK: scf.for
  // CHECK: tensor.extract
  // CHECK: scf.if
  // CHECK: tensor.insert
  %res = hfusion.histogram %arg0, 4, %mask
         : tensor<8xi32>, tensor<8xi1> -> tensor<4xi32>
  return %res : tensor<4xi32>
}
