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
  // CHECK: hfusion.cast {{.*}}
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
  // CHECK: arith.index_cast
  // CHECK: tensor.insert
  %res = hfusion.histogram %arg0, 4 : tensor<8xi32> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----
module {
  func.func @dot_scale_kernel_2D(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf8E5M2>, %arg3: i32, %arg4: memref<?xi8>, %arg5: memref<?xf8E5M2>, %arg6: i32, %arg7: memref<?xi8>, %arg8: memref<?xf32>, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf8E5M2> to memref<64x64xf8E5M2, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<64x64xf8E5M2>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf8E5M2, strided<[64, 1]>> to memref<64x64xf8E5M2>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf8E5M2>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 2], strides: [2, 1] : memref<?xi8> to memref<64x2xi8, strided<[2, 1]>>
    %alloc_1 = memref.alloc() : memref<64x2xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x2xi8, strided<[2, 1]>> to memref<64x2xi8>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x2xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf8E5M2> to memref<64x64xf8E5M2, strided<[64, 1]>>
    %alloc_3 = memref.alloc() : memref<64x64xf8E5M2>
    memref.copy %reinterpret_cast_2, %alloc_3 : memref<64x64xf8E5M2, strided<[64, 1]>> to memref<64x64xf8E5M2>
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<64x64xf8E5M2>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [64, 2], strides: [2, 1] : memref<?xi8> to memref<64x2xi8, strided<[2, 1]>>
    %alloc_5 = memref.alloc() : memref<64x2xi8>
    memref.copy %reinterpret_cast_4, %alloc_5 : memref<64x2xi8, strided<[2, 1]>> to memref<64x2xi8>
    %5 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x2xi8>
    // CHECK: %[[CST_7:.*]] = arith.constant 7 : i16
    // CHECK: %[[EMPTY_SCALE_A_I16:.*]] = tensor.empty() : tensor<64x2xi16>
    // CHECK: %[[SCALE_A_I16:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<64x2xi8>) outs(%[[EMPTY_SCALE_A_I16]] : tensor<64x2xi16>) -> tensor<64x2xi16>
    // CHECK: %[[EMPTY_SCALE_B_I16:.*]] = tensor.empty() : tensor<64x2xi16>
    // CHECK: %[[SCALE_B_I16:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%{{.*}}: tensor<64x2xi8>) outs(%[[EMPTY_SCALE_B_I16]] : tensor<64x2xi16>) -> tensor<64x2xi16>
    // CHECK: %[[SEVEN_A:.*]] = linalg.fill ins(%[[CST_7]] : i16) outs(%[[EMPTY_SCALE_A_I16]] : tensor<64x2xi16>) -> tensor<64x2xi16>
    // CHECK: %[[SCALE_A_SHL_7:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[SCALE_A_I16]], %[[SEVEN_A]] : tensor<64x2xi16>, tensor<64x2xi16>) outs(%[[EMPTY_SCALE_A_I16]] : tensor<64x2xi16>) -> tensor<64x2xi16>
    // CHECK: %[[SEVEN_B:.*]] = linalg.fill ins(%[[CST_7]] : i16) outs(%[[EMPTY_SCALE_B_I16]] : tensor<64x2xi16>) -> tensor<64x2xi16>
    // CHECK: %[[SCALE_B_SHL_7:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[SCALE_B_I16]], %[[SEVEN_B]] : tensor<64x2xi16>, tensor<64x2xi16>) outs(%[[EMPTY_SCALE_B_I16]] : tensor<64x2xi16>) -> tensor<64x2xi16>
    // CHECK: %[[EMPTY_SCALE_A_BF16:.*]] = tensor.empty() : tensor<64x2xbf16>
    // CHECK: %[[SCALE_A_BF16:.*]] = hfusion.bitcast ins(%[[SCALE_A_SHL_7]] : tensor<64x2xi16>) outs(%[[EMPTY_SCALE_A_BF16]] : tensor<64x2xbf16>) -> tensor<64x2xbf16>
    // CHECK: %[[EMPTY_SCALE_B_BF16:.*]] = tensor.empty() : tensor<64x2xbf16>
    // CHECK: %[[SCALE_B_BF16:.*]] = hfusion.bitcast ins(%[[SCALE_B_SHL_7]] : tensor<64x2xi16>) outs(%[[EMPTY_SCALE_B_BF16]] : tensor<64x2xbf16>) -> tensor<64x2xbf16>
    // CHECK: %[[EMPTY_SCALE_A_F32:.*]] = tensor.empty() : tensor<64x2xf32>
    // CHECK: %[[SCALE_A_F32:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[SCALE_A_BF16]] : tensor<64x2xbf16>) outs(%[[EMPTY_SCALE_A_F32]] : tensor<64x2xf32>) -> tensor<64x2xf32>
    // CHECK: %[[EMPTY_SCALE_B_F32:.*]] = tensor.empty() : tensor<64x2xf32>
    // CHECK: %[[SCALE_B_F32:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[SCALE_B_BF16]] : tensor<64x2xbf16>) outs(%[[EMPTY_SCALE_B_F32]] : tensor<64x2xf32>) -> tensor<64x2xf32>
    // CHECK: %[[EMPTY_A_F16:.*]] = tensor.empty() : tensor<64x64xf16>
    // CHECK: %[[A_F16:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<64x64xf8E5M2>) outs(%[[EMPTY_A_F16]] : tensor<64x64xf16>) -> tensor<64x64xf16>
    // CHECK: %[[EMPTY_B_F16:.*]] = tensor.empty() : tensor<64x64xf16>
    // CHECK: %[[B_F16:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<64x64xf8E5M2>) outs(%[[EMPTY_B_F16]] : tensor<64x64xf16>) -> tensor<64x64xf16>
    // CHECK: %[[EMPTY_SCALE_A_F16:.*]] = tensor.empty() : tensor<64x2xf16>
    // CHECK: %[[SCALE_A_F16:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[SCALE_A_F32]] : tensor<64x2xf32>) outs(%[[EMPTY_SCALE_A_F16]] : tensor<64x2xf16>) -> tensor<64x2xf16>
    // CHECK: %[[EMPTY_SCALE_B_F16:.*]] = tensor.empty() : tensor<64x2xf16>
    // CHECK: %[[SCALE_B_F16:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[SCALE_B_F32]] : tensor<64x2xf32>) outs(%[[EMPTY_SCALE_B_F16]] : tensor<64x2xf16>) -> tensor<64x2xf16>
    // CHECK: %[[EMPTY_SCALE_A_BRC:.*]] = tensor.empty() : tensor<64x2x32xf16>
    // CHECK: %[[SCALE_A_BRC:.*]] = linalg.broadcast ins(%[[SCALE_A_F16]] : tensor<64x2xf16>) outs(%[[EMPTY_SCALE_A_BRC]] : tensor<64x2x32xf16>) dimensions = [2] 
    // CHECK: %[[SCALE_A_COLLAPSED:.*]] = tensor.collapse_shape %[[SCALE_A_BRC]] {{\[}}[0], [1, 2]] : tensor<64x2x32xf16> into tensor<64x64xf16>
    // CHECK: %[[EMPTY_SCALE_B_BRC:.*]] = tensor.empty() : tensor<64x2x32xf16>
    // CHECK: %[[SCALE_B_BRC:.*]] = linalg.broadcast ins(%[[SCALE_B_F16]] : tensor<64x2xf16>) outs(%[[EMPTY_SCALE_B_BRC]] : tensor<64x2x32xf16>) dimensions = [2] 
    // CHECK: %[[SCALE_B_COLLAPSED:.*]] = tensor.collapse_shape %[[SCALE_B_BRC]] {{\[}}[0], [1, 2]] : tensor<64x2x32xf16> into tensor<64x64xf16>
    // CHECK: %[[EMPTY_SCALE_B_TRANSPOSED:.*]] = tensor.empty() : tensor<64x64xf16>
    // CHECK: %[[SCALE_B_TRANSPOSED:.*]] = linalg.transpose ins(%[[SCALE_B_COLLAPSED]] : tensor<64x64xf16>) outs(%[[EMPTY_SCALE_B_TRANSPOSED]] : tensor<64x64xf16>) permutation = [1, 0] 
    // CHECK: %[[EMPTY_A_FINAL:.*]] = tensor.empty() : tensor<64x64xf16>
    // CHECK: %[[EMPTY_B_FINAL:.*]] = tensor.empty() : tensor<64x64xf16>
    // CHECK: %[[A_FINAL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[A_F16]], %[[SCALE_A_COLLAPSED]] : tensor<64x64xf16>, tensor<64x64xf16>) outs(%[[EMPTY_A_FINAL]] : tensor<64x64xf16>) -> tensor<64x64xf16>
    // CHECK: %[[B_FINAL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[B_F16]], %[[SCALE_B_TRANSPOSED]] : tensor<64x64xf16>, tensor<64x64xf16>) outs(%[[EMPTY_B_FINAL]] : tensor<64x64xf16>) -> tensor<64x64xf16>
    // CHECK: %[[RES:.*]] = linalg.matmul ins(%[[A_FINAL]], %[[B_FINAL]] : tensor<64x64xf16>, tensor<64x64xf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %6 = hfusion.matmul_mx ins(%2, %4, %3, %5 : tensor<64x64xf8E5M2>, tensor<64x64xf8E5M2>, tensor<64x2xi8>, tensor<64x2xi8>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_6 : (tensor<64x64xf32>, memref<64x64xf32, strided<[64, 1]>>) -> ()
    return
  }
}