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

// CHECK-LABEL:   func.func @histogram_nomask(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<8xi32>) -> tensor<4xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[VAL_9:.*]] = linalg.fill ins(%[[VAL_7]] : i32) outs(%[[VAL_8]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_3]] to %[[VAL_2]] step %[[VAL_4]] iter_args(%[[VAL_12:.*]] = %[[VAL_9]]) -> (tensor<4xi32>) {
// CHECK:             %[[VAL_13:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_11]]] : tensor<8xi32>
// CHECK:             %[[VAL_14:.*]] = arith.cmpi ult, %[[VAL_13]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_15:.*]] = arith.index_castui %[[VAL_13]] : i32 to index
// CHECK:             %[[VAL_16:.*]] = arith.select %[[VAL_14]], %[[VAL_3]], %[[VAL_15]] : index
// CHECK:             %[[VAL_17:.*]] = arith.cmpi uge, %[[VAL_16]], %[[VAL_5]] : index
// CHECK:             %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_3]], %[[VAL_16]] : index
// CHECK:             %[[VAL_19:.*]] = arith.ori %[[VAL_14]], %[[VAL_17]] : i1
// CHECK:             %[[VAL_20:.*]] = arith.xori %[[VAL_19]], %[[VAL_1]] : i1
// CHECK:             %[[VAL_21:.*]] = tensor.extract %[[VAL_12]]{{\[}}%[[VAL_18]]] : tensor<4xi32>
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_23:.*]] = tensor.insert %[[VAL_22]] into %[[VAL_12]]{{\[}}%[[VAL_18]]] : tensor<4xi32>
// CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_20]], %[[VAL_23]], %[[VAL_12]] : tensor<4xi32>
// CHECK:             scf.yield %[[VAL_24]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return %[[VAL_10]] : tensor<4xi32>
// CHECK:         }
func.func @histogram_nomask(%arg0: tensor<8xi32>) -> tensor<4xi32> {
  %res = hfusion.histogram %arg0, 4 : tensor<8xi32> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----

// CHECK-LABEL:   func.func @histogram_mask(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<8xi32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: tensor<8xi1>) -> tensor<4xi32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_8]] : i32) outs(%[[VAL_9]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (tensor<4xi32>) {
// CHECK:             %[[VAL_14:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_12]]] : tensor<8xi32>
// CHECK:             %[[VAL_15:.*]] = arith.cmpi ult, %[[VAL_14]], %[[VAL_8]] : i32
// CHECK:             %[[VAL_16:.*]] = arith.index_castui %[[VAL_14]] : i32 to index
// CHECK:             %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_4]], %[[VAL_16]] : index
// CHECK:             %[[VAL_18:.*]] = arith.cmpi uge, %[[VAL_17]], %[[VAL_6]] : index
// CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_4]], %[[VAL_17]] : index
// CHECK:             %[[VAL_20:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_12]]] : tensor<8xi1>
// CHECK:             %[[VAL_21:.*]] = arith.ori %[[VAL_15]], %[[VAL_18]] : i1
// CHECK:             %[[VAL_22:.*]] = arith.xori %[[VAL_21]], %[[VAL_2]] : i1
// CHECK:             %[[VAL_23:.*]] = arith.andi %[[VAL_20]], %[[VAL_22]] : i1
// CHECK:             %[[VAL_24:.*]] = tensor.extract %[[VAL_13]]{{\[}}%[[VAL_19]]] : tensor<4xi32>
// CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_26:.*]] = tensor.insert %[[VAL_25]] into %[[VAL_13]]{{\[}}%[[VAL_19]]] : tensor<4xi32>
// CHECK:             %[[VAL_27:.*]] = arith.select %[[VAL_23]], %[[VAL_26]], %[[VAL_13]] : tensor<4xi32>
// CHECK:             scf.yield %[[VAL_27]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return %[[VAL_11]] : tensor<4xi32>
// CHECK:         }
func.func @histogram_mask(%arg0: tensor<8xi32>, %mask: tensor<8xi1>)
    -> tensor<4xi32> {
  %res = hfusion.histogram %arg0, 4, %mask
         : tensor<8xi32>, tensor<8xi1> -> tensor<4xi32>
  return %res : tensor<4xi32>
}


// -----

// CHECK-LABEL:   func.func @histogram_nomask_i16(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<16xi16>) -> tensor<4xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i16
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_8]] : i32) outs(%[[VAL_9]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (tensor<4xi32>) {
// CHECK:             %[[VAL_14:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_12]]] : tensor<16xi16>
// CHECK:             %[[VAL_15:.*]] = arith.cmpi ult, %[[VAL_14]], %[[VAL_2]] : i16
// CHECK:             %[[VAL_16:.*]] = arith.index_castui %[[VAL_14]] : i16 to index
// CHECK:             %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_4]], %[[VAL_16]] : index
// CHECK:             %[[VAL_18:.*]] = arith.cmpi uge, %[[VAL_17]], %[[VAL_6]] : index
// CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_4]], %[[VAL_17]] : index
// CHECK:             %[[VAL_20:.*]] = arith.ori %[[VAL_15]], %[[VAL_18]] : i1
// CHECK:             %[[VAL_21:.*]] = arith.xori %[[VAL_20]], %[[VAL_1]] : i1
// CHECK:             %[[VAL_22:.*]] = tensor.extract %[[VAL_13]]{{\[}}%[[VAL_19]]] : tensor<4xi32>
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_24:.*]] = tensor.insert %[[VAL_23]] into %[[VAL_13]]{{\[}}%[[VAL_19]]] : tensor<4xi32>
// CHECK:             %[[VAL_25:.*]] = arith.select %[[VAL_21]], %[[VAL_24]], %[[VAL_13]] : tensor<4xi32>
// CHECK:             scf.yield %[[VAL_25]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return %[[VAL_11]] : tensor<4xi32>
// CHECK:         }
func.func @histogram_nomask_i16(%arg0: tensor<16xi16>) -> tensor<4xi32> {
  %res = hfusion.histogram %arg0, 4 : tensor<16xi16> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----

// CHECK-LABEL:   func.func @histogram_nomask_i8(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<8xi8>) -> tensor<6xi64> {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<6xi64>
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_8]] : i64) outs(%[[VAL_9]] : tensor<6xi64>) -> tensor<6xi64>
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (tensor<6xi64>) {
// CHECK:             %[[VAL_14:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_12]]] : tensor<8xi8>
// CHECK:             %[[VAL_15:.*]] = arith.cmpi ult, %[[VAL_14]], %[[VAL_2]] : i8
// CHECK:             %[[VAL_16:.*]] = arith.index_castui %[[VAL_14]] : i8 to index
// CHECK:             %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_4]], %[[VAL_16]] : index
// CHECK:             %[[VAL_18:.*]] = arith.cmpi uge, %[[VAL_17]], %[[VAL_6]] : index
// CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_4]], %[[VAL_17]] : index
// CHECK:             %[[VAL_20:.*]] = arith.ori %[[VAL_15]], %[[VAL_18]] : i1
// CHECK:             %[[VAL_21:.*]] = arith.xori %[[VAL_20]], %[[VAL_1]] : i1
// CHECK:             %[[VAL_22:.*]] = tensor.extract %[[VAL_13]]{{\[}}%[[VAL_19]]] : tensor<6xi64>
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_7]] : i64
// CHECK:             %[[VAL_24:.*]] = tensor.insert %[[VAL_23]] into %[[VAL_13]]{{\[}}%[[VAL_19]]] : tensor<6xi64>
// CHECK:             %[[VAL_25:.*]] = arith.select %[[VAL_21]], %[[VAL_24]], %[[VAL_13]] : tensor<6xi64>
// CHECK:             scf.yield %[[VAL_25]] : tensor<6xi64>
// CHECK:           }
// CHECK:           return %[[VAL_11]] : tensor<6xi64>
// CHECK:         }
func.func @histogram_nomask_i8(%arg0: tensor<8xi8>) -> tensor<6xi64> {
  %res = hfusion.histogram %arg0, 6 : tensor<8xi8> -> tensor<6xi64>
  return %res : tensor<6xi64>
}

// -----

// CHECK-LABEL:   func.func @histogram_nomask_i64(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<128xi64>) -> tensor<16xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_8]] : i32) outs(%[[VAL_9]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (tensor<16xi32>) {
// CHECK:             %[[VAL_14:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_12]]] : tensor<128xi64>
// CHECK:             %[[VAL_15:.*]] = arith.cmpi ult, %[[VAL_14]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_16:.*]] = arith.index_castui %[[VAL_14]] : i64 to index
// CHECK:             %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_4]], %[[VAL_16]] : index
// CHECK:             %[[VAL_18:.*]] = arith.cmpi uge, %[[VAL_17]], %[[VAL_6]] : index
// CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_4]], %[[VAL_17]] : index
// CHECK:             %[[VAL_20:.*]] = arith.ori %[[VAL_15]], %[[VAL_18]] : i1
// CHECK:             %[[VAL_21:.*]] = arith.xori %[[VAL_20]], %[[VAL_1]] : i1
// CHECK:             %[[VAL_22:.*]] = tensor.extract %[[VAL_13]]{{\[}}%[[VAL_19]]] : tensor<16xi32>
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_24:.*]] = tensor.insert %[[VAL_23]] into %[[VAL_13]]{{\[}}%[[VAL_19]]] : tensor<16xi32>
// CHECK:             %[[VAL_25:.*]] = arith.select %[[VAL_21]], %[[VAL_24]], %[[VAL_13]] : tensor<16xi32>
// CHECK:             scf.yield %[[VAL_25]] : tensor<16xi32>
// CHECK:           }
// CHECK:           return %[[VAL_11]] : tensor<16xi32>
// CHECK:         }
func.func @histogram_nomask_i64(%arg0: tensor<128xi64>) -> tensor<16xi32> {
  %res = hfusion.histogram %arg0, 16 : tensor<128xi64> -> tensor<16xi32>
  return %res : tensor<16xi32>
}
