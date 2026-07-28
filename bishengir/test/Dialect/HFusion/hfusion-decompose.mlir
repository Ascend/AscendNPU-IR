// RUN: bishengir-opt --hfusion-decompose="hfusion-decompose-phase=after-hfusion-flatten" %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt --hfusion-decompose="hfusion-decompose-phase=before-lower-to-loops" %s -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CPU



// CHECK-LABEL: func.func @test_isfinite
func.func @test_isfinite() -> tensor<8192xi1> {
  // CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<8192xf32>
  %0 = tensor.empty() : tensor<8192xf32>
  // CHECK: %[[ISINF:.*]] = hfusion.isinf
  // CHECK: %[[ISNAN:.*]] = hfusion.isnan %[[ZERO]]
  // CHECK: %[[VOR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[ISINF]], %[[ISNAN]]
  // CHECK: %[[VNOT:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VOR]]
  // CHECK: return %[[VNOT]]
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
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[INIT:.*]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EMPTY]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[RES:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<4xi32>) {
// CHECK:             %[[EXTRACTED:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[IDX]]] : tensor<8xi32>
// CHECK:             %[[LT0:.*]] = arith.cmpi ult, %[[EXTRACTED]], %[[C0_I32]] : i32
// CHECK:             %[[CAST:.*]] = arith.index_castui %[[EXTRACTED]] : i32 to index
// CHECK:             %[[IDX_CLAMP_LOW:.*]] = arith.select %[[LT0]], %[[C0]], %[[CAST]] : index
// CHECK:             %[[GE4:.*]] = arith.cmpi uge, %[[IDX_CLAMP_LOW]], %[[C4]] : index
// CHECK:             %[[IDX_CLAMP_HIGH:.*]] = arith.select %[[GE4]], %[[C0]], %[[IDX_CLAMP_LOW]] : index
// CHECK:             %[[OR:.*]] = arith.ori %[[LT0]], %[[GE4]] : i1
// CHECK:             %[[IF_RES:.*]] = scf.if %[[OR]] -> (tensor<4xi32>) {
// CHECK:               scf.yield %[[ACC]] : tensor<4xi32>
// CHECK:             } else {
// CHECK:               %[[OLD_VAL:.*]] = tensor.extract %[[ACC]]{{\[}}%[[IDX_CLAMP_HIGH]]] : tensor<4xi32>
// CHECK:               %[[NEW_VAL:.*]] = arith.addi %[[OLD_VAL]], %[[C1_I32]] : i32
// CHECK:               %[[UPDATED:.*]] = tensor.insert %[[NEW_VAL]] into %[[ACC]]{{\[}}%[[IDX_CLAMP_HIGH]]] : tensor<4xi32>
// CHECK:               scf.yield %[[UPDATED]] : tensor<4xi32>
// CHECK:             }
// CHECK:             scf.yield %[[IF_RES]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return %[[RES]] : tensor<4xi32>
// CHECK:         }
func.func @histogram_nomask(%arg0: tensor<8xi32>) -> tensor<4xi32> {
  %res = hfusion.histogram %arg0, 4 : tensor<8xi32> -> tensor<4xi32>
  return %res : tensor<4xi32>
}


// -----

// CHECK-LABEL:   func.func @histogram_mask(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<8xi32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: tensor<8xi1>) -> tensor<4xi32> {
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[C0_I16:.*]] = arith.constant 0 : i16
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[INIT:.*]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EMPTY]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[CAST_DST:.*]] = tensor.empty() : tensor<8xi16>
// CHECK:           %[[MASK_CAST:.*]] = hfusion.cast {{.*}} ins(%[[VAL_1]] : tensor<8xi1>) outs(%[[CAST_DST]] : tensor<8xi16>) -> tensor<8xi16>
// CHECK:           %[[RES:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<4xi32>) {
// CHECK:             %[[X:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[IDX]]] : tensor<8xi32>
// CHECK:             %[[LT0:.*]] = arith.cmpi ult, %[[X]], %[[C0_I32]] : i32
// CHECK:             %[[CAST_IDX:.*]] = arith.index_castui %[[X]] : i32 to index
// CHECK:             %[[CLAMP_LOW:.*]] = arith.select %[[LT0]], %[[C0]], %[[CAST_IDX]] : index
// CHECK:             %[[GE4:.*]] = arith.cmpi uge, %[[CLAMP_LOW]], %[[C4]] : index
// CHECK:             %[[CLAMP_HIGH:.*]] = arith.select %[[GE4]], %[[C0]], %[[CLAMP_LOW]] : index
// CHECK:             %[[MASK_EXTRACT:.*]] = tensor.extract %[[MASK_CAST]]{{\[}}%[[IDX]]] : tensor<8xi16>
// CHECK:             %[[MASK_COND:.*]] = arith.cmpi ne, %[[MASK_EXTRACT]], %[[C0_I16]] : i16
// CHECK:             %[[OR_INVALID:.*]] = arith.ori %[[LT0]], %[[GE4]] : i1
// CHECK:             %[[VALID:.*]] = arith.xori %[[OR_INVALID]], %[[TRUE]] : i1
// CHECK:             %[[COND:.*]] = arith.andi %[[MASK_COND]], %[[VALID]] : i1
// CHECK:             %[[IF_RES:.*]] = scf.if %[[COND]] -> (tensor<4xi32>) {
// CHECK:               %[[OLD:.*]] = tensor.extract %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<4xi32>
// CHECK:               %[[ADD:.*]] = arith.addi %[[OLD]], %[[C1_I32]] : i32
// CHECK:               %[[UPDATED:.*]] = tensor.insert %[[ADD]] into %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<4xi32>
// CHECK:               scf.yield %[[UPDATED]] : tensor<4xi32>
// CHECK:             } else {
// CHECK:               scf.yield %[[ACC]] : tensor<4xi32>
// CHECK:             }
// CHECK:             scf.yield %[[IF_RES]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return %[[RES]] : tensor<4xi32>
// CHECK:         }
func.func @histogram_mask(%arg0: tensor<8xi32>, %mask: tensor<8xi1>)
    -> tensor<4xi32> {
  %res = hfusion.histogram %arg0, 4, %mask
         : tensor<8xi32>, tensor<8xi1> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----
// CHECK-LABEL: func.func @test_isinf_decompose
// CHECK: hfusion.isinf

// CPU-LABEL: func.func @test_isinf_decompose
// CPU-NOT: hfusion.isinf
module {
  func.func @test_isinf_decompose(%arg0: tensor<4xf32>) -> tensor<4xi1> {
    // CPU-DAG: %[[POS_INF:.*]] = arith.constant 0x7F800000 : f32
    // CPU-DAG: %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
    // CPU: linalg.generic
    // CPU: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: i1):
    // CPU:   %[[IS_POS:.*]] = arith.cmpf oeq, %[[IN]], %[[POS_INF]] : f32
    // CPU:   %[[IS_NEG:.*]] = arith.cmpf oeq, %[[IN]], %[[NEG_INF]] : f32
    // CPU:   %[[RES:.*]] = arith.ori %[[IS_POS]], %[[IS_NEG]] : i1
    // CPU:   linalg.yield %[[RES]] : i1
    %0 = hfusion.isinf %arg0 : tensor<4xf32> -> tensor<4xi1>
    return %0 : tensor<4xi1>
  }
}

// -----

// CHECK-LABEL:   func.func @histogram_nomask_i16(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<16xi16>) -> tensor<4xi32> {
// CHECK:           %[[C0_I16:.*]] = arith.constant 0 : i16
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[INIT:.*]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EMPTY]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[RES:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<4xi32>) {
// CHECK:             %[[X:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[IDX]]] : tensor<16xi16>
// CHECK:             %[[LT0:.*]] = arith.cmpi ult, %[[X]], %[[C0_I16]] : i16
// CHECK:             %[[CAST:.*]] = arith.index_castui %[[X]] : i16 to index
// CHECK:             %[[CLAMP_LOW:.*]] = arith.select %[[LT0]], %[[C0]], %[[CAST]] : index
// CHECK:             %[[GE4:.*]] = arith.cmpi uge, %[[CLAMP_LOW]], %[[C4]] : index
// CHECK:             %[[CLAMP_HIGH:.*]] = arith.select %[[GE4]], %[[C0]], %[[CLAMP_LOW]] : index
// CHECK:             %[[OR_INVALID:.*]] = arith.ori %[[LT0]], %[[GE4]] : i1
// CHECK:             %[[IF_RES:.*]] = scf.if %[[OR_INVALID]] -> (tensor<4xi32>) {
// CHECK:               scf.yield %[[ACC]] : tensor<4xi32>
// CHECK:             } else {
// CHECK:               %[[OLD:.*]] = tensor.extract %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<4xi32>
// CHECK:               %[[ADD:.*]] = arith.addi %[[OLD]], %[[C1_I32]] : i32
// CHECK:               %[[UPDATED:.*]] = tensor.insert %[[ADD]] into %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<4xi32>
// CHECK:               scf.yield %[[UPDATED]] : tensor<4xi32>
// CHECK:             }
// CHECK:             scf.yield %[[IF_RES]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return %[[RES]] : tensor<4xi32>
// CHECK:         }
func.func @histogram_nomask_i16(%arg0: tensor<16xi16>) -> tensor<4xi32> {
  %res = hfusion.histogram %arg0, 4 : tensor<16xi16> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----

// CHECK-LABEL:   func.func @histogram_nomask_i8(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<8xi8>) -> tensor<6xi64> {
// CHECK:           %[[C0_I8:.*]] = arith.constant 0 : i8
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C6:.*]] = arith.constant 6 : index
// CHECK:           %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:           %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<6xi64>
// CHECK:           %[[INIT:.*]] = linalg.fill ins(%[[C0_I64]] : i64) outs(%[[EMPTY]] : tensor<6xi64>) -> tensor<6xi64>
// CHECK:           %[[RES:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<6xi64>) {
// CHECK:             %[[X:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[IDX]]] : tensor<8xi8>
// CHECK:             %[[LT0:.*]] = arith.cmpi ult, %[[X]], %[[C0_I8]] : i8
// CHECK:             %[[CAST:.*]] = arith.index_castui %[[X]] : i8 to index
// CHECK:             %[[CLAMP_LOW:.*]] = arith.select %[[LT0]], %[[C0]], %[[CAST]] : index
// CHECK:             %[[GE6:.*]] = arith.cmpi uge, %[[CLAMP_LOW]], %[[C6]] : index
// CHECK:             %[[CLAMP_HIGH:.*]] = arith.select %[[GE6]], %[[C0]], %[[CLAMP_LOW]] : index
// CHECK:             %[[OR_INVALID:.*]] = arith.ori %[[LT0]], %[[GE6]] : i1
// CHECK:             %[[IF_RES:.*]] = scf.if %[[OR_INVALID]] -> (tensor<6xi64>) {
// CHECK:               scf.yield %[[ACC]] : tensor<6xi64>
// CHECK:             } else {
// CHECK:               %[[OLD:.*]] = tensor.extract %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<6xi64>
// CHECK:               %[[ADD:.*]] = arith.addi %[[OLD]], %[[C1_I64]] : i64
// CHECK:               %[[UPDATED:.*]] = tensor.insert %[[ADD]] into %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<6xi64>
// CHECK:               scf.yield %[[UPDATED]] : tensor<6xi64>
// CHECK:             }
// CHECK:             scf.yield %[[IF_RES]] : tensor<6xi64>
// CHECK:           }
// CHECK:           return %[[RES]] : tensor<6xi64>
// CHECK:         }
func.func @histogram_nomask_i8(%arg0: tensor<8xi8>) -> tensor<6xi64> {
  %res = hfusion.histogram %arg0, 6 : tensor<8xi8> -> tensor<6xi64>
  return %res : tensor<6xi64>
}


// -----

// CHECK-LABEL:   func.func @histogram_nomask_i64(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<128xi64>) -> tensor<16xi32> {
// CHECK:           %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:           %[[C128:.*]] = arith.constant 128 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[INIT:.*]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EMPTY]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK:           %[[RES:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[C128]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<16xi32>) {
// CHECK:             %[[X:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[IDX]]] : tensor<128xi64>
// CHECK:             %[[LT0:.*]] = arith.cmpi ult, %[[X]], %[[C0_I64]] : i64
// CHECK:             %[[CAST:.*]] = arith.index_castui %[[X]] : i64 to index
// CHECK:             %[[CLAMP_LOW:.*]] = arith.select %[[LT0]], %[[C0]], %[[CAST]] : index
// CHECK:             %[[GE16:.*]] = arith.cmpi uge, %[[CLAMP_LOW]], %[[C16]] : index
// CHECK:             %[[CLAMP_HIGH:.*]] = arith.select %[[GE16]], %[[C0]], %[[CLAMP_LOW]] : index
// CHECK:             %[[OR_INVALID:.*]] = arith.ori %[[LT0]], %[[GE16]] : i1
// CHECK:             %[[IF_RES:.*]] = scf.if %[[OR_INVALID]] -> (tensor<16xi32>) {
// CHECK:               scf.yield %[[ACC]] : tensor<16xi32>
// CHECK:             } else {
// CHECK:               %[[OLD:.*]] = tensor.extract %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<16xi32>
// CHECK:               %[[ADD:.*]] = arith.addi %[[OLD]], %[[C1_I32]] : i32
// CHECK:               %[[UPDATED:.*]] = tensor.insert %[[ADD]] into %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<16xi32>
// CHECK:               scf.yield %[[UPDATED]] : tensor<16xi32>
// CHECK:             }
// CHECK:             scf.yield %[[IF_RES]] : tensor<16xi32>
// CHECK:           }
// CHECK:           return %[[RES]] : tensor<16xi32>
// CHECK:         }
func.func @histogram_nomask_i64(%arg0: tensor<128xi64>) -> tensor<16xi32> {
  %res = hfusion.histogram %arg0, 16 : tensor<128xi64> -> tensor<16xi32>
  return %res : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @test_flip_decompose
// CHECK: hfusion.flip

// CPU-LABEL: func.func @test_flip_decompose
// CPU-NOT: hfusion.flip
// CPU: linalg.generic
// CPU: linalg.yield
func.func @test_flip_decompose(%arg0: tensor<4x8x8xf32>) -> tensor<4x8x8xf32> {
  %0 = hfusion.flip %arg0 : tensor<4x8x8xf32> flip_axis = 2 -> tensor<4x8x8xf32>
  return %0 : tensor<4x8x8xf32>
}

// -----

// i8 mask: passthrough (no extsi), extract i8 + cmpi ne 0 in the loop body.
// CHECK-LABEL:   func.func @histogram_mask_i8(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<8xi32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: tensor<8xi8>) -> tensor<4xi32> {
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[C0_I8:.*]] = arith.constant 0 : i8
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[INIT:.*]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EMPTY]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK-NOT:       hfusion.cast
// CHECK:           %[[RES:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<4xi32>) {
// CHECK:             %[[X:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[IDX]]] : tensor<8xi32>
// CHECK:             %[[LT0:.*]] = arith.cmpi ult, %[[X]], %[[C0_I32]] : i32
// CHECK:             %[[CAST:.*]] = arith.index_castui %[[X]] : i32 to index
// CHECK:             %[[CLAMP_LOW:.*]] = arith.select %[[LT0]], %[[C0]], %[[CAST]] : index
// CHECK:             %[[GE4:.*]] = arith.cmpi uge, %[[CLAMP_LOW]], %[[C4]] : index
// CHECK:             %[[CLAMP_HIGH:.*]] = arith.select %[[GE4]], %[[C0]], %[[CLAMP_LOW]] : index
// CHECK:             %[[MASK_VAL:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[IDX]]] : tensor<8xi8>
// CHECK:             %[[MASK_COND:.*]] = arith.cmpi ne, %[[MASK_VAL]], %[[C0_I8]] : i8
// CHECK:             %[[OR_INVALID:.*]] = arith.ori %[[LT0]], %[[GE4]] : i1
// CHECK:             %[[VALID:.*]] = arith.xori %[[OR_INVALID]], %[[TRUE]] : i1
// CHECK:             %[[COND:.*]] = arith.andi %[[MASK_COND]], %[[VALID]] : i1
// CHECK:             %[[IF_RES:.*]] = scf.if %[[COND]] -> (tensor<4xi32>) {
// CHECK:               %[[OLD:.*]] = tensor.extract %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<4xi32>
// CHECK:               %[[ADD:.*]] = arith.addi %[[OLD]], %[[C1_I32]] : i32
// CHECK:               %[[UPDATED:.*]] = tensor.insert %[[ADD]] into %[[ACC]]{{\[}}%[[CLAMP_HIGH]]] : tensor<4xi32>
// CHECK:               scf.yield %[[UPDATED]] : tensor<4xi32>
// CHECK:             } else {
// CHECK:               scf.yield %[[ACC]] : tensor<4xi32>
// CHECK:             }
// CHECK:             scf.yield %[[IF_RES]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return %[[RES]] : tensor<4xi32>
// CHECK:         }
func.func @histogram_mask_i8(%arg0: tensor<8xi32>, %mask: tensor<8xi8>)
    -> tensor<4xi32> {
  %res = hfusion.histogram %arg0, 4, %mask
         : tensor<8xi32>, tensor<8xi8> -> tensor<4xi32>
  return %res : tensor<4xi32>
}
