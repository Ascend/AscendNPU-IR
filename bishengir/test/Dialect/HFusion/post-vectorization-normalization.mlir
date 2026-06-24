// RUN: bishengir-opt %s -hfusion-post-vectorization-normalization -split-input-file | FileCheck %s

// Test: i1 broadcast transfer_read → decomposed into non-broadcast read +
//       sitofp to f16 + vector.broadcast + arith.subf(x,x) + arith.cmpf une.
// CHECK-LABEL: func.func @fix_i1_broadcast_transfer_read
// CHECK: %[[READ:.*]] = vector.transfer_read {{.*}} tensor<1xi1>, vector<1xi1>
// CHECK: %[[CAST:.*]] = arith.sitofp %[[READ]] : vector<1xi1> to vector<1xf16>
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[CAST]] : vector<1xf16> to vector<64xf16>
// CHECK: arith.cmpf une, %[[BCAST]]
func.func @fix_i1_broadcast_transfer_read(%arg0: tensor<1xi1>, %arg1: tensor<64xi1>) -> tensor<64xi1> {
  %c0 = arith.constant 0 : index
  %false = arith.constant false
  %0 = vector.transfer_read %arg0[%c0], %false {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>} : tensor<1xi1>, vector<64xi1>
  %1 = vector.transfer_write %0, %arg1[%c0] {in_bounds = [true]} : vector<64xi1>, tensor<64xi1>
  return %1 : tensor<64xi1>
}

// -----
// Non-bit type (i8) with broadcast — should NOT be rewritten.
// CHECK-LABEL: func.func @skip_i8
// CHECK: vector.transfer_read {{.*}} vector<64xi8>
// CHECK-NOT: sitofp
func.func @skip_i8(%arg0: tensor<1xi8>, %arg1: tensor<64xi8>) -> tensor<64xi8> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i8
  %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>} : tensor<1xi8>, vector<64xi8>
  %1 = vector.transfer_write %0, %arg1[%c0] {in_bounds = [true]} : vector<64xi8>, tensor<64xi8>
  return %1 : tensor<64xi8>
}
