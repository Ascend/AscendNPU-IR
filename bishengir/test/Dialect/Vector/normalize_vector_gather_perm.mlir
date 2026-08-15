// RUN: bishengir-opt %s -normalize-vector -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @gather_3d_cyclic_perm_masked
// CHECK-DAG:     %[[PAD:.*]] = arith.constant dense<0.000000e+00> : vector<64xf32>
// CHECK-DAG:     %[[INDEX:.*]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 0,{{.*}}]> : vector<64xi32>
// CHECK:         %[[MASK:.*]] = vector.constant_mask [16] : vector<64xi1>
// CHECK:         %[[GATHER:.*]] = vector.gather %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] [%[[INDEX]]], %[[MASK]], %[[PAD]] : memref<16x1x1xf32, strided<[16, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
// CHECK:         vector.transfer_write %[[GATHER]]
func.func @gather_3d_cyclic_perm_masked(%arg0: memref<16x1x16xf32, #hivm.address_space<ub>>, %arg1: memref<1x16x16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg1[0, %arg2, 0] [1, 1, 16] [1, 1, 1] : memref<1x16x16xf32, #hivm.address_space<ub>> to memref<1x1x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[0, 0, %arg2] [16, 1, 1] [1, 1, 1] : memref<16x1x16xf32, #hivm.address_space<ub>> to memref<16x1x1xf32, strided<[16, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.constant_mask [16, 1, 1] : vector<64x1x1xi1>
    %1 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst, %0 {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<16x1x1xf32, strided<[16, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xf32>
    %2 = vector.constant_mask [16] : vector<64xi1>
    %subview_1 = memref.subview %subview[0, 0, 0] [1, 1, 16] [1, 1, 1] : memref<1x1x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %3 = vector.shape_cast %1 : vector<1x1x64xf32> to vector<64xf32>
    vector.transfer_write %3, %subview_1[%c0], %2 {in_bounds = [true]} : vector<64xf32>, memref<16xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @gather_3d_cyclic_perm_full_mask
// CHECK-DAG:     %[[PAD:.*]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:     %[[INDEX:.*]] = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31]> : vector<32xi32>
// CHECK:         %[[MASK:.*]] = vector.constant_mask [32] : vector<32xi1>
// CHECK:         %[[GATHER:.*]] = vector.gather %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] [%[[INDEX]]], %[[MASK]], %[[PAD]] : memref<8x4x1xf32, #hivm.address_space<ub>>, vector<32xi32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
// CHECK:         vector.transfer_write %[[GATHER]]
func.func @gather_3d_cyclic_perm_full_mask(%arg0: memref<8x4x1xf32, #hivm.address_space<ub>>, %arg1: memref<32xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : memref<8x4x1xf32, #hivm.address_space<ub>>, vector<4x1x8xf32>
  %1 = vector.shape_cast %0 : vector<4x1x8xf32> to vector<32xf32>
  vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, #hivm.address_space<ub>>
  return
}
