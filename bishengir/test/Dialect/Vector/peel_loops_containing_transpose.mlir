// RUN: bishengir-opt %s -peel-loops-containing-transpose -canonicalize -normalize-vector -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @peel_loops_containing_transpose
// CHECK: %c256 = arith.constant 256 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %c32 = arith.constant 32 : index
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %c64 = arith.constant 64 : index
// CHECK-NEXT: scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:   scf.for %arg3 = %c0 to %c256 step %c64 {
// CHECK-NEXT:     %subview_4 = memref.subview %arg0[%arg3, %arg2] [64, 1] [1, 1] : memref<273x32xf32, #hivm.address_space<ub>> to memref<64x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
// CHECK-NEXT:     %subview_5 = memref.subview %arg1[%arg2, %arg3] [1, 64] [1, 1] : memref<32x273xf32, strided<[280, 1]>, #hivm.address_space<ub>> to memref<1x64xf32, strided<[280, 1], offset: ?>, #hivm.address_space<ub>>
// CHECK-NEXT:     %3 = vector.constant_mask [64] : vector<64xi1>
// CHECK-NEXT:     %4 = vector.gather %subview_4[%c0, %c0] [%cst_1], %3, %cst_0 : memref<64x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
// CHECK-NEXT:     %subview_6 = memref.subview %subview_5[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[280, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
// CHECK-NEXT:     vector.transfer_write %4, %subview_6[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #map, #hivm.address_space<ub>>
// CHECK-NEXT:   }
// CHECK-NEXT:   %subview = memref.subview %arg0[256, %arg2] [17, 1] [1, 1] : memref<273x32xf32, #hivm.address_space<ub>> to memref<17x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
// CHECK-NEXT:   %subview_2 = memref.subview %arg1[%arg2, 256] [1, 17] [1, 1] : memref<32x273xf32, strided<[280, 1]>, #hivm.address_space<ub>> to memref<1x17xf32, strided<[280, 1], offset: ?>, #hivm.address_space<ub>>
// CHECK-NEXT:   %0 = vector.constant_mask [17] : vector<64xi1>
// CHECK-NEXT:   %1 = vector.gather %subview[%c0, %c0] [%cst], %0, %cst_0 : memref<17x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
// CHECK-NEXT:   %2 = vector.constant_mask [17] : vector<64xi1>
// CHECK-NEXT:   %subview_3 = memref.subview %subview_2[0, 0] [1, 17] [1, 1] : memref<1x17xf32, strided<[280, 1], offset: ?>, #hivm.address_space<ub>> to memref<17xf32, #map, #hivm.address_space<ub>>
// CHECK-NEXT:   vector.transfer_write %1, %subview_3[%c0], %2 {in_bounds = [true]} : vector<64xf32>, memref<17xf32, #map, #hivm.address_space<ub>>
// CHECK-NEXT: }
func.func @peel_loops_containing_transpose(%arg0: memref<273x32xf32, #hivm.address_space<ub>>, %arg1: memref<32x273xf32, strided<[280, 1]>, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c273 = arith.constant 273 : index
  scf.for %arg2 = %c0 to %c32 step %c1 {
    scf.for %arg3 = %c0 to %c273 step %c64 {
      %0 = affine.min affine_map<(d0) -> (-d0 + 273, 64)>(%arg3)
      %subview = memref.subview %arg0[%arg3, %arg2] [%0, 1] [1, 1] : memref<273x32xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3] [1, %0] [1, 1] : memref<32x273xf32, strided<[280, 1]>, #hivm.address_space<ub>> to memref<1x?xf32, strided<[280, 1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.create_mask %0, %c1 : vector<64x1xi1>
      %2 = vector.transfer_read %subview[%c0, %c0], %cst, %1 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<?x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %3 = vector.create_mask %c1, %0 : vector<1x64xi1>
      vector.transfer_write %2, %subview_0[%c0, %c0], %3 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[280, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}