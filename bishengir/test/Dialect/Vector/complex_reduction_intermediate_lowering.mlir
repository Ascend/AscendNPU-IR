// RUN: bishengir-opt %s -hivm-complex-reduction-intermediate-lowering -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s


// CHECK-LABEL: func.func @test_xori_reduction
func.func @test_xori_reduction(%arg0: memref<?x128xi32, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<?xi32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %arg1 step %c1 {
    %subview = memref.subview %arg2[%arg3] [1] [1] : memref<?xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true]} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xi32>
    %1 = builtin.unrealized_conversion_cast %0 : vector<1xi32> to vector<64xi32>
    %2 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %1) -> (vector<64xi32>) {
      %subview_0 = memref.subview %arg0[%arg3, %arg4] [1, 64] [1, 1] : memref<?x128xi32, #hivm.address_space<ub>> to memref<1x64xi32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xi32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %4 = vector.transfer_read %subview_1[%c0], %c0_i32 {in_bounds = [true]} : memref<64xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi32>
      %5 = vector.constant_mask [64] : vector<64xi1>
      // CHECK: ave.hir.vintlv 
      // CHECK: ave.hir.vxor 
      // CHECK: ave.hir.vintlv 
      // CHECK: ave.hir.vxor 
      // CHECK: ave.hir.vintlv 
      // CHECK: ave.hir.vxor 
      // CHECK: ave.hir.vintlv 
      // CHECK: ave.hir.vxor 
      // CHECK: ave.hir.vintlv 
      // CHECK: ave.hir.vxor 
      // CHECK: ave.hir.vintlv 
      // CHECK: ave.hir.vxor 
      %6 = ave.hir.reduction <xori>, %4, %5 : vector<64xi32>, vector<64xi1> -> vector<64xi32>
      %7 = arith.xori %arg5, %6 : vector<64xi32>
      scf.yield %7 : vector<64xi32>
    }
    %3 = builtin.unrealized_conversion_cast %2 : vector<64xi32> to vector<1xi32>
    vector.transfer_write %3, %subview[%c0] {in_bounds = [true]} : vector<1xi32>, memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}