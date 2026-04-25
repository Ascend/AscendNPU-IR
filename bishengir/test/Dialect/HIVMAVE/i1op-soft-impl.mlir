// RUN: bishengir-opt -ave-i1op-soft-impl %s -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @test_1
// CHECK: ave.hir.vload
// CHECK: ave.hir.vsel
// CHECK: ave.hir.plt
// CHECK: ave.hir.plt
// CHECK: ave.hir.preg.xor
// CHECK: ave.hir.reduction <xori>
// CHECK: ave.hir.vector_broadcast
// CHECK: ave.hir.vcmp
func.func @test_1(%arg0: memref<16xi1, #hivm.address_space<ub>>, %arg1: memref<16x16xi1, strided<[256, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %0 = ave.hir.pge <ALLF> : vector<256xi1>
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg0[%arg2] [1] [1] : memref<16xi1, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2, 0] [1, 16] [1, 1] : memref<16x16xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1x16xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi1>
    %1 = ave.hir.pge <ALL> : vector<256xi1>
    %2 = ave.hir.vcmp <NE> %res, %0, %1 : vector<256xi1>, vector<256xi1> -> vector<256xi1>
    %3 = ave.hir.pge <VL16> {mask_op_idx = 0 : i32} : vector<256xi1>
    annotation.mark %3 {mask_op_idx = 0 : i32} : vector<256xi1>
    %subview_1 = memref.subview %subview_0[0, 0] [1, 16] [1, 1] : memref<1x16xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B8> %subview_1[%c0], %3, %2 {hivm.is_continuous} : memref<16xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi1>
  }
  return
}

// CHECK-LABEL: func.func @test_2
// CHECK: ave.hir.vload
// CHECK: ave.hir.vsel
// CHECK: ave.hir.plt
// CHECK: ave.hir.plt
// CHECK: ave.hir.preg.xor
// CHECK: ave.hir.reduction <xori>
// CHECK: ave.hir.vector_broadcast
// CHECK: ave.hir.vcmp
func.func @test_2(%arg0: memref<128xi32, #hivm.address_space<ub>>, %arg1: memref<1xi1, #hivm.address_space<ub>>, %arg2: memref<1xi32, #hivm.address_space<ub>>, %arg3: memref<128xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0> : vector<64xi32>
  %c0_i32 = arith.constant 0 : i32
  %false = arith.constant false
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  scf.for %arg4 = %c0 to %c128 step %c64 {
    %subview = memref.subview %arg0[%arg4] [64] [1] : memref<128xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg3[%arg4] [64] [1] : memref<128xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %arg1[%c0] : memref<1xi1, #hivm.address_space<ub>> into vector<64xi1>
    %res_1 = ave.hir.vload <BRC_B32> %arg2[%c0] : memref<1xi32, #hivm.address_space<ub>> into vector<64xi32>
    %res_2 = ave.hir.vload <NORM> %subview[%c0] : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
    %0 = arith.select %res, %res_1, %cst : vector<64xi1>, vector<64xi32>
    %1 = arith.addi %0, %res_2 : vector<64xi32>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_0[%c0], %2, %1 {hivm.is_continuous} : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  }
  return
}