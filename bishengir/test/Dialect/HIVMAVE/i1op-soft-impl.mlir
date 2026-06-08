// RUN: bishengir-opt -ave-i1op-soft-impl %s -o %t.mlir -mlir-print-vector-layout-attr
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
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.xor
// CHECK-NOT: ave.hir.reduction <xori>
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

// CHECK-LABEL: func.func @test_3
// CHECK: ave.hir.vload
// CHECK: ave.hir.vsel
// CHECK: ave.hir.plt
// CHECK: ave.hir.plt
// CHECK: ave.hir.preg.xor
// CHECK: ave.hir.reduction <xori>
// CHECK: ave.hir.vector_broadcast
// CHECK: ave.hir.vcmp
// CHECK: ave.hir.preg.and
// CHECK: ave.hir.masked_store
func.func @test_3(%arg0: memref<168xi1, #hivm.address_space<ub>>, %arg1: memref<2xi1, #hivm.address_space<ub>>, %arg2: memref<4xi1, #hivm.address_space<ub>>, %arg3: memref<168x2x4xi8, strided<[64, 32, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i16 = arith.constant 0 : i16
  %0 = ave.hir.pge <ALL> : vector<256xi1>
  %1 = ave.hir.broadcast %c0_i16, %0 : i16, vector<256xi1> -> vector<256xi8>
  %c1_i16 = arith.constant 1 : i16
  %2 = ave.hir.pge <ALL> : vector<256xi1>
  %3 = ave.hir.broadcast %c1_i16, %2 : i16, vector<256xi1> -> vector<256xi8>
  %4 = ave.hir.pge <ALLF> : vector<256xi1>
  %false = arith.constant false
  %c168 = arith.constant 168 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %5 = ave.hir.pge <VL4> {mask_op_idx = 0 : i32} : vector<256xi1>
  annotation.mark %5 {mask_op_idx = 0 : i32} : vector<256xi1>
  scf.for %arg4 = %c0 to %c168 step %c1 {
    %subview = memref.subview %arg0[%arg4] [1] [1] : memref<168xi1, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      %subview_0 = memref.subview %arg1[%arg5] [1] [1] : memref<2xi1, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg3[%arg4, %arg5, 0] [1, 1, 4] [1, 1, 1] : memref<168x2x4xi8, strided<[64, 32, 1]>, #hivm.address_space<ub>> to memref<1x1x4xi8, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi1>
      %6 = builtin.unrealized_conversion_cast %res : vector<256xi1> to vector<1x1x256xi1>
      %res_2 = ave.hir.vload <NORM> %subview_0[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi1>
      %7 = builtin.unrealized_conversion_cast %res_2 : vector<256xi1> to vector<1x1x256xi1>
      %res_3 = ave.hir.vload <NORM> %arg2[%c0] : memref<4xi1, #hivm.address_space<ub>> into vector<256xi1>
      %8 = builtin.unrealized_conversion_cast %res_3 : vector<256xi1> to vector<1x1x256xi1>
      annotation.mark %8 {reached_mask_ops_idx = 0 : i32} : vector<1x1x256xi1>
      %9 = builtin.unrealized_conversion_cast %8 : vector<1x1x256xi1> to vector<256xi1>
      annotation.mark %9 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %10 = ave.hir.vcmp <NE> %9, %4, %5 : vector<256xi1>, vector<256xi1> -> vector<256xi1>
      annotation.mark %10 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %11 = builtin.unrealized_conversion_cast %7 : vector<1x1x256xi1> to vector<256xi1>
      annotation.mark %11 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %12 = ave.hir.vcmp <NE> %11, %4, %5 : vector<256xi1>, vector<256xi1> -> vector<256xi1>
      annotation.mark %12 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %13 = builtin.unrealized_conversion_cast %6 : vector<1x1x256xi1> to vector<256xi1>
      annotation.mark %13 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %14 = ave.hir.vcmp <NE> %13, %4, %5 : vector<256xi1>, vector<256xi1> -> vector<256xi1>
      annotation.mark %14 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %15 = ave.hir.preg.and <b8> %14, %12, %5 : vector<256xi1>
      annotation.mark %15 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %16 = ave.hir.vcmp <NE> %15, %4, %5 : vector<256xi1>, vector<256xi1> -> vector<256xi1>
      annotation.mark %16 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %17 = ave.hir.preg.and <b8> %16, %10, %5 : vector<256xi1>
      annotation.mark %17 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
      %18 = ave.hir.vsel %17, %3, %1 : vector<256xi1>, vector<256xi8>
      annotation.mark %18 {reached_mask_ops_idx = 0 : i32} : vector<256xi8>
      %subview_4 = memref.subview %subview_1[0, 0, 0] [1, 1, 4] [1, 1, 1] : memref<1x1x4xi8, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>> to memref<4xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B8> %subview_4[%c0], %5, %18 : memref<4xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
    }
  }
  return
}

// CHECK-LABEL: @test_constraint_layout
// CHECK: ave.hir.vload
// CHECK: ave.hir.vsel
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.xor
// CHECK-NOT: ave.hir.reduction <xori>
// CHECK: ave.hir.vector_broadcast
// CHECK: ave.hir.vcmp
func.func @test_constraint_layout(%arg0: memref<64xi1, strided<[256]>, #hivm.address_space<ub>>, %arg1: memref<64x64xf16, #hivm.address_space<ub>>, %arg2: memref<64x64xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  %1 = ave.hir.broadcast %cst, %0 : f16, vector<128xi1> -> vector<128xf16>
  %2 = ave.hir.pge <ALLF> : vector<128xi1>
  %cst_0 = arith.constant 0.000000e+00 : f16
  %false = arith.constant false
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<64xi1, strided<[256]>, #hivm.address_space<ub>> to memref<1xi1, strided<[256], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg1[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] : memref<1xi1, strided<[256], offset: ?>, #hivm.address_space<ub>> into vector<128xi1>
    %3 = builtin.unrealized_conversion_cast %res : vector<128xi1> to vector<1x128xi1>
    %4 = ave.hir.pge <VL64> {mask_op_idx = 0 : i32} : vector<128xi1>
    annotation.mark %4 {mask_op_idx = 0 : i32} : vector<128xi1>
    %subview_3 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<128xf16>
    annotation.mark %res_4 {reached_mask_ops_idx = 0 : i32} : vector<128xf16>
    %5 = builtin.unrealized_conversion_cast %3 : vector<1x128xi1> to vector<128xi1>
    annotation.mark %5 {reached_mask_ops_idx = 0 : i32} : vector<128xi1>
    %6 = ave.hir.vcmp <NE> %5, %2, %4 : vector<128xi1>, vector<128xi1> -> vector<128xi1>
    // CHECK: %[[VAL_2:.*]] = ave.hir.vector.layout_cast %[[VAL_1:.*]] : vector<256xi1> -> vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
    // CHECK-NEXT: %[[VAL_3:.*]] = ave.hir.vector.layout_cast %[[VAL_2:.*]] : vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>> -> vector<256xi1>
    annotation.mark %6 {reached_mask_ops_idx = 0 : i32} : vector<128xi1>
    %7 = ave.hir.vsel %6, %res_4, %1 : vector<128xi1>, vector<128xf16>
    annotation.mark %7 {reached_mask_ops_idx = 0 : i32} : vector<128xf16>
    %subview_5 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B16> %subview_5[%c0], %4, %7 : memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
  }
  return
}

// CHECK-LABEL: func.func @test_offset_beyond_vl
// CHECK: arith.addi
// CHECK: arith.index_cast
// CHECK: arith.divsi
// CHECK: arith.muli
// CHECK: arith.subi
// CHECK: arith.divsi
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: ave.hir.vload
// CHECK: ave.hir.vsel
// CHECK: ave.hir.plt
// CHECK: ave.hir.plt
// CHECK: ave.hir.preg.xor
// CHECK: ave.hir.reduction <xori>
// CHECK: ave.hir.vector_broadcast
// CHECK: ave.hir.vcmp
func.func @test_offset_beyond_vl(%arg0: memref<512xi1, #hivm.address_space<ub>>, %arg1: memref<512x4xi32, strided<[32, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %0 = ave.hir.pge <VL4> {mask_op_idx = 0 : i32} : vector<256xi1>
  annotation.mark %0 {mask_op_idx = 0 : i32} : vector<256xi1>
  scf.for %arg2 = %c0 to %c512 step %c1 {
    %subview = memref.subview %arg0[%arg2] [1] [1] : memref<512xi1, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2, 0] [1, 4] [1, 1] : memref<512x4xi32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<1x4xi32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi1>
    %1 = ave.hir.pge <ALL> : vector<256xi1>
    %2 = ave.hir.vcmp <NE> %res, %1, %0 : vector<256xi1>, vector<256xi1> -> vector<256xi1>
    %subview_1 = memref.subview %subview_0[0, 0] [1, 4] [1, 1] : memref<1x4xi32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> to memref<4xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_1[%c0], %0, %2 {hivm.is_continuous} : memref<4xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi1>
  }
  return
}
