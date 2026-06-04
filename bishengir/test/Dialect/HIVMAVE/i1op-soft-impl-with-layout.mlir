// RUN: bishengir-opt -analyze-vector-layout -analyze-alignment-bitwidth -ave-normalize-ops \
// RUN: -remove-vector-layout-attr -convert-hivmave-to-ave-intrin %s | FileCheck %s

// CHECK-LABEL: @test_layout_constraint
//
// CHECK: intr.hivm.pge.b16
// CHECK: intr.hivm.pge.b16
//
// CHECK: intr.hivm.plds.b8
// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK-NEXT: builtin.unrealized_conversion_cast
//
// CHECK: intr.hivm.pge.b8
// CHECK: intr.hivm.plt.b8.v300
// CHECK: intr.hivm.plt.b8.v300
//
// CHECK: intr.hivm.vintlv
// CHECK: intr.hivm.vintlv
//
// CHECK: "hivm_regbaseintrins.intr.hivm.punpack"
//
// CHECK: intr.hivm.pge.b16{{.*}}mask_op_idx
//
// CHECK: intr.hivm.vldsx1.v128f16
// CHECK: intr.hivm.vstsx1.v128f16
#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @test_layout_constraint(%arg0: memref<64xi1, strided<[256]>, #hivm.address_space<ub>>, %arg1: memref<64x64xf16, #hivm.address_space<ub>>, %arg2: memref<64x64xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = ave.hir.pge <ALL> : vector<128xi1>
    %1 = ave.hir.broadcast %cst, %0 : f16, vector<128xi1> -> vector<128xf16>
    %2 = ave.hir.pge <ALLF> : vector<128xi1>
    scf.for %arg3 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg2[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %arg0[%c0] : memref<64xi1, strided<[256]>, #hivm.address_space<ub>> into vector<128xi1>
      %3 = builtin.unrealized_conversion_cast %res : vector<128xi1> to vector<256xi1>
      %4 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
      %5 = ave.hir.scalar_broadcast %c1_i8 : i8 -> vector<256xi8>
      %6 = ave.hir.vsel %3, %5, %4 : vector<256xi1>, vector<256xi8>
      %7 = ave.hir.pge <ALL> : vector<256xi1>
      %8 = arith.addi %arg3, %c1 : index
      %res_1, %new_true_shape = ave.hir.plt %8 : vector<256xi1>, index
      %res_2, %new_true_shape_3 = ave.hir.plt %arg3 : vector<256xi1>, index
      %9 = ave.hir.preg.xor <b8> %res_1, %res_2, %7 : vector<256xi1>
      %10 = ave.hir.pge <ALL> : vector<256xi1>
      %11 = ave.hir.broadcast %c0_i8, %10 : i8, vector<256xi1> -> vector<256xi8>
      %12 = ave.hir.vor %6, %11, %9 : vector<256xi8>, vector<256xi1>
      %13 = ave.hir.pge <ALL> : vector<256xi1>
      %res1, %res2 = ave.hir.vintlv %12, %11 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %14 = ave.hir.vxor %res1, %res2, %13 : vector<256xi8>, vector<256xi1>
      %res1_4, %res2_5 = ave.hir.vintlv %14, %11 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %15 = ave.hir.vxor %res1_4, %res2_5, %13 : vector<256xi8>, vector<256xi1>
      %res1_6, %res2_7 = ave.hir.vintlv %15, %11 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %16 = ave.hir.vxor %res1_6, %res2_7, %13 : vector<256xi8>, vector<256xi1>
      %res1_8, %res2_9 = ave.hir.vintlv %16, %11 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %17 = ave.hir.vxor %res1_8, %res2_9, %13 : vector<256xi8>, vector<256xi1>
      %res1_10, %res2_11 = ave.hir.vintlv %17, %11 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %18 = ave.hir.vxor %res1_10, %res2_11, %13 : vector<256xi8>, vector<256xi1>
      %res1_12, %res2_13 = ave.hir.vintlv %18, %11 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %19 = ave.hir.vxor %res1_12, %res2_13, %13 : vector<256xi8>, vector<256xi1>
      %res1_14, %res2_15 = ave.hir.vintlv %19, %11 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %20 = ave.hir.vxor %res1_14, %res2_15, %13 : vector<256xi8>, vector<256xi1>
      %res1_16, %res2_17 = ave.hir.vintlv %20, %11 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %21 = ave.hir.vxor %res1_16, %res2_17, %13 : vector<256xi8>, vector<256xi1>
      %22 = ave.hir.vector_broadcast %21, %7, true : vector<256xi8>, vector<256xi1> -> vector<256xi8>
      %23 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
      %24 = ave.hir.vcmp <NE> %22, %23, %7 : vector<256xi8>, vector<256xi1> -> vector<256xi1>
      %25 = ave.hir.vector.layout_cast %24 : vector<256xi1> -> vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
      %26 = ave.hir.vector.layout_cast %25 : vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>> -> vector<256xi1>
      %27 = builtin.unrealized_conversion_cast %26 : vector<256xi1> to vector<128xi1>
      %28 = ave.hir.pge <VL64> {mask_op_idx = 0 : i32} : vector<128xi1>
      %subview_18 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, #map, #hivm.address_space<ub>>
      %res_19 = ave.hir.vload <NORM> %subview_18[%c0] : memref<64xf16, #map, #hivm.address_space<ub>> into vector<128xf16>
      %29 = ave.hir.preg.xor <b8> %27, %2, %28 : vector<128xi1>
      %30 = ave.hir.vsel %29, %res_19, %1 : vector<128xi1>, vector<128xf16>
      %subview_20 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, #map, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B16> %subview_20[%c0], %28, %30 : memref<64xf16, #map, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    }
    return
  }
//
// CHECK-LABEL: @test_layout_cast_no_change
// CHECK: intr.hivm.pge.b8
// CHECK: intr.hivm.plds.b8
// CHECK: intr.hivm.pge.b8
// CHECK: intr.hivm.plt.b8.v300
// CHECK: intr.hivm.plt.b8.v300
// CHECK: intr.hivm.vintlv
// CHECK: intr.hivm.vintlv
// CHECK: intr.hivm.vcmp.ne.s.z
// CHECK: intr.hivm.pge.b8{{.*}}mask_op_idx
// CHECK: intr.hivm.psts.b8
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.punpack"
  func.func @test_layout_cast_no_change(%arg0: memref<16xi1, strided<[256]>, #hivm.address_space<ub>>, %arg1: memref<16x64xi1, strided<[256, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = ave.hir.pge <ALLF> : vector<256xi1>
    scf.for %arg2 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg1[%arg2, 0] [1, 64] [1, 1] : memref<16x64xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1x64xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %arg0[%c0] : memref<16xi1, strided<[256]>, #hivm.address_space<ub>> into vector<256xi1>
      %1 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
      %2 = ave.hir.scalar_broadcast %c1_i8 : i8 -> vector<256xi8>
      %3 = ave.hir.vsel %res, %2, %1 : vector<256xi1>, vector<256xi8>
      %4 = ave.hir.pge <ALL> : vector<256xi1>
      %5 = arith.addi %arg2, %c1 : index
      %res_0, %new_true_shape = ave.hir.plt %5 : vector<256xi1>, index
      %res_1, %new_true_shape_2 = ave.hir.plt %arg2 : vector<256xi1>, index
      %6 = ave.hir.preg.xor <b8> %res_0, %res_1, %4 : vector<256xi1>
      %7 = ave.hir.pge <ALL> : vector<256xi1>
      %8 = ave.hir.broadcast %c0_i8, %7 : i8, vector<256xi1> -> vector<256xi8>
      %9 = ave.hir.vor %3, %8, %6 : vector<256xi8>, vector<256xi1>
      %10 = ave.hir.pge <ALL> : vector<256xi1>
      %res1, %res2 = ave.hir.vintlv %9, %8 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %11 = ave.hir.vxor %res1, %res2, %10 : vector<256xi8>, vector<256xi1>
      %res1_3, %res2_4 = ave.hir.vintlv %11, %8 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %12 = ave.hir.vxor %res1_3, %res2_4, %10 : vector<256xi8>, vector<256xi1>
      %res1_5, %res2_6 = ave.hir.vintlv %12, %8 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %13 = ave.hir.vxor %res1_5, %res2_6, %10 : vector<256xi8>, vector<256xi1>
      %res1_7, %res2_8 = ave.hir.vintlv %13, %8 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %14 = ave.hir.vxor %res1_7, %res2_8, %10 : vector<256xi8>, vector<256xi1>
      %res1_9, %res2_10 = ave.hir.vintlv %14, %8 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %15 = ave.hir.vxor %res1_9, %res2_10, %10 : vector<256xi8>, vector<256xi1>
      %res1_11, %res2_12 = ave.hir.vintlv %15, %8 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %16 = ave.hir.vxor %res1_11, %res2_12, %10 : vector<256xi8>, vector<256xi1>
      %res1_13, %res2_14 = ave.hir.vintlv %16, %8 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %17 = ave.hir.vxor %res1_13, %res2_14, %10 : vector<256xi8>, vector<256xi1>
      %res1_15, %res2_16 = ave.hir.vintlv %17, %8 {layout_change = #ave<layout_change UNCHANGED>} : vector<256xi8>, vector<256xi8>
      %18 = ave.hir.vxor %res1_15, %res2_16, %10 : vector<256xi8>, vector<256xi1>
      %19 = ave.hir.vector_broadcast %18, %4, true : vector<256xi8>, vector<256xi1> -> vector<256xi8>
      %20 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
      %21 = ave.hir.vcmp <NE> %19, %20, %4 : vector<256xi8>, vector<256xi1> -> vector<256xi1>
      %22 = ave.hir.vector.layout_cast %21 : vector<256xi1> -> vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
      %23 = ave.hir.vector.layout_cast %22 : vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>> -> vector<256xi1>
      %24 = ave.hir.pge <ALL> : vector<256xi1>
      %25 = ave.hir.preg.xor <b8> %23, %0, %24 : vector<256xi1>
      %26 = ave.hir.pge <VL64> {mask_op_idx = 0 : i32} : vector<256xi1>
      %subview_17 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B8> %subview_17[%c0], %26, %25 {hivm.is_continuous} : memref<64xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi1>
    }
    return
  }
}