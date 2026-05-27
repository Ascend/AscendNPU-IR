// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -analyze-vector-layout -analyze-alignment-bitwidth -remove-vector-layout-attr \
// RUN: -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s

// CHECK-LABEL: @test_preg_arith_op_lowering
func.func @test_preg_arith_op_lowering(%arg0: memref<8xi1, #hivm.address_space<ub>>, %arg1: memref<8x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %arg2: memref<8x8xi32, #hivm.address_space<ub>>, %arg3: memref<i32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i16 = arith.constant 0 : i16
  %c1_i8 = arith.constant 1 : i8
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b32"
  %1 = ave.hir.broadcast %c0_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
  %2 = ave.hir.pge <ALLF> : vector<64xi1>
  // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b32"
  %res = ave.hir.vload <BRC_B32> %arg3[] : memref<i32, #hivm.address_space<ub>> into vector<1xi32>
  %3 = builtin.unrealized_conversion_cast %res : vector<1xi32> to vector<i32>
  %4 = builtin.unrealized_conversion_cast %3 : vector<i32> to vector<64xi32>
  %5:2 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %3, %arg6 = %4) -> (vector<i32>, vector<64xi32>) {
    %9 = builtin.unrealized_conversion_cast %arg6 : vector<64xi32> to vector<i32>
    %subview = memref.subview %arg1[%arg4, 0] [1, 8] [1, 1] : memref<8x8xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1x8xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg2[%arg4, 0] [1, 8] [1, 1] : memref<8x8xi32, #hivm.address_space<ub>> to memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %res_1 = ave.hir.vload <NORM> %arg0[%c0] : memref<8xi1, #hivm.address_space<ub>> into vector<64xi1>
    %10 = builtin.unrealized_conversion_cast %res_1 : vector<64xi1> to vector<256xi1>
    %11 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
    %12 = ave.hir.scalar_broadcast %c1_i8 : i8 -> vector<256xi8>
    %13 = ave.hir.vsel %10, %12, %11 : vector<256xi1>, vector<256xi8>
    %14 = ave.hir.pge <ALL> : vector<256xi1>
    %15 = arith.addi %arg4, %c1 : index
    %res_2, %new_true_shape = ave.hir.plt %15 : vector<256xi1>, index
    %res_3, %new_true_shape_4 = ave.hir.plt %arg4 : vector<256xi1>, index
    %16 = ave.hir.preg.xor <b8> %res_2, %res_3, %14 : vector<256xi1>
    %17 = ave.hir.pge <ALL> : vector<256xi1>
    %18 = ave.hir.broadcast %c0_i16, %17 : i16, vector<256xi1> -> vector<256xi8>
    %19 = ave.hir.vor %13, %18, %16 : vector<256xi8>, vector<256xi1>
    %20 = ave.hir.pge <ALL> : vector<256xi1>
    %res1, %res2 = ave.hir.vintlv %19, %18 : vector<256xi8>, vector<256xi8>
    %21 = ave.hir.vxor %res1, %res2, %20 : vector<256xi8>, vector<256xi1>
    %res1_5, %res2_6 = ave.hir.vintlv %21, %18 : vector<256xi8>, vector<256xi8>
    %22 = ave.hir.vxor %res1_5, %res2_6, %20 : vector<256xi8>, vector<256xi1>
    %res1_7, %res2_8 = ave.hir.vintlv %22, %18 : vector<256xi8>, vector<256xi8>
    %23 = ave.hir.vxor %res1_7, %res2_8, %20 : vector<256xi8>, vector<256xi1>
    %res1_9, %res2_10 = ave.hir.vintlv %23, %18 : vector<256xi8>, vector<256xi8>
    %24 = ave.hir.vxor %res1_9, %res2_10, %20 : vector<256xi8>, vector<256xi1>
    %res1_11, %res2_12 = ave.hir.vintlv %24, %18 : vector<256xi8>, vector<256xi8>
    %25 = ave.hir.vxor %res1_11, %res2_12, %20 : vector<256xi8>, vector<256xi1>
    %res1_13, %res2_14 = ave.hir.vintlv %25, %18 : vector<256xi8>, vector<256xi8>
    %26 = ave.hir.vxor %res1_13, %res2_14, %20 : vector<256xi8>, vector<256xi1>
    %res1_15, %res2_16 = ave.hir.vintlv %26, %18 : vector<256xi8>, vector<256xi8>
    %27 = ave.hir.vxor %res1_15, %res2_16, %20 : vector<256xi8>, vector<256xi1>
    %res1_17, %res2_18 = ave.hir.vintlv %27, %18 : vector<256xi8>, vector<256xi8>
    %28 = ave.hir.vxor %res1_17, %res2_18, %20 : vector<256xi8>, vector<256xi1>
    %29 = ave.hir.vector_broadcast %28, %14, true : vector<256xi8>, vector<256xi1> -> vector<256xi8>
    %30 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
    %31 = ave.hir.vcmp <NE> %29, %30, %14 : vector<256xi8>, vector<256xi1> -> vector<256xi1>
    %32 = builtin.unrealized_conversion_cast %31 : vector<256xi1> to vector<64xi1>
    %33 = ave.hir.pge <VL8> : vector<64xi1>
    %subview_19 = memref.subview %subview[0, 0] [1, 8] [1, 1] : memref<1x8xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<8xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_20 = ave.hir.vload <NORM> %subview_19[%c0] : memref<8xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi1>
    // CHECK : "hivm_regbaseintrins.intr.hivm.plds.b8"
    %subview_21 = memref.subview %subview_0[0, 0] [1, 8] [1, 1] : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> to memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_22 = ave.hir.vload <NORM> %subview_21[%c0] : memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi32>
    %34 = ave.hir.pge <ALL> : vector<64xi1>
    %35 = ave.hir.preg.xor <b8> %32, %2, %34 : vector<64xi1>
    %36 = ave.hir.pge <ALL> : vector<64xi1>
    %37 = ave.hir.preg.or <b8> %35, %res_20, %36 : vector<64xi1>
    // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b8"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pintlv.b8"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pintlv.b8"
    // CHECK : "hivm_regbaseintrins.intr.hivm.vsel"
    %38 = ave.hir.vsel %37, %res_22, %1 : vector<64xi1>, vector<64xi32>
    // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b8"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pintlv.b8"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pintlv.b8"
    // CHECK : "hivm_regbaseintrins.intr.hivm.vsel"
    %39 = builtin.unrealized_conversion_cast %9 : vector<i32> to i32
    %40 = ave.hir.vsel %33, %38, %1 : vector<64xi1>, vector<64xi32>
    %41 = ave.hir.pge <ALL> : vector<64xi1>
    %42 = builtin.unrealized_conversion_cast %39 : i32 to vector<1xi32>
    %43 = builtin.unrealized_conversion_cast %42 : vector<1xi32> to vector<64xi32>
    %44 = ave.hir.reduction <add>, %40, %41 : vector<64xi32>, vector<64xi1> -> vector<64xi32>
    %45 = ave.hir.pge <ALL> : vector<64xi1>
    %46 = ave.hir.vadd %43, %44, %45 : vector<64xi32>, vector<64xi1>
    %47 = builtin.unrealized_conversion_cast %46 : vector<64xi32> to vector<1xi32>
    %48 = builtin.unrealized_conversion_cast %47 : vector<1xi32> to i32
    %49 = builtin.unrealized_conversion_cast %48 : i32 to vector<i32>
    %50 = builtin.unrealized_conversion_cast %49 : vector<i32> to vector<64xi32>
    scf.yield %49, %50 : vector<i32>, vector<64xi32>
  }
  %6 = builtin.unrealized_conversion_cast %5#1 : vector<64xi32> to vector<i32>
  %7 = builtin.unrealized_conversion_cast %6 : vector<i32> to vector<1xi32>
  %8 = ave.hir.pge <ALL> : vector<1xi1>
  ave.hir.masked_store <ONEPT_B32> %arg3[], %8, %7 : memref<i32, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi32>
  return
}

// -----

// CHECK-LABEL : @plds_i1_as_msk_of_vsel_f32
// CHECK : %[[PLDS_B8:.*]] = "hivm_regbaseintrins.intr.hivm.plds.b8"
// CHECK : %[[PGE_B8:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b8"
// CHECK : %[[PINTLV0:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b8"(%[[PLDS_B8]], %[[PGE_B8]]) : (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK : %[[PINTLV00:.*]] = llvm.extractvalue %[[PINTLV0]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK : %[[PINTLV1:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b8"(%[[PINTLV00]], %[[PGE_B8]]) : (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK : %[[PINTLV10:.*]] = llvm.extractvalue %[[PINTLV1]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK : "hivm_regbaseintrins.intr.hivm.vsel"(%[[LHS:.*]], %[[RHS:.*]], %[[PINTLV10]]) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
#map4 = affine_map<()[s0] -> (s0 * 16)>
#map15 = affine_map<(d0)[s0] -> (d0 + s0)>
#map16 = affine_map<()[s0] -> (s0 * 256)>
#map18 = affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>
func.func @plds_i1_as_msk_of_vsel_f32(%arg0: memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %arg5: memref<4x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c1040 = arith.constant 1040 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %T = ave.hir.pge <ALL> : vector<64xi1>
  %0 = ave.hir.broadcast %cst_0, %T : f32, vector<64xi1> -> vector<64xf32>
  %1 = ave.hir.broadcast %cst_1, %T : f32, vector<64xi1> -> vector<64xf32>
  scf.for %arg6 = %c0 to %c64 step %c1 {
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg0 : memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index
    %2 = affine.apply #map16()[%arg6]
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%2], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map15, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %reinterpret_cast_5[%c0] : memref<64xi1, #map15, #hivm.address_space<ub>> into vector<64xi1>
    %9 = ave.hir.vsel %res, %0, %1 : vector<64xi1>, vector<64xf32>
    %11 = ave.hir.vtruncf %9, <rint>, false, <part_even>, %T : vector<64xf32>, vector<128xbf16>, vector<64xi1>
    %base_buffer_19, %offset_20, %sizes_21:3, %strides_22:3 = memref.extract_strided_metadata %arg5 : memref<4x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index
    %13 = affine.apply #map4()[%arg6]
    %14 = ave.hir.pge <VL64> : vector<128xi1>
    %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer_19 to offset: [%13], sizes: [4, 16], strides: [1040, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x16xbf16, #map18, #hivm.address_space<ub>>
    ave.hir.store_with_stride %reinterpret_cast_23[%c0, %c0], %c1040, %14, %11 : memref<4x16xbf16, #map18, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
  }
  return
}