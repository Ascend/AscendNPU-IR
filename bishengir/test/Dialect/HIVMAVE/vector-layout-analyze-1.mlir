// RUN: bishengir-opt -analyze-vector-layout -analyze-alignment-bitwidth %s -split-input-file | FileCheck %s

// CHECK-LABEL: @test_preg_arith_op_lowering
func.func @test_preg_arith_op_lowering(%arg0: memref<8xi1, #hivm.address_space<ub>>, %arg1: memref<8x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %arg2: memref<8x8xi32, #hivm.address_space<ub>>, %arg3: memref<i32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i16 = arith.constant 0 : i16
  %c1_i8 = arith.constant 1 : i8
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %c0_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
  // CHECK: %1 = ave.hir.broadcast %c0_i32, %0 {element_alignment_bit_width = 32 : i32}
  %2 = ave.hir.pge <ALLF> : vector<64xi1>
  %res = ave.hir.vload <BRC_B32> %arg3[] : memref<i32, #hivm.address_space<ub>> into vector<1xi32>
  // CHECK: %res = ave.hir.vload <BRC_B32> %arg3[] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>}
  %3 = builtin.unrealized_conversion_cast %res : vector<1xi32> to vector<i32>
  %4 = builtin.unrealized_conversion_cast %3 : vector<i32> to vector<64xi32>
  %5:2 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %3, %arg6 = %4) -> (vector<i32>, vector<64xi32>) {
    %9 = builtin.unrealized_conversion_cast %arg6 : vector<64xi32> to vector<i32>
    %subview = memref.subview %arg1[%arg4, 0] [1, 8] [1, 1] : memref<8x8xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1x8xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg2[%arg4, 0] [1, 8] [1, 1] : memref<8x8xi32, #hivm.address_space<ub>> to memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %res_1 = ave.hir.vload <NORM> %arg0[%c0] : memref<8xi1, #hivm.address_space<ub>> into vector<64xi1>
    // CHECK: %res_1 = ave.hir.vload <NORM> %arg0[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>}
    %10 = builtin.unrealized_conversion_cast %res_1 : vector<64xi1> to vector<256xi1>
    %11 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
    %12 = ave.hir.scalar_broadcast %c1_i8 : i8 -> vector<256xi8>
    %13 = ave.hir.vsel %10, %12, %11 : vector<256xi1>, vector<256xi8>
    // CHECK: %11 = ave.hir.scalar_broadcast %c0_i8 {element_alignment_bit_width = 32 : i32}
    // CHECK-NEXT: %12 = ave.hir.scalar_broadcast %c1_i8 {element_alignment_bit_width = 32 : i32}
    // CHECK-NEXT: %13 = ave.hir.vsel %10, %12, %11 {element_alignment_bit_width = 32 : i32}
    %14 = ave.hir.pge <ALL> : vector<256xi1>
    %15 = arith.addi %arg4, %c1 : index
    %res_2, %new_true_shape = ave.hir.plt %15 : vector<256xi1>, index
    %res_3, %new_true_shape_4 = ave.hir.plt %arg4 : vector<256xi1>, index
    %16 = ave.hir.preg.xor <b8> %res_2, %res_3, %14 : vector<256xi1>
    // CHECK: %16 = ave.hir.preg.xor <b8> %res_2, %res_3, %14 {element_alignment_bit_width = 32 : i32}
    %17 = ave.hir.pge <ALL> : vector<256xi1>
    %18 = ave.hir.broadcast %c0_i16, %17 : i16, vector<256xi1> -> vector<256xi8>
    %19 = ave.hir.vor %13, %18, %16 : vector<256xi8>, vector<256xi1>
    // CHECK: %18 = ave.hir.broadcast %c0_i16, %17 {element_alignment_bit_width = 32 : i32}
    // CHECK-NEXT: %19 = ave.hir.vor %13, %18, %16 {element_alignment_bit_width = 32 : i32}
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
    // CHECK: %31 = ave.hir.vcmp <NE> %29, %30, %14 {element_alignment_bit_width = 32 : i32}
    %32 = builtin.unrealized_conversion_cast %31 : vector<256xi1> to vector<64xi1>
    %33 = ave.hir.pge <VL8> : vector<64xi1>
    %subview_19 = memref.subview %subview[0, 0] [1, 8] [1, 1] : memref<1x8xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<8xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_20 = ave.hir.vload <NORM> %subview_19[%c0] : memref<8xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi1>
    // CHECK: %res_20 = ave.hir.vload <NORM> %subview_19[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>}
    %subview_21 = memref.subview %subview_0[0, 0] [1, 8] [1, 1] : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> to memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_22 = ave.hir.vload <NORM> %subview_21[%c0] : memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi32>
    // CHECK: %res_22 = ave.hir.vload <NORM> %subview_21[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>}
    %34 = ave.hir.pge <ALL> : vector<64xi1>
    %35 = ave.hir.preg.xor <b8> %32, %2, %34 : vector<64xi1>
    // CHECK: %35 = ave.hir.preg.xor <b8> %32, %2, %34 {element_alignment_bit_width = 32 : i32}
    %36 = ave.hir.pge <ALL> : vector<64xi1>
    %37 = ave.hir.preg.or <b8> %35, %res_20, %36 : vector<64xi1>
    %38 = ave.hir.vsel %37, %res_22, %1 : vector<64xi1>, vector<64xi32>
    // CHECK: %37 = ave.hir.preg.or <b8> %35, %res_20, %36 {element_alignment_bit_width = 32 : i32}
    // CHECK: %38 = ave.hir.vsel %37, %res_22, %1 {element_alignment_bit_width = 32 : i32}
    %39 = builtin.unrealized_conversion_cast %9 : vector<i32> to i32
    %40 = ave.hir.vsel %33, %38, %1 : vector<64xi1>, vector<64xi32>
    // CHECK: %40 = ave.hir.vsel %33, %38, %1 {element_alignment_bit_width = 32 : i32}
    %41 = ave.hir.pge <ALL> : vector<64xi1>
    %42 = builtin.unrealized_conversion_cast %39 : i32 to vector<1xi32>
    %43 = builtin.unrealized_conversion_cast %42 : vector<1xi32> to vector<64xi32>
    %44 = ave.hir.reduction <add>, %40, %41 : vector<64xi32>, vector<64xi1> -> vector<64xi32>
    // CHECK: %44 = ave.hir.reduction <add>, %40, %41 {element_alignment_bit_width = 32 : i32}
    %45 = ave.hir.pge <ALL> : vector<64xi1>
    %46 = ave.hir.vadd %43, %44, %45 : vector<64xi32>, vector<64xi1>
    // CHECK: %46 = ave.hir.vadd %43, %44, %45 {element_alignment_bit_width = 32 : i32}
    %47 = builtin.unrealized_conversion_cast %46 : vector<64xi32> to vector<1xi32>
    %48 = builtin.unrealized_conversion_cast %47 : vector<1xi32> to i32
    %49 = builtin.unrealized_conversion_cast %48 : i32 to vector<i32>
    %50 = builtin.unrealized_conversion_cast %49 : vector<i32> to vector<64xi32>
    scf.yield %49, %50 : vector<i32>, vector<64xi32>
  } {element_alignment_bit_width = -1 : i32}
  %6 = builtin.unrealized_conversion_cast %5#1 : vector<64xi32> to vector<i32>
  %7 = builtin.unrealized_conversion_cast %6 : vector<i32> to vector<1xi32>
  %8 = ave.hir.pge <ALL> : vector<1xi1>
  ave.hir.masked_store <ONEPT_B32> %arg3[], %8, %7 : memref<i32, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi32>
  // CHECK: ave.hir.masked_store <ONEPT_B32> %arg3[], %8, %7 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>}
  return
}


// -----
// CHECK-LABEL:@plds_i1_as_msk_of_vsel_f32
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
  %0 = ave.hir.broadcast %cst_0, %T {element_alignment_bit_width = 32 : i32} : f32, vector<64xi1> -> vector<64xf32>
  // CHECK: %1 = ave.hir.broadcast %cst, %0 {element_alignment_bit_width = 32 : i32}
  %1 = ave.hir.broadcast %cst_1, %T {element_alignment_bit_width = 32 : i32} : f32, vector<64xi1> -> vector<64xf32>
  // CHECK-NEXT: %2 = ave.hir.broadcast %cst_0, %0 {element_alignment_bit_width = 32 : i32}
  scf.for %arg6 = %c0 to %c64 step %c1 {
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg0 : memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index
    %2 = affine.apply #map16()[%arg6]
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%2], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map15, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %reinterpret_cast_5[%c0] {element_alignment_bit_width = 8 : i32} : memref<64xi1, #map15, #hivm.address_space<ub>> into vector<64xi1>
    // CHECK: %res = ave.hir.vload <NORM> %reinterpret_cast[%c0] {element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb32>}
    %9 = ave.hir.vsel %res, %0, %1 {element_alignment_bit_width = 32 : i32} : vector<64xi1>, vector<64xf32>
    // CHECK-NEXT: %4 = ave.hir.vsel %res, %1, %2 {element_alignment_bit_width = 32 : i32}
    %11 = ave.hir.vtruncf %9, <rint>, false, <part_even>, %T {element_alignment_bit_width = 32 : i32} : vector<64xf32>, vector<128xbf16>, vector<64xi1>
    // CHECK-NEXT: %5 = ave.hir.vtruncf %4, <rint>, false, <part_even>, %0 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<even>}
    %base_buffer_19, %offset_20, %sizes_21:3, %strides_22:3 = memref.extract_strided_metadata %arg5 : memref<4x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index
    %13 = affine.apply #map4()[%arg6]
    %14 = ave.hir.pge <VL64> : vector<128xi1>
    %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer_19 to offset: [%13], sizes: [4, 16], strides: [1040, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x16xbf16, #map18, #hivm.address_space<ub>>
    ave.hir.store_with_stride %reinterpret_cast_23[%c0, %c0], %c1040, %14, %11 {element_alignment_bit_width = 16 : i32} : memref<4x16xbf16, #map18, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
    // CHECK: ave.hir.store_with_stride %reinterpret_cast_5[%c0, %c0], %c1040, %7, %5 {element_alignment_bit_width = 16 : i32, functionType = #ave.func_dist_type<dintlv2>}
  } {element_alignment_bit_width = -1 : i32}
  return
}

// -----
// CHECK-LABEL: @test_exti32
// CHECK: %0 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %1 = ave.hir.broadcast %c0_i64, %0 {element_alignment_bit_width = 32 : i32} : i64, vector<64xi1> -> vector<64xi64>
// CHECK-NEXT: %2 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %3 = ave.hir.broadcast %c-1_i64, %2 {element_alignment_bit_width = 32 : i32} : i64, vector<64xi1> -> vector<64xi64>
// CHECK: %res = ave.hir.vload <NORM> %subview[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
// CHECK-NEXT: %4 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %5 = ave.hir.vextsi %res, %4 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<even>, part = #ave.vcvt_part_type<part_even>} : vector<64xi32>, vector<64xi64>, vector<64xi1>
// CHECK-NEXT: %6 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %7 = ave.hir.vcmp <EQ> %5, %3, %6 {element_alignment_bit_width = 32 : i32} : vector<64xi64>, vector<64xi1> -> vector<64xi1>
// CHECK-NEXT: %8 = ave.hir.vsel %7, %1, %5 {element_alignment_bit_width = 32 : i32} : vector<64xi1>, vector<64xi64>
// CHECK-NEXT: %9 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %10 = ave.hir.vtrunci %8, false, %9 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<even>, part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<si2si>} : vector<64xi64>, vector<64xi32>, vector<64xi1>
// CHECK-NEXT: %11 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: ave.hir.masked_store <NORM_B32> %subview_0[%c0], %11, %10 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
func.func @test_exti32(%arg0: memref<2112xi32, #hivm.address_space<ub>>, %arg1: memref<2112xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c2112 = arith.constant 2112 : index
  %c64 = arith.constant 64 : index
  %c-1_i64 = arith.constant -1 : i64
  %c0_i64 = arith.constant 0 : i64
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %c0_i64, %0 : i64, vector<64xi1> -> vector<64xi64>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %c-1_i64, %2 : i64, vector<64xi1> -> vector<64xi64>
  scf.for %arg2 = %c0 to %c2112 step %c64 {
    %subview = memref.subview %arg0[%arg2] [64] [1] : memref<2112xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [64] [1] : memref<2112xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
    %4 = ave.hir.pge <ALL> : vector<64xi1>
    %5 = ave.hir.vextsi %res, %4 {part = #ave.vcvt_part_type<part_even>} : vector<64xi32>, vector<64xi64>, vector<64xi1>
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    %7 = ave.hir.vcmp <EQ> %5, %3, %6 : vector<64xi64>, vector<64xi1> -> vector<64xi1>
    %8 = ave.hir.vsel %7, %1, %5 : vector<64xi1>, vector<64xi64>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    %10 = ave.hir.vtrunci %8, false, %9 {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<si2si>} : vector<64xi64>, vector<64xi32>, vector<64xi1>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_0[%c0], %11, %10 {hivm.is_continuous} : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  } {element_alignment_bit_width = 32 : i32}
  return
}

// -----
// CHECK-LABEL: @test_i8_4vl_load_used_by_vcmp
// CHECK: %0 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %1 = ave.hir.broadcast %c2147483647_i32, %0 {element_alignment_bit_width = 32 : i32} : i32, vector<64xi1> -> vector<64xi32>
// CHECK-NEXT: %2 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %3 = ave.hir.broadcast %cst, %2 {element_alignment_bit_width = 32 : i32} : f32, vector<64xi1> -> vector<64xf32>
// CHECK-NEXT: %4 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %5 = ave.hir.broadcast %c0_i16, %4 {element_alignment_bit_width = 32 : i32} : i16, vector<64xi1> -> vector<64xi8>
// CHECK: %6 = ave.hir.pge <VL16> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>, mask_op_idx = 0 : i32} : vector<64xi1>
// CHECK: %res = ave.hir.vload <NORM> %subview_2[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pk4>} : memref<16xi8, #map, #hivm.address_space<ub>> into vector<64xi8>
// CHECK: %res_4 = ave.hir.vload <NORM> %subview_3[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<16xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
// CHECK-NEXT: %7 = ave.hir.vmuls %res_4, %arg2, %6 {element_alignment_bit_width = 32 : i32} : vector<64xf32>, f32, vector<64xi1>
// CHECK-NEXT: %8 = ave.hir.vcmp <NE> %res, %5, %6 {element_alignment_bit_width = 32 : i32} : vector<64xi8>, vector<64xi1> -> vector<64xi1>
// CHECK-NEXT: %9 = ave.hir.vsel %8, %7, %3 {element_alignment_bit_width = 32 : i32} : vector<64xi1>, vector<64xf32>
// CHECK: ave.hir.masked_store <NORM_B32> %subview_5[%c0], %6, %9 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<16xf32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
// CHECK: ave.hir.masked_store <NORM_B32> %subview_7[%c0], %6, %1 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<16xi32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>


func.func @test_i8_4vl_load_used_by_vcmp(%arg0: memref<8x16xi8, strided<[32, 1]>, #hivm.address_space<ub>>, %arg1: memref<8x16xf32, #hivm.address_space<ub>>, %arg2: f32, %arg3: memref<8x16xf32, #hivm.address_space<ub>>, %arg4: memref<8x16xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0_i16 = arith.constant 0 : i16
  %cst = arith.constant 0xFF800000 : f32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %c2147483647_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %cst, %2 : f32, vector<64xi1> -> vector<64xf32>
  %4 = ave.hir.pge <ALL> : vector<64xi1>
  %5 = ave.hir.broadcast %c0_i16, %4 : i16, vector<64xi1> -> vector<64xi8>
  scf.for %arg5 = %c0 to %c8 step %c1 {
    %subview = memref.subview %arg3[%arg5, 0] [1, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg5, 0] [1, 16] [1, 1] : memref<8x16xi8, strided<[32, 1]>, #hivm.address_space<ub>> to memref<1x16xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg5, 0] [1, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %6 = ave.hir.pge <VL16> {mask_op_idx = 0 : i32} : vector<64xi1>
    %subview_2 = memref.subview %subview_0[0, 0] [1, 16] [1, 1] : memref<1x16xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_2[%c0] : memref<16xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi8>
    %subview_3 = memref.subview %subview_1[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<16xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %7 = ave.hir.vmuls %res_4, %arg2, %6 : vector<64xf32>, f32, vector<64xi1>
    %8 = ave.hir.vcmp <NE> %res, %5, %6 : vector<64xi8>, vector<64xi1> -> vector<64xi1>
    %9 = ave.hir.vsel %8, %7, %3 : vector<64xi1>, vector<64xf32>
    %subview_5 = memref.subview %subview[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_5[%c0], %6, %9 : memref<16xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %subview_6 = memref.subview %arg4[%arg5, 0] [1, 16] [1, 1] : memref<8x16xi32, #hivm.address_space<ub>> to memref<1x16xi32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %subview_6[0, 0] [1, 16] [1, 1] : memref<1x16xi32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_7[%c0], %6, %1 : memref<16xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  } {element_alignment_bit_width = -1 : i32}
  return
}

// -----
// CHECK-LABEL: @test_iterarg_not_used
// CHECK: %0 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %1 = ave.hir.broadcast %cst, %0 {element_alignment_bit_width = 32 : i32} : f32, vector<64xi1> -> vector<64xf32>
// CHECK: %res = ave.hir.vload <NORM> %arg0[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
// CHECK-NEXT: %5 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: ave.hir.masked_store <NORM_B32> %arg1[%c0], %5, %res {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
// CHECK: %3 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK-NEXT: %4 = ave.hir.reduction <add>, %2, %3 {element_alignment_bit_width = 32 : i32} : vector<64xf32>, vector<64xi1> -> vector<64xf32>
// CHECK-NEXT: ave.hir.masked_store <NORM_B32> %arg2[%c0], %3, %4 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
func.func @test_iterarg_not_used(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
  %27 = scf.for %arg11 = %c0 to %c64 step %c64 iter_args(%arg12 = %1) -> (vector<64xf32>) {
    %res_10 = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %33 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg1[%c0], %33, %res_10 {hivm.is_continuous} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    scf.yield %res_10 : vector<64xf32>
  } {element_alignment_bit_width = 32 : i32}
  %29 = ave.hir.pge <ALL> : vector<64xi1>
  %30 = ave.hir.reduction <add>, %27, %29 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  ave.hir.masked_store <NORM_B32> %arg2[%c0], %29, %30 {hivm.is_continuous} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  return
}

// CHECK-LABEL:@test_pstore
// CHECK: %0 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK: %1 = ave.hir.broadcast %cst, %0 {element_alignment_bit_width = 32 : i32} : f32, vector<64xi1> -> vector<64xf32>
// CHECK: %res = ave.hir.vload <NORM> %subview_1[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
// CHECK: %2 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK: %3 = ave.hir.vcmp <GE> %res, %1, %2 {element_alignment_bit_width = 32 : i32} : vector<64xf32>, vector<64xi1> -> vector<64xi1>
// CHECK: %4 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK: ave.hir.masked_store <NORM_B8> %subview_2[%c0], %4, %3 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
func.func @test_pstore(%arg0: memref<64x64xf32, #hivm.address_space<ub>>, %arg1: memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
  scf.for %arg2 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg0[%arg2, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2, 0] [1, 64] [1, 1] : memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1x64xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_1[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    %3 = ave.hir.vcmp <GE> %res, %1, %2 : vector<64xf32>, vector<64xi1> -> vector<64xi1>
    %subview_2 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %4 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B8> %subview_2[%c0], %4, %3 : memref<64xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  } {element_alignment_bit_width = 32 : i32}
  return
}

// -----
// CHECK-LABEL: @test_mull
// CHECK: %0 = ave.hir.pge <ALL> {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
// CHECK: %res = ave.hir.vload <NORM> %arg0[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<6xi32, #hivm.address_space<ub>> into vector<64xi32>
// CHECK: %res_0 = ave.hir.vload <NORM> %arg0[%c0] {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<6xi32, #hivm.address_space<ub>> into vector<64xi32>
// CHECK: %res1, %res2 = ave.hir.mull %res, %res_0, %0 {element_alignment_bit_width = 32 : i32} : vector<64xi32>, vector<64xi1>
// CHECK: ave.hir.masked_store <NORM_B32> %arg0[%c0], %0, %res2 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<norm>} : memref<6xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
func.func @test_mull(%arg0: memref<6xi32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %v1 = ave.hir.vload <NORM> %arg0[%c0] : memref<6xi32, #hivm.address_space<ub>> into vector<64xi32>
  %v2 = ave.hir.vload <NORM> %arg0[%c0] : memref<6xi32, #hivm.address_space<ub>> into vector<64xi32>
  %res1, %res2 = ave.hir.mull %v1, %v2, %mask : vector<64xi32>, vector<64xi1>
  ave.hir.masked_store <NORM_B32> %arg0[%c0], %mask, %res2 : memref<6xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}

// -----
// CHECK-LABEL@test_dintlv_before_vsstb
func.func @test_dintlv_before_vsstb(%arg0: memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64x128xf32, #hivm.address_space<ub>>, %arg3: memref<64x128xf32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 8 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline, noinline, outline = true, vector_mode = "simd"} {
  %c1040 = arith.constant 1040 : index
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  scf.for %arg4 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
    %0 = arith.index_cast %arg4 : i32 to index
    %subview = memref.subview %arg1[%0] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg2[%0, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg3[%0, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_2[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %res_3 = ave.hir.vload <BRC_B32> %subview[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %1 = ave.hir.pge <ALL> : vector<64xi1>
    %2 = ave.hir.vsub %res, %res_3, %1 : vector<64xf32>, vector<64xi1>
    %subview_4 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_5 = ave.hir.vload <NORM> %subview_4[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %3 = ave.hir.pge <ALL> : vector<64xi1>
    %4 = ave.hir.vmul %res_5, %2, %3 : vector<64xf32>, vector<64xi1>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.vtruncf %4, <rint>, false, <part_even>, %5 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    %subview_6 = memref.subview %arg0[0, %0, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xbf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %subview_6[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xbf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xbf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %arg2[%0, 64] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_9 = memref.subview %arg3[%0, 64] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_10 = memref.subview %subview_9[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_11 = ave.hir.vload <NORM> %subview_10[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vsub %res_11, %res_3, %7 : vector<64xf32>, vector<64xi1>
    %subview_12 = memref.subview %subview_8[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_13 = ave.hir.vload <NORM> %subview_12[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    %10 = ave.hir.vmul %res_13, %8, %9 : vector<64xf32>, vector<64xi1>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    %12 = ave.hir.vtruncf %10, <rint>, false, <part_even>, %11 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    %13 = ave.hir.pge <ALL> : vector<128xi1>
    %res1, %res2 = ave.hir.vdintlv %6, %12 {layout_change = #ave<layout_change DENSE>}: vector<64xbf16>, vector<128xbf16>
    ave.hir.store_with_stride %subview_7[%c0, %c0], %c1040, %13, %res1 : memref<4x16xbf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
    // CHECK: ave.hir.store_with_stride %subview_7[%c0, %c0], %c1040, %13, %res1 {element_alignment_bit_width = 16 : i32, functionType = #ave.func_dist_type<norm>} : memref<4x16xbf16, #map1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
  } {element_alignment_bit_width = -1 : i32}
  return
}

// -----
// CHECK-LABEL: @test_dintlv_after_vgather
func.func @test_dintlv_after_vgather(%arg0: memref<7x5x3x4xf8E5M2, strided<[480, 96, 32, 1]>, #hivm.address_space<ub>>, %arg1: memref<4x5x3x7xf8E5M2, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 16 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i8 = arith.constant 0 : i8
  %c480_i16 = arith.constant 480 : i16
  %c7 = arith.constant 7 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %c960_i16 = arith.constant 960 : i16
  %c0_i16 = arith.constant 0 : i16
  %0 = ave.hir.vci %c0_i16, <INCREASE> : i16, vector<128xi16>
  %1 = ave.hir.pge <ALL> : vector<128xi1>
  %2 = ave.hir.vmuls %0, %c960_i16, %1 : vector<128xi16>, i16, vector<128xi1>
  %3 = ave.hir.pge <ALL> : vector<128xi1>
  %4 = ave.hir.broadcast %c0_i16, %3 : i16, vector<128xi1> -> vector<128xi16>
  %5 = ave.hir.pge <VL4> : vector<128xi1>
  %6 = ave.hir.vsel %5, %2, %4 : vector<128xi1>, vector<128xi16>
  scf.for %arg2 = %c0 to %c4 step %c1 {
    scf.for %arg3 = %c0 to %c5 step %c1 {
      scf.for %arg4 = %c0 to %c3 step %c1 {
        %subview = memref.subview %arg0[0, %arg3, %arg4, %arg2] [7, 1, 1, 1] [1, 1, 1, 1] : memref<7x5x3x4xf8E5M2, strided<[480, 96, 32, 1]>, #hivm.address_space<ub>> to memref<7x1x1x1xf8E5M2, strided<[480, 96, 32, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg1[%arg2, %arg3, %arg4, 0] [1, 1, 1, 7] [1, 1, 1, 1] : memref<4x5x3x7xf8E5M2, #hivm.address_space<ub>> to memref<1x1x1x7xf8E5M2, strided<[105, 21, 7, 1], offset: ?>, #hivm.address_space<ub>>
        %res, %new_true_shape = ave.hir.plt %c7 {mask_op_idx = 0 : i32} : vector<256xi1>, index
        %7 = ave.hir.vci %c0_i16, <INCREASE> : i16, vector<128xi16>
        %8 = ave.hir.pge <ALL> : vector<128xi1>
        %9 = ave.hir.vmuls %7, %c960_i16, %8 : vector<128xi16>, i16, vector<128xi1>
        %10 = ave.hir.vadds %9, %c480_i16, %8 : vector<128xi16>, i16, vector<128xi1>
        %11 = ave.hir.pge <ALL> : vector<128xi1>
        %12 = ave.hir.broadcast %c0_i16, %11 : i16, vector<128xi1> -> vector<128xi16>
        %13 = ave.hir.pge <VL3> : vector<128xi1>
        %14 = ave.hir.vsel %13, %10, %12 : vector<128xi1>, vector<128xi16>
        %15 = ave.hir.vgather %subview[%c0, %c0, %c0, %c0] [%6], %res : memref<7x1x1x1xf8E5M2, strided<[480, 96, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xf8E5M2>
        %16 = ave.hir.vgather %subview[%c0, %c0, %c0, %c0] [%14], %res : memref<7x1x1x1xf8E5M2, strided<[480, 96, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xf8E5M2>
        %17 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
        %18 = vector.bitcast %17 : vector<256xi8> to vector<256xf8E5M2>
        %res1, %res2 = ave.hir.vdintlv %15, %18 {layout_change = #ave<layout_change DENSE>} : vector<256xf8E5M2>, vector<256xf8E5M2>
        %res1_1, %res2_2 = ave.hir.vdintlv %16, %18 {layout_change = #ave<layout_change DENSE>} : vector<256xf8E5M2>, vector<256xf8E5M2>
        %res1_3, %res2_4 = ave.hir.vintlv %res1, %res1_1 : vector<256xf8E5M2>, vector<256xf8E5M2>
        %subview_5 = memref.subview %subview_0[0, 0, 0, 0] [1, 1, 1, 7] [1, 1, 1, 1] : memref<1x1x1x7xf8E5M2, strided<[105, 21, 7, 1], offset: ?>, #hivm.address_space<ub>> to memref<7xf8E5M2, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        ave.hir.masked_store <NORM_B8> %subview_5[%c0], %res, %res1_3 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<7xf8E5M2, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xf8E5M2>
      // CHECK: %res, %new_true_shape = ave.hir.plt %c7 {element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb8>, mask_op_idx = 0 : i32} : vector<256xi1>, index
      // CHECK: %7 = ave.hir.vci %c0_i16, <INCREASE> {element_alignment_bit_width = 16 : i32} : i16, vector<128xi16>
      // CHECK: %8 = ave.hir.pge <ALL> {element_alignment_bit_width = 16 : i32, functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
      // CHECK: %9 = ave.hir.vmuls %7, %c960_i16, %8 {element_alignment_bit_width = 16 : i32} : vector<128xi16>, i16, vector<128xi1>
      // CHECK: %10 = ave.hir.vadds %9, %c480_i16, %8 {element_alignment_bit_width = 16 : i32} : vector<128xi16>, i16, vector<128xi1>
      // CHECK: %11 = ave.hir.pge <ALL> {element_alignment_bit_width = 16 : i32, functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
      // CHECK: %12 = ave.hir.broadcast %c0_i16, %11 {element_alignment_bit_width = 16 : i32} : i16, vector<128xi1> -> vector<128xi16>
      // CHECK: %13 = ave.hir.pge <VL3> {element_alignment_bit_width = 16 : i32, functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
      // CHECK: %14 = ave.hir.vsel %13, %10, %12 {element_alignment_bit_width = 16 : i32} : vector<128xi1>, vector<128xi16>
      // CHECK: %15 = ave.hir.vgather %subview[%c0, %c0, %c0, %c0] [%6], %res {element_alignment_bit_width = 8 : i32} : memref<7x1x1x1xf8E5M2, strided<[480, 96, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xf8E5M2>
      // CHECK: %16 = ave.hir.vgather %subview[%c0, %c0, %c0, %c0] [%14], %res {element_alignment_bit_width = 8 : i32} : memref<7x1x1x1xf8E5M2, strided<[480, 96, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xf8E5M2>
      // CHECK: %17 = ave.hir.scalar_broadcast %c0_i8 {element_alignment_bit_width = 16 : i32} : i8 -> vector<256xi8>
      // CHECK: %18 = vector.bitcast %17 {element_alignment_bit_width = 16 : i32} : vector<256xi8> to vector<256xf8E5M2>
      // CHECK: %res1, %res2 = ave.hir.vdintlv %15, %18 {element_alignment_bit_width = 16 : i32, layout_change = #ave<layout_change DENSE>} : vector<256xf8E5M2>, vector<256xf8E5M2>
      // CHECK: %res1_1, %res2_2 = ave.hir.vdintlv %16, %18 {element_alignment_bit_width = 16 : i32, layout_change = #ave<layout_change DENSE>} : vector<256xf8E5M2>, vector<256xf8E5M2>
      // CHECK: %res1_3, %res2_4 = ave.hir.vintlv %res1, %res1_1 {element_alignment_bit_width = 8 : i32} : vector<256xf8E5M2>, vector<256xf8E5M2>
      // CHECK: ave.hir.masked_store <NORM_B8> %subview_5[%c0], %res, %res1_3 {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<norm>} : memref<7xf8E5M2, #map, #hivm.address_space<ub>>, vector<256xi1>, vector<256xf8E5M2>
      } {element_alignment_bit_width = -1 : i32}
    } {element_alignment_bit_width = 8 : i32}
  } {element_alignment_bit_width = 8 : i32}
  return
}

// -----
// CHECK-LABEL: @test_vor_dense
#map = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>
func.func @test_vor_dense(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64x128xf32, #hivm.address_space<ub>>, %arg3: memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline, noinline, outline = true, vector_mode = "simd"} {
  %c1040 = arith.constant 1040 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 5.000000e-01 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst_0, %0 : f32, vector<64xi1> -> vector<64xf32>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %cst, %2 : f32, vector<64xi1> -> vector<64xf32>
  scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
    %4 = arith.index_cast %arg5 : i32 to index
    %subview = memref.subview %arg2[%4, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_1[%c0] : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.vmul %res, %1, %5 : vector<64xf32>, vector<64xi1>
    %subview_2 = memref.subview %arg0[%4, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_3 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_3[%c0], %7, %6 : memref<64xf32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %subview_4 = memref.subview %arg2[%4, 64] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %subview_4[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %res_6 = ave.hir.vload <NORM> %subview_5[%c0] : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
    %8 = ave.hir.pge <ALL> : vector<64xi1>
    %9 = ave.hir.vmul %res_6, %1, %8 : vector<64xf32>, vector<64xi1>
    %subview_7 = memref.subview %arg0[%4, 64] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %subview_7[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %10 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_8[%c0], %10, %9 : memref<64xf32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    %12 = ave.hir.vmax %6, %9, %11 : vector<64xf32>, vector<64xi1>
    %13 = ave.hir.pge <ALL> : vector<64xi1>
    %14 = ave.hir.reduction <max>, %12, %13 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    %15 = builtin.unrealized_conversion_cast %14 : vector<64xf32> to vector<1xf32>
    %subview_9 = memref.subview %arg1[%4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_10 = memref.subview %subview_9[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>
    %16 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B32> %subview_10[], %16, %15 {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous} : memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  }
  annotation.mark %c0_i64 {SYNC_IN_VF = "VST_VLD"} : i64
  scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
    %4 = arith.index_cast %arg5 : i32 to index
    %subview = memref.subview %arg1[%4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg0[%4, 0] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %5 = memref.load %subview[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %res, %res1 = ave.hir.vload <DINTLV_B32> %subview_2[%c0] : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>, vector<64xf32>
    %6 = ave.hir.scalar_broadcast %5 : f32 -> vector<64xf32>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vsub %res, %6, %7 : vector<64xf32>, vector<64xi1>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    %10 = ave.hir.vsub %res1, %6, %9 : vector<64xf32>, vector<64xi1>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    %12 = ave.hir.vexp %8, %11 : vector<64xf32>, vector<64xi1>
    %13 = ave.hir.pge <ALL> : vector<64xi1>
    %14 = ave.hir.vexp %10, %13 : vector<64xf32>, vector<64xi1>
    %15 = ave.hir.pge <ALL> : vector<64xi1>
    %subview_3 = memref.subview %arg3[0, %4, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_4 = memref.subview %subview_3[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, #map1, #hivm.address_space<ub>>
    %16 = ave.hir.pge <ALL> : vector<64xi1>
    %17 = ave.hir.pge <ALL> : vector<128xi1>
    %18 = ave.hir.vtruncf %12, <rint>, false, <part_even>, %15 {layout_change = #ave<layout_change DENSE>} : vector<64xf32>, vector<128xf16>, vector<64xi1>
    %19 = ave.hir.vtruncf %14, <rint>, false, <part_odd>, %16 {layout_change = #ave<layout_change DENSE>} : vector<64xf32>, vector<128xf16>, vector<64xi1>
    %20 = ave.hir.vor %18, %19, %17 : vector<128xf16>, vector<128xi1>
    ave.hir.store_with_stride %subview_4[%c0, %c0], %c1040, %17, %20 : memref<4x16xf16, #map1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    // CHECK:       %18 = ave.hir.vtruncf %12, <rint>, false, <part_even>, %15 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<c2c>, layout_change = #ave<layout_change DENSE>} : vector<64xf32>, vector<128xf16>, vector<64xi1>
    // CHECK:       %19 = ave.hir.vtruncf %14, <rint>, false, <part_odd>, %16 {element_alignment_bit_width = 32 : i32, functionType = #ave.func_dist_type<c2c>, layout_change = #ave<layout_change DENSE>} : vector<64xf32>, vector<128xf16>, vector<64xi1>
    // CHECK:       %20 = ave.hir.vor %18, %19, %17 {element_alignment_bit_width = 16 : i32} : vector<128xf16>, vector<128xi1>
    // CHECK:       ave.hir.store_with_stride %subview_4[%c0, %c0], %c1040, %17, %20 {element_alignment_bit_width = 16 : i32, functionType = #ave.func_dist_type<norm>} : memref<4x16xf16, #map1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
%21 = ave.hir.pge <ALL> : vector<64xi1>
    %22 = ave.hir.vadd %12, %14, %21 : vector<64xf32>, vector<64xi1>
    %23 = ave.hir.pge <ALL> : vector<64xi1>
    %24 = ave.hir.reduction <add>, %22, %23 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    %25 = ave.hir.pge <ALL> : vector<64xi1>
    %26 = ave.hir.vadd %24, %3, %25 : vector<64xf32>, vector<64xi1>
    %27 = builtin.unrealized_conversion_cast %26 : vector<64xf32> to vector<1xf32>
    %subview_5 = memref.subview %arg4[%4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %subview_5[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>
    %28 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B32> %subview_6[], %28, %27 {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous} : memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  }
  return
}