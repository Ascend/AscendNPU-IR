// RUN: bishengir-opt -analyze-vector-layout -ave-normalize-ops \
// RUN:  -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -cse %s -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @no_unroll
// CHECK: scf.for %arg4
// CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vdintlv"
// CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v128f16"
func.func @no_unroll(%arg0: memref<64x8x16xf32, #hivm.address_space<ub>>, %arg1: memref<64x8x16xf16, #hivm.address_space<ub>>, %arg2: memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c1040 = arith.constant 1040 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c4 {
      %subview = memref.subview %arg0[%arg3, %arg4, 0] [1, 4, 16] [1, 1, 1] : memref<64x8x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %subview[0, 0, 0] [1, 4, 16] [1, 1, 1] : memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_0[%c0, %c0] : memref<4x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      %0 = ave.hir.pge <ALL> : vector<64xi1>
      %1 = ave.hir.vtruncf %res, <round>, false, <part_even>, %0 : vector<64xf32>, vector<64xf16>, vector<64xi1>
      %subview_1 = memref.subview %arg2[%arg4, %arg3, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview_1[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.store_with_stride %subview_2[%c0, %c0], %c1040, %2, %1 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    }
  }
  return
}

// CHECK-LABEL: func.func @bf16_vsstb
// CHECK: scf.for %arg4
// CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vdintlv"
// CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v128bf16"
func.func @bf16_vsstb(%arg0: memref<64x8x16xf32, #hivm.address_space<ub>>, %arg1: memref<64x8x16xbf16, #hivm.address_space<ub>>, %arg2: memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c1040 = arith.constant 1040 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c4 {
      %subview = memref.subview %arg0[%arg3, %arg4, 0] [1, 4, 16] [1, 1, 1] : memref<64x8x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %subview[0, 0, 0] [1, 4, 16] [1, 1, 1] : memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_0[%c0, %c0] : memref<4x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      %0 = ave.hir.pge <ALL> : vector<64xi1>
      %1 = ave.hir.vtruncf %res, <round>, false, <part_even>, %0 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
      %subview_1 = memref.subview %arg2[%arg4, %arg3, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xbf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview_1[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xbf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xbf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.store_with_stride %subview_2[%c0, %c0], %c1040, %2, %1 : memref<4x16xbf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    }
  }
  return
}

// CHECK-LABEL: func.func @narrow_mask
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v128f16"
// CHECK: "hivm_regbaseintrins.intr.hivm.plt.b16.v300"
// CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v128f16"
func.func @narrow_mask(%arg0: memref<64x7x16xf16, #hivm.address_space<ub>>, %arg1: memref<64x7x16xf32, #hivm.address_space<ub>>, %arg2: memref<7x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c48 = arith.constant 48 : index
  %c1040 = arith.constant 1040 : index
  %c112 = arith.constant 112 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg1[%arg3, 0, 0] [1, 4, 16] [1, 1, 1] : memref<64x7x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[112, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %subview[0, 0, 0] [1, 4, 16] [1, 1, 1] : memref<1x4x16xf32, strided<[112, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_0[%c0, %c0] : memref<4x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.vtruncf %res, <rint>, false, <part_even>, %0 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2]] : memref<64x7x16xf16, #hivm.address_space<ub>> into memref<7168xf16, #hivm.address_space<ub>>
    %2 = arith.muli %arg3, %c112 : index
    %subview_1 = memref.subview %collapse_shape[%2] [64] [1] : memref<7168xf16, #hivm.address_space<ub>> to memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %3 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B16> %subview_1[%c0], %3, %1 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    %subview_2 = memref.subview %arg2[0, %arg3, 0] [4, 1, 16] [1, 1, 1] : memref<7x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_3 = memref.subview %arg0[%arg3, 0, 0] [1, 4, 16] [1, 1, 1] : memref<64x7x16xf16, #hivm.address_space<ub>> to memref<4x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %res_4 = ave.hir.vload <NORM> %subview_3[%c0, %c0] : memref<4x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf16>
    %subview_5 = memref.subview %subview_2[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
    %4 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.store_with_stride %subview_5[%c0, %c0], %c1040, %4, %res_4 : memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    %subview_6 = memref.subview %arg1[%arg3, 4, 0] [1, 3, 16] [1, 1, 1] : memref<64x7x16xf32, #hivm.address_space<ub>> to memref<1x3x16xf32, strided<[112, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %res_7, %new_true_shape = ave.hir.plt %c48 : vector<64xi1>, index
    %5 = builtin.unrealized_conversion_cast %res_7 : vector<64xi1> to vector<4x16xi1>
    %subview_8 = memref.subview %subview_6[0, 0, 0] [1, 3, 16] [1, 1, 1] : memref<1x3x16xf32, strided<[112, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<3x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>>
    %res_9 = ave.hir.vload <NORM> %subview_8[%c0, %c0] : memref<3x16xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    %7 = ave.hir.vtruncf %res_9, <rint>, false, <part_even>, %6 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %8 = arith.addi %2, %c64 : index
    %subview_10 = memref.subview %collapse_shape[%8] [48] [1] : memref<7168xf16, #hivm.address_space<ub>> to memref<48xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B16> %subview_10[%c0], %9, %7 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<48xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    %subview_11 = memref.subview %arg2[4, %arg3, 0] [3, 1, 16] [1, 1, 1] : memref<7x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<3x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_12 = memref.subview %arg0[%arg3, 4, 0] [1, 3, 16] [1, 1, 1] : memref<64x7x16xf16, #hivm.address_space<ub>> to memref<3x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %res_13 = ave.hir.vload <NORM> %subview_12[%c0, %c0] : memref<3x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf16>
    %subview_14 = memref.subview %subview_11[0, 0, 0] [3, 1, 16] [1, 1, 1] : memref<3x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<3x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>
    %10 = builtin.unrealized_conversion_cast %res_7 : vector<64xi1> to vector<16xi1>
    ave.hir.store_with_stride %subview_14[%c0, %c0], %c1040, %10, %res_13 : memref<3x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>, #hivm.address_space<ub>>, vector<16xi1>, vector<64xf16>
  }
  return
}

// CHECK-LABEL: func.func @narrow_mask1
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.pge.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
func.func @narrow_mask1(%arg0: memref<32x32xf32, #hivm.address_space<ub>>, %arg1: memref<2x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c528 = arith.constant 528 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c32 step %c1 {
    %subview = memref.subview %arg1[0, %arg2, 0] [2, 1, 16] [1, 1, 1] : memref<2x32x16xbf16, strided<[528, 16, 1]>, #hivm.address_space<ub>> to memref<2x1x16xbf16, strided<[528, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg2, 0] [1, 32] [1, 1] : memref<32x32xf32, #hivm.address_space<ub>> to memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = ave.hir.pge <VL32> : vector<64xi1>
    %subview_1 = memref.subview %subview_0[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> to memref<32xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_1[%c0] : memref<32xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %1 = ave.hir.pge <ALL> : vector<64xi1>
    %2 = ave.hir.vtruncf %res, <rint>, false, <part_even>, %1 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    %3 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<4x16xi1>
    %subview_2 = memref.subview %subview[0, 0, 0] [2, 1, 16] [1, 1, 1] : memref<2x1x16xbf16, strided<[528, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<2x16xbf16, affine_map<(d0, d1)[s0] -> (d0 * 528 + d1 + s0)>, #hivm.address_space<ub>>
    %4 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<16xi1>
    ave.hir.store_with_stride %subview_2[%c0, %c0], %c528, %4, %2 : memref<2x16xbf16, affine_map<(d0, d1)[s0] -> (d0 * 528 + d1 + s0)>, #hivm.address_space<ub>>, vector<16xi1>, vector<64xbf16>
  }
  return
}

// CHECK-LABEL: func.func @no_merge_vsstb
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"
// CHECK: "hivm_regbaseintrins.intr.hivm.vdintlv"
// CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v128f16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"
// CHECK: "hivm_regbaseintrins.intr.hivm.vdintlv"
// CHECK: "hivm_regbaseintrins.intr.hivm.vsstb.v128f16"
func.func @no_merge_vsstb(%arg0: memref<16x32xf32, #hivm.address_space<ub>>, %arg1: memref<16xf32, #hivm.address_space<ub>>, %arg2: memref<1xf32, #hivm.address_space<ub>>, %arg3: memref<16x32xf32, #hivm.address_space<ub>>, %arg4: memref<16xf32, #hivm.address_space<ub>>, %arg5: f32, %arg6: memref<2x16x16xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>, %arg7: memref<2x16x16xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c272 = arith.constant 272 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.693147182 : f32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
  scf.for %arg8 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg6[0, %arg8, 0] [2, 1, 16] [1, 1, 1] : memref<2x16x16xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>> to memref<2x1x16xf16, strided<[272, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg8, 0] [1, 32] [1, 1] : memref<16x32xf32, #hivm.address_space<ub>> to memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg8] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %2 = ave.hir.pge <VL32> : vector<64xi1>
    %subview_2 = memref.subview %subview_0[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> to memref<32xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_2[%c0] : memref<32xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %res_3 = ave.hir.vload <BRC_B32> %arg2[%c0] : memref<1xf32, #hivm.address_space<ub>> into vector<64xf32>
    %res_4 = ave.hir.vload <BRC_B32> %subview_1[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %3 = ave.hir.pge <ALL> : vector<64xi1>
    %4 = ave.hir.vmul %res, %res_3, %3 : vector<64xf32>, vector<64xi1>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.vsub %4, %res_4, %5 : vector<64xf32>, vector<64xi1>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vmul %6, %1, %7 : vector<64xf32>, vector<64xi1>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    %10 = ave.hir.vexp %8, %9 : vector<64xf32>, vector<64xi1>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    %12 = ave.hir.vtruncf %10, <rint>, false, <part_even>, %11 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %13 = builtin.unrealized_conversion_cast %2 : vector<64xi1> to vector<4x16xi1>
    %subview_5 = memref.subview %subview[0, 0, 0] [2, 1, 16] [1, 1, 1] : memref<2x1x16xf16, strided<[272, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<2x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 272 + d1 + s0)>, #hivm.address_space<ub>>
    %14 = builtin.unrealized_conversion_cast %2 : vector<64xi1> to vector<16xi1>
    ave.hir.store_with_stride %subview_5[%c0, %c0], %c272, %14, %12 : memref<2x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 272 + d1 + s0)>, #hivm.address_space<ub>>, vector<16xi1>, vector<64xf16>
    %subview_6 = memref.subview %arg7[0, %arg8, 0] [2, 1, 16] [1, 1, 1] : memref<2x16x16xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>> to memref<2x1x16xf16, strided<[272, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %arg3[%arg8, 0] [1, 32] [1, 1] : memref<16x32xf32, #hivm.address_space<ub>> to memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %arg4[%arg8] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_9 = memref.subview %subview_7[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> to memref<32xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_10 = ave.hir.vload <NORM> %subview_9[%c0] : memref<32xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %res_11 = ave.hir.vload <BRC_B32> %subview_8[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %15 = ave.hir.pge <ALL> : vector<64xi1>
    %16 = ave.hir.vsub %res_10, %res_11, %15 : vector<64xf32>, vector<64xi1>
    %17 = ave.hir.scalar_broadcast %arg5 : f32 -> vector<64xf32>
    %18 = ave.hir.pge <ALL> : vector<64xi1>
    %19 = ave.hir.vmul %10, %17, %18 : vector<64xf32>, vector<64xi1>
    %20 = ave.hir.pge <ALL> : vector<64xi1>
    %21 = ave.hir.vmul %19, %16, %20 : vector<64xf32>, vector<64xi1>
    %22 = ave.hir.pge <ALL> : vector<64xi1>
    %23 = ave.hir.vtruncf %21, <rint>, false, <part_even>, %22 : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %subview_12 = memref.subview %subview_6[0, 0, 0] [2, 1, 16] [1, 1, 1] : memref<2x1x16xf16, strided<[272, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<2x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 272 + d1 + s0)>, #hivm.address_space<ub>>
    %24 = builtin.unrealized_conversion_cast %2 : vector<64xi1> to vector<16xi1>
    ave.hir.store_with_stride %subview_12[%c0, %c0], %c272, %24, %23 : memref<2x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 272 + d1 + s0)>, #hivm.address_space<ub>>, vector<16xi1>, vector<64xf16>
  }
  return
}
