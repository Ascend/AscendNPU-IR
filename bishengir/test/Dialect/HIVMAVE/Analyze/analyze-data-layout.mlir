// RUN: bishengir-opt --data-layout-analyze %s | FileCheck %s

// Test that data-layout-analyze correctly sets element_alignment_bit_width = -1
// for a func that has nested scf.for loops with conflicting alignments.
//
// Background:
// When lowering HIVMAVEToAVEIntrin.cpp, the getElementAlignmentBitWidth function
// (lines 190-196) prioritizes the parent operation's alignment over the current
// operation's alignment. If a top-level operation (like scf.for) uses data in a
// for-loop with alignment 16, but the containing func has alignment 32, the
// lowering would incorrectly use 32 instead of 16.
// To prevent this mismatch, when the analysis detects that a func contains
// scf.for loops with different alignments (e.g., one with bf16=16 and another
// with f32=32), it marks the func's alignment as -1 so it inherits from children.

module {
  func.func @test_conflicting_alignment_in_for_loops(%arg0: memref<259x16xbf16, #hivm.address_space<ub>>, %arg1: memref<16x4xbf16, #hivm.address_space<ub>>, %arg2: memref<256x16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c16_i16 = arith.constant 16 : i16
    %c0_i16 = arith.constant 0 : i16
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.vci %c0_i16, <INCREASE> : i16, vector<128xi16>
    %3 = ave.hir.pge <ALL> : vector<128xi1>
    %4 = ave.hir.vmuls %2, %c16_i16, %3 : vector<128xi16>, i16, vector<128xi1>
    %5 = ave.hir.pge <ALL> : vector<128xi1>
    %6 = ave.hir.broadcast %c0_i16, %5 : i16, vector<128xi1> -> vector<128xi16>
    %7 = ave.hir.pge <VL4> : vector<128xi1>
    %8 = ave.hir.vsel %7, %4, %6 : vector<128xi1>, vector<128xi16>
    scf.for %arg3 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg0[31, %arg3] [4, 1] [1, 1] : memref<259x16xbf16, #hivm.address_space<ub>> to memref<4x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg3, 0] [1, 4] [1, 1] : memref<16x4xbf16, #hivm.address_space<ub>> to memref<1x4xbf16, strided<[4, 1], offset: ?>, #hivm.address_space<ub>>
      %9 = ave.hir.pge <VL4> {mask_op_idx = 0 : i32} : vector<128xi1>
      %10 = ave.hir.vgather %subview[%c0, %c0] [%8], %9 : memref<4x1xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xbf16>
      %subview_1 = memref.subview %subview_0[0, 0] [1, 4] [1, 1] : memref<1x4xbf16, strided<[4, 1], offset: ?>, #hivm.address_space<ub>> to memref<4xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B16> %subview_1[%c0], %9, %10 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<4xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
    }
    scf.for %arg3 = %c0 to %c256 step %c1 {
      %subview = memref.subview %arg2[%arg3, 0] [1, 16] [1, 1] : memref<256x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %9 = ave.hir.pge <VL16> {mask_op_idx = 1 : i32} : vector<64xi1>
      %subview_0 = memref.subview %subview[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B32> %subview_0[%c0], %9, %1 : memref<16xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
}

// CHECK: func.func @test_conflicting_alignment_in_for_loops
// CHECK-SAME: attributes {element_alignment_bit_width = -1 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function}