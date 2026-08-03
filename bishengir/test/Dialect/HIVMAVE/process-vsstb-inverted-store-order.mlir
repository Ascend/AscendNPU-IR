// RUN: bishengir-opt -ave-process-vsstb %s -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// Regression test: the store order in the block can be inverted relative to
// the trunc/load order. Here truncB's store comes first, even though
// truncA/resA are physically earlier in the block.
//
// This makes oldLoad1 (found from the first store's trunc, truncB) be resB,
// which is defined *after* truncA already consumes resA. No single point
// works for a merged load then: it would need to sit after resB but before
// truncA. The pass must detect this and fall back to the safe vdintlv
// (VFDeInterleaveOp) path instead of merging the loads, rather than emitting
// a dominance violation (the original bug).
// CHECK-LABEL: func.func @inverted_store_order
// CHECK: ave.hir.vdintlv
// CHECK: ave.hir.store_with_stride
func.func @inverted_store_order(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<8x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c1040 = arith.constant 1040 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    %resA = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %pA = ave.hir.pge <ALL> : vector<64xi1>
    %truncA = ave.hir.vtruncf %resA, <rint>, false, <part_even>, %pA : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %resB = ave.hir.vload <NORM> %arg1[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %pB = ave.hir.pge <ALL> : vector<64xi1>
    %truncB = ave.hir.vtruncf %resB, <rint>, false, <part_even>, %pB : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %subviewB = memref.subview %arg2[4, 0] [4, 16] [1, 1] : memref<8x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>> to memref<4x16xf16, strided<[1040, 1], offset: 4160>, #hivm.address_space<ub>>
    %pStoreB = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.store_with_stride %subviewB[%c0, %c0], %c1040, %pStoreB, %truncB : memref<4x16xf16, strided<[1040, 1], offset: 4160>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    %subviewA = memref.subview %arg2[0, 0] [4, 16] [1, 1] : memref<8x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>> to memref<4x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>>
    %pStoreA = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.store_with_stride %subviewA[%c0, %c0], %c1040, %pStoreA, %truncA : memref<4x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  }
  return
}

// Same inverted store order as above, but both loads are issued before
// either trunc, so oldLoad1 (resB) *is* before resA's user (truncA) --
// the load-merge safety window is satisfiable here. This exercises the
// trunc/mask/or/store anchor fix specifically (through the merged-load path)
// without hitting the safety bail-out covered by the case above.
// CHECK-LABEL: func.func @inverted_store_order_safe_merge
// CHECK: ave.hir.vload <DINTLV_B32>
// CHECK: ave.hir.vtruncf
// CHECK-NEXT: ave.hir.vtruncf
// CHECK: ave.hir.vor
// CHECK: ave.hir.store_with_stride
func.func @inverted_store_order_safe_merge(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<8x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c1040 = arith.constant 1040 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    %resA = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %pA = ave.hir.pge <ALL> : vector<64xi1>
    %resB = ave.hir.vload <NORM> %arg1[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %pB = ave.hir.pge <ALL> : vector<64xi1>
    %truncA = ave.hir.vtruncf %resA, <rint>, false, <part_even>, %pA : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %truncB = ave.hir.vtruncf %resB, <rint>, false, <part_even>, %pB : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %subviewB = memref.subview %arg2[4, 0] [4, 16] [1, 1] : memref<8x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>> to memref<4x16xf16, strided<[1040, 1], offset: 4160>, #hivm.address_space<ub>>
    %pStoreB = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.store_with_stride %subviewB[%c0, %c0], %c1040, %pStoreB, %truncB : memref<4x16xf16, strided<[1040, 1], offset: 4160>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
    %subviewA = memref.subview %arg2[0, 0] [4, 16] [1, 1] : memref<8x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>> to memref<4x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>>
    %pStoreA = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.store_with_stride %subviewA[%c0, %c0], %c1040, %pStoreA, %truncA : memref<4x16xf16, strided<[1040, 1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  }
  return
}
