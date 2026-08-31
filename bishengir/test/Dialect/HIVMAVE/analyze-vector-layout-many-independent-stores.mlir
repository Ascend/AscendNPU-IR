// RUN: bishengir-opt %s --analyze-vector-layout | FileCheck %s
// RUN: bishengir-opt %s --analyze-vector-layout --debug-only=analyze-vector-layout -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

// CHECK-LABEL: func.func @many_independent_i1_stores
// CHECK-COUNT-15: ave.hir.masked_store
// DEBUG: [solveIt]: Start vector layout search:
// DEBUG: [advanceWithCandidates]: Branch at ave.hir.masked_store:
// DEBUG: [recordFailure]: Reject path at ave.hir.vcmp:
// DEBUG: [tryConflictBackjump]: Conflict-directed backjump from ave.hir.vcmp to ave.hir.masked_store:
// Keep all f16 compares before the stores. The reverse solver sees 15
// three-way i1 store branches before it can reject B8 at the compare producers.
func.func @many_independent_i1_stores(
    %arg0: memref<64xi1, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %lhs = ave.hir.scalar_broadcast %cst : f16 -> vector<64xf16>
  %cmpMask0 = ave.hir.pge <ALL> : vector<64xi1>
  %value0 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask0
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask1 = ave.hir.pge <ALL> : vector<64xi1>
  %value1 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask1
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask2 = ave.hir.pge <ALL> : vector<64xi1>
  %value2 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask2
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask3 = ave.hir.pge <ALL> : vector<64xi1>
  %value3 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask3
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask4 = ave.hir.pge <ALL> : vector<64xi1>
  %value4 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask4
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask5 = ave.hir.pge <ALL> : vector<64xi1>
  %value5 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask5
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask6 = ave.hir.pge <ALL> : vector<64xi1>
  %value6 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask6
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask7 = ave.hir.pge <ALL> : vector<64xi1>
  %value7 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask7
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask8 = ave.hir.pge <ALL> : vector<64xi1>
  %value8 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask8
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask9 = ave.hir.pge <ALL> : vector<64xi1>
  %value9 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask9
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask10 = ave.hir.pge <ALL> : vector<64xi1>
  %value10 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask10
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask11 = ave.hir.pge <ALL> : vector<64xi1>
  %value11 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask11
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask12 = ave.hir.pge <ALL> : vector<64xi1>
  %value12 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask12
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask13 = ave.hir.pge <ALL> : vector<64xi1>
  %value13 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask13
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %cmpMask14 = ave.hir.pge <ALL> : vector<64xi1>
  %value14 = ave.hir.vcmp <NE> %lhs, %lhs, %cmpMask14
      : vector<64xf16>, vector<64xi1> -> vector<64xi1>
  %mask0 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask0, %value0
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask1 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask1, %value1
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask2 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask2, %value2
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask3 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask3, %value3
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask4 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask4, %value4
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask5 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask5, %value5
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask6 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask6, %value6
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask7 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask7, %value7
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask8 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask8, %value8
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask9 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask9, %value9
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask10 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask10, %value10
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask11 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask11, %value11
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask12 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask12, %value12
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask13 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask13, %value13
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %mask14 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask14, %value14
      : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
