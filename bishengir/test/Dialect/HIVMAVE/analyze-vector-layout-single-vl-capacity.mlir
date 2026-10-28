// RUN: bishengir-opt %s -analyze-vector-layout -mlir-print-vector-layout-attr \
// RUN:   | FileCheck %s --implicit-check-not=b8_4vl \
// RUN:       --implicit-check-not=b16_2vl \
// RUN:       --implicit-check-not="func_dist_type<pk4>"

// A masked store is lowered by a single VSTS that consumes exactly one vector
// register, so the value it stores has to fit into one register (VL = 256 B =
// 2048 bit).  Both store rows below hold 128 elements:
//   * vector<128xi8>  -> B8_2VL is 128 x 2 B = 256 B (fits), B8_4VL would need
//     512 B and can not be lowered (the PK4 store would only write half a row),
//   * vector<128xf16> -> B16 is 128 x 2 B = 256 B (fits), B16_2VL would need
//     512 B.
// The layouts used to be selected anyway for two independent reasons:
//   1. the constraint solver only looked at (element bit width, layout) and
//      never at the vector length, so a layout overflowing the register was
//      accepted, and
//   2. conflict-directed backjumping dropped the branch that resolves the
//      conflict: the f16 value stored in the second loop is also the vcmp
//      operand of the first loop, and that compare fails at B8, so the search
//      used to settle on the infeasible (PK4, PK) combination instead of
//      backtracking to (PK, NORM).
//
// CHECK-LABEL: func.func @vector_layout_regression
// CHECK: ave.hir.vector.layout_cast {{.*}} {functionType = #ave.func_dist_type<dintlv2>}
// CHECK-SAME: vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
// CHECK-SAME: -> vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
// The i8 row stores B8_2VL data with a B16 predicate (PK packing).
// CHECK: ave.hir.masked_store <NORM_B8> {{.*}} {functionType = #ave.func_dist_type<pk>, hivm.is_continuous}
// CHECK-SAME: vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
// CHECK-SAME: vector<128xi8, #ave.vector_layout<{mem = #ave.vec_mem_type<b8_2vl>}>>
// The f16 row stores dense B16 data.
// CHECK: ave.hir.masked_store <NORM_B16> {{.*}} {functionType = #ave.func_dist_type<norm>, hivm.is_continuous}
// CHECK-SAME: vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
// CHECK-SAME: vector<128xf16, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>

module {
func.func @vector_layout_regression(%i8buf: memref<32x128xi8, #hivm.address_space<ub>>, %f16buf: memref<32x128xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %ci0 = arith.constant 0 : i8
  %cf0 = arith.constant 0.000000e+00 : f16
  %p1 = ave.hir.pge <ALL> : vector<1xi1>
  %f1 = ave.hir.broadcast %cf0, %p1 : f16, vector<1xi1> -> vector<1xf16>
  %pst = ave.hir.pge <ALL> : vector<128xi1>
  %f16v = ave.hir.broadcast %cf0, %pst : f16, vector<128xi1> -> vector<128xf16>
  scf.for %iv = %c0 to %c32 step %c1 {
    %pm = ave.hir.pge <ALL> : vector<256xi1>
    %lc1 = ave.hir.vector.layout_cast %pm : vector<256xi1> -> vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
    %lc2 = ave.hir.vector.layout_cast %lc1 : vector<256xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>> -> vector<256xi1>
    %narrow = builtin.unrealized_conversion_cast %lc2 : vector<256xi1> to vector<1xi1>
    %fv = ave.hir.vsel %narrow, %f1, %f1 : vector<1xi1>, vector<1xf16>
    %fwide = builtin.unrealized_conversion_cast %fv : vector<1xf16> to vector<128xf16>
    %pb = ave.hir.pge <ALL> : vector<128xi1>
    %fcmp = ave.hir.vcmp <NE> %fwide, %f16v, %pb : vector<128xf16>, vector<128xi1> -> vector<128xi1>
    %and = ave.hir.preg.and <b8> %fcmp, %pb, %pb : vector<128xi1>
    %i8z = ave.hir.broadcast %ci0, %pb : i8, vector<128xi1> -> vector<128xi8>
    %stv = ave.hir.vsel %and, %i8z, %i8z : vector<128xi1>, vector<128xi8>
    %sub1 = memref.subview %i8buf[%iv, 0] [1, 128] [1, 1] : memref<32x128xi8, #hivm.address_space<ub>> to memref<1x128xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %sub2 = memref.subview %sub1[0, 0] [1, 128] [1, 1] : memref<1x128xi8, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<128xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B8> %sub2[%c0], %and, %stv {hivm.is_continuous} : memref<128xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi8>
  }
  scf.for %iv2 = %c0 to %c32 step %c1 {
    %subf1 = memref.subview %f16buf[%iv2, 0] [1, 128] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<1x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subf2 = memref.subview %subf1[0, 0] [1, 128] [1, 1] : memref<1x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<128xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B16> %subf2[%c0], %pst, %f16v {hivm.is_continuous} : memref<128xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
  }
  return
}
}
