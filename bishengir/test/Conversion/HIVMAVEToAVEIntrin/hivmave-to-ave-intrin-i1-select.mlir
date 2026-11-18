// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -convert-hivmave-to-ave-intrin -cse %s -split-input-file | FileCheck %s

// Regression tests for the i1 (predicate) branch of `HIVMSelectOpLowering`.
//
// The hardware `vsel` instruction only accepts b16/b32 data registers, so a
// `ave.hir.vsel` whose element type is i1 operates on predicate registers and
// must be expanded into predicate logic instead:
//
//   vsel(mask, t, f) = por(pand(mask, t), pand(pnot(mask), f))
//
// Before this expansion existed the op survived to LLVM translation and the
// compile failed with "LLVM Translation failed for operation: ave.hir.vsel",
// which is what made 122 of the FlagGems `test_quantile.py` cases fail.

// -----

// A predicate select must lower to pnot + two pand + por, and must *not*
// leave a `vsel` intrinsic (that one cannot take predicate registers) nor a
// residual `ave.hir.vsel` behind.

// CHECK-LABEL: @vsel_i1_predicate
// CHECK-NOT: ave.hir.vsel
// CHECK-DAG: %[[ALLTRUE:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b8"
// CHECK: %[[NOT:.*]] = "hivm_regbaseintrins.intr.hivm.pnot.z"(%[[MASK:.*]], %[[ALLTRUE]])
// CHECK-DAG: %[[ANDT:.*]] = "hivm_regbaseintrins.intr.hivm.pand.z"(%[[MASK]], %{{.*}}, %[[ALLTRUE]])
// CHECK-DAG: %[[ANDF:.*]] = "hivm_regbaseintrins.intr.hivm.pand.z"(%[[NOT]], %{{.*}}, %[[ALLTRUE]])
// CHECK: "hivm_regbaseintrins.intr.hivm.por.z"(%[[ANDT]], %[[ANDF]], %[[ALLTRUE]])
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vsel"
// CHECK: return
func.func @vsel_i1_predicate(%dst: memref<512xi1, #hivm.address_space<ub>>,
                             %mask: vector<256xi1>,
                             %t: vector<256xi1>,
                             %f: vector<256xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %res = ave.hir.vsel %mask, %t, %f : vector<256xi1>, vector<256xi1>
  %store_mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %store_mask, %res {functionType = #ave.func_dist_type<pb8>} : memref<512xi1, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi1>
  return
}

// -----

// A partial predicate mask takes the same path: the expansion is per-lane
// boolean algebra, so it does not care how many lanes the mask enables.  Only
// the lane order matters, and `pand`/`por`/`pnot` are all lane-wise.

// CHECK-LABEL: @vsel_i1_partial_mask
// CHECK-NOT: ave.hir.vsel
// CHECK: "hivm_regbaseintrins.intr.hivm.pnot.z"
// CHECK-COUNT-2: "hivm_regbaseintrins.intr.hivm.pand.z"
// CHECK: "hivm_regbaseintrins.intr.hivm.por.z"
// CHECK: return
func.func @vsel_i1_partial_mask(%dst: memref<512xi1, #hivm.address_space<ub>>,
                                %t: vector<256xi1>,
                                %f: vector<256xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <H> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  %res = ave.hir.vsel %mask, %t, %f : vector<256xi1>, vector<256xi1>
  %store_mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<256xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %store_mask, %res {functionType = #ave.func_dist_type<pb8>} : memref<512xi1, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi1>
  return
}

// -----

// A data-typed select must keep using the hardware `vsel` instruction: the
// predicate expansion is only for i1 and must not steal the cheaper path.

// CHECK-LABEL: @vsel_i8_still_uses_vsel
// CHECK-NOT: ave.hir.vsel
// CHECK: "hivm_regbaseintrins.intr.hivm.vsel"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.por.z"
// CHECK: return
func.func @vsel_i8_still_uses_vsel(%dst: memref<512xi8, #hivm.address_space<ub>>,
                                   %mask: vector<256xi1>,
                                   %t: vector<256xi8>,
                                   %f: vector<256xi8>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %res = ave.hir.vsel %mask, %t, %f : vector<256xi1>, vector<256xi8>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %res {functionType = #ave.func_dist_type<pb8>} : memref<512xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}
