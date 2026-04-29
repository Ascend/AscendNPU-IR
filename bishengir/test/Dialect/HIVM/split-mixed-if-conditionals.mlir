// RUN: bishengir-opt %s -hivm-split-mixed-if-conditionals -split-input-file -verify-diagnostics | FileCheck %s

// =============================================================================
// Group A — Universal short-circuits & no-op cases
// =============================================================================

// scf.if body has only non-core ops (tensor.empty, constants). Pattern bails
// at the `!hasC && !hasV` early-exit; IR is left untouched.
// CHECK-LABEL: @noop_no_core_ops
// CHECK:         scf.if %{{.*}} -> (tensor<32xf32>) {
// CHECK-NEXT:      tensor.empty() : tensor<32xf32>
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NOT:     hivm.cube_only
// CHECK-NOT:     hivm.vec_only
// CHECK-NOT:     hivm.branch_split_done
// CHECK-NOT:     hivm.core_split_done
func.func @noop_no_core_ops(%cond: i1, %a: tensor<32xf32>) -> tensor<32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %0 = scf.if %cond -> tensor<32xf32> {
    %t = tensor.empty() : tensor<32xf32>
    scf.yield %t : tensor<32xf32>
  } else {
    scf.yield %a : tensor<32xf32>
  }
  return %0 : tensor<32xf32>
}

// -----

// Pre-existing `hivm.cube_only` triggers the universal short-circuit; pass
// is a no-op even though the body would otherwise be eligible.
// CHECK-LABEL: @noop_already_cube_only
// CHECK:         scf.if {{.*}} -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {hivm.cube_only}
// CHECK-NOT:     hivm.branch_split_done
// CHECK-NOT:     hivm.core_split_done
// CHECK-NOT:     arith.select
func.func @noop_already_cube_only(%cond: i1, %a: tensor<32x32xf16>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %ce = tensor.empty() : tensor<32x32xf32>
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %mm : tensor<32x32xf32>
  } else {
    scf.yield %ce : tensor<32x32xf32>
  } {hivm.cube_only}
  return %0 : tensor<32x32xf32>
}

// -----

// Pre-existing both `branch_split_done` AND `core_split_done` triggers the
// universal short-circuit; pass is a no-op even on a mixed-core body.
// CHECK-LABEL: @noop_already_split_done
// CHECK:         scf.if {{.*}} -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:           hivm.hir.vadd
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done}
// CHECK-NOT:     hivm.cube_only
// CHECK-NOT:     hivm.vec_only
// CHECK-NOT:     arith.select
func.func @noop_already_split_done(%cond: i1, %a: tensor<32x32xf16>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %ce = tensor.empty() : tensor<32x32xf32>
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%mm, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  } else {
    scf.yield %ce : tensor<32x32xf32>
  } {hivm.branch_split_done, hivm.core_split_done}
  return %0 : tensor<32x32xf32>
}

// =============================================================================
// Group B — Stage C only (uniform-core attribute attachment)
// =============================================================================

// Single-branch (no-else) scf.if with only CUBE-typed core ops. Stage C
// attaches `hivm.cube_only`; nothing else changes.
// CHECK-LABEL: @stage_c_uniform_cube_no_else
// CHECK:         scf.if %{{.*}} {
// CHECK:           hivm.hir.mmadL1
// CHECK:           "test.consume"
// CHECK:         } {hivm.cube_only}
// CHECK-NOT:     hivm.vec_only
// CHECK-NOT:     hivm.branch_split_done
// CHECK-NOT:     hivm.core_split_done
// CHECK-NOT:     arith.select
func.func @stage_c_uniform_cube_no_else(%cond: i1, %a: tensor<32x32xf16>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %ce = tensor.empty() : tensor<32x32xf32>
  scf.if %cond {
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    "test.consume"(%mm) : (tensor<32x32xf32>) -> ()
  }
  return
}

// -----

// Both branches contain only VECTOR-typed core ops. Stage C attaches
// `hivm.vec_only`.
// CHECK-LABEL: @stage_c_uniform_vector_with_else
// CHECK:         scf.if %{{.*}} -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.vadd
// CHECK:         } else {
// CHECK:           hivm.hir.vadd
// CHECK:         } {hivm.vec_only}
// CHECK-NOT:     hivm.cube_only
// CHECK-NOT:     hivm.branch_split_done
// CHECK-NOT:     arith.select
func.func @stage_c_uniform_vector_with_else(%cond: i1, %a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  } else {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%b, %a : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  }
  return %0 : tensor<32x32xf32>
}

// =============================================================================
// Group C — Stage A only (each branch already uniform after split)
// =============================================================================

// then = pure CUBE, else = pure VECTOR. Stage A clones both sides; Stage C
// then tags thenIf with `cube_only`, elseIf with `vec_only`. arith.select
// merges per-result.
// CHECK-LABEL: @stage_a_cube_then_vec_else
// CHECK:         %[[CUBE:.*]] = scf.if %[[COND:.*]] -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.cube_only}
// CHECK:         %[[VEC:.*]] = scf.if %[[COND]] -> (tensor<32x32xf32>) {
// CHECK:           tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.vadd
// CHECK:         } {hivm.branch_split_done, hivm.vec_only}
// CHECK:         %[[SEL:.*]] = arith.select %[[COND]], %[[CUBE]], %[[VEC]]
// CHECK:         return %[[SEL]]
func.func @stage_a_cube_then_vec_else(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %mm : tensor<32x32xf32>
  } else {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  }
  return %0 : tensor<32x32xf32>
}

// -----

// then = pure VECTOR, else = pure CUBE. Mirror of the previous test.
// CHECK-LABEL: @stage_a_vec_then_cube_else
// CHECK:         %[[VEC:.*]] = scf.if %[[COND:.*]] -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.vadd
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.vec_only}
// CHECK:         %[[CUBE:.*]] = scf.if %[[COND]] -> (tensor<32x32xf32>) {
// CHECK:           tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {hivm.branch_split_done, hivm.cube_only}
// CHECK:         %[[SEL:.*]] = arith.select %[[COND]], %[[VEC]], %[[CUBE]]
// CHECK:         return %[[SEL]]
func.func @stage_a_vec_then_cube_else(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  } else {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %mm : tensor<32x32xf32>
  }
  return %0 : tensor<32x32xf32>
}

// =============================================================================
// Group D — Stage B only (no-else mixed; chain of per-core ifs, no select)
// =============================================================================

// no-else scf.if with mixed CUBE+VECTOR ops, no results. Stage A skipped
// (no else); Stage B partitions into per-core ifs sharing the original cond.
// The `test.consume` calls land in the remainder WorkItem (no specific core),
// emitted as a third per-core if without a uniform-core marker.
// CHECK-LABEL: @stage_b_no_else_mixed
// CHECK:         %[[CUBE:.*]] = scf.if %[[COND:.*]] -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.cube_only}
// CHECK:         %[[VEC:.*]] = scf.if %[[COND]] -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.vadd
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.vec_only}
// CHECK:         scf.if %[[COND]] {
// CHECK:           "test.consume"(%[[CUBE]])
// CHECK:           "test.consume"(%[[VEC]])
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done}
// CHECK-NOT:     arith.select
func.func @stage_b_no_else_mixed(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  scf.if %cond {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    "test.consume"(%mm) : (tensor<32x32xf32>) -> ()
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    "test.consume"(%va) : (tensor<32x32xf32>) -> ()
  }
  return
}

// =============================================================================
// Group E — Stage A + Stage B (mixed branches, decomposed to per-core chains)
// =============================================================================

// then = mixed CUBE+VECTOR (the cube result feeds a vec op), else = pure CUBE.
// Stage A → 2 clones; thenIf live=mixed triggers Stage B (cube WI → vec WI);
// elseIf is uniform-CUBE so Stage C tags it.
// CHECK-LABEL: @stage_ab_mix_then_cube_else
// CHECK:         %[[T_CUBE:.*]]:2 = scf.if %[[COND:.*]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.cube_only}
// CHECK:         %[[T_VEC:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           hivm.hir.vadd ins(%[[T_CUBE]]#1
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.vec_only}
// CHECK:         %[[E_CUBE:.*]] = scf.if %[[COND]] -> (tensor<32x32xf32>) {
// CHECK:           tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {hivm.branch_split_done, hivm.cube_only}
// CHECK:         %[[SEL:.*]] = arith.select %[[COND]], %[[T_VEC]]#1, %[[E_CUBE]]
// CHECK:         return %[[SEL]]
func.func @stage_ab_mix_then_cube_else(%cond: i1, %a: tensor<32x32xf16>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%mm, %bias : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  } else {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %mm : tensor<32x32xf32>
  }
  return %0 : tensor<32x32xf32>
}

// -----

// Both branches mixed: Stage A clones, then Stage B fires on each clone
// (one with then=live mixed, one with else=live mixed). Final shape is
// 4 per-core scf.ifs + arith.select.
// CHECK-LABEL: @stage_ab_mix_mix_branches
// CHECK:         %[[T_CUBE:.*]]:2 = scf.if %[[COND:.*]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.cube_only}
// CHECK:         %[[T_VEC:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           hivm.hir.vadd ins(%[[T_CUBE]]#1
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.vec_only}
// CHECK:         %[[E_CUBE:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.cube_only}
// CHECK:         %[[E_VEC:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.vadd ins(%[[E_CUBE]]#1
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.vec_only}
// CHECK:         %[[SEL:.*]] = arith.select %[[COND]], %[[T_VEC]]#1, %[[E_VEC]]#1
// CHECK:         return %[[SEL]]
func.func @stage_ab_mix_mix_branches(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%mm, %v1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  } else {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%mm, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  }
  return %0 : tensor<32x32xf32>
}

// -----

// Cross-WorkItem dependency in Stage B: a single-branch (no-else) mixed if
// where the vector op consumes the cube op's result. Stage B emits a CUBE
// per-core if followed by a VECTOR per-core if; the vec one references the
// cube one's result through globalMap.
// CHECK-LABEL: @stage_b_cross_wi_dep
// CHECK:         %[[CUBE:.*]]:2 = scf.if %[[COND:.*]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.cube_only}
// CHECK:         %[[VEC:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           hivm.hir.vadd ins(%[[CUBE]]#1
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.vec_only}
// CHECK:         scf.if %[[COND]] {
// CHECK:           "test.consume"(%[[VEC]]#1)
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done}
// CHECK-NOT:     arith.select
func.func @stage_b_cross_wi_dep(%cond: i1, %a: tensor<32x32xf16>, %bias: tensor<32x32xf32>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  scf.if %cond {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%mm, %bias : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    "test.consume"(%va) : (tensor<32x32xf32>) -> ()
  }
  return
}

// Regression: when Stage A produces an elseIf (live=else, dummy=then) that is
// still mixed-core, Stage B re-fires with `liveIsThen=false`, so the dummy
// region is `thenRegion` (a SizedRegion<1>). If a WorkItem in the live else
// block has no externally-escaping outputs (e.g., a non-core WorkItem whose
// only op is a side-effect terminal like `test.consume`), the per-core
// scf.if has no results — and an earlier version skipped creating any block
// for the dummy thenRegion in that case, hitting the verifier error
// "'scf.if' op region #0 ('thenRegion') failed to verify constraint: region
// with 1 blocks". The fix always creates an empty dummy block (with bare
// `scf.yield`) so the structural invariant holds even with empty outputs.
// The trailing `scf.if %[[COND]] { } else { "test.consume" ... }` below is
// the load-bearing assertion: an empty dummy thenRegion on a no-result if.
// CHECK-LABEL: @stage_b_else_live_empty_vec_outputs
// CHECK:         %[[CE:.*]] = tensor.empty
// CHECK:         %[[T_VEC:.*]] = scf.if %[[COND:.*]] -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.vadd
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.vec_only}
// CHECK:         %[[E_CUBE:.*]] = scf.if %[[COND]] -> (tensor<32x32xf32>) {
// CHECK-NEXT:      tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.cube_only}
// CHECK:         %[[E_VEC:.*]] = scf.if %[[COND]] -> (tensor<32x32xf32>) {
// CHECK-NEXT:      tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.vadd ins(%[[E_CUBE]]
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.vec_only}
// CHECK:         scf.if %[[COND]] {
// CHECK-NEXT:    } else {
// CHECK:           "test.consume"(%[[E_VEC]])
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done}
// CHECK:         %[[SEL:.*]] = arith.select %[[COND]], %[[T_VEC]], %[[CE]]
// CHECK:         return %[[SEL]]
func.func @stage_b_else_live_empty_vec_outputs(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %ce = tensor.empty() : tensor<32x32xf32>
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %va = hivm.hir.vadd ins(%v1, %v1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  } else {
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    %fake = hivm.hir.vadd ins(%mm, %v1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    "test.consume"(%fake) : (tensor<32x32xf32>) -> ()
    scf.yield %ce : tensor<32x32xf32>
  }
  return %0 : tensor<32x32xf32>
}

// -----

// =============================================================================
// Group F — Yield-type variants
// =============================================================================

// scf.if yields multiple tensors; arith.select must wire each result
// independently. then=cube, else=vec, two yields.
// CHECK-LABEL: @multi_value_yield
// CHECK:         %[[CUBE:.*]]:2 = scf.if %[[COND:.*]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.cube_only}
// CHECK:         %[[VEC:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.vadd
// CHECK:         } {hivm.branch_split_done, hivm.vec_only}
// CHECK:         %[[SEL0:.*]] = arith.select %[[COND]], %[[CUBE]]#0, %[[VEC]]#0
// CHECK:         %[[SEL1:.*]] = arith.select %[[COND]], %[[CUBE]]#1, %[[VEC]]#1
// CHECK:         return %[[SEL0]], %[[SEL1]]
func.func @multi_value_yield(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0:2 = scf.if %cond -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %mm, %ce : tensor<32x32xf32>, tensor<32x32xf32>
  } else {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va, %ve : tensor<32x32xf32>, tensor<32x32xf32>
  }
  return %0#0, %0#1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----

// scf.if yields a scalar (i32) alongside a tensor. Per the dummy-policy spec,
// non-tensor inactive yields are not fabricated as zero; instead, the producer
// chain of the *adjacent* original branch's matching yield is cloned into the
// dummy block. Here both scalar values (%c7, %c11) are defined outside the
// original if, so cloneProducerInto returns them directly without re-cloning.
// CHECK-LABEL: @scalar_and_tensor_yield
// CHECK:         %[[K7:.*]] = arith.constant 7 : i32
// CHECK:         %[[K11:.*]] = arith.constant 11 : i32
// CHECK:         %[[CUBE:.*]]:2 = scf.if %[[COND:.*]] -> (tensor<32x32xf32>, i32) {
// CHECK:           hivm.hir.mmadL1
// CHECK:           scf.yield %{{.*}}, %[[K7]] : tensor<32x32xf32>, i32
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:           scf.yield %{{.*}}, %[[K11]] : tensor<32x32xf32>, i32
// CHECK:         } {hivm.branch_split_done, hivm.cube_only}
// CHECK:         %[[VEC:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, i32) {
// CHECK:           tensor.empty
// CHECK:           scf.yield %{{.*}}, %[[K7]] : tensor<32x32xf32>, i32
// CHECK:         } else {
// CHECK:           hivm.hir.vadd
// CHECK:           scf.yield %{{.*}}, %[[K11]] : tensor<32x32xf32>, i32
// CHECK:         } {hivm.branch_split_done, hivm.vec_only}
// CHECK:         %[[SEL_T:.*]] = arith.select %[[COND]], %[[CUBE]]#0, %[[VEC]]#0 : tensor<32x32xf32>
// CHECK:         %[[SEL_I:.*]] = arith.select %[[COND]], %[[CUBE]]#1, %[[VEC]]#1 : i32
// CHECK:         return %[[SEL_T]], %[[SEL_I]]
func.func @scalar_and_tensor_yield(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> (tensor<32x32xf32>, i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %k_then = arith.constant 7 : i32
  %k_else = arith.constant 11 : i32
  %0:2 = scf.if %cond -> (tensor<32x32xf32>, i32) {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %mm, %k_then : tensor<32x32xf32>, i32
  } else {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va, %k_else : tensor<32x32xf32>, i32
  }
  return %0#0, %0#1 : tensor<32x32xf32>, i32
}

// =============================================================================
// Group G — Recursion / structural cases
// =============================================================================

// Outer mixed scf.if contains an inner mixed scf.if; the greedy driver
// recurses until both are decomposed. Each level produces its own per-core
// chain, with cube/vec markers and arith.select wiring at every level.
// CHECK-LABEL: @nested_mixed_ifs
// CHECK:         %[[OUT_CUBE:.*]] = scf.if %[[C1:.*]] -> (tensor<32x32xf32>) {
// CHECK:           scf.if %[[C2:.*]] -> (tensor<32x32xf32>) {
// CHECK:             hivm.hir.mmadL1
// CHECK:           } else {
// CHECK:             tensor.empty
// CHECK:           } {hivm.branch_split_done, hivm.cube_only
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.cube_only}
// CHECK:         %[[OUT_VEC:.*]] = scf.if %[[C1]] -> (tensor<32x32xf32>) {
// CHECK:           scf.if %[[C2]] -> (tensor<32x32xf32>) {
// CHECK:             tensor.empty
// CHECK:           } else {
// CHECK:             hivm.hir.vadd
// CHECK:           } {hivm.branch_split_done, hivm.vec_only
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.vec_only}
// CHECK:         arith.select %[[COND:.*]], %[[OUT_CUBE]], %[[OUT_VEC]]
func.func @nested_mixed_ifs(%cond1: i1, %cond2: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0 = scf.if %cond1 -> tensor<32x32xf32> {
    %inner = scf.if %cond2 -> tensor<32x32xf32> {
      %ce = tensor.empty() : tensor<32x32xf32>
      %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond2, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm : tensor<32x32xf32>
    } else {
      %ve = tensor.empty() : tensor<32x32xf32>
      %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %va : tensor<32x32xf32>
    }
    scf.yield %inner : tensor<32x32xf32>
  } else {
    %ce = tensor.empty() : tensor<32x32xf32>
    scf.yield %ce : tensor<32x32xf32>
  }
  return %0 : tensor<32x32xf32>
}

// -----

// scf.for body contains a mixed scf.if — the if must split, the for-loop
// must remain intact (the pass operates on scf.if only).
// CHECK-LABEL: @if_inside_for
// CHECK:         scf.for {{.*}} iter_args(%{{.*}} = %{{.*}})
// CHECK:           %[[CUBE:.*]] = scf.if %[[COND:.*]] -> (tensor<32x32xf32>) {
// CHECK:             hivm.hir.mmadL1
// CHECK:           } else {
// CHECK:             tensor.empty
// CHECK:           } {hivm.branch_split_done, hivm.cube_only}
// CHECK:           %[[VEC:.*]] = scf.if %[[COND]] -> (tensor<32x32xf32>) {
// CHECK:             tensor.empty
// CHECK:           } else {
// CHECK:             hivm.hir.vadd
// CHECK:           } {hivm.branch_split_done, hivm.vec_only}
// CHECK:           %[[SEL:.*]] = arith.select %[[COND]], %[[CUBE]], %[[VEC]]
// CHECK:           scf.yield %[[SEL]]
// CHECK:         }
func.func @if_inside_for(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %init = tensor.empty() : tensor<32x32xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<32x32xf32> {
    %0 = scf.if %cond -> tensor<32x32xf32> {
      %ce = tensor.empty() : tensor<32x32xf32>
      %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %mm : tensor<32x32xf32>
    } else {
      %ve = tensor.empty() : tensor<32x32xf32>
      %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %va : tensor<32x32xf32>
    }
    scf.yield %0 : tensor<32x32xf32>
  }
  return %r : tensor<32x32xf32>
}

// -----

// Two sibling scf.ifs in the same block; each evaluated independently.
// First is mixed (split into Stage A); second is uniform-VEC (Stage C only).
// CHECK-LABEL: @sibling_ifs
// CHECK:         %[[I1_CUBE:.*]] = scf.if %[[C1:.*]] -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:         } else {
// CHECK:           tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.cube_only}
// CHECK:         %[[I1_VEC:.*]] = scf.if %[[C1]] -> (tensor<32x32xf32>) {
// CHECK:           tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.vadd
// CHECK:         } {hivm.branch_split_done, hivm.vec_only}
// CHECK:         %[[SEL:.*]] = arith.select %[[C1]], %[[I1_CUBE]], %[[I1_VEC]]
// CHECK:         %[[I2:.*]] = scf.if %[[C2:.*]] -> (tensor<32x32xf32>) {
// CHECK:           hivm.hir.vadd
// CHECK:         } else {
// CHECK:           hivm.hir.vadd
// CHECK:         } {hivm.vec_only}
// CHECK:         return %[[SEL]], %[[I2]]
func.func @sibling_ifs(%c1: i1, %c2: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %r1 = scf.if %c1 -> tensor<32x32xf32> {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %c1, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %mm : tensor<32x32xf32>
  } else {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  }
  %r2 = scf.if %c2 -> tensor<32x32xf32> {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  } else {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v2, %v1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  }
  return %r1, %r2 : tensor<32x32xf32>, tensor<32x32xf32>
}

// =============================================================================
// Group H — Realistic / corner cases
// =============================================================================

// A non-core helper (tensor.empty) is used by both a CUBE and a VECTOR op in
// the live branch. WorklistBuilder's traceDependentOps clones the helper into
// each consuming WorkItem so each per-core if has its own copy.
// CHECK-LABEL: @shared_noncore_dep
// CHECK:         %[[CUBE:.*]]:2 = scf.if %[[COND:.*]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           %[[SHARED_C:.*]] = tensor.empty
// CHECK:           hivm.hir.mmadL1 {{.*}} outs(%[[SHARED_C]]
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.cube_only}
// CHECK:         %[[VEC:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           %[[SHARED_V:.*]] = tensor.empty
// CHECK:           hivm.hir.vadd ins(%[[CUBE]]#1, %{{.*}}) outs(%[[SHARED_V]]
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.core_split_done, hivm.vec_only}
// CHECK:         arith.select
func.func @shared_noncore_dep(%cond: i1, %a: tensor<32x32xf16>, %bias: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0 = scf.if %cond -> tensor<32x32xf32> {
    %shared = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%shared : tensor<32x32xf32>) -> tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%mm, %bias : tensor<32x32xf32>, tensor<32x32xf32>) outs(%shared : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va : tensor<32x32xf32>
  } else {
    %ce = tensor.empty() : tensor<32x32xf32>
    scf.yield %ce : tensor<32x32xf32>
  }
  return %0 : tensor<32x32xf32>
}

// -----

// One yield is a value defined outside the if (passthrough). No WorkItem
// produces it; the live yield value is not in any localOutputs, so
// finalReplacements falls back to the original SSA value via
// IRMapping::lookupOrDefault.
// CHECK-LABEL: @passthrough_yield
// CHECK-SAME:    %[[COND:[a-zA-Z0-9_]+]]: i1, %{{.*}}: tensor<32x32xf16>, %[[EXT:[a-zA-Z0-9_]+]]: tensor<32x32xf32>
// CHECK:         %[[CUBE:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:           scf.yield %{{.*}}, %[[EXT]]
// CHECK:         } else {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } {hivm.branch_split_done, hivm.cube_only}
// CHECK:         %[[VEC:.*]]:2 = scf.if %[[COND]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
// CHECK-COUNT-2:   tensor.empty
// CHECK:         } else {
// CHECK:           hivm.hir.vadd
// CHECK:           scf.yield %{{.*}}, %[[EXT]]
// CHECK:         } {hivm.branch_split_done, hivm.vec_only}
// CHECK:         arith.select %[[COND]], %[[CUBE]]#0, %[[VEC]]#0
// CHECK:         arith.select %[[COND]], %[[CUBE]]#1, %[[VEC]]#1
func.func @passthrough_yield(%cond: i1, %a: tensor<32x32xf16>, %ext: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0:2 = scf.if %cond -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %mm, %ext : tensor<32x32xf32>, tensor<32x32xf32>
  } else {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%ext, %ext : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %va, %ext : tensor<32x32xf32>, tensor<32x32xf32>
  }
  return %0#0, %0#1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----

// scf.if yields a memref. arith.select cannot select memrefs, so the Stage A
// combiner takes the then-side; the else-side memref alloc becomes dead.
// Stage A is the only stage that fires (each branch is uniform after split).
// CHECK-LABEL: @memref_yield_take_then_side
// CHECK:         %[[CUBE:.*]] = scf.if %[[COND:.*]] -> (memref<32x32xf32>) {
// CHECK:           hivm.hir.mmadL1
// CHECK:           memref.alloc
// CHECK:           hivm.hir.fixpipe
// CHECK:         } else {
// CHECK:           memref.alloc
// CHECK:         } {hivm.branch_split_done, hivm.cube_only}
// CHECK:         %{{.*}} = scf.if %[[COND]] -> (memref<32x32xf32>) {
// CHECK:           memref.alloc
// CHECK:         } else {
// CHECK:           hivm.hir.vadd
// CHECK:           memref.alloc
// CHECK:           bufferization.materialize_in_destination
// CHECK:         } {hivm.branch_split_done, hivm.vec_only}
// CHECK-NOT:     arith.select
// CHECK:         return %[[CUBE]]
func.func @memref_yield_take_then_side(%cond: i1, %a: tensor<32x32xf16>, %v1: tensor<32x32xf32>, %v2: tensor<32x32xf32>) -> memref<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c32 = arith.constant 32 : index
  %0 = scf.if %cond -> memref<32x32xf32> {
    %ce = tensor.empty() : tensor<32x32xf32>
    %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%a, %a, %cond, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%ce : tensor<32x32xf32>) -> tensor<32x32xf32>
    %buf = memref.alloc() : memref<32x32xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mm : tensor<32x32xf32>) outs(%buf : memref<32x32xf32>)
    scf.yield %buf : memref<32x32xf32>
  } else {
    %ve = tensor.empty() : tensor<32x32xf32>
    %va = hivm.hir.vadd ins(%v1, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ve : tensor<32x32xf32>) -> tensor<32x32xf32>
    %buf = memref.alloc() : memref<32x32xf32>
    bufferization.materialize_in_destination %va in writable %buf : (tensor<32x32xf32>, memref<32x32xf32>) -> ()
    scf.yield %buf : memref<32x32xf32>
  }
  return %0 : memref<32x32xf32>
}
