// RUN: bishengir-opt %s -transform-interpreter -split-input-file 2>&1 | FileCheck %s
//
// Test transform.loop.fuse_nested_siblings: after outer LoopFuseSiblingOp,
// inner sibling loops are collected, grouped, and (with `recursive = true`)
// recursively fused.

// Test 1: bounds_normalization + n_way_fusion + recursive_fusion
//
// 20 inner loops over 7 iters groups, 2-level nesting for recursive fusion.
// Groups (iters -> members -> tags):
//   G1(8):  X,Y with XA,XB,YA,YB -> {X,Y} -> {XA,XB,YA,YB}  (recursive)
//   G2(10): S,R,M,N -> {M,N,R,S}  (gcd=2, mixed +/- steps)
//   G3(12): V,W -> {V,W}  (V affine.apply, W identity skip)
//   G4(12): Z -> {Z}  (sibling singleton, Z consumes V)
//   G5(fail): BAD -> {BAD}  (cnt*st<0, skipped)
//   G6(fail): DYNA,DYNB -> {DYNA},{DYNB}  (non-constant bounds, skipped)
//   G7(3):  Q -> {Q}  (iters singleton)
//   G8(tail): P1,P2,P3 -> {P1,P2,P3}  (identical tail-block bounds 0..100/32, exact-bounds pass)
//
// CHECK-DAG: #[[MAP_S:.*]] = affine_map<(d0) -> (-d0 + 10)>
// CHECK-DAG: #[[MAP_M:.*]] = affine_map<(d0) -> (d0 * -2 + 15)>
// CHECK-DAG: #[[MAP_R:.*]] = affine_map<(d0) -> (d0 * -5)>
// CHECK-DAG: #[[MAP_N_V:.*]] = affine_map<(d0) -> (d0 * 2)>

// CHECK: func.func @bounds_normalization(
// CHECK: scf.for

// CHECK: %[[C0_G2:.*]] = arith.constant 0 : index
// CHECK: %[[C20_G2:.*]] = arith.constant 20 : index
// CHECK: %[[C2_G2:.*]] = arith.constant 2 : index

// G1: recursive fusion -- {X, Y} -> {XA, XB, YA, YB}
// CHECK: %[[C0_G1:.*]] = arith.constant 0 : index
// CHECK: %[[C128_G1:.*]] = arith.constant 128 : index
// CHECK: %[[C16_G1:.*]] = arith.constant 16 : index
// CHECK: scf.for %[[G1_IV:.*]] = %[[C0_G1]] to %[[C128_G1]] step %[[C16_G1]]
// CHECK: %[[C0_G1_IN:.*]] = arith.constant 0 : index
// CHECK: %[[C128_G1_IN:.*]] = arith.constant 128 : index
// CHECK: %[[C16_G1_IN:.*]] = arith.constant 16 : index
// CHECK: scf.for %[[G1_IN_IV:.*]] = %[[C0_G1_IN]] to %[[C128_G1_IN]] step %[[C16_G1_IN]]
// CHECK-COUNT-4: arith.
// CHECK: scf.yield
// CHECK: } {XA, XB, YA, YB}
// CHECK: scf.yield
// CHECK: } {X, Y}

// G2: {M, N, R, S} (iters=10, gcd=2)
// CHECK: scf.for %[[G2_IV:.*]] = %[[C0_G2]] to %[[C20_G2]] step %[[C2_G2]]
// CHECK: affine.apply #[[MAP_S]](%[[G2_IV]])
// CHECK: affine.apply #[[MAP_M]](%[[G2_IV]])
// CHECK: affine.apply #[[MAP_R]](%[[G2_IV]])
// CHECK: affine.apply #[[MAP_N_V]](%[[G2_IV]])
// CHECK: scf.yield
// CHECK: } {M, N, R, S}

// G3: {V, W} (iters=12, V affine.apply, W identity)
// CHECK: %[[C0_G3:.*]] = arith.constant 0 : index
// CHECK: %[[C12_G3:.*]] = arith.constant 12 : index
// CHECK: %[[C1_G3:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[G3_IV:.*]] = %[[C0_G3]] to %[[C12_G3]] step %[[C1_G3]]
// CHECK: affine.apply #[[MAP_N_V]](%[[G3_IV]])
// CHECK-NOT: affine.apply
// CHECK: scf.yield
// CHECK: } {V, W}

// G4: {Z} (iters=12, sibling singleton)
// CHECK: scf.for
// CHECK: arith.
// CHECK: } {Z}

// G5: {BAD} (cnt*st<0, skipped by groupByFactorCompatibility)
// CHECK: scf.for
// CHECK: arith.
// CHECK: } {BAD}

// G6: {DYNA}, {DYNB} (dynamic ub, skipped by groupByFactorCompatibility)
// CHECK: scf.for
// CHECK: arith.
// CHECK: } {DYNA}
// CHECK: scf.for
// CHECK: arith.
// CHECK: } {DYNB}

// G7: {Q} (iters=3, iters singleton)
// CHECK: scf.for
// CHECK: arith.
// CHECK: } {Q}

// G8: {P1,P2,P3} (identical tail-block bounds 0..100/32, fused by exact-bounds pass)
// CHECK: scf.for
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: } {P1, P2, P3}

// CHECK: } {outer_loop}

func.func @bounds_normalization(
    %arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>,
    %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>,
    %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128xf32>,
    %arg9: tensor<128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>,
    %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>,
    %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>)
    -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %c10 = arith.constant 10 : index
  %c12 = arith.constant 12 : index
  %c15 = arith.constant 15 : index
  %c16 = arith.constant 16 : index
  %c24 = arith.constant 24 : index
  %c40 = arith.constant 40 : index
  %c100 = arith.constant 100 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %cn2 = arith.constant -2 : index
  %cn4 = arith.constant -4 : index
  %cn10 = arith.constant -10 : index
  %cn24 = arith.constant -24 : index
  %cn25 = arith.constant -25 : index
  %cn100 = arith.constant -100 : index
  %dynamic_ub = arith.addi %c0, %c100 : index
  %cst = arith.constant 0.0 : f32

  // Outer target: 4 inner loops (S,M from iters=10, X,Y from iters=8)
  %r0:6 = scf.for %iv0 = %c0 to %c128 step %c16
      iter_args(%aS = %arg0, %aM = %arg1, %aXA = %arg2, %aXB = %arg3, %aYA = %arg4, %aYB = %arg5)
      -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
          tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
    // S: 10..-10/-2, addf (neg+lb>0, iters=10)
    %iS = scf.for %ivS = %c10 to %cn10 step %cn2 iter_args(%iaS = %aS) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaS : tensor<128xf32> } {S}
    // M: 15..-25/-4, addf (neg+lb>0, iters=10)
    %iM = scf.for %ivM = %c15 to %cn25 step %cn4 iter_args(%iaM = %aM) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaM : tensor<128xf32> } {M}
    // X: 0..128/16 with nested XA, XB (iters=8, recursive fusion)
    %iX:2 = scf.for %ivX = %c0 to %c128 step %c16 iter_args(%iaX_A = %aXA, %iaX_B = %aXB) -> (tensor<128xf32>, tensor<128xf32>) {
      %iXA = scf.for %ivXA = %c0 to %c128 step %c16 iter_args(%iaXA = %iaX_A) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaXA : tensor<128xf32> } {XA}
      %iXB = scf.for %ivXB = %c0 to %c128 step %c16 iter_args(%iaXB = %iaX_B) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaXB : tensor<128xf32> } {XB}
      scf.yield %iXA, %iXB : tensor<128xf32>, tensor<128xf32>
    } {X}
    // Y: 0..128/16 with nested YA, YB (iters=8, recursive fusion)
    %iY:2 = scf.for %ivY = %c0 to %c128 step %c16 iter_args(%iaY_A = %aYA, %iaY_B = %aYB) -> (tensor<128xf32>, tensor<128xf32>) {
      %iYA = scf.for %ivYA = %c0 to %c128 step %c16 iter_args(%iaYA = %iaY_A) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaYA : tensor<128xf32> } {YA}
      %iYB = scf.for %ivYB = %c0 to %c128 step %c16 iter_args(%iaYB = %iaY_B) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaYB : tensor<128xf32> } {YB}
      scf.yield %iYA, %iYB : tensor<128xf32>, tensor<128xf32>
    } {Y}
    scf.yield %iS, %iM, %iX#0, %iX#1, %iY#0, %iY#1
        : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
          tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
  } {outer_loop}

  // Outer source: 6 inner loops + 1 bad loop (cnt*st<0, skipped)
  %r1:12 = scf.for %iv1 = %c0 to %c128 step %c16
      iter_args(%aR = %arg6, %aN = %arg7, %aV = %arg8, %aW = %arg9,
                %aZ = %arg10, %aBAD = %arg11, %aDYNA = %arg12, %aDYNB = %arg13, %aQ = %arg14,
                %aP1 = %arg15, %aP2 = %arg16, %aP3 = %arg17)
      -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
          tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
          tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
          tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
    // R: 0..-100/-10, addf (neg, iters=10)
    %iR = scf.for %ivR = %c0 to %cn100 step %cn10 iter_args(%iaR = %aR) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaR : tensor<128xf32> } {R}
    // N: 0..40/4, addf (iters=10)
    %iN = scf.for %ivN = %c0 to %c40 step %c4 iter_args(%iaN = %aN) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaN : tensor<128xf32> } {N}
    // V: 0..24/2, addf (iters=12)
    %iV = scf.for %ivV = %c0 to %c24 step %c2 iter_args(%iaV = %aV) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaV : tensor<128xf32> } {V}
    // W: 0..12/1, addf (iters=12, identity skip)
    %iW = scf.for %ivW = %c0 to %c12 step %c1 iter_args(%iaW = %aW) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaW : tensor<128xf32> } {W}
    // Z: 0..-24/-2, addf (iters=12, sibling singleton -- consumes V's result via iter_arg)
    %iZ = scf.for %ivZ = %c0 to %cn24 step %cn2 iter_args(%iaZ = %iV) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaZ : tensor<128xf32> } {Z}
    // BAD: 0..100/-10, addf (cnt=100, st=-10, cnt*st<0 -> skipped by groupByFactorCompatibility)
    %iBAD = scf.for %ivBAD = %c0 to %c100 step %cn10 iter_args(%iaBAD = %aBAD) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaBAD : tensor<128xf32> } {BAD}
    // DYNA: dynamic ub (non-constant, skipped by groupByFactorCompatibility)
    %iDYNA = scf.for %ivDA = %c0 to %dynamic_ub step %c1 iter_args(%iaDA = %aDYNA) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaDA : tensor<128xf32> } {DYNA}
    // DYNB: dynamic ub (non-constant, skipped by groupByFactorCompatibility)
    %iDYNB = scf.for %ivDB = %c0 to %dynamic_ub step %c1 iter_args(%iaDB = %aDYNB) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaDB : tensor<128xf32> } {DYNB}
    // Q: 0..6/2, addf (iters=3, singleton, size<2 -> NOT fused)
    %iQ = scf.for %ivQ = %c0 to %c6 step %c2 iter_args(%iaQ = %aQ) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaQ : tensor<128xf32> } {Q}
    // P1,P2,P3: 0..100/32 (tail block 100%32=4, fused by exact-bounds pass, no normalization)
    %iP1 = scf.for %ivP1 = %c0 to %c100 step %c32 iter_args(%iaP1 = %aP1) -> (tensor<128xf32>) { %s = arith.addf %cst, %cst : f32 scf.yield %iaP1 : tensor<128xf32> } {P1}
    %iP2 = scf.for %ivP2 = %c0 to %c100 step %c32 iter_args(%iaP2 = %aP2) -> (tensor<128xf32>) { %s = arith.mulf %cst, %cst : f32 scf.yield %iaP2 : tensor<128xf32> } {P2}
    %iP3 = scf.for %ivP3 = %c0 to %c100 step %c32 iter_args(%iaP3 = %aP3) -> (tensor<128xf32>) { %s = arith.subf %cst, %cst : f32 scf.yield %iaP3 : tensor<128xf32> } {P3}
    scf.yield %iR, %iN, %iV, %iW, %iZ, %iBAD, %iDYNA, %iDYNB, %iQ, %iP1, %iP2, %iP3
        : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
          tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
          tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
          tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
  } {outer_loop}

  return %r0#0, %r0#1, %r0#2, %r0#3, %r0#4, %r0#5,
         %r1#0, %r1#1, %r1#2, %r1#3, %r1#4, %r1#5, %r1#6, %r1#7, %r1#8,
         %r1#9, %r1#10, %r1#11
      : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %outer = transform.structured.match ops{["scf.for"]} attributes{outer_loop} in %arg0 : (!transform.any_op) -> (!transform.any_op)
    %t, %s = transform.split_handle %outer : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %t into %s
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    %result = transform.loop.fuse_nested_siblings %fused {recursive = true}
        : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}


// -----

// Test 2: dependency_chain
//
// Three intermediate op topologies placed between outer target and outer source.
// adjustEarlierLoopUsers must move all three groups after the fused loop.
//
// CHECK-LABEL: func.func @dependency_chain(
// CHECK: scf.for
// CHECK: } {outer_loop}
// Intermediates should be moved after the fused loop
// CHECK: tensor.cast
// CHECK: linalg.generic
// CHECK: scf.if
// CHECK: tensor.cast
// CHECK: return
func.func @dependency_chain(%A: tensor<128xf32>, %B: tensor<128xf32>)
    -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %true = arith.constant 1 : i1

  // Outer target (earlier loop) with one inner loop.
  %r0 = scf.for %iv0 = %c0 to %c128 step %c16 iter_args(%acc0 = %A) -> (tensor<128xf32>) {
    %inner = scf.for %iv = %c0 to %c128 step %c16 iter_args(%ia = %acc0) -> (tensor<128xf32>) {
      %v = vector.transfer_read %ia[%iv], %cst : tensor<128xf32>, vector<16xf32>
      %w = vector.transfer_write %v, %ia[%iv] : vector<16xf32>, tensor<128xf32>
      scf.yield %w : tensor<128xf32>
    }
    scf.yield %inner : tensor<128xf32>
  } {outer_loop}

  // Pattern 1: Chained user (tensor.cast)
  %chain = tensor.cast %r0 : tensor<128xf32> to tensor<128xf32>

  // Pattern 2: Diamond users (fork + linalg.generic convergence)
  %u1 = tensor.cast %r0 : tensor<128xf32> to tensor<128xf32>
  %u2 = tensor.cast %r0 : tensor<128xf32> to tensor<128xf32>
  %empty = tensor.empty() : tensor<128xf32>
  %u3 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%u1, %u2 : tensor<128xf32>, tensor<128xf32>)
    outs(%empty : tensor<128xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<128xf32>

  // Pattern 3: Container climb (scf.if with internal user + external consumer)
  %ifr = scf.if %true -> (tensor<128xf32>) {
    %inner_cast = tensor.cast %r0 : tensor<128xf32> to tensor<128xf32>
    scf.yield %inner_cast : tensor<128xf32>
  } else {
    scf.yield %A : tensor<128xf32>
  }
  %consumer = tensor.cast %ifr : tensor<128xf32> to tensor<128xf32>

  // Outer source (later loop) with one inner loop.
  %r1 = scf.for %iv1 = %c0 to %c128 step %c16 iter_args(%acc1 = %B) -> (tensor<128xf32>) {
    %inner = scf.for %iv = %c0 to %c128 step %c16 iter_args(%ia = %acc1) -> (tensor<128xf32>) {
      %v = vector.transfer_read %ia[%iv], %cst : tensor<128xf32>, vector<16xf32>
      %w = vector.transfer_write %v, %ia[%iv] : vector<16xf32>, tensor<128xf32>
      scf.yield %w : tensor<128xf32>
    }
    scf.yield %inner : tensor<128xf32>
  } {outer_loop}

  return %chain, %u3, %consumer, %r1
      : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %outer = transform.structured.match ops{["scf.for"]} attributes{outer_loop} in %arg0 : (!transform.any_op) -> (!transform.any_op)
    %t, %s = transform.split_handle %outer : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %t into %s
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    %result = transform.loop.fuse_nested_siblings %fused {recursive = true}
        : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
