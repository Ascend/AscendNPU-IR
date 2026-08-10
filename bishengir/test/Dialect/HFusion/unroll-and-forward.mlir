// RUN: bishengir-opt %s -hfusion-unroll-and-forward -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// Test 1: a single isolated loop (no sibling chain) is NOT unrolled.
// CHECK-LABEL: func.func @reject_isolated
// CHECK: scf.for
func.func @reject_isolated(%arg0: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x150xf32>
  %r = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %r : tensor<1x150xf32>
}

// Test 2: producer→consumer chain is unrolled. The sum read's sole user is
// select(mask, read, 0), so the pass forwards it to the exp vector — the
// select becomes select(mask, <exp>, 0) and the sum transfer_read is gone.
// CHECK-LABEL: func.func @unroll_and_relay
// CHECK-NOT: scf.for
// CHECK: math.exp
// CHECK: arith.select
// CHECK-NOT: vector.transfer_read
// CHECK: arith.addf
func.func @unroll_and_relay(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x150xf32>
  // exp loop: produces intermediate tensor %t
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  // sum loop: reads %t via a masked read whose sole user is
  // select(mask, read, 0) — the pass forwards the read to the exp vector,
  // turning the select into select(mask, exp, 0) (OOB lanes stay zeroed).
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %cst0 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %s = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %zero) -> vector<1x64xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %g = arith.select %mask, %v, %cst0 : vector<1x64xi1>, vector<1x64xf32>
    %red = arith.addf %g, %acc : vector<1x64xf32>
    scf.yield %red : vector<1x64xf32>
  }
  %final = vector.transfer_write %s, %out[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x150xf32>
  return %final : tensor<1x150xf32>
}

// Test 3: nested loop (not innermost) → rejected.
// CHECK-LABEL: func.func @reject_nested
// CHECK: scf.for
func.func @reject_nested(%arg0: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x150xf32>
  %r = scf.for %outer = %c0 to %c3 step %c1 iter_args(%oacc = %init) -> tensor<1x150xf32> {
    %inner = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %oacc) -> tensor<1x150xf32> {
      scf.yield %acc : tensor<1x150xf32>
    }
    scf.yield %inner : tensor<1x150xf32>
  }
  return %r : tensor<1x150xf32>
}

// Test 4: tripCount > maxUnroll (65 > 4) → rejected.
// CHECK-LABEL: func.func @reject_trip_overflow
// CHECK: scf.for
// CHECK: scf.for
func.func @reject_trip_overflow(%arg0: tensor<1x260xf32>, %out: tensor<1x260xf32>) -> tensor<1x260xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c260 = arith.constant 260 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x260xf32>
  // trip = ceil(260/4) = 65 > 4 → rejected
  %t = scf.for %iv = %c0 to %c260 step %c4 iter_args(%acc = %init) -> tensor<1x260xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 260, 4)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x260xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x260xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x4xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x4xf32>
    %e = math.exp %v : vector<1x4xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x260xf32>
    scf.yield %ins : tensor<1x260xf32>
  }
  %zero = arith.constant dense<0.000000e+00> : vector<1x4xf32>
  %s = scf.for %iv = %c0 to %c260 step %c4 iter_args(%acc = %zero) -> vector<1x4xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 260, 4)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x260xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x4xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x4xf32>
    %red = arith.addf %v, %acc : vector<1x4xf32>
    scf.yield %red : vector<1x4xf32>
  }
  %final = vector.transfer_write %s, %out[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x260xf32>
  return %final : tensor<1x260xf32>
}


// -----

// Test 5: register-pressure structural check. A trip=4 chain (256 cols / 64 =
// 4 segments) is the maxUnroll boundary. After full unroll the body holds
// exactly `trip` copies of the relayed exp vector (the long-lived values
// simultaneously in scope). The 5 eliminability conditions (simple write/read
// path, pure intermediate, disjoint writes, mask-safe) keep the structure
// straight-line, so the long-lived register count is bounded by
// relayed_vectors × trip = 1 × 4 = 4. This test asserts that count.
// CHECK-LABEL: func.func @reg_pressure_trip4
// CHECK-NOT: scf.for
// After post-unroll CSE+canonicalize, consumer reads are forwarded (no mask),
// so addf eats exp directly.
// CHECK: math.exp
// CHECK: arith.addf
func.func @reg_pressure_trip4(%arg0: tensor<1x256xf32>, %out: tensor<1x256xf32>) -> tensor<1x256xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x256xf32>
  // exp loop: trip = ceil(256/64) = 4
  %t = scf.for %iv = %c0 to %c256 step %c64 iter_args(%acc = %init) -> tensor<1x256xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 256, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x256xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x256xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x256xf32>
    scf.yield %ins : tensor<1x256xf32>
  }
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %s = scf.for %iv = %c0 to %c256 step %c64 iter_args(%acc = %zero) -> vector<1x64xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 256, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x256xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %red = arith.addf %v, %acc : vector<1x64xf32>
    scf.yield %red : vector<1x64xf32>
  }
  %final = vector.transfer_write %s, %out[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x256xf32>
  return %final : tensor<1x256xf32>
}

// -----

// Test 6: trip count exactly 5 > maxUnroll=4 → rejected, both loops preserved.
// 320 cols / step 64 = trip 5, just past the max-unroll boundary.
// CHECK-LABEL: func.func @reject_trip5
// CHECK: scf.for
// CHECK: scf.for
func.func @reject_trip5(%arg0: tensor<1x320xf32>, %out: tensor<1x320xf32>) -> tensor<1x320xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c320 = arith.constant 320 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x320xf32>
  %t = scf.for %iv = %c0 to %c320 step %c64 iter_args(%acc = %init) -> tensor<1x320xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 320, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x320xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x320xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x320xf32>
    scf.yield %ins : tensor<1x320xf32>
  }
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %s = scf.for %iv = %c0 to %c320 step %c64 iter_args(%acc = %zero) -> vector<1x64xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 320, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x320xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %red = arith.addf %v, %acc : vector<1x64xf32>
    scf.yield %red : vector<1x64xf32>
  }
  %final = vector.transfer_write %s, %out[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x320xf32>
  return %final : tensor<1x320xf32>
}

// -----

// Test 7: non-contiguous group — an unrelated scf.for is interleaved between
// the producer and consumer loops (same bounds). Conservative: reject, both
// loops preserved.
// CHECK-LABEL: func.func @reject_interleaved
// CHECK: scf.for
// CHECK: scf.for
func.func @reject_interleaved(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x150xf32>
  // exp loop (producer)
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  // interleaved unrelated scf.for with different bounds — a non-group
  // loop between the producer and consumer breaks contiguity: the chain
  // (exp + sum) is not contiguous in the block, so the whole group is
  // rejected and both chain loops stay unrolled.
  %c16 = arith.constant 16 : index
  %inter = scf.for %iv = %c0 to %c16 step %c1 iter_args(%acc = %init) -> tensor<1x150xf32> {
    scf.yield %acc : tensor<1x150xf32>
  }
  // sum loop (consumer) — same bounds as exp, but not contiguous
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %s = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %zero) -> vector<1x64xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %red = arith.addf %v, %acc : vector<1x64xf32>
    scf.yield %red : vector<1x64xf32>
  }
  %final = vector.transfer_write %s, %out[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x150xf32>
  return %final : tensor<1x150xf32>
}



// -----

// Test 8: masksAreEqual — after unroll, the write and read masks are
// distinct SSA ops (different create_mask) but have identical constant
// operands. The read has no select(mask, read, 0) wrapping, so forward
// must rely on masksAreEqual comparing the create_mask operands.
// Without it, the read would survive (not forwarded).
// CHECK-LABEL: func.func @relay_distinct_mask_ssa
// CHECK-NOT: scf.for
// CHECK: math.exp
// Forward bails: a consumer transfer_read appears after math.exp (divf reads
// through UB, not forwarded to exp). If forward succeeded, divf would follow
// exp with no intervening transfer_read.
// CHECK: vector.transfer_read
// CHECK: arith.divf
func.func @relay_distinct_mask_ssa(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %init = tensor.empty() : tensor<1x150xf32>
  // exp loop: writes exp into %t, with mask
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  // div loop: reads %t with a *distinct* create_mask (same values, different SSA).
  // No select(mask, read, 0). Result accumulates (divf v into acc) so every
  // iteration's divf is live — no DCE.
  %d = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %zero) -> vector<1x64xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask2 = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask2 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %red = arith.divf %v, %acc : vector<1x64xf32>
    scf.yield %red : vector<1x64xf32>
  }
  %final = vector.transfer_write %d, %out[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x150xf32>
  return %final : tensor<1x150xf32>
}

// -----

// Test 9: a masked read whose sole user is NOT select(mask, read, 0) is NOT
// forwarded — the pass declines to synthesize a select guard, so the read
// stays (precision over register-relay). Here the sum read's only user is
// arith.addf, so forward bails and transfer_read survives. Uses 150/64
// (trip=3, tail 22) so the dangerous partial-mask case is exercised, not just
// full-VL.
// CHECK-LABEL: func.func @reject_masked_read_no_select_guard
// CHECK-NOT: scf.for
// CHECK: math.exp
// CHECK: vector.transfer_read
// CHECK: arith.addf
func.func @reject_masked_read_no_select_guard(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x150xf32>
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %s = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %zero) -> vector<1x64xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %red = arith.addf %v, %acc : vector<1x64xf32>
    scf.yield %red : vector<1x64xf32>
  }
  %final = vector.transfer_write %s, %out[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x150xf32>
  return %final : tensor<1x150xf32>
}

// -----// Test 10: masked read whose sole user is arith.divf (not select) but
// the div transfer_write carries the SAME mask. OOB-lane divf results (stale
// exp / possible inf) are discarded by the masked write, so forward is safe:
// divf eats the exp vector directly, the div transfer_read is gone.
// %sum = 2.0 (not 1.0) so divf isn't constant-folded to identity.
// CHECK-LABEL: func.func @relay_masked_divf_same_mask_write
// CHECK-NOT: scf.for
// CHECK-COUNT-3: math.exp
// CHECK-COUNT-3: arith.divf
func.func @relay_masked_divf_same_mask_write(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %sum = arith.constant dense<2.000000e+00> : vector<1x64xf32>
  %init = tensor.empty() : tensor<1x150xf32>
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  %d = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask2 = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask2 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %div = arith.divf %v, %sum : vector<1x64xf32>
    %w = vector.transfer_write %div, %dsl[%c0, %c0], %mask2 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %d : tensor<1x150xf32>
}

// -----// Test 12: partial-overlap insert chain → bail (don't forward).
// exp loop trip=3 step=32, slice size=64: adjacent iters insert at [0,0],
// [0,32], [0,64] (size 64 each) — iter1 overlaps iter0 on [32,64). A div read
// of [0,64) of the final tensor would need vA[0,32)+vB[32,64), not any single
// write vector. The overlaps check must bail; the div read survives.
// CHECK-LABEL: func.func @reject_partial_overlap
// CHECK: vector.transfer_read
// CHECK: arith.divf
func.func @reject_partial_overlap(%arg0: tensor<1x96xf32>, %out: tensor<1x96xf32>) -> tensor<1x96xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c96 = arith.constant 96 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %sum = arith.constant dense<2.000000e+00> : vector<1x64xf32>
  %init = tensor.empty() : tensor<1x96xf32>
  %t = scf.for %iv = %c0 to %c96 step %c32 iter_args(%acc = %init) -> tensor<1x96xf32> {
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %c64] [1, 1] : tensor<1x96xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %c64] [1, 1] : tensor<1x96xf32> to tensor<1x?xf32>
    %v = vector.transfer_read %sl[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %c64] [1, 1] : tensor<1x?xf32> into tensor<1x96xf32>
    scf.yield %ins : tensor<1x96xf32>
  }
  %d = scf.for %iv = %c0 to %c96 step %c32 iter_args(%acc = %init) -> tensor<1x96xf32> {
    %sl = tensor.extract_slice %t[0, %iv] [1, %c64] [1, 1] : tensor<1x96xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %c64] [1, 1] : tensor<1x96xf32> to tensor<1x?xf32>
    %v = vector.transfer_read %sl[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %div = arith.divf %v, %sum : vector<1x64xf32>
    %w = vector.transfer_write %div, %dsl[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %c64] [1, 1] : tensor<1x?xf32> into tensor<1x96xf32>
    scf.yield %ins : tensor<1x96xf32>
  }
  return %d : tensor<1x96xf32>
}

// -----// Test 11: masked read + sole user divf + div write with NO mask → bail.
// OOB-lane divf results would be written back and pollute; forward must decline.
// CHECK-LABEL: func.func @reject_masked_divf_write_no_mask
// CHECK: vector.transfer_read
// CHECK: arith.divf
func.func @reject_masked_divf_write_no_mask(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %sum = arith.constant dense<2.000000e+00> : vector<1x64xf32>
  %init = tensor.empty() : tensor<1x150xf32>
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  %d = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask2 = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask2 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %div = arith.divf %v, %sum : vector<1x64xf32>
    %w = vector.transfer_write %div, %dsl[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %d : tensor<1x150xf32>
}

// -----// Test 13: softmax-faithful exp→sum→div with multi-user tail read.
// This replicates the real softmax structure: exp producer, then two consumer
// loops — sum (select guard + addf reduction) and div (divf + masked write).
// The tail segment's div read has TWO users (sum's select + div's divf).
// all-users-discard must forward all three segments including the tail.
// Key assertion: no transfer_read between the last math.exp and the first
// arith.divf (i.e. divf eats exp directly, not through UB).
// CHECK-LABEL: func.func @softmax_faithful_multi_user_forward
// CHECK-NOT: scf.for
// CHECK: math.exp
// CHECK: arith.divf
// NOTE: FileCheck cannot verify divf's operand is exp (not transfer_read).
// Mock compile + IR grep (bishengir-compile --mlir-print-ir-after-all) is the
// authoritative check: grep "arith.divf" and verify its %operand matches a
// math.exp result line. If any segment bails, divf eats transfer_read instead.
func.func @softmax_faithful_multi_user_forward(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %sum = arith.constant dense<2.000000e+00> : vector<1x64xf32>
  %init = tensor.empty() : tensor<1x150xf32>
  // exp producer: writes exp into %t
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  // sum consumer: select(mask, read, 0) + addf — path A user of div read
  %s = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %zero) -> vector<1x64xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %g = arith.select %mask, %v, %zero : vector<1x64xi1>, vector<1x64xf32>
    %red = arith.addf %g, %acc : vector<1x64xf32>
    scf.yield %red : vector<1x64xf32>
  }
  // div consumer: reads %t (same source, multi-user with sum) + divf + masked write
  // %sum = 2.0 to avoid constant fold.
  %d = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask2 = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask2 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %div = arith.divf %v, %sum : vector<1x64xf32>
    %w = vector.transfer_write %div, %dsl[%c0, %c0], %mask2 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %d : tensor<1x150xf32>
}

// -----// Test 14: a func.call between producer and consumer loops breaks
// contiguity — the call may have side effects or influence data flow, so the
// chain is rejected and both loops stay unrolled.
// CHECK-LABEL: func.func @reject_call_between_loops
// CHECK: scf.for
// CHECK: call @side_effect
// CHECK: scf.for
func.func @reject_call_between_loops(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %sum = arith.constant dense<2.000000e+00> : vector<1x64xf32>
  %init = tensor.empty() : tensor<1x150xf32>
  // exp loop (producer)
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  // func.call between producer and consumer — breaks contiguity
  func.call @side_effect(%t) : (tensor<1x150xf32>) -> ()
  // div loop (consumer) — same bounds as exp, but a func.call intervenes
  %d = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask2 = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask2 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %div = arith.divf %v, %sum : vector<1x64xf32>
    %w = vector.transfer_write %div, %dsl[%c0, %c0], %mask2 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %d : tensor<1x150xf32>
}
func.func @side_effect(%arg0: tensor<1x150xf32>) -> () {
  return
}

// -----// Test 15: an scf.if between producer and consumer loops breaks
// contiguity — control flow between chain loops can influence data flow,
// so the chain is rejected and both loops stay unrolled.
// CHECK-LABEL: func.func @reject_if_between_loops
// CHECK: scf.for
// CHECK: scf.if
// CHECK: scf.for
func.func @reject_if_between_loops(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>, %cond: i1) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %sum = arith.constant dense<2.000000e+00> : vector<1x64xf32>
  %init = tensor.empty() : tensor<1x150xf32>
  // exp loop (producer)
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  // scf.if between producer and consumer — breaks contiguity
  %dummy = scf.if %cond -> (tensor<1x150xf32>) {
    scf.yield %t : tensor<1x150xf32>
  } else {
    scf.yield %init : tensor<1x150xf32>
  }
  // div loop (consumer) — same bounds as exp, but an scf.if intervenes
  %d = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask2 = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask2 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %div = arith.divf %v, %sum : vector<1x64xf32>
    %w = vector.transfer_write %div, %dsl[%c0, %c0], %mask2 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %d : tensor<1x150xf32>
}

// -----// Test 16: mismatched vector types between producer write and
// consumer read (vector<1x64xf32> vs vector<64xf32>) — validateWrite rejects
// forwarding, so the consumer's transfer_read survives (loops are still
// unrolled, but the read is not replaced by the write vector).
// CHECK-LABEL: func.func @reject_mismatched_vector_type
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.divf
func.func @reject_mismatched_vector_type(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %sum = arith.constant dense<2.000000e+00> : vector<64xf32>
  %init = tensor.empty() : tensor<1x150xf32>
  // exp loop (producer): writes vector<1x64xf32>
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  // div loop (consumer): reads with vector<64xf32> — mismatched shape
  %d = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask2 = vector.create_mask %m : vector<64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask2 {in_bounds = [true]} : tensor<1x?xf32>, vector<64xf32>
    %div = arith.divf %v, %sum : vector<64xf32>
    %w = vector.transfer_write %div, %dsl[%c0, %c0], %mask2 {in_bounds = [true]} : vector<64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %d : tensor<1x150xf32>
}

// -----

// T returned directly via func.return escapes the chain → rejected.
// CHECK-LABEL: func.func @probe_t_direct_return
// CHECK: scf.for
func.func @probe_t_direct_return(%arg0: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x150xf32>
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %t : tensor<1x150xf32>
}

// -----// Test: masked read with NON-zero padding (-inf) is rejected. Forwarding
// would replace the read's OOB lanes (padding = -inf) with the producer's
// exp(-inf) = 0, changing OOB lane semantics. The chain must bail so the
// original scf.for loops survive untouched.
// CHECK-LABEL: func.func @reject_nonzero_padding
// CHECK: scf.for
func.func @reject_nonzero_padding(%arg0: tensor<1x150xf32>, %out: tensor<1x150xf32>) -> tensor<1x150xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c150 = arith.constant 150 : index
  %c64 = arith.constant 64 : index
  %cst_ninf = arith.constant 0xFF800000 : f32
  %sum = arith.constant dense<2.000000e+00> : vector<1x64xf32>
  %init = tensor.empty() : tensor<1x150xf32>
  %t = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %arg0[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst_ninf, %mask {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %e = math.exp %v : vector<1x64xf32>
    %w = vector.transfer_write %e, %dsl[%c0, %c0], %mask {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  %d = scf.for %iv = %c0 to %c150 step %c64 iter_args(%acc = %init) -> tensor<1x150xf32> {
    %m = affine.min affine_map<(d0) -> (-d0 + 150, 64)>(%iv)
    %sl = tensor.extract_slice %t[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %dsl = tensor.extract_slice %acc[0, %iv] [1, %m] [1, 1] : tensor<1x150xf32> to tensor<1x?xf32>
    %mask2 = vector.create_mask %c1, %m : vector<1x64xi1>
    %v = vector.transfer_read %sl[%c0, %c0], %cst_ninf, %mask2 {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x64xf32>
    %div = arith.divf %v, %sum : vector<1x64xf32>
    %w = vector.transfer_write %div, %dsl[%c0, %c0], %mask2 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %ins = tensor.insert_slice %w into %acc[0, %iv] [1, %m] [1, 1] : tensor<1x?xf32> into tensor<1x150xf32>
    scf.yield %ins : tensor<1x150xf32>
  }
  return %d : tensor<1x150xf32>
}
