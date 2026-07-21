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

// Test 2: producer→consumer chain is unrolled, sum uses exp via relay.
// CHECK-LABEL: func.func @unroll_and_relay
// CHECK-NOT: scf.for
// CHECK: math.exp
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
  // sum loop: reads %t — relayed to exp vector after unroll+forward
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
// CHECK-COUNT-4: math.exp
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
// CHECK-COUNT-3: math.exp
// CHECK-COUNT-3: arith.divf
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
