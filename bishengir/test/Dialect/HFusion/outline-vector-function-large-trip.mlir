// RUN: bishengir-opt %s -split-input-file -outline-vector-function -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// ccec packs the trip count of a VF-outlined vector loop into a 16-bit field
// of the launch parameter block, so any outlined loop with a static trip
// count above 65535 would launch its VF with a truncated (modulo 2^16)
// count and silently corrupt the kernel output. Outline first, preserving
// existing VF boundaries, then split large loops inside each VF. Absolute
// index bounds and chained iter-args preserve the original loop semantics.
// The exact bound constants below pin every chunk's trip count.

module {
  // Case 1: trip 65536 (2^16, the failing broadcast shape) -> one VF, two
  // chunk loops [0, 65535) + [65535, 65536), nested broadcast loop kept.
  func.func @fill_65536(%arg0: tensor<65536xi8>) -> tensor<65536x2xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %c65536 = arith.constant 65536 : index
    %c0_i8 = arith.constant 0 : i8
    %init = tensor.empty() : tensor<65536x2xi8>
    %r = scf.for %iv = %c0 to %c65536 step %c1 iter_args(%acc = %init) -> (tensor<65536x2xi8>) {
      %src = tensor.extract_slice %arg0[%iv] [1] [1] : tensor<65536xi8> to tensor<1xi8>
      %inner = scf.for %jv = %c0 to %c2 step %c256 iter_args(%iacc = %acc) -> (tensor<65536x2xi8>) {
        %dst = tensor.extract_slice %iacc[%iv, %jv] [1, 2] [1, 1] : tensor<65536x2xi8> to tensor<1x2xi8>
        %v = vector.transfer_read %src[%c0], %c0_i8 {in_bounds = [false, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : tensor<1xi8>, vector<1x256xi8>
        %w = vector.transfer_write %v, %dst[%c0, %c0] : vector<1x256xi8>, tensor<1x2xi8>
        %ins = tensor.insert_slice %w into %iacc[%iv, %jv] [1, 2] [1, 1] : tensor<1x2xi8> into tensor<65536x2xi8>
        scf.yield %ins : tensor<65536x2xi8>
      }
      scf.yield %inner : tensor<65536x2xi8>
    } {outlinedLoopWithFill}
    return %r : tensor<65536x2xi8>
  }

  // Case 2: trip 65535 (largest representable count) -> one VF, one loop,
  // bound unchanged; no wrapped 2^16 bound may appear.
  func.func @fill_65535(%arg0: tensor<65535xi8>) -> tensor<65535x2xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c65535 = arith.constant 65535 : index
    %c0_i8 = arith.constant 0 : i8
    %init = tensor.empty() : tensor<65535x2xi8>
    %r = scf.for %iv = %c0 to %c65535 step %c1 iter_args(%acc = %init) -> (tensor<65535x2xi8>) {
      %src = tensor.extract_slice %arg0[%iv] [1] [1] : tensor<65535xi8> to tensor<1xi8>
      %v = vector.transfer_read %src[%c0], %c0_i8 {in_bounds = [false, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : tensor<1xi8>, vector<1x256xi8>
      %dst = tensor.extract_slice %acc[%iv, 0] [1, 2] [1, 1] : tensor<65535x2xi8> to tensor<1x2xi8>
      %w = vector.transfer_write %v, %dst[%c0, %c0] : vector<1x256xi8>, tensor<1x2xi8>
      %ins = tensor.insert_slice %w into %acc[%iv, 0] [1, 2] [1, 1] : tensor<1x2xi8> into tensor<65535x2xi8>
      scf.yield %ins : tensor<65535x2xi8>
    } {outlinedLoopWithFill}
    return %r : tensor<65535x2xi8>
  }

  // Case 3: trip 131072 (2^17) -> one VF, three chunk loops
  // [0, 65535) + [65535, 131070) + [131070, 131072).
  func.func @fill_131072(%arg0: tensor<131072xi8>) -> tensor<131072x2xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c131072 = arith.constant 131072 : index
    %c0_i8 = arith.constant 0 : i8
    %init = tensor.empty() : tensor<131072x2xi8>
    %r = scf.for %iv = %c0 to %c131072 step %c1 iter_args(%acc = %init) -> (tensor<131072x2xi8>) {
      %src = tensor.extract_slice %arg0[%iv] [1] [1] : tensor<131072xi8> to tensor<1xi8>
      %v = vector.transfer_read %src[%c0], %c0_i8 {in_bounds = [false, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : tensor<1xi8>, vector<1x256xi8>
      %dst = tensor.extract_slice %acc[%iv, 0] [1, 2] [1, 1] : tensor<131072x2xi8> to tensor<1x2xi8>
      %w = vector.transfer_write %v, %dst[%c0, %c0] : vector<1x256xi8>, tensor<1x2xi8>
      %ins = tensor.insert_slice %w into %acc[%iv, 0] [1, 2] [1, 1] : tensor<1x2xi8> into tensor<131072x2xi8>
      scf.yield %ins : tensor<131072x2xi8>
    } {outlinedLoopWithFill}
    return %r : tensor<131072x2xi8>
  }

  // Case 4: two adjacent large loops from DIFFERENT source loops. Each
  // source loop is split, but merging is limited to chunks sharing the
  // original VF boundaries: two VFs (one per source loop).
  func.func @fill_two_sources(%arg0: tensor<65536xi8>, %arg1: tensor<65536xi8>) -> (tensor<65536x2xi8>, tensor<65536x2xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c65536 = arith.constant 65536 : index
    %c0_i8 = arith.constant 0 : i8
    %init0 = tensor.empty() : tensor<65536x2xi8>
    %r0 = scf.for %iv = %c0 to %c65536 step %c1 iter_args(%acc = %init0) -> (tensor<65536x2xi8>) {
      %src = tensor.extract_slice %arg0[%iv] [1] [1] : tensor<65536xi8> to tensor<1xi8>
      %v = vector.transfer_read %src[%c0], %c0_i8 {in_bounds = [false, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : tensor<1xi8>, vector<1x256xi8>
      %dst = tensor.extract_slice %acc[%iv, 0] [1, 2] [1, 1] : tensor<65536x2xi8> to tensor<1x2xi8>
      %w = vector.transfer_write %v, %dst[%c0, %c0] : vector<1x256xi8>, tensor<1x2xi8>
      %ins = tensor.insert_slice %w into %acc[%iv, 0] [1, 2] [1, 1] : tensor<1x2xi8> into tensor<65536x2xi8>
      scf.yield %ins : tensor<65536x2xi8>
    } {outlinedLoopWithFill}
    %init1 = tensor.empty() : tensor<65536x2xi8>
    %r1 = scf.for %iv = %c0 to %c65536 step %c1 iter_args(%acc = %init1) -> (tensor<65536x2xi8>) {
      %src = tensor.extract_slice %arg1[%iv] [1] [1] : tensor<65536xi8> to tensor<1xi8>
      %v = vector.transfer_read %src[%c0], %c0_i8 {in_bounds = [false, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : tensor<1xi8>, vector<1x256xi8>
      %dst = tensor.extract_slice %acc[%iv, 0] [1, 2] [1, 1] : tensor<65536x2xi8> to tensor<1x2xi8>
      %w = vector.transfer_write %v, %dst[%c0, %c0] : vector<1x256xi8>, tensor<1x2xi8>
      %ins = tensor.insert_slice %w into %acc[%iv, 0] [1, 2] [1, 1] : tensor<1x2xi8> into tensor<65536x2xi8>
      scf.yield %ins : tensor<65536x2xi8>
    } {outlinedLoopWithFill}
    return %r0, %r1 : tensor<65536x2xi8>, tensor<65536x2xi8>
  }

  // Case 5: plain outlined-loop-target (no fill) with trip 65536 is split
  // inside its outlined VF. The existing plain-loop grouping stays unchanged.
  func.func @plain_large_trip(%arg0: tensor<65536xi8>) -> tensor<65536x2xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c65536 = arith.constant 65536 : index
    %c0_i8 = arith.constant 0 : i8
    %init = tensor.empty() : tensor<65536x2xi8>
    %r = scf.for %iv = %c0 to %c65536 step %c1 iter_args(%acc = %init) -> (tensor<65536x2xi8>) {
      %src = tensor.extract_slice %arg0[%iv] [1] [1] : tensor<65536xi8> to tensor<1xi8>
      %v = vector.transfer_read %src[%c0], %c0_i8 {in_bounds = [false, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : tensor<1xi8>, vector<1x256xi8>
      %dst = tensor.extract_slice %acc[%iv, 0] [1, 2] [1, 1] : tensor<65536x2xi8> to tensor<1x2xi8>
      %w = vector.transfer_write %v, %dst[%c0, %c0] : vector<1x256xi8>, tensor<1x2xi8>
      %ins = tensor.insert_slice %w into %acc[%iv, 0] [1, 2] [1, 1] : tensor<1x2xi8> into tensor<65536x2xi8>
      scf.yield %ins : tensor<65536x2xi8>
    } {"outlined-loop-target-0"}
    return %r : tensor<65536x2xi8>
  }

  // Case 6: lb != 0, step != 1, (ub - lb) not a multiple of step:
  // lb=1, step=4, ub=262143 -> trip = ceil(262142/4) = 65536, so two chunks
  // [1, 262141) + [262141, 262143). The last chunk keeps the ORIGINAL ub;
  // lb + trip * step = 262145 would overshoot the tensor range and must not
  // appear as a chunk bound.
  func.func @fill_lbx_step4(%arg0: tensor<262143xi8>) -> tensor<262143x2xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c262143 = arith.constant 262143 : index
    %c0_i8 = arith.constant 0 : i8
    %init = tensor.empty() : tensor<262143x2xi8>
    %r = scf.for %iv = %c1 to %c262143 step %c4 iter_args(%acc = %init) -> (tensor<262143x2xi8>) {
      %src = tensor.extract_slice %arg0[%iv] [1] [1] : tensor<262143xi8> to tensor<1xi8>
      %v = vector.transfer_read %src[%c0], %c0_i8 {in_bounds = [false, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : tensor<1xi8>, vector<1x256xi8>
      %dst = tensor.extract_slice %acc[%iv, 0] [1, 2] [1, 1] : tensor<262143x2xi8> to tensor<1x2xi8>
      %w = vector.transfer_write %v, %dst[%c0, %c0] : vector<1x256xi8>, tensor<1x2xi8>
      %ins = tensor.insert_slice %w into %acc[%iv, 0] [1, 2] [1, 1] : tensor<1x2xi8> into tensor<262143x2xi8>
      scf.yield %ins : tensor<262143x2xi8>
    } {outlinedLoopWithFill}
    return %r : tensor<262143x2xi8>
  }

  // Case 7: i32-bounds loop (scf.for ... : i32) with trip 65536. Chunk
  // bounds keep the i32 type -- scf.for requires lowerBound/upperBound/step
  // to share one type, so emitting index bounds here would produce IR that
  // only fails much later in the pipeline. Both chunks still merge into one
  // i32 VF.
  func.func @fill_i32(%arg0: tensor<65536xi8>) -> tensor<65536x2xi8> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c65536 = arith.constant 65536 : i32
    %c0_idx = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %init = tensor.empty() : tensor<65536x2xi8>
    %r = scf.for %iv = %c0 to %c65536 step %c1 iter_args(%acc = %init) -> (tensor<65536x2xi8>) : i32 {
      %idx = arith.index_cast %iv : i32 to index
      %src = tensor.extract_slice %arg0[%idx] [1] [1] : tensor<65536xi8> to tensor<1xi8>
      %v = vector.transfer_read %src[%c0_idx], %c0_i8 {in_bounds = [false, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : tensor<1xi8>, vector<1x256xi8>
      %dst = tensor.extract_slice %acc[%idx, 0] [1, 2] [1, 1] : tensor<65536x2xi8> to tensor<1x2xi8>
      %w = vector.transfer_write %v, %dst[%c0_idx, %c0_idx] : vector<1x256xi8>, tensor<1x2xi8>
      %ins = tensor.insert_slice %w into %acc[%idx, 0] [1, 2] [1, 1] : tensor<1x2xi8> into tensor<65536x2xi8>
      scf.yield %ins : tensor<65536x2xi8>
    } {outlinedLoopWithFill}
    return %r : tensor<65536x2xi8>
  }
}

// Case 1: one VF, two chunk loops [0, 65535) + [65535, 65536), nested
// broadcast loop kept. No second chunk function may appear.
// CHECK-LABEL: func.func @fill_65536_outlined_vf_0
// CHECK: scf.for %{{.+}} = %{{.+}} to %c65535{{(_[0-9]+)?}}
// CHECK: scf.for %{{.+}} = %c65535{{(_[0-9]+)?}} to %c65536{{(_[0-9]+)?}}
// CHECK-NOT: func.func @fill_65536_outlined_vf_1

// Case 2: one VF, one loop, bound unchanged; no wrapped 2^16 bound.
// CHECK-LABEL: func.func @fill_65535_outlined_vf_0
// CHECK: scf.for %{{.+}} = %{{.+}} to %c65535{{(_[0-9]+)?}}
// CHECK-NOT: scf.for %{{.+}} = %{{.+}} to %c65536{{(_[0-9]+)?}}
// CHECK-NOT: func.func @fill_65535_outlined_vf_1

// Case 3: one VF, three chunk loops
// [0, 65535) + [65535, 131070) + [131070, 131072).
// CHECK-LABEL: func.func @fill_131072_outlined_vf_0
// CHECK: scf.for %{{.+}} = %{{.+}} to %c65535{{(_[0-9]+)?}}
// CHECK: scf.for %{{.+}} = %c65535{{(_[0-9]+)?}} to %c131070{{(_[0-9]+)?}}
// CHECK: scf.for %{{.+}} = %c131070{{(_[0-9]+)?}} to %c131072{{(_[0-9]+)?}}
// CHECK-NOT: func.func @fill_131072_outlined_vf_1

// Case 4: each source loop keeps its own VF (two chunks inside each);
// a cross-source merge into one VF would drop the vf_1 label entirely.
// CHECK-LABEL: func.func @fill_two_sources_outlined_vf_0
// CHECK: scf.for %{{.+}} = %{{.+}} to %c65535{{(_[0-9]+)?}}
// CHECK: scf.for %{{.+}} = %c65535{{(_[0-9]+)?}} to %c65536{{(_[0-9]+)?}}
// CHECK-LABEL: func.func @fill_two_sources_outlined_vf_1
// CHECK: scf.for %{{.+}} = %{{.+}} to %c65535{{(_[0-9]+)?}}
// CHECK: scf.for %{{.+}} = %c65535{{(_[0-9]+)?}} to %c65536{{(_[0-9]+)?}}
// CHECK-NOT: func.func @fill_two_sources_outlined_vf_2

// Case 5: one plain VF holding both chunks; a second VF (or an unsplit
// 65536-trip loop) would fail the label/bound pins below.
// CHECK-LABEL: func.func @plain_large_trip_outlined_vf_0
// CHECK: scf.for %{{.+}} = %{{.+}} to %c65535{{(_[0-9]+)?}}
// CHECK: scf.for %{{.+}} = %c65535{{(_[0-9]+)?}} to %c65536{{(_[0-9]+)?}}
// CHECK-NOT: func.func @plain_large_trip_outlined_vf_1

// Case 6: two chunks with the last bound exactly the original ub (262143);
// the overshooting lb + trip * step = 262145 may not appear as a bound.
// CHECK-LABEL: func.func @fill_lbx_step4_outlined_vf_0
// CHECK: scf.for %{{.+}} = %c1{{(_[0-9]+)?}} to %c262141{{(_[0-9]+)?}}
// CHECK: scf.for %{{.+}} = %c262141{{(_[0-9]+)?}} to %c262143{{(_[0-9]+)?}}
// CHECK-NOT: func.func @fill_lbx_step4_outlined_vf_1

// Case 7: two : i32 chunk loops inside ONE VF; bounds stay i32 (the call
// bounds remain integer constants inside the VF), never mixed with index.
// CHECK-LABEL: func.func @fill_i32_outlined_vf_0
// CHECK: arith.constant 65535 : i32
// CHECK: scf.for %{{.+}} step %c1_i32{{(_[0-9]+)?}} {{.*}} : i32
// CHECK: scf.for %{{.+}} step %c1_i32{{(_[0-9]+)?}} {{.*}} : i32
// CHECK-NOT: func.func @fill_i32_outlined_vf_1

// -----

// Existing VFs also need legalization, including unannotated nested loops.
// CHECK-LABEL: func.func @existing_vf
// CHECK: %[[FIRST:.*]] = scf.for {{.*}} to %c65535{{(_[0-9]+)?}} {{.*}}iter_args
// CHECK: %[[LAST:.*]] = scf.for {{.*}} = %c65535{{(_[0-9]+)?}} to %c65536{{(_[0-9]+)?}} {{.*}}iter_args({{.*}} = %[[FIRST]])
// CHECK: scf.yield %[[LAST]]
module {
  func.func @existing_vf(%init: i32) -> i32 attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c65536 = arith.constant 65536 : index
    %one = arith.constant 1 : i32
    %outer = scf.for %i = %c0 to %c1 step %c1 iter_args(%x = %init) -> i32 {
      %inner = scf.for %j = %c0 to %c65536 step %c1 iter_args(%y = %x) -> i32 {
        %next = arith.addi %y, %one : i32
        scf.yield %next : i32
      }
      scf.yield %inner : i32
    }
    return %outer : i32
  }
  func.func @existing_caller(%init: i32) -> i32 {
    %r = func.call @existing_vf(%init) : (i32) -> i32
    return %r : i32
  }
  // A scalar loop outside VF must remain untouched.
  // CHECK-LABEL: func.func @scalar_loop
  // CHECK: scf.for {{.*}} to %c65536{{(_[0-9]+)?}}
  // CHECK-NOT: scf.for
  func.func @scalar_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c65536 = arith.constant 65536 : index
    scf.for %i = %c0 to %c65536 step %c1 {
    }
    return
  }
}

// -----
// The signed bound difference itself exceeds INT64_MAX, but trip is 65536.
// CHECK-LABEL: func.func @wide_range
// CHECK: arith.constant 4611545280939032576 : index
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: scf.for
module {
  func.func @wide_caller() {
    func.call @wide_range() : () -> ()
    return
  }
  func.func @wide_range() attributes {hivm.vector_function, no_inline} {
    %lb = arith.constant -4611686018427387904 : index
    %ub = arith.constant 4611686018427387904 : index
    %step = arith.constant 140737488355328 : index
    scf.for %i = %lb to %ub step %step {
    }
    return
  }
}
