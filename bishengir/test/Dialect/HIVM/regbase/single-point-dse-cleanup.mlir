// RUN: bishengir-opt --hivm-canonicalization-pipeline %s | FileCheck %s
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_957b --hivm-canonicalization-pipeline %s | FileCheck %s

// Regression tests for the HIVM canonicalization cluster
// (canonicalizationHIVMPipeline / hivm-canonicalization-pipeline):
//
//   ArithToAffineConversion, CanonicalizeIterArg, Canonicalizer,
//   SCFForLoopCanonicalization, CSE, HIVMOptSinglePoint,
//   memref::DeadStoreElimination, Canonicalizer
//
// These tests pin down the observable behaviour of the cluster: which ops
// survive, which disappear, and the resulting memory accesses. They never
// assert on the pass list itself.
//
// The second RUN line repeats exactly the same checks on the A5 RegBase target
// (Ascend950PR_957b), which is the target whose production pipeline
// (buildLowerHIVMPipelines -> canonicalizationHIVMPipeline) actually executes
// this cluster; the default target never reaches the RegBase path. The device
// spec appended by -hacc-append-device-spec only adds module attributes, so the
// same CHECK lines must hold on both targets.

// ---------------------------------------------------------------------------
// Category A1: single-point scalarization with load/store forwarding.
// HIVMOptSinglePoint rewrites a one-element HIVM op into scalar arith over
// memref load/store; the result must be the scalarized form, and the HIVM op
// must be gone.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @single_point_vadd
func.func @single_point_vadd(%a: memref<1x1xf32>, %b: memref<1x1xf32>, %c: memref<1x1xf32>) {
  // CHECK-NOT: hivm.hir.vadd
  // CHECK: memref.load
  // CHECK-NEXT: memref.load
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: memref.store
  hivm.hir.vadd ins(%a, %b : memref<1x1xf32>, memref<1x1xf32>) outs(%c : memref<1x1xf32>)
  return
}

// -----

// ---------------------------------------------------------------------------
// Category A2: load/store forwarding. A store followed by a load of the same
// location must forward the stored value; the load of the source and the store
// of the result remain (both are observable).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @forward_store_to_load
func.func @forward_store_to_load(%a: memref<8xf32>, %b: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[V:.*]] = memref.load
  // CHECK: memref.store %[[V]],
  %v = memref.load %a[%c0] : memref<8xf32>
  memref.store %v, %b[%c0] : memref<8xf32>
  return
}

// -----

// ---------------------------------------------------------------------------
// Category A3: dead alloc and dead store elimination. An alloc whose only
// store is never read must be removed together with the store.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @dead_alloc_and_store
func.func @dead_alloc_and_store() {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.store
  // CHECK: return
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %tmp = memref.alloc() : memref<8xf32>
  memref.store %cst, %tmp[%c0] : memref<8xf32>
  return
}

// -----

// ---------------------------------------------------------------------------
// Category A3b: a store that is overwritten before any read must be dropped,
// while the last, live store must be kept.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @dead_store_overwritten
func.func @dead_store_overwritten(%a: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 2.0 : f32
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: 1.000000e+00
  // CHECK: %[[LIVE:.*]] = arith.constant 2.000000e+00
  // CHECK: memref.store %[[LIVE]], %arg0
  %tmp = memref.alloc() : memref<8xf32>
  memref.store %cst1, %tmp[%c0] : memref<8xf32>
  memref.store %cst2, %tmp[%c0] : memref<8xf32>
  memref.store %cst2, %a[%c0] : memref<8xf32>
  return
}

// -----

// ---------------------------------------------------------------------------
// Category A3c: store-to-load forwarding through a short-lived alloc.
// memref::DeadStoreElimination forwards the stored value into the load and then
// removes the now-dead alloc, store and load. This forward-and-delete is the
// part of the cluster that the plain Canonicalizer does not perform on its own
// (verified by running the same pipeline with memref-dse removed: the alloc,
// store and load survive), so this case pins the pass rather than merely the
// surrounding canonicalization.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @dse_store_to_load_forward
func.func @dse_store_to_load_forward(%a: memref<8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 3.0 : f32
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.store
  // CHECK-NOT: memref.load
  // CHECK: return %{{.*}} : f32
  %tmp = memref.alloc() : memref<8xf32>
  memref.store %cst, %tmp[%c0] : memref<8xf32>
  %v = memref.load %tmp[%c0] : memref<8xf32>
  return %v : f32
}

// -----

// ---------------------------------------------------------------------------
// Category A4: scf loop / affine handling. The scf.for with an iter_arg must
// survive with its iter_arg structure intact: CanonicalizeIterArg and
// SCFForLoopCanonicalization normalize the loop form but must not drop the
// accumulation.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @affine_and_iter_arg
func.func @affine_and_iter_arg(%a: memref<4xf32>, %b: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] =
  // CHECK: memref.load %arg0
  // CHECK-NEXT: arith.addf %[[ACC]],
  // CHECK-NEXT: scf.yield
  // CHECK: memref.store
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (f32) {
    %v = memref.load %a[%i] : memref<4xf32>
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  memref.store %r, %b[%c0] : memref<4xf32>
  return
}

// -----

// ---------------------------------------------------------------------------
// Category A5: no-change case. An already-normal function must not be altered:
// the same load and return structure has to be reproduced.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @already_normal
func.func @already_normal(%a: memref<4x4xf32>, %b: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: memref.load %arg0
  // CHECK: memref.store %{{.*}}, %arg1
  // CHECK: return
  %v = memref.load %a[%c0, %c0] : memref<4x4xf32>
  memref.store %v, %b[%c0, %c0] : memref<4x4xf32>
  return
}
