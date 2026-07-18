// RUN: bishengir-opt %s                                                       \
// RUN:   -pass-pipeline="builtin.module(                                      \
// RUN:     hacc-append-device-spec{target=Ascend910B1},                       \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true}),               \
// RUN:     func.func(hivm-plan-memory,hivm-graph-sync-solver,                  \
// RUN:               hivm-enable-multi-buffer,hivm-lower-multi-buffer-counter))" \
// RUN:   -split-input-file                                                    \
// RUN:   | FileCheck %s

// -----
// Regression: the alloca-based counter scheme MUST NOT alter the scf.while
// op's signature.  We use a non-trivial signature `(i1, i32, i64) -> (i1,
// i32, i64)` and explicitly CHECK that:
//   * The exact result-type list shows up unchanged in the output.
//   * No fourth iter_arg or fourth result was injected (CHECK-NOT
//     ", i64, i64)" rules out an i64 counter being appended).
//   * The before-region keeps yielding exactly 3 values via scf.condition.
//
// This locks in the key invariant of the alloca-based design vs. the
// previously-considered iter_args-based alternative.

// CHECK-LABEL: func.func @while_signature_immutability(
func.func @while_signature_immutability(%arg0: memref<8xf32, #hivm.address_space<gm>>,
                                        %arg1: memref<8xf32, #hivm.address_space<gm>>,
                                        %arg2: i32,
                                        %arg3: i64) {
  %true = arith.constant true

  // CHECK: scf.while ({{.*}}) : (i1, i32, i64) -> (i1, i32, i64) {
  // CHECK-NOT: (i1, i32, i64, i64) -> (i1, i32, i64, i64)
  // CHECK-NOT: (i1, i32, i64, i32, i64) -> (i1, i32, i64, i32, i64)
  %r:3 = scf.while (%c0_i1 = %true, %c0_i32 = %arg2, %c0_i64 = %arg3) :
      (i1, i32, i64) -> (i1, i32, i64) {
    // CHECK: scf.condition(%{{.*}}) %{{.*}}, %{{.*}}, %{{.*}} : i1, i32, i64
    scf.condition(%c0_i1) %c0_i1, %c0_i32, %c0_i64 : i1, i32, i64
  } do {
  ^bb0(%c1_i1: i1, %c1_i32: i32, %c1_i64: i64):
    %tmp = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%tmp : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<8xf32, #hivm.address_space<gm>>)
    // CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}} : i1, i32, i64
    scf.yield %c1_i1, %c1_i32, %c1_i64 : i1, i32, i64
  }
  return
}
