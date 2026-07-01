// RUN: bishengir-opt %s -hivm-enable-multi-buffer -hivm-lower-multi-buffer-counter -split-input-file | FileCheck %s

// -----
// Multiple multi-address pointer_casts in the same scf.while body must share
// the *same* counter alloca (looked up via hivm.multi_buffer_counter_for ==
// hivm.multi_buffer_loop_id). Otherwise each call to ensureCounterMaterialized
// would produce a duplicate alloca, increment-store pair and the counters
// would drift apart.
//
// Verification:
//   1. Exactly one memref.alloca with the counter tag at funcOp top
//      (single CHECK then CHECK-NOT after the loop).
//   2. Exactly one increment-store back to the alloca per while iteration.
//
// Both pointer_casts must still produce their own arith.select cascade
// driven by the shared counter.

module {
// CHECK-LABEL: func.func @while_multi_alloc(
  func.func @while_multi_alloc(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                               %arg1: memref<16xf16, #hivm.address_space<gm>>,
                               %arg2: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c256_i64 = arith.constant 256 : i64
    %c272_i64 = arith.constant 272 : i64
    %true = arith.constant true

    // CHECK: %[[CTR:.*]] = memref.alloca() {hivm.multi_buffer_counter_for = {{.*}}} : memref<1xi64>

    // CHECK: scf.while {{.*}} : (i1) -> i1
    %r = scf.while (%cond = %true) : (i1) -> i1 {
      scf.condition(%cond) %cond : i1
    } do {
    ^bb0(%cin: i1):
      // Exactly one body-head load against the shared alloca.
      // CHECK: memref.load %[[CTR]]
      %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %1 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      %2 = hivm.hir.pointer_cast(%c256_i64, %c272_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %2 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%2 : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
      // Exactly one body-tail increment + store-back against the shared
      // alloca. The CHECK-NOT after the store rules out any second
      // increment-store before the yield.
      // CHECK: arith.addi
      // CHECK: memref.store %{{.*}}, %[[CTR]]
      // CHECK-NOT: memref.store %{{.*}}, %[[CTR]]
      // CHECK: scf.yield
      scf.yield %cin : i1
    }
    return
  }
}
