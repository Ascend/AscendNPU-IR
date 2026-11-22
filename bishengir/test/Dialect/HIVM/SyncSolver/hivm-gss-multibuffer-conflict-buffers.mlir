// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" -split-input-file %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" -split-input-file %s | FileCheck %s
// clang-format off

// Only the buffers the two ends of a handshake conflict on pick the counter
// block.
// A scratchpad that one end merely reads takes no part in the rotation, so
// counting it would pull the common ancestor up to the loop body and drop the
// guard counter the handshake is entitled to.

// -----
module {
  // CHECK-LABEL: func.func @scratchpad_outside_guard_keeps_guard_counter(
  func.func @scratchpad_outside_guard_keeps_guard_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>,
      %arg2: memref<16xf16, #hivm.address_space<gm>>,
      %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %c96_i64 = arith.constant 96 : i64
    %c128_i64 = arith.constant 128 : i64
    %c160_i64 = arith.constant 160 : i64
    %c192_i64 = arith.constant 192 : i64
    %c224_i64 = arith.constant 224 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    scf.for %iv = %c0 to %c8 step %c1 {
      %shared = hivm.hir.pointer_cast(%c64_i64, %c96_i64) : memref<16xf16, #hivm.address_space<ub>>
      %scratch = hivm.hir.pointer_cast(%c128_i64, %c160_i64) : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%shared : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%shared, %shared : memref<16xf16, #hivm.address_space<ub>>,
                                memref<16xf16, #hivm.address_space<ub>>)
                    outs(%scratch : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: scf.if
      scf.if %cond {
        // CHECK: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
        %buf = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
        annotation.mark %buf {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
        %out = hivm.hir.pointer_cast(%c192_i64, %c224_i64) : memref<16xf16, #hivm.address_space<ub>>
        // CHECK: %[[MOD:.*]] = arith.remui %[[CTR]], %{{.*}} : i64
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%buf, %shared : memref<16xf16, #hivm.address_space<ub>>,
                                        memref<16xf16, #hivm.address_space<ub>>)
                      outs(%out : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.store ins(%out : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      }
      // CHECK: return
    }
    return
  }
}
// clang-format on
