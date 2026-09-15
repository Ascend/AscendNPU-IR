// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v1 -split-input-file %s | FileCheck %s
// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v2 -split-input-file %s | FileCheck %s

// A multi_buffer pointer_cast lives in the inner for, is yielded out, and the
// outer for consumes it. getParentLoop anchors the double buffer on the outer
// loop, so the wait/set flags around the buffer rotate with a counter-driven
// (SSA) event id instead of a static EVENT_ID*.

module {
// CHECK-LABEL: func.func @test_inner_yielded_db_anchored_outer(
  func.func @test_inner_yielded_db_anchored_outer(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    %init = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>

    scf.for %arg2 = %c0 to %c4 step %c2 {
      %res = scf.for %arg3 = %c0 to %c4 step %c2 iter_args(%arg4 = %init) -> (memref<16xf16, #hivm.address_space<ub>>) {
        %buf = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
        annotation.mark %buf {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
        // Double buffer: rotated (SSA) event id, not a static EVENT_ID*.
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
        hivm.hir.store ins(%buf : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
        scf.yield %buf : memref<16xf16, #hivm.address_space<ub>>
      }
      hivm.hir.store ins(%res : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----

// A dynamically bounded loop may execute zero times, in which case its result
// aliases the corresponding init argument rather than the yielded value.

module {
// CHECK-LABEL: func.func @test_zero_trip_result_aliases_init(
  func.func @test_zero_trip_result_aliases_init(%ub: index, %arg0: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
    %yielded = hivm.hir.pointer_cast(%c32_i64) : memref<16xf16, #hivm.address_space<ub>>

    %res = scf.for %iv = %c0 to %ub step %c1 iter_args(%iter = %init) -> (memref<16xf16, #hivm.address_space<ub>>) {
      // Keep %iter unused: traceback starts from the loop result, so it must
      // find %init without relying on the region block-argument path.
      scf.yield %yielded : memref<16xf16, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <[[EVENT:EVENT_ID[0-7]]]>]
    hivm.hir.store ins(%res : memref<16xf16, #hivm.address_space<ub>>) outs(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <[[EVENT]]>]
    hivm.hir.vadd ins(%yielded, %yielded : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%init : memref<16xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A statically non-empty loop result cannot alias its init argument. Do not
// introduce a false MTE3-to-V conflict between %init and the loop result.

module {
// CHECK-LABEL: func.func @test_nonzero_trip_result_excludes_init(
  func.func @test_nonzero_trip_result_excludes_init(%arg0: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
    %yielded = hivm.hir.pointer_cast(%c32_i64) : memref<16xf16, #hivm.address_space<ub>>

    %res = scf.for %iv = %c0 to %c1 step %c1 iter_args(%iter = %init) -> (memref<16xf16, #hivm.address_space<ub>>) {
      scf.yield %yielded : memref<16xf16, #hivm.address_space<ub>>
    }
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%res : memref<16xf16, #hivm.address_space<ub>>) outs(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
    // CHECK-NOT: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>
    // CHECK: hivm.hir.vadd
    hivm.hir.vadd ins(%yielded, %yielded : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%init : memref<16xf16, #hivm.address_space<ub>>)
    return
  }
}
