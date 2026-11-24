// clang-format off
// RUN: bishengir-opt %s -hivm-enable-multi-buffer -hivm-lower-multi-buffer-counter -split-input-file | FileCheck %s

// A marked pointer_cast inside an scf.if has a definition block that counts
// guard entries, not loop iterations.
// No counter owned there => the pair keeps the loop counter:
// handing the definition block to the adapter would open a counter inside the
// guard, so the slot would rotate per entry while the flags (inject-sync path)
// rotate per iteration.
//
// Assertions: counter load, slot select and increment stay in the loop body;
// the guard holds only the uses.

module {
  // CHECK-LABEL: func.func @guarded_cast(
  func.func @guarded_cast(%cond: i1,
                          %arg0: memref<16xf16, #hivm.address_space<gm>>,
                          %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    // counter alloca: function scope, initialised before the loop.
    // CHECK-DAG: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
    // CHECK-DAG: memref.store %{{.*}}, %[[CTR]]
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: scf.for
      // CHECK-NOT: scf.if
      // CHECK: %[[CUR:.*]] = memref.load %[[CTR]]
      // slot index -> from the loop-level load, not from a counter in the guard
      // => both selects precede the guard.
      // CHECK: arith.remui %[[CUR]], %{{.*}} : i64
      // CHECK: arith.select {{.*}} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: arith.select {{.*}} : memref<16xf16, #hivm.address_space<ub>>
      scf.if %cond {
        // CHECK: scf.if
        %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
        annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
        %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>,
                                memref<16xf16, #hivm.address_space<ub>>)
                      outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      }
      // CHECK: arith.addi %[[CUR]], %{{.*}} : i64
      // CHECK: memref.store {{.*}} : memref<1xi64>
    }
    return
  }
}
// clang-format on
