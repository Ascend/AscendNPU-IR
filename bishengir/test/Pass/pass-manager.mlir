// REQUIRES: hivmc-a5
// REQUIRES: enable-lir-compile

// RUN: bishengir-compile -enable-lir-compile=false \
// RUN:   -enable-hivm-graph-sync-solver=true -mlir-print-ir-after='hivm-graph-sync-solver' \
// RUN:   -mlir-print-ir-before='hfusion-auto-schedule' %s 2>&1 | FileCheck %s
// CHECK: IR Dump After GraphSyncSolver (hivm-graph-sync-solver)
// CHECK-NOT: IR Dump After AutoSchedule (hfusion-auto-schedule)

// RUN: bishengir-compile -enable-lir-compile=false \
// RUN:   -enable-hivm-delayed-cross-core-gss=true -mlir-print-ir-after=hivm-delayed-cross-core-gss %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CROSS-CORE
// CROSS-CORE: IR Dump After DelayedCrossCoreGSS (hivm-delayed-cross-core-gss)

module {
  func.func @foo(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>, %arg2: memref<16xf16, #hivm.address_space<gm>>) attributes {hacc.entry} {
    %alloc = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16xf16, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xf16, #hivm.address_space<ub>>)
    %alloc_1 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
    return
  }
}
