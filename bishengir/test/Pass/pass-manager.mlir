// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-lir-compile=false \
// RUN:   -bishengir-print-ir-after='hivm-inject-sync' \
// RUN:   -bishengir-print-ir-before='hfusion-auto-schedule' %s 2>&1 | FileCheck %s

// CHECK: IR Dump After InjectSync (hivm-inject-sync)
// CHECK-NOT: IR Dump After AutoSchedule (hfusion-auto-schedule)
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
