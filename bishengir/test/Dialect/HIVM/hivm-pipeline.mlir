// RUN: bishengir-opt -optimize-hivm-pipeline %s | FileCheck %s
// RUN: bishengir-opt -optimize-hivm-pipeline="enable-hivm-inject-barrier-all-sync=true" %s | FileCheck %s -check-prefix=CHECK-BAR-ALL

module {
// CHECK: hivm.hir.set_mask_norm

// CHECK-BAR-ALL-NOT: hivm.hir.set_flag
// CHECK-BAR-ALL-NOT: hivm.hir.wait_flag
  func.func @test(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>, %arg2: memref<16xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
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
