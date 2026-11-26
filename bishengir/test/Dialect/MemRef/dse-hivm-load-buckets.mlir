// REQUIRES: system-linux
// RUN: bishengir-opt --memref-dse %s | FileCheck %s
// RUN: env MALLOC_PERTURB_=165 bishengir-opt --memref-dse %s > /dev/null
//
// Regression test for a use-after-free in MemrefDeadStoreElimination.
//
// ValueDependencyAnalyzer::OpsOfAlloc records raw Operation* (one bucket per
// value).  A single hivm.load appears in *several* buckets: the source and the
// destination allocation, plus any other operand.  eraseUnusedLoad() destroys
// the load while only popping it from the bucket it is currently walking, so
// the buckets visited later still hold a dangling pointer and dyn_cast on it
// reads freed memory.
//
// The destination allocation is declared first on purpose: buckets are visited
// in value order, so the destination bucket must be visited (and the load
// erased) before the source bucket is scanned.
//
// The second RUN line poisons freed heap memory on glibc (MALLOC_PERTURB_ fills
// freed chunks with 0xa5), which turns the use-after-free into a deterministic
// crash instead of an intermittent one.

// CHECK-LABEL:   func.func @dse_dst_allocated_first
// CHECK-NOT:     hivm.hir.load
// CHECK:         return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dse_dst_allocated_first() {
    %t = memref.alloc() : memref<16xf32>
    %s = memref.alloc() : memref<16xf32>
    hivm.hir.load ins(%s : memref<16xf32>) outs(%t : memref<16xf32>)
    return
  }
}
