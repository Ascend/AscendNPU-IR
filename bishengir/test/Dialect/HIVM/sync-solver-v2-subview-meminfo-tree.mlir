// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" %s | FileCheck %s

// V2 groups memory accesses in a MemInfoTree. Subviews of the same allocation
// must remain separate nodes so that their individual ranges are preserved.
module {
  func.func @distinct_subviews_in_meminfo_tree(
      %arg0: memref<32x256xf32, #hivm.address_space<gm>>,
      %arg1: memref<32x256xf32, #hivm.address_space<gm>>,
      %arg2: memref<32x256xf32, #hivm.address_space<gm>>,
      %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<32x512xf32, #hivm.address_space<ub>>
    %left = memref.subview %ub[0, 0] [32, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    %right = memref.subview %ub[0, 256] [32, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<32x256xf32, strided<[512, 1], offset: 256>, #hivm.address_space<ub>>
    scf.if %cond {
      hivm.hir.load ins(%arg0 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%left : memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.load ins(%arg1 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%right : memref<32x256xf32, strided<[512, 1], offset: 256>, #hivm.address_space<ub>>)
    }
    // The first access in the if must not hide the conflicting access to
    // %right when V2 summarizes the if in its MemInfoTree.
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    // CHECK-NEXT: hivm.hir.load
    hivm.hir.load ins(%arg2 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%right : memref<32x256xf32, strided<[512, 1], offset: 256>, #hivm.address_space<ub>>)
    return
  }
}
