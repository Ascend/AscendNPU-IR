// RUN: bishengir-opt -hivm-graph-sync-solver -split-input-file %s | FileCheck %s

// A pointer_cast result that is loop-carried and yielded by an scf.while used
// to abort getParentLoop (WhileOp::getLoopResults() is std::nullopt). The
// solver must now run to completion.

module {
// CHECK-LABEL: func.func @while_yielded_pointer_cast(
// CHECK: scf.while
  func.func @while_yielded_pointer_cast(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                        %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %true = arith.constant true

    %init = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>

    %r = scf.while (%it = %init) : (memref<16xf16, #hivm.address_space<ub>>) -> (memref<16xf16, #hivm.address_space<ub>>) {
      scf.condition(%true) %it : memref<16xf16, #hivm.address_space<ub>>
    } do {
    ^bb0(%a: memref<16xf16, #hivm.address_space<ub>>):
      %buf = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%buf : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      scf.yield %buf : memref<16xf16, #hivm.address_space<ub>>
    }
    return
  }
}
