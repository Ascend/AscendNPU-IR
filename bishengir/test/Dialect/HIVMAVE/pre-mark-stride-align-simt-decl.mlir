// RUN: bishengir-opt -hivm-pre-mark-stride-align %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-NOT: annotation.mark
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @simt_scope_0(memref<8xi64, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline}

  func.func @main(%arg0: memref<8xi64, #hivm.address_space<gm>>) {
    %alloc = memref.alloc() : memref<8xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<8xi64, #hivm.address_space<gm>>) outs(%alloc : memref<8xi64, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    func.call @simt_scope_0(%alloc)
        : (memref<8xi64, #hivm.address_space<ub>>) -> ()
    return
  }
}
