// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v1 %s | FileCheck %s
// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v2 %s | FileCheck %s

// The atomic custom op reads and updates its first GM input. The solver must
// synchronize the preceding GM load with the atomic add and the following GM
// load with the atomic add before the final store.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @indirect_atomic_add_read_modify_write
  func.func @indirect_atomic_add_read_modify_write(
      %src: memref<16xf32, #hivm.address_space<gm>>,
      %atomic_dst: memref<16xf32, #hivm.address_space<gm>>,
      %offsets: memref<16xi64, #hivm.address_space<ub>>,
      %mask: memref<16xi8, #hivm.address_space<ub>>,
      %result: memref<16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry,
                  hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i64 = arith.constant 0 : i64
    %value = hivm.hir.pointer_cast(%c0_i64) :
      memref<16xf32, #hivm.address_space<ub>>
    %atomic_out = hivm.hir.pointer_cast(%c0_i64) :
      memref<16xf32, #hivm.address_space<ub>>
    %reloaded = hivm.hir.pointer_cast(%c0_i64) :
      memref<16xf32, #hivm.address_space<ub>>

    // CHECK: hivm.hir.load
    hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>)
      outs(%value : memref<16xf32, #hivm.address_space<ub>>)

    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID{{[0-7]}}>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID{{[0-7]}}>]

    // CHECK: hivm.hir.custom
    // CHECK-SAME: gm_addr_args_indices = array<i32: 0>
    // CHECK-SAME: "__builtin_indirect_atomic"
    hivm.hir.custom {extra_attr = "operate=add",
                     gm_addr_args_indices = array<i32: 0>,
                     hivm.pipe = #hivm.pipe<PIPE_V>,
                     hivm.tcore_type = #hivm.tcore_type<VECTOR>,
                     hivm.vf_mode = #hivm.vf_mode<SIMT>,
                     symbol = "__builtin_indirect_atomic"}
        "__builtin_indirect_atomic"
        ins(%atomic_dst, %offsets, %value, %mask :
            memref<16xf32, #hivm.address_space<gm>>,
            memref<16xi64, #hivm.address_space<ub>>,
            memref<16xf32, #hivm.address_space<ub>>,
            memref<16xi8, #hivm.address_space<ub>>)
        outs(%atomic_out : memref<16xf32, #hivm.address_space<ub>>)

    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID{{[0-7]}}>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID{{[0-7]}}>]

    hivm.hir.load ins(%atomic_dst : memref<16xf32, #hivm.address_space<gm>>)
      outs(%reloaded : memref<16xf32, #hivm.address_space<ub>>)

    hivm.hir.store ins(%reloaded : memref<16xf32, #hivm.address_space<ub>>)
      outs(%result : memref<16xf32, #hivm.address_space<gm>>)
    return
  }
}
