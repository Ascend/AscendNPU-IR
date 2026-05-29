// RUN: bishengir-opt -hivm-graph-sync-solver -split-input-file %s | FileCheck %s

module {
  func.func private @simple_indirect_load_kernel_scope_0(
      memref<?xi64, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>},
      memref<8xi64, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<read>},
      memref<?xf32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>},
      i32,
      memref<8xf32, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<write>}
    ) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.vf_mode = #hivm.vf_mode<SIMT>,
                  no_inline, outline}

  // CHECK-LABEL: func.func @simple_indirect_load_kernel
  func.func @simple_indirect_load_kernel(
      %arg_indices_gm: memref<?xi64, #hivm.address_space<gm>>,
      %arg_data_gm:    memref<?xf32, #hivm.address_space<gm>>,
      %arg_out_gm:     memref<?xf32, #hivm.address_space<gm>>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %idx_ub = hivm.hir.pointer_cast(%c0_i64) : memref<8xi64, #hivm.address_space<ub>>
    %out_ub = hivm.hir.pointer_cast(%c8192_i64) : memref<8xf32, #hivm.address_space<ub>>
    %idx_strided = memref.reinterpret_cast %arg_indices_gm to offset: [0], sizes: [8], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<8xi64, strided<[1]>, #hivm.address_space<gm>>
    %idx_cast = memref.cast %idx_strided : memref<8xi64, strided<[1]>, #hivm.address_space<gm>> to memref<8xi64, #hivm.address_space<gm>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%idx_cast : memref<8xi64, #hivm.address_space<gm>>)
                  outs(%idx_ub : memref<8xi64, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: call @simple_indirect_load_kernel_scope_0
    call @simple_indirect_load_kernel_scope_0(%arg_indices_gm, %idx_ub, %arg_data_gm, %c1_i32, %out_ub)
        : (memref<?xi64, #hivm.address_space<gm>>,
           memref<8xi64, #hivm.address_space<ub>>,
           memref<?xf32, #hivm.address_space<gm>>,
           i32,
           memref<8xf32, #hivm.address_space<ub>>) -> ()
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %out_strided = memref.reinterpret_cast %arg_out_gm to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
    %out_cast = memref.cast %out_strided : memref<8xf32, strided<[1]>, #hivm.address_space<gm>> to memref<8xf32, #hivm.address_space<gm>>
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%out_ub : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%out_cast : memref<8xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func private @simple_indirect_store_kernel_scope_0(
      memref<8xf32, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<read>},
      memref<8xi64, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<read>},
      memref<?xf32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<write>},
      i32
    ) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.vf_mode = #hivm.vf_mode<SIMT>,
                  no_inline, outline}

  // CHECK-LABEL: func.func @simple_indirect_store_kernel
  func.func @simple_indirect_store_kernel(
      %arg_val_gm:     memref<?xf32, #hivm.address_space<gm>>,
      %arg_indices_gm: memref<?xi64, #hivm.address_space<gm>>,
      %arg_out_gm:     memref<?xf32, #hivm.address_space<gm>>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %val_ub = hivm.hir.pointer_cast(%c0_i64) : memref<8xf32, #hivm.address_space<ub>>
    %idx_ub = hivm.hir.pointer_cast(%c8192_i64) : memref<8xi64, #hivm.address_space<ub>>
    %val_strided = memref.reinterpret_cast %arg_val_gm to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
    %val_cast = memref.cast %val_strided : memref<8xf32, strided<[1]>, #hivm.address_space<gm>> to memref<8xf32, #hivm.address_space<gm>>
    %idx_strided = memref.reinterpret_cast %arg_indices_gm to offset: [0], sizes: [8], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<8xi64, strided<[1]>, #hivm.address_space<gm>>
    %idx_cast = memref.cast %idx_strided : memref<8xi64, strided<[1]>, #hivm.address_space<gm>> to memref<8xi64, #hivm.address_space<gm>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%val_cast : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%val_ub : memref<8xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%idx_cast : memref<8xi64, #hivm.address_space<gm>>)
                  outs(%idx_ub : memref<8xi64, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: call @simple_indirect_store_kernel_scope_0
    call @simple_indirect_store_kernel_scope_0(%val_ub, %idx_ub, %arg_out_gm, %c1_i32)
        : (memref<8xf32, #hivm.address_space<ub>>,
           memref<8xi64, #hivm.address_space<ub>>,
           memref<?xf32, #hivm.address_space<gm>>,
           i32) -> ()
    return
  }
}
