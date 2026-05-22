// RUN: bishengir-opt %s --convert-hivm-to-std | FileCheck %s

// CHECK-LABEL: triton_cos
// CHECK: call @cos_fp32
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_cos(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<MIX>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1024_i32 = arith.constant 1024 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.remsi %1, %arg4 : i32
    %3 = arith.muli %2, %c1024_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%5 : memref<1024xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %6 = hivm.hir.pointer_cast(%c4096_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%6 : memref<1024xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.custom {arg_attrs = [], bitcode = "/usr/local/python3.11.13/lib/python3.11/site-packages/triton/backends/ascend/lib/libdevice.10.bc", hivm.pipe = #hivm.pipe<PIPE_MTE3>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>, symbol = "cos_fp32"} "my_custom_op" ins(%5 : memref<1024xf32, #hivm.address_space<ub>>) outs(%6 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%6 : memref<1024xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
}
