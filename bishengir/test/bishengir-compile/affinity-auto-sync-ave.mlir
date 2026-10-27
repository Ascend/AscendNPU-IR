// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --save-temps=%t -o %t.o
// RUN: FileCheck %s --input-file=%t/module.hivm.opt.mlir --implicit-check-not=global_kernel
//
// No intra-core synchronization is provided in the input. Affinity compilation
// must generate the DMA/VF dependencies and the completion barrier by default.
// Module, entry-function and VF core types are explicitly provided by the input.
// The Triton global_kernel marker must become HACC entry and argument attributes
// before the launch ABI is adapted from ten arguments to seven.
// CHECK: hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @auto_sync_ave_kernel(
// CHECK-SAME: hacc.arg_type = #hacc.arg_type<sync_block_lock>
// CHECK-SAME: hacc.arg_type = #hacc.arg_type<workspace>
// CHECK-SAME: func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>
// CHECK-SAME: hacc.entry
// CHECK-SAME: hacc.function_kind = #hacc.function_kind<DEVICE>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK: call @load_gm_to_ubuf_1d_float(
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
// CHECK: call @auto_sync_ave_kernel_scope_0({{.*}}) {hivm.vector_function}
// CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: call @store_ubuf_to_gm_1d_float(
// CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// CHECK-NEXT: return

#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>

module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @auto_sync_ave_kernel(
      %sync: memref<?xi8>,
      %workspace: memref<?xi8>,
      %inputBase: memref<?xf32> {
        tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32
      },
      %outputBase: memref<?xf32> {
        tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32
      },
      %gridX: i32, %gridY: i32, %gridZ: i32,
      %programIdX: i32, %programIdY: i32, %programIdZ: i32)
      attributes {
        SyncBlockLockArgIdx = 0 : i64,
        WorkspaceArgIdx = 1 : i64,
        global_kernel = "local",
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vf_mode = #hivm.vf_mode<SIMD>,
        mix_mode = "aiv",
        parallel_mode = "simd"
      } {
    scope.scope : () -> () {
      %inputGM = memref.reinterpret_cast %inputBase to
          offset: [0], sizes: [16], strides: [1]
          : memref<?xf32> to memref<16xf32, strided<[1]>>
      %outputGM = memref.reinterpret_cast %outputBase to
          offset: [0], sizes: [16], strides: [1]
          : memref<?xf32> to memref<16xf32, strided<[1]>>
      %inputStorage = memref.alloc()
          : memref<64xf32, #hivm.address_space<ub>>
      %outputStorage = memref.alloc()
          : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %inputStorage {mem_unique}
          : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %outputStorage {mem_unique}
          : memref<64xf32, #hivm.address_space<ub>>
      %inputUB = memref.subview %inputStorage[0] [16] [1]
          : memref<64xf32, #hivm.address_space<ub>> to
            memref<16xf32, #hivm.address_space<ub>>
      %outputUB = memref.subview %outputStorage[0] [16] [1]
          : memref<64xf32, #hivm.address_space<ub>> to
            memref<16xf32, #hivm.address_space<ub>>

      hivm.hir.load
          ins(%inputGM : memref<16xf32, strided<[1]>>)
          outs(%inputUB : memref<16xf32, #hivm.address_space<ub>>)
          eviction_policy = <EvictFirst> core_type = <VECTOR>

      scope.scope : () -> () {
        %c0 = arith.constant 0 : index
        %mask = ave.hir.pge <VL16>
            {functionType = #ave.func_dist_type<pb32>}
            : vector<64xi1, #b32>
        %inputVec = ave.hir.vload <NORM> %inputStorage[%c0]
            {functionType = #ave.func_dist_type<norm>}
            : memref<64xf32, #hivm.address_space<ub>> into
              vector<64xf32, #b32>
        %outputVec = ave.hir.vexp %inputVec, %mask
            : vector<64xf32, #b32>, vector<64xi1, #b32>
        ave.hir.masked_store <NORM_B32> %outputStorage[%c0], %mask, %outputVec
            {functionType = #ave.func_dist_type<norm>}
            : memref<64xf32, #hivm.address_space<ub>>,
              vector<64xi1, #b32>, vector<64xf32, #b32>
        scope.return
      } {
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vector_function,
        hivm.vf_mode = #hivm.vf_mode<SIMD>,
        no_inline,
        outline
      }

      hivm.hir.store
          ins(%outputUB : memref<16xf32, #hivm.address_space<ub>>)
          outs(%outputGM : memref<16xf32, strided<[1]>>)
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    return
  }
}
