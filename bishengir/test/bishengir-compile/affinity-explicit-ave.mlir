// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --disable-hivm-auto-inject-sync=true --disable-ffts --save-temps=%t -o %t.o
// RUN: test -s %t.o
// RUN: FileCheck %s --input-file=%t/module.hivm.opt.mlir --implicit-check-not=vexpdif --implicit-check-not=ave.hir. --implicit-check-not=vector.transfer
//
// Explicit AVE must compile without layout search and preserve the chosen
// instruction sequence and VL16/PB32 computation/store predicate.
// The function's no_inline marker keeps the VF outlined through InlineScope's
// inliner, without duplicating the marker on the call.
// CHECK-LABEL: func.func @explicit_ave_kernel_scope_0(
// CHECK-SAME: no_inline
// CHECK: %[[VL16:.*]] = llvm.mlir.constant(6 : i32)
// CHECK: %[[PRED:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"(%[[VL16]], {{.*}}) {mask_bit_width = 32 : i32}
// CHECK: %[[INPUT:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"
// CHECK: %[[DIFF:.*]] = "hivm_regbaseintrins.intr.hivm.vsub.s.x"(%[[INPUT]], %[[INPUT]], %[[PRED]])
// CHECK: %[[EXP:.*]] = "hivm_regbaseintrins.intr.hivm.vexp.x"(%[[DIFF]], %[[PRED]])
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%[[EXP]], {{.*}}, %[[PRED]])
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
// CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
// CHECK-NEXT: call @explicit_ave_kernel_scope_0({{.*}}) {hivm.vector_function}
// CHECK-NEXT: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID5>]
// CHECK-NEXT: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID5>]
// CHECK: call @store_ubuf_to_gm_1d_float(
// CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// CHECK-NEXT: return

#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>

module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @explicit_ave_kernel(
      %sync: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
      %workspace: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
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
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
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

      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]

      scope.scope : () -> () {
        %c0 = arith.constant 0 : index
        %mask = ave.hir.pge <VL16>
            {functionType = #ave.func_dist_type<pb32>}
            : vector<64xi1, #b32>
        %inputVec = ave.hir.vload <NORM> %inputStorage[%c0]
            {functionType = #ave.func_dist_type<norm>}
            : memref<64xf32, #hivm.address_space<ub>> into
              vector<64xf32, #b32>
        // Keep the user-selected SUB then EXP sequence; the standard AVE
        // pipeline would fuse this into EXPDIF.
        %difference = ave.hir.vsub %inputVec, %inputVec, %mask
            : vector<64xf32, #b32>, vector<64xi1, #b32>
        %outputVec = ave.hir.vexp %difference, %mask
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

      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID5>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID5>]

      hivm.hir.store
          ins(%outputUB : memref<16xf32, #hivm.address_space<ub>>)
          outs(%outputGM : memref<16xf32, strided<[1]>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    return
  }
}
