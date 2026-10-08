// RUN: bishengir-opt %s -hivm-enable-stride-align | FileCheck %s

// Minimal reproduction extracted from the original bisheng compile log
// (state right before hivm-enable-stride-align in the A5 regbase pipeline,
// stride-align multibuffer dot/where case; stride-align marks already in place).
// Before the strip-align-marks-deep fix, the plain-init fallback of
// propagateScfForYieldOp left a dense yield behind while the outer-if branch
// allocs were still aligned: the loop type was switched to strided via its
// init and the mismatch materialized as a redundant strided->dense->strided
// copy round trip plus an unpadded strided UB alloc (994B span, counted as
// 64B by plan memory). With the fix the fallback gives up loop-carried
// The load-bearing invariant: no unpadded strided-layout UB alloc may survive
// (its 994B physical span is counted as 64B by plan memory and lets neighbors
// be packed inside the view extent). The materialization must use a padded
// dense alloc + subview instead. This invariant holds regardless of the
// greedy rewrite order between the inner-if and loop-boundary unifications.

// alignment for real: the outer-if branch allocs stay dense and the loop
// tail keeps a single strided->dense copy.

// CHECK-LABEL: func.func @calc_cube_vector_mix_aiv(
// CHECK: memref.alloc() : memref<2x2x2x1x2x2x32x1x1xi8, #hivm.address_space<ub>>
// CHECK: memref<2x2x2x1x2x2x2x1xi8, strided<[512, 256, 128, 128, 64, 32, 1, 1]>, #hivm.address_space<ub>>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
  func.func @calc_cube_vector_mix_aiv_fused_0_outlined_vf_0(%arg0: memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>, %arg1: memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>, %arg2: memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>, %arg3: memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x1x1x1x1x256xi8>
    %c0_i8 = arith.constant 0 : i8
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 1, 1, 1, 1] : vector<1x1x1x1x1x1x1x256xi1>
    scf.for %arg4 = %c0 to %c2 step %c1 {
      scf.for %arg5 = %c0 to %c2 step %c1 {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            scf.for %arg8 = %c0 to %c2 step %c1 {
              scf.for %arg9 = %c0 to %c2 step %c1 {
                %subview = memref.subview %arg0[%arg4, %arg5, %arg6, 0, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>> to memref<1x1x1x1x1x1x1x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>
                %subview_0 = memref.subview %arg1[%arg4, %arg5, %arg6, 0, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>> to memref<1x1x1x1x1x1x1x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>
                %subview_1 = memref.subview %arg2[%arg4, %arg5, %arg6, 0, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>> to memref<1x1x1x1x1x1x1x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>
                %subview_2 = memref.subview %arg3[%arg4, %arg5, %arg6, 0, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>> to memref<1x1x1x1x1x1x1x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>
                %1 = vector.transfer_read %subview[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %c0_i8, %0 {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x1x1x1x1x1x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x1x1x256xi8>
                %2 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %c0_i8, %0 {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x1x1x1x1x1x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x1x1x256xi8>
                %3 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %c0_i8, %0 {in_bounds = [true, true, true, true, true, true, true, true]} : memref<1x1x1x1x1x1x1x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x1x1x256xi8>
                %4 = arith.cmpi ne, %1, %cst : vector<1x1x1x1x1x1x1x256xi8>
                %5 = arith.select %4, %2, %3 : vector<1x1x1x1x1x1x1x256xi1>, vector<1x1x1x1x1x1x1x256xi8>
                vector.transfer_write %5, %subview_2[%c0, %c0, %c0, %c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x1x256xi8>, memref<1x1x1x1x1x1x1x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>
              }
            }
          }
        }
      }
    }
    return
  }
func.func @calc_cube_vector_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: memref<?xi8, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xi8, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg8: memref<?xi8, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xi8, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c18_i32 = arith.constant 18 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  hivm.hir.anchor {id = 0 : i64}
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [2, 2, 2, 1, 2, 2, 2, 1], strides: [32, 16, 8, 8, 4, 2, 1, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [2, 2, 2, 1, 2, 2, 2, 1], strides: [32, 16, 8, 8, 4, 2, 1, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [2, 2, 2, 1, 2, 2, 2, 1], strides: [32, 16, 8, 8, 4, 2, 1, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [2, 2, 2, 1, 2, 2, 2, 1], strides: [32, 16, 8, 8, 4, 2, 1, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>
  hivm.hir.anchor {id = 1 : i64}
  %2 = hivm.hir.get_sub_block_idx -> i64
  %3 = arith.index_cast %2 : i64 to index
  %4 = arith.cmpi eq, %3, %c0 : index
  scf.for %arg13 = %c0_i32 to %c18_i32 step %c1_i32  : i32 {
    hivm.hir.anchor {id = 2 : i64}
    hivm.hir.anchor {id = 3 : i64}
    hivm.hir.anchor {id = 4 : i64}
    hivm.hir.anchor {id = 5 : i64}
    hivm.hir.anchor {id = 6 : i64}
    hivm.hir.anchor {id = 7 : i64}
    scf.for %arg14 = %c0 to %c2 step %c1 {
      hivm.hir.anchor {id = 8 : i64}
      hivm.hir.anchor {id = 9 : i64}
      hivm.hir.anchor {id = 10 : i64}
      hivm.hir.anchor {id = 11 : i64}
      hivm.hir.anchor {id = 12 : i64}
    } {fixpipe_for_mmad_result_already_inserted = true}
    %alloc = memref.alloc() : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.skip_stride_align_for_vload = #hivm.skip_stride_align_for_vload} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    hivm.hir.anchor {id = 13 : i64}
    hivm.hir.load ins(%reinterpret_cast : memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>) outs(%alloc : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    hivm.hir.anchor {id = 14 : i64}
    %alloc_3 = memref.alloc() : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    annotation.mark %alloc_3 {hivm.stride_align_dims = array<i32: 6>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    annotation.mark %alloc_3 {hivm.skip_stride_align_for_vload = #hivm.skip_stride_align_for_vload} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    hivm.hir.anchor {id = 15 : i64}
    hivm.hir.load ins(%reinterpret_cast_0 : memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>) outs(%alloc_3 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    hivm.hir.anchor {id = 16 : i64}
    %5 = arith.cmpi sgt, %arg13, %c0_i32 : i32
    hivm.hir.anchor {id = 17 : i64}
    %6 = scf.if %5 -> (memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) {
      hivm.hir.anchor {id = 18 : i64}
      %alloc_4 = memref.alloc() : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {hivm.stride_align_dims = array<i32: 6>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 19 : i64}
      hivm.hir.load ins(%reinterpret_cast_0 : memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>) outs(%alloc_4 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
      hivm.hir.anchor {id = 20 : i64}
      hivm.hir.anchor {id = 21 : i64}
      scf.yield %alloc_4 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    } else {
      hivm.hir.anchor {id = 22 : i64}
      %alloc_4 = memref.alloc() : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {hivm.stride_align_dims = array<i32: 6>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 23 : i64}
      hivm.hir.load ins(%reinterpret_cast_1 : memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>) outs(%alloc_4 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
      hivm.hir.anchor {id = 24 : i64}
      hivm.hir.anchor {id = 25 : i64}
      scf.yield %alloc_4 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    }
    hivm.hir.anchor {id = 26 : i64}
    %7 = scf.for %arg14 = %c0_i32 to %c18_i32 step %c1_i32 iter_args(%arg15 = %6) -> (memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>)  : i32 {
      hivm.hir.anchor {id = 27 : i64}
      %alloc_4 = memref.alloc() : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {hivm.skip_stride_align_for_vload = #hivm.skip_stride_align_for_vload} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 28 : i64}
      hivm.hir.load ins(%reinterpret_cast_1 : memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>) outs(%alloc_4 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
      hivm.hir.anchor {id = 29 : i64}
      %8 = arith.cmpi eq, %arg14, %c0_i32 : i32
      hivm.hir.anchor {id = 30 : i64}
      %9 = scf.if %8 -> (memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) {
        hivm.hir.anchor {id = 31 : i64}
        %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
        annotation.mark %alloc_5 {hivm.stride_align_dims = array<i32: 6>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
        hivm.hir.copy ins(%alloc_3 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) outs(%alloc_5 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>)
        scf.yield %alloc_5 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
      } else {
        hivm.hir.anchor {id = 32 : i64}
        hivm.hir.anchor {id = 33 : i64}
        hivm.hir.anchor {id = 34 : i64}
        %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
        func.call @calc_cube_vector_mix_aiv_fused_0_outlined_vf_0(%alloc, %alloc_3, %alloc_4, %alloc_5) {hivm.vector_function, no_inline} : (memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>, memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>, memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>, memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) -> ()
        hivm.hir.anchor {id = 35 : i64}
        scf.yield %alloc_5 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
      }
      hivm.hir.anchor {id = 36 : i64}
      scf.yield %9 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    }
    annotation.mark %7 {hivm.stride_align_dims = array<i32: 6>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>
    hivm.hir.anchor {id = 37 : i64}
    scf.if %4 {
      hivm.hir.store ins(%7 : memref<2x2x2x1x2x2x2x1xi8, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<2x2x2x1x2x2x2x1xi8, strided<[32, 16, 8, 8, 4, 2, 1, 1]>, #hivm.address_space<gm>>)
    } {limit_sub_block_id0}
    hivm.hir.anchor {id = 38 : i64}
  }
  hivm.hir.set_ctrl true at ctrl[60]
  hivm.hir.anchor {id = 39 : i64}
  return
}

}

// CHECK-NOT: memref.alloc() : memref<2x2x2x1x2x2x2x1xi8, strided<[512, 256, 128, 128, 64, 32, 1, 1]>
