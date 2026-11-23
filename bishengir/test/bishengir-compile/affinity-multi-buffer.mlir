// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --enable-auto-multi-buffer=false --enable-lib-call-no-inline=false --hivm-sync-solver-version=v1 --mlir-disable-threading --mlir-print-ir-after=hivm-lower-multi-buffer-counter --save-temps=%t.v1 -o %t.v1.o 2>&1 | FileCheck %s --check-prefix=EXPANDED --implicit-check-not=hivm.hir.multi_buffer_counter
// RUN: test -s %t.v1.o
// RUN: FileCheck %s --check-prefix=FINAL --input-file=%t.v1/module.hivm.opt.mlir
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --enable-lib-call-no-inline=false --hivm-sync-solver-version=v2 --mlir-disable-threading --mlir-print-ir-after=hivm-lower-multi-buffer-counter -o %t.v2.o 2>&1 | FileCheck %s --check-prefix=EXPANDED --implicit-check-not=hivm.hir.multi_buffer_counter
// RUN: test -s %t.v2.o
//
// Runtime N affects tail accesses, while the user-declared slot count stays 2.
// No automatic blockification or multibuffer marking is needed. GSS v1/v2 and
// buffer selection share one counter. Frontends must reject multibuffer hints
// when auto-sync is disabled; users then manage buffer slots explicitly.
// EXPANDED-LABEL: func.func @exp_kernel_multicore(
// EXPANDED: %[[COUNT:.*]] = memref.alloca() : memref<1xi64>
// EXPANDED: memref.store {{.*}}, %[[COUNT]][{{.*}}] : memref<1xi64>
// EXPANDED-COUNT-4: hivm.hir.pointer_cast({{[^,)]*}}) : memref<256xf32, #hivm.address_space<ub>>
// EXPANDED-NOT: hivm.hir.pointer_cast
// EXPANDED: scf.for
// EXPANDED: %[[ITER:.*]] = memref.load %[[COUNT]][{{.*}}] : memref<1xi64>
// EXPANDED: arith.remui %[[ITER]], {{.*}} : i64
// EXPANDED: arith.select {{.*}} : memref<256xf32, #hivm.address_space<ub>>
// EXPANDED: arith.select {{.*}} : memref<256xf32, #hivm.address_space<ub>>
// EXPANDED: func.call @exp_kernel_multicore_outlined_vf_1
// EXPANDED: %[[NEXT:.*]] = arith.addi %[[ITER]], {{.*}} : i64
// EXPANDED: memref.store %[[NEXT]], %[[COUNT]][{{.*}}] : memref<1xi64>
// FINAL-LABEL: func.func @exp_kernel_multicore(
// FINAL-SAME: func_dyn_memref_args = dense<[true, true, true, true, false, false, false, false]> : vector<8xi1>
// FINAL: hivm.hir.set_flag
// FINAL: call @exp_kernel_multicore_outlined_vf_1
// FINAL: hivm.hir.wait_flag
// FINAL: hivm.hir.pipe_barrier[<PIPE_ALL>]

#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>
#map = affine_map<()[s0] -> (s0 + 256)>
#map1 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @exp_kernel_multicore_outlined_vf_0(
      %arg0: memref<256xf32, #hivm.address_space<ub>>)
      attributes {
        hfusion.has_fill,
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vector_function,
        hivm.vf_mode = #hivm.vf_mode<SIMD>,
        no_inline
      } {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #b32>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1, #b32> -> vector<64xf32, #b32>
    scf.for %arg1 = %c0 to %c256 step %c64 {
      %subview = memref.subview %arg0[%arg1] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #b32>
      ave.hir.masked_store <NORM_B32> %subview[%c0], %2, %1 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1, #b32>, vector<64xf32, #b32>
    }
    return
  }
  func.func @exp_kernel_multicore_outlined_vf_1(
      %arg0: memref<256xf32, #hivm.address_space<ub>>,
      %arg1: memref<256xf32, #hivm.address_space<ub>>)
      attributes {
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vector_function,
        hivm.vf_mode = #hivm.vf_mode<SIMD>,
        no_inline
      } {
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c256 step %c64 {
      %subview = memref.subview %arg0[%arg2] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32, #b32>
      %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #b32>
      %1 = ave.hir.vexp %res, %0 : vector<64xf32, #b32>, vector<64xi1, #b32>
      %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1, #b32>
      ave.hir.masked_store <NORM_B32> %subview_0[%c0], %2, %1 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1, #b32>, vector<64xf32, #b32>
    }
    return
  }
  func.func @exp_kernel_multicore(
      %sync: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
      %workspace: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>},
      %inputBase: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32},
      %outputBase: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32},
      %N: i32,
      %gridX: i32,
      %gridY: i32,
      %gridZ: i32,
      %programIdX: i32,
      %programIdY: i32,
      %programIdZ: i32)
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
    %c256_i64 = arith.constant 256 : i64
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    // Explicit logical-program mapping for the target's 56 AIV cores.
    %c56_i32 = arith.constant 56 : i32
    %0 = arith.muli %gridX, %gridY : i32
    %1 = arith.muli %0, %gridZ : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.index_cast %N : i32 to index
    scf.for %pid = %3 to %1 step %c56_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %5 = arith.remsi %pid, %gridX : i32
      %6 = arith.extsi %5 : i32 to i64
      %7 = arith.muli %6, %c256_i64 : i64
      %8 = arith.index_cast %7 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %inputBase to offset: [%8], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      // Each allocation describes ONE 1024-byte slot; the annotation requests
      // two physical slots. The compiler generates counter % 2 and selection.
      %alloc = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {hivm.multi_buffer = 2 : i32, mem_unique} : memref<256xf32, #hivm.address_space<ub>>
      %9 = affine.apply #map()[%8]
      // valid = min(256, max(N - offset, 0)). Keep the dynamic tail explicit.
      %10 = arith.maxsi %8, %4 : index
      %11 = arith.minsi %9, %10 : index
      %12 = affine.apply #map1()[%11, %8]
      %13 = arith.cmpi slt, %12, %c256 : index
      scf.if %13 {
        func.call @exp_kernel_multicore_outlined_vf_0(%alloc) {hivm.vector_function, no_inline} : (memref<256xf32, #hivm.address_space<ub>>) -> ()
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0] [%12] [1] : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %alloc[0] [%12] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32, mem_unique} : memref<256xf32, #hivm.address_space<ub>>
      func.call @exp_kernel_multicore_outlined_vf_1(%alloc, %alloc_1) {hivm.vector_function, no_inline} : (memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) -> ()
      %reinterpret_cast_2 = memref.reinterpret_cast %outputBase to offset: [%8], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %alloc_1[0] [%12] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_2[0] [%12] [1] : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_ctrl true at ctrl[60]
    }
    return
  }
}
