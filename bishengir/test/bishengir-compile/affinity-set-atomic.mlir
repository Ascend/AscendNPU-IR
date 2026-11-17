// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --disable-hivm-auto-inject-sync=true --mlir-disable-threading --mlir-print-ir-after=hivm-decompose-op --save-temps=%t -o %t.o 2>&1 | FileCheck %s --check-prefix=DECOMPOSE --implicit-check-not=hivm.hir.set_atomic
// RUN: test -s %t.o
// RUN: FileCheck %s --check-prefix=FINAL --implicit-check-not=hivm.hir.set_atomic --input-file=%t/module.hivm.opt.mlir
//
// Explicit atomic control operations must lower to set_ctrl in affinity mode.
// Check that atomic ADD is enabled before the store and reset after completion.
// DECOMPOSE-LABEL: func.func @atomic_store(
// DECOMPOSE: hivm.hir.set_ctrl true at ctrl[6]
// DECOMPOSE-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// DECOMPOSE-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// DECOMPOSE-NEXT: hivm.hir.set_ctrl false at ctrl[9]
// DECOMPOSE-NEXT: hivm.hir.set_ctrl false at ctrl[10]
// DECOMPOSE-NEXT: hivm.hir.store
// DECOMPOSE-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// DECOMPOSE-NEXT: hivm.hir.set_ctrl false at ctrl[6]
// DECOMPOSE-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// DECOMPOSE-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// FINAL-LABEL: func.func @atomic_store(
// FINAL: call @load_gm_to_ubuf_1d_float(
// FINAL: hivm.hir.set_ctrl true at ctrl[6]
// FINAL-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// FINAL-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// FINAL-NEXT: hivm.hir.set_ctrl false at ctrl[9]
// FINAL-NEXT: hivm.hir.set_ctrl false at ctrl[10]
// FINAL: call @store_ubuf_to_gm_1d_float(
// FINAL-NEXT: hivm.hir.pipe_barrier[<PIPE_ALL>]
// FINAL-NEXT: hivm.hir.set_ctrl false at ctrl[6]
// FINAL-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// FINAL-NEXT: hivm.hir.set_ctrl false at ctrl[8]

module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @atomic_store(
      %input: memref<64xf32, #hivm.address_space<gm>>,
      %output: memref<64xf32, #hivm.address_space<gm>>,
      %gridX: i32, %gridY: i32, %gridZ: i32,
      %programIdX: i32, %programIdY: i32, %programIdZ: i32)
      attributes {
        global_kernel = "local",
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        mix_mode = "aiv",
        parallel_mode = "simd"
      } {
    scope.scope : () -> () {
      %buffer = memref.alloc()
          : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %buffer {mem_unique}
          : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.load
          ins(%input : memref<64xf32, #hivm.address_space<gm>>)
          outs(%buffer : memref<64xf32, #hivm.address_space<ub>>)
          core_type = <VECTOR>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.set_atomic kind = <add>[type = f32]
      hivm.hir.store
          ins(%buffer : memref<64xf32, #hivm.address_space<ub>>)
          outs(%output : memref<64xf32, #hivm.address_space<gm>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.set_atomic kind = <none>[type = f32]
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
