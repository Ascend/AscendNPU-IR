// RUN: split-file %s %t
// RUN: bishengir-opt %t/planned.mlir -create-preload="bypass-shape-registry=true" -hivm-lower-multi-buffer-counter | FileCheck %s
// RUN: (bishengir-compile %t/staged-copy.mlir --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --enable-tla-cv-pipelining --disable-auto-inject-block-sync=true --enable-tuning-mode --enable-lir-compile=false --mlir-disable-threading --mlir-print-ir-after=hivm-lower-multi-buffer-counter -o %t/affinity.ll 2>&1 || true) | FileCheck %s --check-prefix=AFFINITY --implicit-check-not=hivm.cv_pipelined_loop --implicit-check-not=scope.scope
// RUN: (bishengir-compile %t/staged-copy.mlir --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --enable-tla-cv-pipelining --disable-auto-inject-block-sync=true --enable-tuning-mode --enable-lir-compile=false --mlir-disable-threading --mlir-print-ir-after=hivm-lower-multi-buffer-counter --disable-hivm-auto-inject-sync=true -o %t/affinity.explicit.ll 2>&1 || true) | FileCheck %s --check-prefix=ROTATE --implicit-check-not=hivm.hir.multi_buffer_counter

//--- planned.mlir
// The input represents the post-PlanMemory boundary. Both same-stage scopes
// must execute; their stage-local buffer needs one address at depth two.
// CHECK-LABEL: func.func @same_stage
// CHECK: scf.for
// CHECK: scf.if
// CHECK: memref.store %arg0
// CHECK: memref.store %arg1
func.func @same_stage(%a: f32, %b: f32, %T: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %addr0 = arith.constant 0 : i64
  scf.for %i = %c0 to %T step %c1 {
    %buffer = hivm.hir.pointer_cast(%addr0)
        : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %buffer {hivm.preload_local_buffer}
        : memref<64xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      memref.store %a, %buffer[%c0] : memref<64xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      memref.store %b, %buffer[%c1] : memref<64xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// A counter created by CCGSS must stay outside guards and be shared by stages.
// CHECK-LABEL: func.func @legacy_counter
// CHECK: scf.for
// CHECK: %[[TICK:.*]] = memref.load
// CHECK-NOT: hivm.hir.multi_buffer_counter
// CHECK: scf.if
// CHECK: memref.store %[[TICK]]
// CHECK-NOT: hivm.hir.multi_buffer_counter
// CHECK: scf.if
// CHECK: memref.store %[[TICK]]
// CHECK-NOT: hivm.hir.multi_buffer_counter
// CHECK: return
func.func @legacy_counter(%T: index, %p: memref<2xi64>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %T step %c1 {
    %tick = hivm.hir.multi_buffer_counter -> i64
    scope.scope : () -> () {
      memref.store %tick, %p[%c0] : memref<2xi64>
      scope.return
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      memref.store %tick, %p[%c1] : memref<2xi64>
      scope.return
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

//--- staged-copy.mlir
// Exercise the real affinity hook, memory planning, guarded skew and GSS.
// In-place tiles avoid relying on any noalias assumption about kernel arguments.
// AFFINITY-LABEL: func.func @staged_copy
// AFFINITY: hivm.hir.set_flag
// AFFINITY: scf.for
// AFFINITY: scf.if
// AFFINITY: hivm.hir.wait_flag
// AFFINITY: hivm.hir.load
// AFFINITY: scf.if
// AFFINITY: hivm.hir.store
// The staged buffer still rotates when both sync solvers are disabled.
// ROTATE-LABEL: func.func @staged_copy(
// ROTATE: %[[A:[^ ]+]] = hivm.hir.pointer_cast(%{{[^,)]*}})
// ROTATE: %[[B:[^ ]+]] = hivm.hir.pointer_cast(%{{[^,)]*}})
// ROTATE: arith.select {{.*}}, %[[B]], %[[A]]

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @staged_copy(
      %sync: memref<?xi8>, %workspace: memref<?xi8>,
      %gm: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
      %gridX: i32, %gridY: i32, %gridZ: i32,
      %pidX: i32, %pidY: i32, %pidZ: i32)
      attributes {
        SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
        global_kernel = "local",
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        mix_mode = "aiv", parallel_mode = "simd"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %tile = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %offset = arith.muli %i, %c64 : index
      %view = memref.reinterpret_cast %gm to offset: [%offset],
          sizes: [64], strides: [1] : memref<?xf32> to
          memref<64xf32, strided<[1], offset: ?>>
      scope.scope : () -> () {
        hivm.hir.load ins(%view : memref<64xf32, strided<[1], offset: ?>>)
            outs(%tile : memref<64xf32, #hivm.address_space<ub>>)
        scope.return
      } {hivm.task_name = "load", hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      scope.scope : () -> () {
        hivm.hir.store ins(%tile : memref<64xf32, #hivm.address_space<ub>>)
            outs(%view : memref<64xf32, strided<[1], offset: ?>>)
        scope.return
      } {hivm.task_name = "store", hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    } {hivm.cv_pipelined_loop}
    return
  }
}
