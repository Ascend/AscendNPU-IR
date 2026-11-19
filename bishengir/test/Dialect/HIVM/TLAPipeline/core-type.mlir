// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --enable-tuning-mode --enable-lir-compile=false --mlir-disable-threading --mlir-print-ir-before=hivm-cross-core-gss --mlir-print-ir-after=hivm-v2-split-kernel -o %t.ll > %t.log 2>&1
// RUN: FileCheck %s --check-prefixes=CHECK,SPLIT --implicit-check-not=memref.store < %t.log
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --enable-tuning-mode --enable-lir-compile=false --mlir-disable-threading --disable-auto-inject-block-sync=true --mlir-print-ir-after=hivm-v2-split-kernel -o %t.explicit.ll > %t.explicit.log 2>&1
// RUN: FileCheck %s --check-prefix=SPLIT --implicit-check-not=memref.store < %t.explicit.log

// Scope inlining must preserve each scalar access's execution core. A shared
// metadata read remains live on both cores when the kernel is split.
// CHECK: IR Dump Before CrossCoreGSS
// CHECK-LABEL: func.func @scalar_owners(
// CHECK: memref.load {{.*}} {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK: memref.load {{.*}} {hivm.tcore_type = #hivm.tcore_type<CUBE>}
// CHECK: memref.store {{.*}} {hivm.tcore_type = #hivm.tcore_type<CUBE>}
// CHECK: memref.load {{.*}} {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: memref.store {{.*}} {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// SPLIT: IR Dump After V2SplitKernel
// SPLIT-LABEL: func.func @scalar_owners_mix_aic(
// SPLIT-NOT: hivm.tcore_type<VECTOR>
// SPLIT: memref.load {{.*}} {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// SPLIT: memref.store {{.*}} {hivm.tcore_type = #hivm.tcore_type<CUBE>}
// SPLIT-NOT: hivm.tcore_type<VECTOR>
// SPLIT-LABEL: func.func @scalar_owners_mix_aiv(
// SPLIT-NOT: hivm.tcore_type<CUBE>
// SPLIT: memref.load {{.*}} {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// SPLIT: memref.store {{.*}} {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// SPLIT-NOT: hivm.tcore_type<CUBE>

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @scalar_owners(
      %sync: memref<?xi8>, %workspace: memref<?xi8>,
      %meta: memref<?xi32>, %cube_out: memref<?xi32>, %vector_out: memref<?xi32>,
      %gridX: i32, %gridY: i32, %gridZ: i32,
      %pidX: i32, %pidY: i32, %pidZ: i32)
      attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                  global_kernel = "local", mix_mode = "mix", parallel_mode = "simd",
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  hivm.core_ratio = #hivm.core_ratio<1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %shared = memref.load %meta[%c0] : memref<?xi32>
    scope.scope : () -> () {
      %cube_value = memref.load %meta[%c1] : memref<?xi32>
      %cube_result = arith.addi %cube_value, %shared : i32
      memref.store %cube_result, %cube_out[%c0] : memref<?xi32>
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %vector_value = memref.load %meta[%c2] : memref<?xi32>
      %vector_result = arith.addi %vector_value, %shared : i32
      memref.store %vector_result, %vector_out[%c0] : memref<?xi32>
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
