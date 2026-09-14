// RUN: bishengir-opt %s --hivm-mark-disable-load -split-input-file | FileCheck %s

// AIV hivm.hir.load of %arg0 writes UB dest only (SAP %arg5 FP) — AIC load
// must not get disableDCache. AIV hir.store to %arg1 still marks. %arg2 load-only.

// CHECK-LABEL: func.func @kernel_mix_aic
// CHECK: memref.load %arg0[%{{.*}}] {markDCacheInvalidatePatternVisited = 0 : i32} :
// CHECK: memref.load %arg1[%{{.*}}] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32}
// CHECK: memref.load %arg2[%{{.*}}] {markDCacheInvalidatePatternVisited = 0 : i32} :
func.func @kernel_mix_aic(
    %arg0: memref<1xi32, #hivm.address_space<gm>>,
    %arg1: memref<1xi32, #hivm.address_space<gm>>,
    %arg2: memref<1xi32, #hivm.address_space<gm>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
  %c0 = arith.constant 0 : index
  %0 = memref.load %arg0[%c0] : memref<1xi32, #hivm.address_space<gm>>
  %1 = memref.load %arg1[%c0] : memref<1xi32, #hivm.address_space<gm>>
  %2 = memref.load %arg2[%c0] : memref<1xi32, #hivm.address_space<gm>>
  return
}

func.func @kernel_mix_aiv(
    %arg0: memref<1xi32, #hivm.address_space<gm>>,
    %arg1: memref<1xi32, #hivm.address_space<gm>>,
    %arg2: memref<1xi32, #hivm.address_space<gm>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
  %ub = memref.alloc() : memref<1xi32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<1xi32, #hivm.address_space<gm>>)
      outs(%ub : memref<1xi32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%ub : memref<1xi32, #hivm.address_space<ub>>)
      outs(%arg1 : memref<1xi32, #hivm.address_space<gm>>)
  return
}

// -----

// Same-function memref.store of a GM arg still disables dcache on its load.

// CHECK-LABEL: func.func @same_func_store
// CHECK: memref.load %arg0[%{{.*}}] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32}
func.func @same_func_store(%arg0: memref<1xi32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %v = memref.load %arg0[%c0] : memref<1xi32, #hivm.address_space<gm>>
  memref.store %v, %arg0[%c0] : memref<1xi32, #hivm.address_space<gm>>
  return
}
