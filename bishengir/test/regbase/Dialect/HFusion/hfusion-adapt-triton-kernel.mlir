// REQUIRES: regbase
// TODO: enable after migrating CustomOp base (gm_addr_args_indices marking via
// the divergent customOpBaseDecl/BuiltinInfoBase) + bufferization dialect loading.
// These cases depend on hivm.hir.custom "__builtin_gather_load" being annotated
// with gm_addr_args_indices by the (not-yet-reconciled) CustomOp base, which the
// AdaptTritonKernel setCustomOpBuiltinGMArgAttrs walk then propagates to
// #hacc.arg_type<gm_addr> on the parent function arg.
// RUN: bishengir-opt -adapt-triton-kernel %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt --canonicalize -adapt-triton-kernel %s -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CHECK-CUSTOM

// CHECK-CUSTOM-LABEL: test_custom
// CHECK-CUSTOM: %arg5: memref<?xf32> {hacc.arg_type = #hacc.arg_type<gm_addr>}
func.func @test_custom(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xf32>, %arg3 : memref<?xi64>, %arg4 : memref<?xf32>, %arg5 : memref<?xf32>, %out : memref<2x2xf32>) {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %alloc = memref.alloc() : memref<2x2xi64>
  %gather_load = hivm.hir.custom "__builtin_gather_load" ins(%arg5, %alloc, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32 : memref<?xf32>, memref<2x2xi64>, i64, i32, i64, i64, i32, i32, i32, i32) outs(%out : memref<2x2xf32>) -> memref<2x2xf32>
  return
}

// -----

// CHECK-CUSTOM-LABEL: test_custom_macro
// CHECK-CUSTOM: %arg2: memref<?xf32> {hacc.arg_type = #hacc.arg_type<gm_addr>}
func.func @test_custom_macro(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xf32>, %arg3 : memref<?xi64>, %arg4 : memref<?xf32>, %out : memref<2x2xf32>) {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %alloc = memref.alloc() : memref<2x2xi64>
  %gather_load = hivm.hir.custom_macro "__builtin_gather_load" ins(%arg2, %alloc, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32 : memref<?xf32>, memref<2x2xi64>, i64, i32, i64, i64, i32, i32, i32, i32) outs(%out : memref<2x2xf32>) -> memref<2x2xf32>
  return
}
