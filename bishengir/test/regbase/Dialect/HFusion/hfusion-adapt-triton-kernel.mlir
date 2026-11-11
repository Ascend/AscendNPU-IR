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

// -----

// A GM operand that folds an offset into the pointer reaches the op as a
// memref.reinterpret_cast rather than as the block argument itself. It still names
// its kernel argument, so it must still be marked.

// The gm-addr walk only runs for a reg-based target, so this case carries the
// device spec the other cases in this file leave out.

// CHECK-LABEL: test_custom_gm_addr_through_reinterpret_cast
// CHECK-SAME: %arg0: memref<?xbf16> {hacc.arg_type = #hacc.arg_type<gm_addr>}
// CHECK-SAME: %arg1: memref<?xi32> {hacc.arg_type = #hacc.arg_type<gm_addr>}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_custom_gm_addr_through_reinterpret_cast(%arg0 : memref<?xbf16>, %arg1 : memref<?xi32>, %off : index, %out : memref<12x8x16x16xbf16, #hivm.address_space<cbuf>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %base = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
    %index = memref.reinterpret_cast %arg1 to offset: [%off], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %gather = hivm.hir.custom {gm_addr_args_indices = array<i32: 0, 1>, hivm.pipe = #hivm.pipe<PIPE_MTE2>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "sparse_gather_load_to_L1buffer_bfloat16_t_int32_t"} "__builtin_sparse_gather_load_to_L1buffer" ins(%base, %index, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64, %c1_i32, %c1_i32, %c0_i32 : memref<1xbf16, strided<[1], offset: ?>>, memref<1xi32, strided<[1], offset: ?>>, i32, i32, i32, i32, i64, i32, i32, i32) outs(%out : memref<12x8x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<12x8x16x16xbf16, #hivm.address_space<cbuf>>
    return
  }
}
