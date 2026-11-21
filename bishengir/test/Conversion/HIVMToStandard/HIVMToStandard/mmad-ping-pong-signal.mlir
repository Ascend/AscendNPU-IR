// RUN: bishengir-opt %s -convert-hivm-to-std -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -convert-hivm-to-std -hivm-lower-to-cpu-backend -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -convert-hivm-to-std -convert-hivm-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -split-input-file | FileCheck %s --check-prefix=LLVM

// Calls with and without a signal must share the same declaration and ABI.
// Test both target lowerings and both encounter orders, plus regbase batch/MX.
// LLVM: llvm.func @_mlir_ciface_mma_tile_half_to_float(!llvm.ptr, !llvm.ptr, i1, i64, i64, i64, !llvm.ptr, !llvm.ptr,
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // CHECK-LABEL: func.func @absent_first
  // CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
  func.func @absent_first(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
      %c: memref<16x16xf32>, %signal: memref<i32>, %init: i1, %n: index) {
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK: %[[NULL:.*]] = hivm.hir.pointer_cast(%[[ZERO]]) {hivm.ping_pong_signal} : memref<i32>
    // CHECK: %[[ARG:.*]] = memref.cast %[[NULL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
    // CHECK: %[[ARG:.*]] = memref.cast %[[SIGNAL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
        ping_pong_signal(%signal : memref<i32>)
    return
  }
}

// -----

// LLVM: llvm.func @_mlir_ciface_mma_tile_half_to_float(!llvm.ptr, !llvm.ptr, i1, i64, i64, i64, !llvm.ptr, !llvm.ptr,
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // CHECK-LABEL: func.func @present_first
  // CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
  func.func @present_first(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
      %c: memref<16x16xf32>, %signal: memref<i32>, %init: i1, %n: index) {
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK: %[[ARG:.*]] = memref.cast %[[SIGNAL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
        ping_pong_signal(%signal : memref<i32>)
    // CHECK: %[[NULL:.*]] = hivm.hir.pointer_cast(%[[ZERO]]) {hivm.ping_pong_signal} : memref<i32>
    // CHECK: %[[ARG:.*]] = memref.cast %[[NULL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
    return
  }
}

// -----

// LLVM: llvm.func @_mlir_ciface_mma_tile_half_to_float(!llvm.ptr, !llvm.ptr, i1, i64, i64, i64, !llvm.ptr, !llvm.ptr,
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @absent_first
  // CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
  func.func @absent_first(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
      %c: memref<16x16xf32>, %signal: memref<i32>, %init: i1, %n: index) {
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK: %[[NULL:.*]] = hivm.hir.pointer_cast(%[[ZERO]]) {hivm.ping_pong_signal} : memref<i32>
    // CHECK: %[[ARG:.*]] = memref.cast %[[NULL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
    // CHECK: %[[ARG:.*]] = memref.cast %[[SIGNAL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
        ping_pong_signal(%signal : memref<i32>)
    return
  }
}

// -----

// LLVM: llvm.func @_mlir_ciface_mma_tile_half_to_float(!llvm.ptr, !llvm.ptr, i1, i64, i64, i64, !llvm.ptr, !llvm.ptr,
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @present_first
  // CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
  func.func @present_first(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
      %c: memref<16x16xf32>, %signal: memref<i32>, %init: i1, %n: index) {
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK: %[[ARG:.*]] = memref.cast %[[SIGNAL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
        ping_pong_signal(%signal : memref<i32>)
    // CHECK: %[[NULL:.*]] = hivm.hir.pointer_cast(%[[ZERO]]) {hivm.ping_pong_signal} : memref<i32>
    // CHECK: %[[ARG:.*]] = memref.cast %[[NULL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
    return
  }
}

// -----

// LLVM: llvm.func @_mlir_ciface_batch_mma_tile_half_to_float(!llvm.ptr, !llvm.ptr, i1, i64, i64, i64, !llvm.ptr, !llvm.ptr,
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @absent_first
  // CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
  func.func @absent_first(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
      %c: memref<16x16xf32>, %signal: memref<i32>, %init: i1, %n: index) {
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK: %[[NULL:.*]] = hivm.hir.pointer_cast(%[[ZERO]]) {hivm.ping_pong_signal} : memref<i32>
    // CHECK: %[[ARG:.*]] = memref.cast %[[NULL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @batch_mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.batchMmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
    // CHECK: %[[ARG:.*]] = memref.cast %[[SIGNAL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @batch_mma_tile_half_to_float({{.*}}, %[[ARG]],
    hivm.hir.batchMmadL1
        ins(%a, %b, %init, %n, %n, %n :
            memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
        ping_pong_signal(%signal : memref<i32>)
    return
  }
}

// -----

// LLVM: llvm.func @_mlir_ciface_mmadmxL1_float8_e5m2_t_float8_e5m2_t_to_float(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i1, i64, i64, i64, !llvm.ptr,
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @absent_first
  // CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
  func.func @absent_first(%a: memref<16x16xf8E5M2>, %b: memref<16x16xf8E5M2>,
      %c: memref<16x16xf32>, %signal: memref<i32>, %init: i1, %n: index, %sa: memref<16x1xui8>, %sb: memref<16x1xui8>) {
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK: %[[NULL:.*]] = hivm.hir.pointer_cast(%[[ZERO]]) {hivm.ping_pong_signal} : memref<i32>
    // CHECK: %[[ARG:.*]] = memref.cast %[[NULL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mmadmxL1_float8_e5m2_t_float8_e5m2_t_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadmxL1
        ins(%a, %b, %sa, %sb, %init, %n, %n, %n :
            memref<16x16xf8E5M2>, memref<16x16xf8E5M2>, memref<16x1xui8>, memref<16x1xui8>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
    // CHECK: %[[ARG:.*]] = memref.cast %[[SIGNAL]] : memref<i32> to memref<i32, strided<[], offset: ?>>
    // CHECK: call @mmadmxL1_float8_e5m2_t_float8_e5m2_t_to_float({{.*}}, %[[ARG]],
    hivm.hir.mmadmxL1
        ins(%a, %b, %sa, %sb, %init, %n, %n, %n :
            memref<16x16xf8E5M2>, memref<16x16xf8E5M2>, memref<16x1xui8>, memref<16x1xui8>, i1, index, index, index)
        outs(%c : memref<16x16xf32>)
        ping_pong_signal(%signal : memref<i32>)
    return
  }
}
