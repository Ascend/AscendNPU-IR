// RUN: bishengir-opt %s -affine-scalrep | FileCheck %s

// Signal reads preserve initialization; writes prevent forwarding stale values.

// CHECK-LABEL: func.func @mmadL1_signal_read(
func.func @mmadL1_signal_read(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
    %c: memref<16x16xf32>, %n: index) {
  %signal = memref.alloca() : memref<i32>
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  // CHECK: hivm.hir.mmadL1
  hivm.hir.mmadL1
      ins(%a, %b, %true, %n, %n, %n :
          memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
      outs(%c : memref<16x16xf32>)
      ping_pong_signal(%signal : memref<i32>)
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  return
}

// CHECK-LABEL: func.func @mmadL1_signal_write(
func.func @mmadL1_signal_write(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
    %c: memref<16x16xf32>, %n: index) -> i32 {
  %signal = memref.alloca() : memref<i32>
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  // CHECK: hivm.hir.mmadL1
  hivm.hir.mmadL1
      ins(%a, %b, %true, %n, %n, %n :
          memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
      outs(%c : memref<16x16xf32>)
      ping_pong_signal(%signal : memref<i32>)
  // CHECK: %[[VALUE:.*]] = affine.load
  %value = affine.load %signal[] : memref<i32>
  // CHECK: return %[[VALUE]]
  return %value : i32
}

// CHECK-LABEL: func.func @batchMmadL1_signal_read(
func.func @batchMmadL1_signal_read(%a: memref<1x16x16xf16>, %b: memref<1x16x16xf16>,
    %c: memref<1x16x16xf32>, %n: index) {
  %signal = memref.alloca() : memref<i32>
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  // CHECK: hivm.hir.batchMmadL1
  hivm.hir.batchMmadL1
      ins(%a, %b, %true, %n, %n, %n :
          memref<1x16x16xf16>, memref<1x16x16xf16>, i1, index, index, index)
      outs(%c : memref<1x16x16xf32>)
      ping_pong_signal(%signal : memref<i32>)
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  return
}

// CHECK-LABEL: func.func @batchMmadL1_signal_write(
func.func @batchMmadL1_signal_write(%a: memref<1x16x16xf16>, %b: memref<1x16x16xf16>,
    %c: memref<1x16x16xf32>, %n: index) -> i32 {
  %signal = memref.alloca() : memref<i32>
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  // CHECK: hivm.hir.batchMmadL1
  hivm.hir.batchMmadL1
      ins(%a, %b, %true, %n, %n, %n :
          memref<1x16x16xf16>, memref<1x16x16xf16>, i1, index, index, index)
      outs(%c : memref<1x16x16xf32>)
      ping_pong_signal(%signal : memref<i32>)
  // CHECK: %[[VALUE:.*]] = affine.load
  %value = affine.load %signal[] : memref<i32>
  // CHECK: return %[[VALUE]]
  return %value : i32
}

// CHECK-LABEL: func.func @mmadMxL1_signal_read(
func.func @mmadMxL1_signal_read(%a: memref<16x16xi8>, %b: memref<16x16xi8>,
    %c: memref<16x16xf32>, %scale: memref<16xi8>, %n: index) {
  %signal = memref.alloca() : memref<i32>
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  // CHECK: hivm.hir.mmadmxL1
  hivm.hir.mmadmxL1
      ins(%a, %b, %scale, %scale, %true, %n, %n, %n :
          memref<16x16xi8>, memref<16x16xi8>, memref<16xi8>, memref<16xi8>, i1,
          index, index, index)
      outs(%c : memref<16x16xf32>)
      ping_pong_signal(%signal : memref<i32>)
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  return
}

// CHECK-LABEL: func.func @mmadMxL1_signal_write(
func.func @mmadMxL1_signal_write(%a: memref<16x16xi8>, %b: memref<16x16xi8>,
    %c: memref<16x16xf32>, %scale: memref<16xi8>, %n: index) -> i32 {
  %signal = memref.alloca() : memref<i32>
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  // CHECK: affine.store
  affine.store %zero, %signal[] : memref<i32>
  // CHECK: hivm.hir.mmadmxL1
  hivm.hir.mmadmxL1
      ins(%a, %b, %scale, %scale, %true, %n, %n, %n :
          memref<16x16xi8>, memref<16x16xi8>, memref<16xi8>, memref<16xi8>, i1,
          index, index, index)
      outs(%c : memref<16x16xf32>)
      ping_pong_signal(%signal : memref<i32>)
  // CHECK: %[[VALUE:.*]] = affine.load
  %value = affine.load %signal[] : memref<i32>
  // CHECK: return %[[VALUE]]
  return %value : i32
}
