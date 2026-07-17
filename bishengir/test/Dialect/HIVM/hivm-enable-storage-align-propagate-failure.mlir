// RUN: bishengir-opt -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @scf_if_with_propagate_failure
func.func @scf_if_with_propagate_failure(%arg0: i32, %arg1: i32, %arg2: memref<32x259xf16, #hivm.address_space<gm>>) {
  %cond = arith.cmpi eq, %arg0, %arg1 : i32
  %15 = scf.if %cond -> (memref<32x259xf16, #hivm.address_space<ub>>) {
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<32x259xf16, #hivm.address_space<ub>>
    annotation.mark %alloc_13 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<32x259xf16, #hivm.address_space<ub>>
    // CHECK: scf.yield %subview : memref<32x259xf16, strided<[4144, 16]>, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.copy
    scf.yield %alloc_13 : memref<32x259xf16, #hivm.address_space<ub>>
  } else {
    %alloc_5 = memref.alloc() : memref<32x259xf16, #hivm.address_space<ub>>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %subview_7 = memref.subview %alloc_5[%1, %2] [%3, %4] [1, 1] : memref<32x259xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[259, 1], offset: ?>, #hivm.address_space<ub>>
    annotation.mark %subview_7 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x?xf16, strided<[259, 1], offset: ?>, #hivm.address_space<ub>>
    // CHECK: scf.yield %subview : memref<32x259xf16, strided<[4144, 16]>, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.copy
    scf.yield %alloc_5 : memref<32x259xf16, #hivm.address_space<ub>>
  }
  hivm.hir.store ins(%15:memref<32x259xf16, #hivm.address_space<ub>>) outs(%arg2: memref<32x259xf16, #hivm.address_space<gm>>)
  return
}
