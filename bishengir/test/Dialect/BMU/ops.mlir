// RUN: bishengir-opt %s -split-input-file | FileCheck %s

// Test BMU (an A6 feature) semantic op parsing and verification.
// All integer types are i64 (BiSheng-CCE maps all integers to llvm_i64_ty).
// FREE does not carry seg (BiSheng-CCE BUF_FREE has no seg parameter).

// CHECK-LABEL: func.func @test_alloc_reg
func.func @test_alloc_reg(%size : i64) -> !llvm.ptr<6> {
  // CHECK: bmu.alloc_reg
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> !llvm.ptr<6>
  return %addr : !llvm.ptr<6>
}

// -----

// CHECK-LABEL: func.func @test_alloc_imm
func.func @test_alloc_imm() -> !llvm.ptr<2> {
  // CHECK: bmu.alloc_imm
  %addr = bmu.alloc_imm {buffer = #bmu.buffer<cbuf>, pipe = #bmu.pipe<PIPE_V>, seg = 0 : i64, size = 4 : i64} : () -> !llvm.ptr<2>
  return %addr : !llvm.ptr<2>
}

// -----

// CHECK-LABEL: func.func @test_free_reg
func.func @test_free_reg(%addr : !llvm.ptr<6>, %size : i64) {
  // CHECK: bmu.free_reg
  bmu.free_reg(%addr, %size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>} : !llvm.ptr<6>, i64
  return
}

// -----

// CHECK-LABEL: func.func @test_free_imm
func.func @test_free_imm(%addr : !llvm.ptr<6>) {
  // CHECK: bmu.free_imm
  bmu.free_imm(%addr) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, size = 4 : i64} : !llvm.ptr<6>
  return
}

// -----

// CHECK-LABEL: func.func @test_set_segm
func.func @test_set_segm(%val : i64) {
  // CHECK: bmu.set_segm
  bmu.set_segm(%val) {buffer = #bmu.buffer<ubuf>} : i64
  return
}

// -----

// CHECK-LABEL: func.func @test_get_segm
func.func @test_get_segm() -> i64 {
  // CHECK: bmu.get_segm
  %val = bmu.get_segm {buffer = #bmu.buffer<cc>} : () -> i64
  return %val : i64
}
