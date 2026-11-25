// RUN: bishengir-opt %s -split-input-file | FileCheck %s
// design: bmu#s.3.2.1

// Test BMU (an A6 feature) semantic op parsing and verification.
// All integer types are i64 (BiSheng-CCE maps all integers to llvm_i64_ty).
// FREE does not carry seg (BiSheng-CCE BUF_FREE has no seg parameter).
//
// A BMU handle is a ranked memref in the buffer's HIVM address space, so a BMU
// allocation mixes with memref/HIVM operations. BT is the exception: it is
// addressed by an i64 scalar and has no address space.

// CHECK-LABEL: func.func @test_alloc_reg
func.func @test_alloc_reg(%size : i64) -> memref<8xf32, #hivm.address_space<ub>> {
  // CHECK: bmu.alloc_reg
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<ub>>
  return %addr : memref<8xf32, #hivm.address_space<ub>>
}

// -----

// CHECK-LABEL: func.func @test_alloc_imm
func.func @test_alloc_imm() -> memref<4xi16, #hivm.address_space<cbuf>> {
  // CHECK: bmu.alloc_imm
  %addr = bmu.alloc_imm {buffer = #bmu.buffer<cbuf>, pipe = #bmu.pipe<PIPE_V>, seg = 0 : i64, size = 4 : i64} : () -> memref<4xi16, #hivm.address_space<cbuf>>
  return %addr : memref<4xi16, #hivm.address_space<cbuf>>
}

// -----

// BT has no address space: its handle is an i64 scalar (proto "ULi").
// CHECK-LABEL: func.func @test_alloc_bt
func.func @test_alloc_bt(%size : i64) -> i64 {
  // CHECK: bmu.alloc_reg
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<bt>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> i64
  return %addr : i64
}

// -----

// CHECK-LABEL: func.func @test_free_reg
func.func @test_free_reg(%addr : memref<8xf32, #hivm.address_space<ub>>, %size : i64) {
  // CHECK: bmu.free_reg
  bmu.free_reg(%addr, %size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>} : memref<8xf32, #hivm.address_space<ub>>, i64
  return
}

// -----

// CHECK-LABEL: func.func @test_free_imm
func.func @test_free_imm(%addr : memref<8xf32, #hivm.address_space<cc>>) {
  // CHECK: bmu.free_imm
  bmu.free_imm(%addr) {buffer = #bmu.buffer<cc>, pipe = #bmu.pipe<PIPE_S>, size = 4 : i64} : memref<8xf32, #hivm.address_space<cc>>
  return
}

// -----

// CHECK-LABEL: func.func @test_free_bt
func.func @test_free_bt(%addr : i64, %size : i64) {
  // CHECK: bmu.free_reg
  bmu.free_reg(%addr, %size) {buffer = #bmu.buffer<bt>, pipe = #bmu.pipe<PIPE_S>} : i64, i64
  return
}

// -----

// A BMU allocation mixes with memref operations: the allocated UB memref is
// written with memref.store and read back with memref.load.
// CHECK-LABEL: func.func @test_mixing_with_memref_ops
func.func @test_mixing_with_memref_ops(%size : i64) {
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<ub>>
  %c0 = arith.constant 0 : index
  %v = arith.constant 1.000000e+00 : f32
  // CHECK: memref.store
  memref.store %v, %addr[%c0] : memref<8xf32, #hivm.address_space<ub>>
  // CHECK: memref.load
  %r = memref.load %addr[%c0] : memref<8xf32, #hivm.address_space<ub>>
  // CHECK: bmu.free_reg
  bmu.free_reg(%addr, %size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>} : memref<8xf32, #hivm.address_space<ub>>, i64
  return
}

// -----

// The capacity check accepts a view that exactly fills the reservation, accepts
// a view smaller than the reservation, and is skipped when the slice size of the
// buffer is unknown or the element type has no unambiguous byte size. The slice
// sizes are derived from the module's device spec: the A6 UB value 3080192 bit
// (376 KiB) plus the 8 KiB compiler reserve over 48 slices is 8 KB, L1 is
// 512 KiB / 64 = 8 KB and L0C is 256 KiB / 64 = 4 KB.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<
    #dlti.dl_entry<"UB_SIZE", 3080192 : i32>,
    #dlti.dl_entry<"L1_SIZE", 4194304 : i32>,
    #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>>>} {
// CHECK-LABEL: func.func @test_capacity_check_accepts
func.func @test_capacity_check_accepts() {
  // Exact fit: one 4 KB L0C slice holds a 4 KB view.
  // CHECK: bmu.alloc_imm
  %0 = bmu.alloc_imm {buffer = #bmu.buffer<cc>, pipe = #bmu.pipe<PIPE_M>, seg = 0 : i64, size = 1 : i64} : () -> memref<16x64xf32, #hivm.address_space<cc>>
  // Over-allocation is legal: a 2 KB view inside one 8 KB UB slice.
  %1 = bmu.alloc_imm {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_V>, seg = 0 : i64, size = 1 : i64} : () -> memref<1024xf16, #hivm.address_space<ub>>
  // L0A has no BMU ring, so no capacity check applies.
  %2 = bmu.alloc_imm {buffer = #bmu.buffer<ca>, pipe = #bmu.pipe<PIPE_MTE1>, seg = 0 : i64, size = 1 : i64} : () -> memref<8xf32, #hivm.address_space<ca>>
  // Vector elements are sized by their total width: 32 x 8 x f32 = 1 KB.
  %3 = bmu.alloc_imm {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_V>, seg = 0 : i64, size = 1 : i64} : () -> memref<32xvector<8xf32>, #hivm.address_space<ub>>
  // Sub-byte element types are not sized, so the check is skipped.
  %4 = bmu.alloc_imm {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64, size = 1 : i64} : () -> memref<8xi1, #hivm.address_space<ub>>
  return
}
}

// -----

// Without a device spec there is no slice size to compare against, so the
// capacity check is skipped instead of falling back to a second set of numbers.
// CHECK-LABEL: func.func @test_capacity_check_without_spec
func.func @test_capacity_check_without_spec() {
  // One slice would be far too small with a spec attached, but there is none.
  // CHECK: bmu.alloc_imm
  %0 = bmu.alloc_imm {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_V>, seg = 0 : i64, size = 1 : i64} : () -> memref<16384xf16, #hivm.address_space<ub>>
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

// -----

// Every buffer maps to its own HIVM address space.
// CHECK-LABEL: func.func @test_all_buffer_address_spaces
func.func @test_all_buffer_address_spaces(%size : i64) {
  // CHECK: bmu.alloc_reg
  %0 = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<ub>>
  %1 = bmu.alloc_reg(%size) {buffer = #bmu.buffer<cbuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<cbuf>>
  %2 = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ca>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<ca>>
  %3 = bmu.alloc_reg(%size) {buffer = #bmu.buffer<cb>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<cb>>
  %4 = bmu.alloc_reg(%size) {buffer = #bmu.buffer<cc>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<cc>>
  %5 = bmu.alloc_reg(%size) {buffer = #bmu.buffer<fbuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<fixbuf>>
  %6 = bmu.alloc_reg(%size) {buffer = #bmu.buffer<bt>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> i64
  return
}
// -----

// A region allocation returns the raw i64 base of a slice span (design 3.3.6).
// CHECK-LABEL: func.func @test_alloc_region_imm
func.func @test_alloc_region_imm() -> i64 {
  // CHECK: bmu.alloc_region_imm {{.*}}size = 3 : i64
  %base = bmu.alloc_region_imm {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_MTE2>, seg = 0 : i64, size = 3 : i64} : () -> i64
  return %base : i64
}

// -----

// CHECK-LABEL: func.func @test_free_region_imm
func.func @test_free_region_imm(%base : i64) {
  // CHECK: bmu.free_region_imm
  bmu.free_region_imm(%base) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_MTE3>, size = 3 : i64} : i64
  return
}
