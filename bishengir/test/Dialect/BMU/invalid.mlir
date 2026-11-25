// RUN: bishengir-opt %s -split-input-file -verify-diagnostics
// design: bmu#s.3.2.1

// Test BMU (an A6 feature) handle verification.
// A handle must match its buffer: a ranked memref with a static shape and an
// identity layout in the buffer's HIVM address space, or an i64 scalar for BT.

// UB requires the UB address space.
func.func @alloc_ub_wrong_address_space(%size : i64) {
  // expected-error@+1 {{buffer ubuf requires a ranked memref with a static shape and an identity layout in #hivm.address_space<ub>, but got 'memref<8xf32, #hivm.address_space<cbuf>>'}}
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, #hivm.address_space<cbuf>>
  return
}

// -----

// A memref without an HIVM address space is not a BMU handle.
func.func @alloc_ub_without_address_space(%size : i64) {
  // expected-error@+1 {{buffer ubuf requires a ranked memref}}
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32>
  return
}

// -----

// The allocation is described by a slice count, so a dynamic shape has no
// operand to carry it.
func.func @alloc_ub_dynamic_shape(%size : i64) {
  // expected-error@+1 {{buffer ubuf requires a ranked memref}}
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<?xf32, #hivm.address_space<ub>>
  return
}

// -----

// A non-identity layout is not an allocation handle.
func.func @alloc_ub_strided_layout(%size : i64) {
  // expected-error@+1 {{buffer ubuf requires a ranked memref}}
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32, strided<[2]>, #hivm.address_space<ub>>
  return
}

// -----

// BT has no address space, so a memref handle is rejected.
func.func @alloc_bt_with_memref(%size : i64) {
  // expected-error@+1 {{buffer bt requires an i64 scalar address (BT has no address space), but got 'memref<8xf32>'}}
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<bt>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> memref<8xf32>
  return
}

// -----

// Every other buffer is addressed through an address space, not by a scalar.
func.func @alloc_ub_with_i64(%size : i64) {
  // expected-error@+1 {{buffer ubuf requires a ranked memref}}
  %addr = bmu.alloc_reg(%size) {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64} : i64 -> i64
  return
}

// -----

// FREE operands are verified against the buffer the same way.
func.func @free_l1_wrong_address_space(%addr : memref<8xf32, #hivm.address_space<ub>>, %size : i64) {
  // expected-error@+1 {{buffer cbuf requires a ranked memref with a static shape and an identity layout in #hivm.address_space<cbuf>, but got 'memref<8xf32, #hivm.address_space<ub>>'}}
  bmu.free_reg(%addr, %size) {buffer = #bmu.buffer<cbuf>, pipe = #bmu.pipe<PIPE_S>} : memref<8xf32, #hivm.address_space<ub>>, i64
  return
}

// -----

// BT allocations are i64 scalars, so a memref cannot be freed as BT.
func.func @free_bt_with_memref(%addr : memref<8xf32, #hivm.address_space<ub>>) {
  // expected-error@+1 {{buffer bt requires an i64 scalar address}}
  bmu.free_imm(%addr) {buffer = #bmu.buffer<bt>, pipe = #bmu.pipe<PIPE_S>, size = 4 : i64} : memref<8xf32, #hivm.address_space<ub>>
  return
}

// -----

// The reservation must hold the memref view. The IMM variants carry a
// compile-time slice count, so an undersized allocation is diagnosed; the slice
// size is derived from the device spec, and the A6 UB value (376 KiB plus the
// 8 KiB compiler reserve, over 48 slices) makes one slice 8 KB, so it cannot
// hold a 16 KB view.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<
    #dlti.dl_entry<"UB_SIZE", 3080192 : i32>>>} {
func.func @alloc_imm_ub_too_small() {
  // expected-error@+1 {{buffer ubuf reserves 8192 bytes, but the handle type views 16384 bytes}}
  %addr = bmu.alloc_imm {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_V>, seg = 0 : i64, size = 1 : i64} : () -> memref<8192xf16, #hivm.address_space<ub>>
  return
}
}

// -----

// L1 is 512 KiB over 64 slices, so one slice is 8 KB and cannot hold a 16 KB
// view.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<
    #dlti.dl_entry<"L1_SIZE", 4194304 : i32>>>} {
func.func @alloc_imm_l1_too_small() {
  // expected-error@+1 {{buffer cbuf reserves 8192 bytes, but the handle type views 16384 bytes}}
  %addr = bmu.alloc_imm {buffer = #bmu.buffer<cbuf>, pipe = #bmu.pipe<PIPE_MTE2>, seg = 0 : i64, size = 1 : i64} : () -> memref<4096xf32, #hivm.address_space<cbuf>>
  return
}
}

// -----

// A free with an immediate size is checked the same way; L0C is 256 KiB over 64
// slices, so one slice is 4 KB.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<
    #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>>>} {
func.func @free_imm_cc_too_small(%addr : memref<2048xf32, #hivm.address_space<cc>>) {
  // expected-error@+1 {{buffer cc reserves 4096 bytes, but the handle type views 8192 bytes}}
  bmu.free_imm(%addr) {buffer = #bmu.buffer<cc>, pipe = #bmu.pipe<PIPE_S>, size = 1 : i64} : memref<2048xf32, #hivm.address_space<cc>>
  return
}
}

// -----

// The slice size follows the device spec, not a number frozen in the dialect:
// with the UB value the pinned A6 spec carries today (2555904 bit = 312 KiB),
// one slice is (312 KiB + 8 KiB) / 48 = 6826 bytes, so a 8 KB view no longer
// fits where it fits under 376 KiB.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<
    #dlti.dl_entry<"UB_SIZE", 2555904 : i32>>>} {
func.func @alloc_imm_ub_slice_follows_spec() {
  // expected-error@+1 {{buffer ubuf reserves 6826 bytes, but the handle type views 8192 bytes}}
  %addr = bmu.alloc_imm {buffer = #bmu.buffer<ubuf>, pipe = #bmu.pipe<PIPE_V>, seg = 0 : i64, size = 1 : i64} : () -> memref<2048xf32, #hivm.address_space<ub>>
  return
}
}

// -----

// A region is sliced by the ring, so BT, which has no slice size, cannot be
// allocated as a region.
func.func @test_alloc_region_bt() -> i64 {
  // expected-error@+1 {{buffer bt has no slice size and cannot be allocated as a region}}
  %base = bmu.alloc_region_imm {buffer = #bmu.buffer<bt>, pipe = #bmu.pipe<PIPE_S>, seg = 0 : i64, size = 1 : i64} : () -> i64
  return %base : i64
}
