// RUN: bishengir-opt %s -verify-diagnostics -split-input-file
func.func @small_metadata(%meta: memref<49xi32, #hivm.address_space<ssbuf>>) {
  // expected-error @+1 {{metadata is too small for the configured pool}}
  hivm.hir.tcb_pool_init %meta {slot_bytes = 32 : i32, slot_count = 5 : i32, slots_per_buffer = array<i32: 2, 1>, search_begin = array<i32: 0, 4>, search_end = array<i32: 4, 5>, flag_mask = 31 : i32} : memref<49xi32, #hivm.address_space<ssbuf>>
  return
}
// -----
func.func @bad_alignment(%meta: memref<50xi32, #hivm.address_space<ssbuf>>) {
  // expected-error @+1 {{slot bytes must be positive and 32-byte aligned, and slot count positive}}
  hivm.hir.tcb_pool_init %meta {slot_bytes = 16 : i32, slot_count = 5 : i32, slots_per_buffer = array<i32: 2, 1>, search_begin = array<i32: 0, 4>, search_end = array<i32: 4, 5>, flag_mask = 31 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>
  return
}
// -----
func.func @too_many_flags(%meta: memref<64xi32, #hivm.address_space<ssbuf>>) {
  // expected-error @+1 {{flag_mask must be nonzero and use only IDs in [0,15]}}
  hivm.hir.tcb_pool_init %meta {slot_bytes = 32 : i32, slot_count = 5 : i32, slots_per_buffer = array<i32: 2, 1>, search_begin = array<i32: 0, 4>, search_end = array<i32: 4, 5>, flag_mask = 65536 : i32} : memref<64xi32, #hivm.address_space<ssbuf>>
  return
}
// -----
func.func @bad_segment(%meta: memref<50xi32, #hivm.address_space<ssbuf>>, %pool: memref<160xi8, #hivm.address_space<ub>>) {
  // expected-error @+1 {{segment must be 0 or 1}}
  %a, %id = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 2 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>, memref<160xi8, #hivm.address_space<ub>> -> (i64, i32)
  return
}
// -----
func.func @wrong_core(%meta: memref<50xi32, #hivm.address_space<ssbuf>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  // expected-error @+1 {{operation is on the wrong core}}
  hivm.hir.tcb_pool_release %meta : memref<50xi32, #hivm.address_space<ssbuf>>
  return
}
// -----
func.func @wrong_metadata_space(%meta: memref<50xi32>) {
  // expected-error @+1 {{metadata must be a static contiguous i32 SSBuffer vector}}
  hivm.hir.tcb_pool_release %meta : memref<50xi32>
  return
}
// -----
func.func @wrong_pool_size(%meta: memref<50xi32, #hivm.address_space<ssbuf>>, %pool: memref<128xi8, #hivm.address_space<ub>>) {
  hivm.hir.tcb_pool_init %meta {slot_bytes = 32 : i32, slot_count = 5 : i32, slots_per_buffer = array<i32: 2, 1>, search_begin = array<i32: 0, 4>, search_end = array<i32: 4, 5>, flag_mask = 31 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>
  // expected-error @+1 {{pool size must match initialized slot geometry}}
  %a, %id = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>, memref<128xi8, #hivm.address_space<ub>> -> (i64, i32)
  return
}
// -----
func.func @different_pools(%meta: memref<50xi32, #hivm.address_space<ssbuf>>, %pool: memref<160xi8, #hivm.address_space<ub>>, %other: memref<160xi8, #hivm.address_space<ub>>) {
  // expected-error @+1 {{shared metadata must use the same pool}}
  %a, %id = hivm.hir.tcb_pool_get %meta, %pool {segment = 0 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>, memref<160xi8, #hivm.address_space<ub>> -> (i64, i32)
  %b, %other_id = hivm.hir.tcb_pool_get %meta, %other {segment = 1 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>, memref<160xi8, #hivm.address_space<ub>> -> (i64, i32)
  return
}
// -----
func.func @bad_region_arity(%meta: memref<50xi32, #hivm.address_space<ssbuf>>) {
  // expected-error @+1 {{search_begin and search_end must each contain two slots}}
  hivm.hir.tcb_pool_init %meta {slot_bytes = 32 : i32, slot_count = 5 : i32, slots_per_buffer = array<i32: 2, 1>, search_begin = array<i32: 0>, search_end = array<i32: 4, 5>, flag_mask = 31 : i32} : memref<50xi32, #hivm.address_space<ssbuf>>
  return
}
// -----
func.func @out_of_bounds_region(%meta: memref<64xi32, #hivm.address_space<ssbuf>>) {
  // expected-error @+1 {{search regions must fit each queue demand within pool capacity}}
  hivm.hir.tcb_pool_init %meta {slot_bytes = 32 : i32, slot_count = 4 : i32, slots_per_buffer = array<i32: 2, 1>, search_begin = array<i32: 0, 4>, search_end = array<i32: 4, 5>, flag_mask = 31 : i32} : memref<64xi32, #hivm.address_space<ssbuf>>
  return
}
