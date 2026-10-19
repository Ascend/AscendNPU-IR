// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @region_id_above_12_bits() {
  %region_id = arith.constant 4096 : i32
  // expected-error @+1 {{region_id must fit in 12 bits}}
  hivm.hir.custom "__builtin_proton_scope_boundary" ins(%region_id : i32)
  return
}

// -----

func.func @negative_region_id() {
  %region_id = arith.constant -1 : i32
  // expected-error @+1 {{region_id must fit in 12 bits}}
  hivm.hir.custom "__builtin_proton_scope_boundary" ins(%region_id : i32)
  return
}

// -----

func.func @dynamic_region_id(%region_id: i32) {
  // expected-error @+1 {{region_id must be a constant}}
  hivm.hir.custom "__builtin_proton_scope_boundary" ins(%region_id : i32)
  return
}
