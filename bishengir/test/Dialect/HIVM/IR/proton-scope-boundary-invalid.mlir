// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @region_id_above_12_bits() {
  // expected-error @+1 {{region_id must fit in 12 bits}}
  hivm.hir.proton_scope_boundary {region_id = 4096 : i32}
  return
}

// -----

func.func @negative_region_id() {
  // expected-error @+1 {{region_id must fit in 12 bits}}
  hivm.hir.proton_scope_boundary {region_id = -1 : i32}
  return
}
