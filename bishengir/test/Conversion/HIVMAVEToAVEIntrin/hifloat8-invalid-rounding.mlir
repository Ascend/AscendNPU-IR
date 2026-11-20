// REQUIRES: bishengir-hifloat8
// RUN: bishengir-opt %s -convert-hivmave-to-ave-intrin -split-input-file -verify-diagnostics

func.func @unsupported_rint(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // expected-error @+1 {{HiFloat8 supports only ROUND or HYBRID}}
  %y = ave.hir.vtruncf %x, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<256xf8HIF>, vector<256xi1>
  return %y : vector<256xf8HIF>
}

// -----

func.func @unsupported_floor(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // expected-error @+1 {{HiFloat8 supports only ROUND or HYBRID}}
  %y = ave.hir.vtruncf %x, <floor>, false, <part_even>, %mask : vector<64xf32>, vector<256xf8HIF>, vector<256xi1>
  return %y : vector<256xf8HIF>
}

// -----

func.func @unsupported_ceil(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // expected-error @+1 {{HiFloat8 supports only ROUND or HYBRID}}
  %y = ave.hir.vtruncf %x, <ceil>, false, <part_even>, %mask : vector<64xf32>, vector<256xf8HIF>, vector<256xi1>
  return %y : vector<256xf8HIF>
}

// -----

func.func @unsupported_trunc(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // expected-error @+1 {{HiFloat8 supports only ROUND or HYBRID}}
  %y = ave.hir.vtruncf %x, <trunc>, false, <part_even>, %mask : vector<64xf32>, vector<256xf8HIF>, vector<256xi1>
  return %y : vector<256xf8HIF>
}

// -----

func.func @unsupported_odd(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // expected-error @+1 {{HiFloat8 supports only ROUND or HYBRID}}
  %y = ave.hir.vtruncf %x, <odd>, false, <part_even>, %mask : vector<64xf32>, vector<256xf8HIF>, vector<256xi1>
  return %y : vector<256xf8HIF>
}

// -----

func.func @unsupported_truncwithoverflow(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // expected-error @+1 {{HiFloat8 supports only ROUND or HYBRID}}
  %y = ave.hir.vtruncf %x, <truncwithoverflow>, false, <part_even>, %mask : vector<64xf32>, vector<256xf8HIF>, vector<256xi1>
  return %y : vector<256xf8HIF>
}

// -----

func.func @hybrid_to_f16(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<128xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // expected-error @+1 {{HYBRID rounding requires a HiFloat8 result}}
  %y = ave.hir.vtruncf %x, <hybrid>, false, <part_even>, %mask : vector<64xf32>, vector<128xf16>, vector<256xi1>
  return %y : vector<128xf16>
}

// -----

func.func @hybrid_to_i32(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<64xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // expected-error @+1 {{HYBRID rounding requires a HiFloat8 result}}
  %y = ave.hir.vfptosi %x, <hybrid>, %mask {sat = false} : vector<64xf32>, vector<256xi1>, vector<64xi32>
  return %y : vector<64xi32>
}
