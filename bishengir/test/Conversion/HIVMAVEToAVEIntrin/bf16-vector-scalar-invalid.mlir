// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @bad_vmaxs_scalar_type(%vec: vector<64xbf16>, %scalar: f32) -> vector<64xbf16> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  // expected-error @+1 {{failed to verify that the scalar should have the same type with the vector's element type}}
  %res = ave.hir.vmaxs %vec, %scalar, %mask : vector<64xbf16>, f32, vector<64xi1>
  return %res : vector<64xbf16>
}

// -----

func.func @bad_vmins_scalar_type(%vec: vector<64xbf16>, %scalar: f32) -> vector<64xbf16> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  // expected-error @+1 {{failed to verify that the scalar should have the same type with the vector's element type}}
  %res = ave.hir.vmins %vec, %scalar, %mask : vector<64xbf16>, f32, vector<64xi1>
  return %res : vector<64xbf16>
}

// -----

func.func @bad_vadds_bf16_scalar(%vec: vector<64xbf16>, %scalar: bf16) -> vector<64xbf16> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  // expected-error @+1 {{operand #1 must be 8-bit signless integer or 16-bit signless integer or 32-bit signless integer or 64-bit signless integer or 16-bit float or 32-bit float, but got 'bf16'}}
  %res = ave.hir.vadds %vec, %scalar, %mask : vector<64xbf16>, bf16, vector<64xi1>
  return %res : vector<64xbf16>
}
