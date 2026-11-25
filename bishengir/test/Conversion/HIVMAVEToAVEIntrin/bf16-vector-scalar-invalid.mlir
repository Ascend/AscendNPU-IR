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

func.func @bad_vadds_scalar_type(%vec: vector<64xbf16>, %scalar: f32) -> vector<64xbf16> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  // expected-error @+1 {{failed to verify that the scalar should have the same type with the vector's element type}}
  %res = ave.hir.vadds %vec, %scalar, %mask : vector<64xbf16>, f32, vector<64xi1>
  return %res : vector<64xbf16>
}
