// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @missing_offset(%base: memref<128xi32, #hivm.address_space<ub>>, %index: vector<64xi32>, %mask: vector<64xi1>) -> vector<64xi32> {
  // expected-error @+1 {{requires one scalar index per buffer dimension}}
  %result = ave.hir.vgatherb %base[][%index], %mask : memref<128xi32, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xi32>
  return %result : vector<64xi32>
}
