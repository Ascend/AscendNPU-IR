// RUN: bishengir-opt %s -execution-engine-convert-hivm-to-upstream -split-input-file -verify-diagnostics

// The ExecutionEngine lowering replaces a plain load with linalg.copy, which
// cannot express a broadcast descriptor (the two sides differ in extent). Fail
// with a diagnostic instead of emitting a copy that no longer verifies.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @broadcast_load_ee(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<4x32xf32, #hivm.address_space<ub>>) {
  // expected-error @+1 {{broadcast_axes is not supported by ExecutionEngine}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}
}
