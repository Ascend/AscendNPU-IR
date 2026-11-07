// RUN: bishengir-opt %s -hivm-decompose-op -split-input-file -verify-diagnostics

// -----
func.func @test_vcast_op_f32_to_i8_incorrect_semantics() {
  %f32 = tensor.empty() : tensor<2x16xf32>
  %s8 = memref.alloc() : memref<2x16xi8>
  // expected-error@+1 {{'hivm.hir.vcast' op VCastOp should have pure buffer or tensor Semantics!}}
  hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<round>
  return
}

// -----
func.func @test_vbrc_op_incorrect_semantics() {
  %0 = tensor.empty() : tensor<1x1xf32>
  %1 = memref.alloc() : memref<2x16xf32>
  // expected-error@+1 {{'hivm.hir.vbrc' op hivm::VBrcOp should have pure buffer or tensor Semantics!}}
  hivm.hir.vbrc ins(%0 : tensor<1x1xf32>) outs(%1 : memref<2x16xf32>) broadcast_dims=[0, 1]
  return
}

// -----
func.func @test_vrec_op_incorrect_semantics() {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = memref.alloc() : memref<16xf32>
  // expected-error@+1 {{'hivm.hir.vrec' op VRecOp should have pure buffer semantics!}}
  hivm.hir.vrec ins(%0 : tensor<16xf32>) outs(%1 : memref<16xf32>)
  return
}

// -----
func.func @test_vREC_op_incorrect_semantics() {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = memref.alloc() : memref<16xf32>
  // expected-error@+1 {{'hivm.hir.vrec' op VRecOp should have pure buffer semantics!}}
  hivm.hir.vrec ins(%0 : tensor<16xf32>) outs(%1 : memref<16xf32>)
  return
}

// -----
// f64 arithmetic has no device lowering (it would need the fp64 soft-float
// helpers) and must be rejected instead of reaching the linker.
func.func @test_f64_addf_unsupported(%0: f64, %1: f64) -> f64 {
  // expected-error@+1 {{f64 is not supported on this target}}
  %2 = arith.addf %0, %1 : f64
  return %2 : f64
}

// -----
// Widening f32 -> f64 is unsupported as well: it lowers to __extendsfdf2.
func.func @test_f64_extf_unsupported(%0: f32) -> f64 {
  // expected-error@+1 {{f64 is not supported on this target}}
  %1 = arith.extf %0 : f32 to f64
  return %1 : f64
}

// -----
// f64 -> f16 is not the supported f64 -> f32 conversion and stays unsupported.
func.func @test_f64_to_f16_truncf_unsupported(%0: f64) -> f16 {
  // expected-error@+1 {{f64 is not supported on this target}}
  %1 = arith.truncf %0 : f64 to f16
  return %1 : f16
}
