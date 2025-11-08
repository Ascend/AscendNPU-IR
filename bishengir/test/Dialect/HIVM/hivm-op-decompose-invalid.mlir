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