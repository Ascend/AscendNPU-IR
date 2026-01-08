// RUN: bishengir-opt %s -hivm-decompose-op -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test VCastOp MemRef
//===----------------------------------------------------------------------===//

func.func @test_vcast_op_memref_f32_to_i8_layout() {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16xf32, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2x16xi8, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>
  %f32 = memref.alloc() : memref<2x16xf32, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>
  %s8 = memref.alloc() : memref<2x16xi8, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>

  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x16xf16, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x16xf32, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>) outs(%[[ALLOC1]] : memref<2x16xf16, #hivm.address_space<ub>>) round_mode = <round>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<2x16xf16, #hivm.address_space<ub>>) outs(%[[ALLOC0]] : memref<2x16xi8, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>) round_mode = <round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>)
                 outs(%s8 : memref<2x16xi8, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>)
                 round_mode = #hivm.round_mode<round>

  return
}

// -----
// CHECK: func.func @test_vcast_op_memref_f32_to_i8_args(%[[ARG0:.*]]: memref<2x16xf32>, %[[ARG1:.*]]: memref<2x16xi8>)
func.func @test_vcast_op_memref_f32_to_i8_args(%f32 : memref<2x16xf32>, %s8 : memref<2x16xi8>) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16xf16>
  // CHECK: hivm.hir.vcast ins(%[[ARG0]] : memref<2x16xf32>) outs(%[[ALLOC]] : memref<2x16xf16>) round_mode = <round>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x16xf16>) outs(%[[ARG1]] : memref<2x16xi8>) round_mode = <round>

  hivm.hir.vcast ins(%f32 : memref<2x16xf32>)
                 outs(%s8 : memref<2x16xi8>)
                 round_mode = #hivm.round_mode<round>

  return
}

// -----
func.func @test_vcast_op_memref_f32_to_i8_dyn(%d : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[ARG0:.*]]) : memref<2x?xf32>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc(%[[ARG0:.*]]) : memref<2x?xi8>
  %f32 = memref.alloc(%d) : memref<2x?xf32>
  %s8 = memref.alloc(%d) : memref<2x?xi8>

  // CHECK: %[[ALLOC1:.*]] = memref.alloc(%[[ARG0]]) : memref<2x?xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x?xf32>) outs(%[[ALLOC1]] : memref<2x?xf16>) round_mode = <round>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<2x?xf16>) outs(%[[ALLOC0]] : memref<2x?xi8>) round_mode = <round>
  hivm.hir.vcast ins(%f32 : memref<2x?xf32>) outs(%s8 : memref<2x?xi8>)
                 round_mode = #hivm.round_mode<round>
  return
}

// -----
// CHECK: func.func @test_vcast_op_memref_f32_to_i8_args_dyn(%[[ARG0:.*]]: memref<2x?xf32>, %[[ARG1:.*]]: memref<2x?xi8>)
func.func @test_vcast_op_memref_f32_to_i8_args_dyn(%f32 : memref<2x?xf32>, %s8 : memref<2x?xi8>) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[Dim:.*]] = memref.dim %arg0, %[[C1]] : memref<2x?xf32>
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[Dim]]) : memref<2x?xf16>
  // CHECK: hivm.hir.vcast ins(%[[ARG0]] : memref<2x?xf32>) outs(%[[ALLOC]] : memref<2x?xf16>) round_mode = <round>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x?xf16>) outs(%[[ARG1]] : memref<2x?xi8>) round_mode = <round>

  hivm.hir.vcast ins(%f32 : memref<2x?xf32>) outs(%s8 : memref<2x?xi8>)
                 round_mode = #hivm.round_mode<round>
  return
}

// -----
func.func @test_vcast_op_memref_i4_to_i8() {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16xi4>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2x16xi8>
  %i4 = memref.alloc() : memref<2x16xi4>
  %s8 = memref.alloc() : memref<2x16xi8>

  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x16xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x16xi4>) outs(%[[ALLOC1]] : memref<2x16xf16>)
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<2x16xf16>) outs(%[[ALLOC0]] : memref<2x16xi8>)
  hivm.hir.vcast ins(%i4 : memref<2x16xi4>) outs(%s8 : memref<2x16xi8>)
  return
}

// -----
func.func @test_vcast_op_memref_i4_to_i8_dyn(%d : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[ALLOC:.*]]  = memref.alloc(%[[ARG0:.*]]) : memref<2x?xi4>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc(%[[ARG0:.*]]) : memref<2x?xi8>
  %i4 = memref.alloc(%d) : memref<2x?xi4>
  %s8 = memref.alloc(%d) : memref<2x?xi8>

  // CHECK: %[[ALLOC1:.*]] = memref.alloc(%[[ARG0]]) : memref<2x?xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x?xi4>) outs(%[[ALLOC1]] : memref<2x?xf16>)
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<2x?xf16>) outs(%[[ALLOC0]] : memref<2x?xi8>)
  hivm.hir.vcast ins(%i4 : memref<2x?xi4>) outs(%s8 : memref<2x?xi8>)
  return
}

// -----
func.func @test_vcast_op_memref_alloca_f32_to_i8() {
  // CHECK: %[[ALLOC:.*]] = memref.alloca() : memref<2x16xf32, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>
  // CHECK: %[[ALLOC0:.*]] = memref.alloca() : memref<2x16xi8, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>
  %f32 = memref.alloca() : memref<2x16xf32, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>
  %s8 = memref.alloca() : memref<2x16xi8, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>

  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x16xf16, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x16xf32, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>) outs(%[[ALLOC1]] : memref<2x16xf16, #hivm.address_space<ub>>) round_mode = <round>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<2x16xf16, #hivm.address_space<ub>>) outs(%[[ALLOC0]] : memref<2x16xi8, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>) round_mode = <round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>)
                 outs(%s8 : memref<2x16xi8, strided<[16, 1], offset: 16>, #hivm.address_space<ub>>)
                 round_mode = #hivm.round_mode<round>

  return
}

// -----
// CHECK: func.func @test_vcast_op_memref_alloca_f32_to_i8_dyn(%[[ARG0:.*]]: index)
func.func @test_vcast_op_memref_alloca_f32_to_i8_dyn(%d : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[ALLOC:.*]] = memref.alloca(%[[ARG0:.*]]) : memref<2x?xf32>
  // CHECK: %[[ALLOC0:.*]] = memref.alloca(%[[ARG0:.*]]) : memref<2x?xi8>
  %f32 = memref.alloca(%d) : memref<2x?xf32>
  %s8 = memref.alloca(%d) : memref<2x?xi8>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc(%[[ARG0]]) : memref<2x?xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x?xf32>) outs(%[[ALLOC1]] : memref<2x?xf16>) round_mode = <round>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<2x?xf16>) outs(%[[ALLOC0]] : memref<2x?xi8>) round_mode = <round>
  hivm.hir.vcast ins(%f32 : memref<2x?xf32>) outs(%s8 : memref<2x?xi8>)
                 round_mode = #hivm.round_mode<round>
  return
}

// -----
func.func @test_vcast_op_memref_unchanged() {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16xf32>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2x16xf16>
  %f32 = memref.alloc() : memref<2x16xf32>
  %f16 = memref.alloc() : memref<2x16xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x16xf32>) outs(%[[ALLOC0]] : memref<2x16xf16>)
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
  return
}

//===----------------------------------------------------------------------===//
// Test VCastOp Tensor
//===----------------------------------------------------------------------===//

// -----
func.func @test_vcast_op_tensor_f32_to_i8() -> tensor<2x16x16xi8> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<2x16x16xf32>
  // CHECK: %[[T1:.*]] = tensor.empty() : tensor<2x16x16xi8>
  %f32 = tensor.empty() : tensor<2x16x16xf32>
  %s8 = tensor.empty() : tensor<2x16x16xi8>

  // f32 to f16 to i8
  // CHECK: %[[T2:.*]] = tensor.empty() : tensor<2x16x16xf16>
  // CHECK: %[[T3:.*]] = hivm.hir.vcast ins(%[[T0]] : tensor<2x16x16xf32>) outs(%[[T2]] : tensor<2x16x16xf16>) round_mode = <round> -> tensor<2x16x16xf16>
  // CHECK: %[[T4:.*]] = hivm.hir.vcast ins(%[[T3]] : tensor<2x16x16xf16>) outs(%[[T1]] : tensor<2x16x16xi8>) round_mode = <round> -> tensor<2x16x16xi8>
  %res = hivm.hir.vcast ins(%f32 : tensor<2x16x16xf32>) outs(%s8 : tensor<2x16x16xi8>)
                 round_mode = #hivm.round_mode<round> -> tensor<2x16x16xi8>
  return %res : tensor<2x16x16xi8>
}

// -----
// CHECK: func.func @test_vcast_op_tensor_f32_to_i8_args(%[[ARG0:.*]]: tensor<2x16x16xf32>, %[[ARG1:.*]]: tensor<2x16x16xi8>)
func.func @test_vcast_op_tensor_f32_to_i8_args(%f32 : tensor<2x16x16xf32>, %s8 : tensor<2x16x16xi8>) -> tensor<2x16x16xi8> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<2x16x16xf16>
  // CHECK: %[[T1:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<2x16x16xf32>) outs(%[[T0]] : tensor<2x16x16xf16>) round_mode = <round> -> tensor<2x16x16xf16>
  // CHECK: %[[T2:.*]] = hivm.hir.vcast ins(%[[T1]] : tensor<2x16x16xf16>) outs(%[[ARG1]] : tensor<2x16x16xi8>) round_mode = <round> -> tensor<2x16x16xi8>

  %res = hivm.hir.vcast ins(%f32 : tensor<2x16x16xf32>) outs(%s8 : tensor<2x16x16xi8>)
                 round_mode = #hivm.round_mode<round> -> tensor<2x16x16xi8>
  return %res : tensor<2x16x16xi8>
}

// -----
// CHECK: func.func @test_vcast_op_tensor_f32_to_i8_dyn(%[[ARG0]]: index)
func.func @test_vcast_op_tensor_f32_to_i8_dyn(%d : index) -> tensor<2x?xi8> attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[T0:.*]] = tensor.empty(%[[ARG0]]) : tensor<2x?xf32>
  // CHECK: %[[T1:.*]] = tensor.empty(%[[ARG0]]) : tensor<2x?xi8>
  // CHECK: %[[DIM:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<2x?xf32>
  // CHECK: %[[T2:.*]] = tensor.empty(%[[DIM]]) : tensor<2x?xf16>
  %f32 = tensor.empty(%d) : tensor<2x?xf32>
  %s8 = tensor.empty(%d) : tensor<2x?xi8>

  // CHECK: %[[T3:.*]] = hivm.hir.vcast ins(%[[T0]] : tensor<2x?xf32>) outs(%[[T2]] : tensor<2x?xf16>) round_mode = <round> -> tensor<2x?xf16>
  // CHECK: %[[T4:.*]] = hivm.hir.vcast ins(%[[T3]] : tensor<2x?xf16>) outs(%[[T1]] : tensor<2x?xi8>) round_mode = <round> -> tensor<2x?xi8>
  %res = hivm.hir.vcast ins(%f32 : tensor<2x?xf32>) outs(%s8 : tensor<2x?xi8>)
                 round_mode = #hivm.round_mode<round> -> tensor<2x?xi8>
  return %res : tensor<2x?xi8>
}

// -----
func.func @test_vcast_op_tensor_i4_to_i8() -> tensor<2x16x16xi8> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<2x16x16xi4>
  // CHECK: %[[T1:.*]] = tensor.empty() : tensor<2x16x16xi8>
  %s4 = tensor.empty() : tensor<2x16x16xi4>
  %s8 = tensor.empty() : tensor<2x16x16xi8>

  // CHECK: %[[T2:.*]] = tensor.empty() : tensor<2x16x16xf16>
  // CHECK: %[[T3:.*]] = hivm.hir.vcast ins(%[[T0]] : tensor<2x16x16xi4>) outs(%[[T2]] : tensor<2x16x16xf16>) -> tensor<2x16x16xf16>
  // CHECK: %[[T4:.*]] = hivm.hir.vcast ins(%[[T3]] : tensor<2x16x16xf16>) outs(%[[T1]] : tensor<2x16x16xi8>) -> tensor<2x16x16xi8>
  %res = hivm.hir.vcast ins(%s4 : tensor<2x16x16xi4>) outs(%s8 : tensor<2x16x16xi8>)
                 round_mode = #hivm.round_mode<rint> -> tensor<2x16x16xi8>
  return %res : tensor<2x16x16xi8>
}

// -----
// CHECK: func.func @test_vcast_op_tensor_i4_to_i8_dyn(%[[ARG0:.*]]: index)
func.func @test_vcast_op_tensor_i4_to_i8_dyn(%d : index) -> tensor<2x?xi8> attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[T0:.*]] = tensor.empty(%[[ARG0]]) : tensor<2x?xi4>
  // CHECK: %[[T1:.*]] = tensor.empty(%[[ARG0]]) : tensor<2x?xi8>
  // CHECK: %[[DIM:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<2x?xi4>
  // CHECK: %[[T2:.*]] = tensor.empty(%[[DIM]]) : tensor<2x?xf16>
  %s4 = tensor.empty(%d) : tensor<2x?xi4>
  %s8 = tensor.empty(%d) : tensor<2x?xi8>

  // CHECK: %[[T3:.*]] = hivm.hir.vcast ins(%[[T0]] : tensor<2x?xi4>) outs(%[[T2]] : tensor<2x?xf16>) -> tensor<2x?xf16>
  // CHECK: %[[T4:.*]] = hivm.hir.vcast ins(%[[T3]] : tensor<2x?xf16>) outs(%[[T1]] : tensor<2x?xi8>) -> tensor<2x?xi8>
  %res = hivm.hir.vcast ins(%s4 : tensor<2x?xi4>) outs(%s8 : tensor<2x?xi8>)
                 round_mode = #hivm.round_mode<rint> -> tensor<2x?xi8>
  return %res : tensor<2x?xi8>
}

// -----
// src op is not tensor empty op
func.func @test_vcast_op_tensor_i4_to_i8_not_empty_dyn(%d : index) -> tensor<2x?xi8> attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[T0:.*]] = tensor.empty(%arg0) : tensor<2x?xi4>
  // CHECK: %[[T1:.*]] = arith.addi %0, %0 : tensor<2x?xi4>
  // CHECK: %[[T2:.*]] = tensor.empty(%arg0) : tensor<2x?xi8>
  // CHECK: %[[DIM:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<2x?xi4>
  // CHECK: %[[T3:.*]] = tensor.empty(%[[DIM]]) : tensor<2x?xf16>
  // CHECK: %[[T4:.*]] = hivm.hir.vcast ins(%[[T1]] : tensor<2x?xi4>) outs(%[[T3]] : tensor<2x?xf16>) -> tensor<2x?xf16>
  // CHECK: %[[T5:.*]] = hivm.hir.vcast ins(%[[T4]] : tensor<2x?xf16>) outs(%[[T2]] : tensor<2x?xi8>) -> tensor<2x?xi8>

  %s40 = tensor.empty(%d) : tensor<2x?xi4>
  %s4 = arith.addi %s40, %s40 : tensor<2x?xi4>

  %s8 = tensor.empty(%d) : tensor<2x?xi8>

  %res = hivm.hir.vcast ins(%s4 : tensor<2x?xi4>) outs(%s8 : tensor<2x?xi8>) -> tensor<2x?xi8>
  return %res : tensor<2x?xi8>
}

//===----------------------------------------------------------------------===//
// Test VCastOp args: transpose broadcast
//===----------------------------------------------------------------------===//

// -----
func.func @test_vcast_op_arg_transpose() {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16x16xf32>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16x2x16xi8>
  %src = memref.alloc() : memref<2x16x16xf32>
  %dst = memref.alloc() : memref<16x2x16xi8>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x16x16xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<2x16x16xf32>) outs(%[[ALLOC1]] : memref<2x16x16xf16>)
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<2x16x16xf16>) outs(%[[ALLOC0]] : memref<16x2x16xi8>) transpose = [1, 0, 2]
  hivm.hir.vcast ins(%src : memref<2x16x16xf32>) outs(%dst : memref<16x2x16xi8>) transpose=[1, 0, 2]
  return
}

// -----
func.func @test_vcast_op_arg_broadcast() {
  // CHECK: %[[ALLOC]] = memref.alloc() : memref<16x1x16xf32>
  // CHECK: %[[ALLOC0]] = memref.alloc() : memref<16x2x16xi8>
  %src1 = memref.alloc() : memref<16x1x16xf32>
  %dst1 = memref.alloc() : memref<16x2x16xi8>
  // CHECK: %[[ALLOC1]] = memref.alloc() : memref<16x1x16xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC]] : memref<16x1x16xf32>) outs(%[[ALLOC1]] : memref<16x1x16xf16>)
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<16x1x16xf16>) outs(%[[ALLOC0]] : memref<16x2x16xi8>) broadcast = [1]
  hivm.hir.vcast ins(%src1 : memref<16x1x16xf32>) outs(%dst1 : memref<16x2x16xi8>) broadcast=[1]
  return
}

//===----------------------------------------------------------------------===//
// Test VCastOp support modes
//===----------------------------------------------------------------------===//

// -----
func.func @test_vcast_op_memref_f32_to_i8_incorrect_mode() {
  %f32 = memref.alloc() : memref<2x16xf32>
  %s8 = memref.alloc() : memref<2x16xi8>

  // support below
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<trunc>

  // expected-error@+1 {{'hivm.hir.vcast' op currently don't support cast float_to_int8_t_oddmode}}
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<odd>
  return
}

// -----
func.func @test_vcast_op_memref_i4_to_i8_incorrect_mode() {
  %i4 = memref.alloc() : memref<2x16xi4>
  %s8 = memref.alloc() : memref<2x16xi8>

  // support rint
  hivm.hir.vcast ins(%i4 : memref<2x16xi4>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<rint>

  // expected-error@+1 {{'hivm.hir.vcast' op currently don't support cast int4_t_to_int8_t_roundmode}}
  hivm.hir.vcast ins(%i4 : memref<2x16xi4>) outs(%s8 : memref<2x16xi8>) round_mode = #hivm.round_mode<round>

  return
}



//===----------------------------------------------------------------------===//
// Test VBrcOp Decompose Buffer static and dynamic
//===----------------------------------------------------------------------===//

// -----
func.func @test_broadcast_op_multi_memref_01() {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x32xi16>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16x8x32xi16>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1x8x32xi16>
  // CHECK: hivm.hir.vbrc ins(%[[ALLOC]] : memref<1x1x32xi16>) outs(%[[ALLOC1]] : memref<1x8x32xi16>) broadcast_dims = [1]
  // CHECK: hivm.hir.vbrc ins(%[[ALLOC1]] : memref<1x8x32xi16>) outs(%[[ALLOC0]] : memref<16x8x32xi16>) broadcast_dims = [0]

  %src = memref.alloc() : memref<1x1x32xi16>
  %dst = memref.alloc() : memref<16x8x32xi16>
  hivm.hir.vbrc ins(%src : memref<1x1x32xi16>) outs(%dst : memref<16x8x32xi16>) broadcast_dims = [0, 1]
  return
}

// -----
// CHECK: func.func @test_broadcast_op_multi_memref_01_args(%[[ARG0:.*]]: memref<1x1x32xi16>, %[[ARG1:.*]]: memref<16x8x32xi16>) {
func.func @test_broadcast_op_multi_memref_01_args(%src : memref<1x1x32xi16>, %dst : memref<16x8x32xi16>) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x8x32xi16>
  // CHECK: hivm.hir.vbrc ins(%[[ARG0]] : memref<1x1x32xi16>) outs(%[[ALLOC]] : memref<1x8x32xi16>) broadcast_dims = [1]
  // CHECK: hivm.hir.vbrc ins(%[[ALLOC]] : memref<1x8x32xi16>) outs(%[[ARG1]] : memref<16x8x32xi16>) broadcast_dims = [0]

  hivm.hir.vbrc ins(%src : memref<1x1x32xi16>) outs(%dst : memref<16x8x32xi16>) broadcast_dims = [0, 1]
  return
}

// -----
// CHECK: func.func @test_broadcast_op_multi_memref_01_dyn(%[[ARG0:.*]]: index)
func.func @test_broadcast_op_multi_memref_01_dyn(%d : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[ARG0]]) : memref<1x1x?xi16>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc(%[[ARG0]]) : memref<16x8x?xi16>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc(%[[ARG0]]) : memref<1x8x?xi16>
  // CHECK: hivm.hir.vbrc ins(%[[ALLOC]] : memref<1x1x?xi16>) outs(%[[ALLOC1]] : memref<1x8x?xi16>) broadcast_dims = [1]
  // CHECK: hivm.hir.vbrc ins(%[[ALLOC1]] : memref<1x8x?xi16>) outs(%[[ALLOC0]] : memref<16x8x?xi16>) broadcast_dims = [0]

  %src = memref.alloc(%d) : memref<1x1x?xi16>
  %dst = memref.alloc(%d) : memref<16x8x?xi16>
  hivm.hir.vbrc ins(%src : memref<1x1x?xi16>) outs(%dst : memref<16x8x?xi16>) broadcast_dims = [0, 1]
  return
}

// -----
// CHECK: func.func @test_broadcast_op_multi_memref_01_dyn2(%[[ARG0:.*]]: index)
func.func @test_broadcast_op_multi_memref_01_dyn2(%d : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x32xi16>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc(%[[ARG0]]) : memref<16x?x32xi16>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc(%[[ARG0]]) : memref<1x?x32xi16>
  // CHECK: hivm.hir.vbrc ins(%[[ALLOC]] : memref<1x1x32xi16>) outs(%[[ALLOC1]] : memref<1x?x32xi16>) broadcast_dims = [1]
  // CHECK: hivm.hir.vbrc ins(%[[ALLOC1]] : memref<1x?x32xi16>) outs(%[[ALLOC0]] : memref<16x?x32xi16>) broadcast_dims = [0]

  %src = memref.alloc() : memref<1x1x32xi16>
  %dst = memref.alloc(%d) : memref<16x?x32xi16>
  hivm.hir.vbrc ins(%src : memref<1x1x32xi16>) outs(%dst : memref<16x?x32xi16>) broadcast_dims = [0, 1]
  return
}

// -----
func.func @test_broadcast_op_multi_memref_013() {
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<1x1x10x1xi16>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<16x8x10x32xi16>
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<1x1x10x32xi16>
  // CHECK: hivm.hir.vbrc ins(%[[alloc]] : memref<1x1x10x1xi16>) outs(%[[alloc_1]] : memref<1x1x10x32xi16>) broadcast_dims = [3]
  // CHECK: %[[alloc_2:.*]] = memref.alloc() : memref<1x8x10x32xi16>
  // CHECK: hivm.hir.vbrc ins(%[[alloc_1]] : memref<1x1x10x32xi16>) outs(%[[alloc_2]] : memref<1x8x10x32xi16>) broadcast_dims = [1]
  // CHECK: hivm.hir.vbrc ins(%[[alloc_2]] : memref<1x8x10x32xi16>) outs(%[[alloc_0]] : memref<16x8x10x32xi16>) broadcast_dims = [0]

  %src = memref.alloc() : memref<1x1x10x1xi16>
  %dst = memref.alloc() : memref<16x8x10x32xi16>
  hivm.hir.vbrc ins(%src : memref<1x1x10x1xi16>) outs(%dst : memref<16x8x10x32xi16>) broadcast_dims = [0, 1, 3]
  return
}

// -----
func.func @test_broadcast_op_multi_memref_023() {
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<1x8x1x1xi16>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<16x8x10x32xi16>
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<1x8x1x32xi16>
  // CHECK: hivm.hir.vbrc ins(%[[alloc]] : memref<1x8x1x1xi16>) outs(%[[alloc_1]] : memref<1x8x1x32xi16>) broadcast_dims = [3]
  // CHECK: %[[alloc_2:.*]] = memref.alloc() : memref<1x8x10x32xi16>
  // CHECK: hivm.hir.vbrc ins(%[[alloc_1]] : memref<1x8x1x32xi16>) outs(%[[alloc_2]] : memref<1x8x10x32xi16>) broadcast_dims = [2]
  // CHECK: hivm.hir.vbrc ins(%[[alloc_2]] : memref<1x8x10x32xi16>) outs(%[[alloc_0]] : memref<16x8x10x32xi16>) broadcast_dims = [0]

  %src = memref.alloc() : memref<1x8x1x1xi16>
  %dst = memref.alloc() : memref<16x8x10x32xi16>
  hivm.hir.vbrc ins(%src : memref<1x8x1x1xi16>) outs(%dst : memref<16x8x10x32xi16>) broadcast_dims = [0, 2, 3]
  return
}

// -----
func.func @test_broadcast_op_multi_memref_0234() {
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<1x8x1x1x1x16xi16>
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<16x8x10x32x16x16xi16>
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<1x8x1x1x16x16xi16>
  // CHECK: hivm.hir.vbrc ins(%[[alloc]] : memref<1x8x1x1x1x16xi16>) outs(%[[alloc_1]] : memref<1x8x1x1x16x16xi16>) broadcast_dims = [4]
  // CHECK: %[[alloc_2:.*]] = memref.alloc() : memref<1x8x1x32x16x16xi16>
  // CHECK: hivm.hir.vbrc ins(%[[alloc_1]] : memref<1x8x1x1x16x16xi16>) outs(%[[alloc_2]] : memref<1x8x1x32x16x16xi16>) broadcast_dims = [3]
  // CHECK: %[[alloc_3:.*]] = memref.alloc() : memref<1x8x10x32x16x16xi16>
  // CHECK: hivm.hir.vbrc ins(%[[alloc_2]] : memref<1x8x1x32x16x16xi16>) outs(%[[alloc_3]] : memref<1x8x10x32x16x16xi16>) broadcast_dims = [2]
  // CHECK: hivm.hir.vbrc ins(%[[alloc_3]] : memref<1x8x10x32x16x16xi16>) outs(%[[alloc_0]] : memref<16x8x10x32x16x16xi16>) broadcast_dims = [0]

  %src = memref.alloc() : memref<1x8x1x1x1x16xi16>
  %dst = memref.alloc() : memref<16x8x10x32x16x16xi16>
  hivm.hir.vbrc ins(%src : memref<1x8x1x1x1x16xi16>) outs(%dst : memref<16x8x10x32x16x16xi16>) broadcast_dims = [0, 2, 3, 4]
  return
}


//===----------------------------------------------------------------------===//
// Test VBrcOp Decompose Tensor static and dynamic
//===----------------------------------------------------------------------===//

// -----
func.func @test_broadcast_op_multi_tensor_01() -> tensor<16x8x32xi16> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<1x1x32xi16>
  // CHECK: %[[T1:.*]] = tensor.empty() : tensor<16x8x32xi16>
  // CHECK: %[[T2:.*]] = tensor.empty() : tensor<1x8x32xi16>
  // CHECK: %[[T3:.*]] = hivm.hir.vbrc ins(%[[T0]] : tensor<1x1x32xi16>) outs(%[[T2]] : tensor<1x8x32xi16>) broadcast_dims = [1] -> tensor<1x8x32xi16>
  // CHECK: %[[T4:.*]] = hivm.hir.vbrc ins(%[[T3]] : tensor<1x8x32xi16>) outs(%[[T1]] : tensor<16x8x32xi16>) broadcast_dims = [0] -> tensor<16x8x32xi16>

  %src = tensor.empty() : tensor<1x1x32xi16>
  %dst = tensor.empty() : tensor<16x8x32xi16>
  %res = hivm.hir.vbrc ins(%src : tensor<1x1x32xi16>) outs(%dst : tensor<16x8x32xi16>) broadcast_dims = [0, 1] -> tensor<16x8x32xi16>
  return %res : tensor<16x8x32xi16>
}

// -----
// CHECK: func.func @test_broadcast_op_multi_tensor_01_args(%[[arg0:.*]]: tensor<1x1x32xi16>, %[[arg1:.*]]: tensor<16x8x32xi16>) -> tensor<16x8x32xi16> {
func.func @test_broadcast_op_multi_tensor_01_args(%src : tensor<1x1x32xi16>, %dst : tensor<16x8x32xi16>) -> tensor<16x8x32xi16> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<1x8x32xi16>
  // CHECK: %[[T1:.*]] = hivm.hir.vbrc ins(%[[arg0]] : tensor<1x1x32xi16>) outs(%[[T0]] : tensor<1x8x32xi16>) broadcast_dims = [1] -> tensor<1x8x32xi16>
  // CHECK: %[[T2:.*]] = hivm.hir.vbrc ins(%[[T1]] : tensor<1x8x32xi16>) outs(%[[arg1]] : tensor<16x8x32xi16>) broadcast_dims = [0] -> tensor<16x8x32xi16>

  %res = hivm.hir.vbrc ins(%src : tensor<1x1x32xi16>) outs(%dst : tensor<16x8x32xi16>) broadcast_dims = [0, 1] -> tensor<16x8x32xi16>
  return %res : tensor<16x8x32xi16>
}

// -----
// CHECK: func.func @test_broadcast_op_multi_tensor_01_dyn(%[[arg0:.*]]: index) -> tensor<16x8x?xi16>
func.func @test_broadcast_op_multi_tensor_01_dyn(%d : index) -> tensor<16x8x?xi16> attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[c2:.*]] = arith.constant 2 : index
  // CHECK: %[[T0:.*]] = tensor.empty(%arg0) : tensor<1x1x?xi16>
  // CHECK: %[[T1:.*]] = tensor.empty(%arg0) : tensor<16x8x?xi16>
  // CHECK: %[[dim:.*]] = tensor.dim %1, %c2 : tensor<16x8x?xi16>
  // CHECK: %[[T2:.*]] = tensor.empty(%dim) : tensor<1x8x?xi16>
  // CHECK: %[[T3:.*]] = hivm.hir.vbrc ins(%[[T0]] : tensor<1x1x?xi16>) outs(%[[T2]] : tensor<1x8x?xi16>) broadcast_dims = [1] -> tensor<1x8x?xi16>
  // CHECK: %[[T4:.*]] = hivm.hir.vbrc ins(%[[T3]] : tensor<1x8x?xi16>) outs(%[[T1]] : tensor<16x8x?xi16>) broadcast_dims = [0] -> tensor<16x8x?xi16>

  %src = tensor.empty(%d) : tensor<1x1x?xi16>
  %dst = tensor.empty(%d) : tensor<16x8x?xi16>
  %res = hivm.hir.vbrc ins(%src : tensor<1x1x?xi16>) outs(%dst : tensor<16x8x?xi16>) broadcast_dims = [0, 1] -> tensor<16x8x?xi16>
  return %res : tensor<16x8x?xi16>
}

// -----
// CHECK: func.func @test_broadcast_op_multi_tensor_01_dyn2(%[[arg0:.*]]: index) -> tensor<16x?x32xi16>
func.func @test_broadcast_op_multi_tensor_01_dyn2(%d : index) -> tensor<16x?x32xi16> attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<1x1x32xi16>
  // CHECK: %[[T1:.*]] = tensor.empty(%[[arg0]]) : tensor<16x?x32xi16>
  // CHECK: %[[dim:.*]] = tensor.dim %[[T1]], %[[c1]] : tensor<16x?x32xi16>
  // CHECK: %[[T2:.*]] = tensor.empty(%[[dim]]) : tensor<1x?x32xi16>
  // CHECK: %[[T3:.*]] = hivm.hir.vbrc ins(%[[T0]] : tensor<1x1x32xi16>) outs(%[[T2]] : tensor<1x?x32xi16>) broadcast_dims = [1] -> tensor<1x?x32xi16>
  // CHECK: %[[T4:.*]] = hivm.hir.vbrc ins(%[[T3]] : tensor<1x?x32xi16>) outs(%[[T1]] : tensor<16x?x32xi16>) broadcast_dims = [0] -> tensor<16x?x32xi16>

  %src = tensor.empty() : tensor<1x1x32xi16>
  %dst = tensor.empty(%d) : tensor<16x?x32xi16>
  %res = hivm.hir.vbrc ins(%src : tensor<1x1x32xi16>) outs(%dst : tensor<16x?x32xi16>) broadcast_dims = [0, 1] -> tensor<16x?x32xi16>
  return %res : tensor<16x?x32xi16>
}

// -----
func.func @test_broadcast_op_multi_tensor_013() -> tensor<16x8x10x32xi16> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<1x1x10x1xi16>
  // CHECK: %[[T1:.*]] = tensor.empty() : tensor<16x8x10x32xi16>
  // CHECK: %[[T2:.*]] = tensor.empty() : tensor<1x1x10x32xi16>
  // CHECK: %[[T3:.*]] = hivm.hir.vbrc ins(%[[T0]] : tensor<1x1x10x1xi16>) outs(%[[T2]] : tensor<1x1x10x32xi16>) broadcast_dims = [3] -> tensor<1x1x10x32xi16>
  // CHECK: %[[T4:.*]] = tensor.empty() : tensor<1x8x10x32xi16>
  // CHECK: %[[T5:.*]] = hivm.hir.vbrc ins(%[[T3]] : tensor<1x1x10x32xi16>) outs(%[[T4]] : tensor<1x8x10x32xi16>) broadcast_dims = [1] -> tensor<1x8x10x32xi16>
  // CHECK: %[[T6:.*]] = hivm.hir.vbrc ins(%[[T5]] : tensor<1x8x10x32xi16>) outs(%[[T1]] : tensor<16x8x10x32xi16>) broadcast_dims = [0] -> tensor<16x8x10x32xi16>

  %src = tensor.empty() : tensor<1x1x10x1xi16>
  %dst = tensor.empty() : tensor<16x8x10x32xi16>
  %res = hivm.hir.vbrc ins(%src : tensor<1x1x10x1xi16>) outs(%dst : tensor<16x8x10x32xi16>) broadcast_dims = [0, 1, 3] -> tensor<16x8x10x32xi16>
  return %res : tensor<16x8x10x32xi16>
}

// -----

// VRecOp decompose

func.func @test_vrec_decompose() {
  // CHECK-DAG: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16x16xf32>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<16x16xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[ALLOC1]] : memref<16x16xf32>)
  // CHECK: hivm.hir.vdiv ins(%[[ALLOC1]], %[[ALLOC0]] : memref<16x16xf32>, memref<16x16xf32>) outs(%[[ALLOC1]] : memref<16x16xf32>)
  %allocIn = memref.alloc() : memref<16x16xf32>
  %allocOut = memref.alloc() : memref<16x16xf32>
  hivm.hir.vrec ins(%allocIn : memref<16x16xf32>) outs(%allocOut : memref<16x16xf32>)
  return
}

// -----

// Test SyncBlockAll
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @kernel_mix_aic() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %ffts_base_addr = arith.constant 0 : i64
  // CHECK: hivm.hir.pipe_barrier[<PIPE_FIX>]
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 10
  // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 13 ffts_base_addr = %c0_i64 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 13
  // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>] flag = 11 ffts_base_addr = %c0_i64 sync_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block[#hivm.sync_block_mode<ALL>, 1 : i16]
                ffts_base_addr = %ffts_base_addr
                tcube_pipe=#hivm.pipe<PIPE_FIX>
                tvector_pipe=#hivm.pipe<PIPE_MTE3>
    return
  }
  func.func @kernel_mix_aiv() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %ffts_base_addr = arith.constant 0 : i64
  // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 10 ffts_base_addr = %c0_i64 sync_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
  // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 11
    hivm.hir.sync_block[#hivm.sync_block_mode<ALL>, 1 : i16]
                ffts_base_addr = %ffts_base_addr
                tcube_pipe=#hivm.pipe<PIPE_FIX>
                tvector_pipe=#hivm.pipe<PIPE_MTE3>
    return
  }
}

// -----
// Test SyncBlockAllBarrierAll
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @kernel_mix_aic() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %ffts_base_addr = arith.constant 0 : i64
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 13 ffts_base_addr = %c0_i64 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 13
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 11 ffts_base_addr = %c0_i64 sync_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block[#hivm.sync_block_mode<ALL>, 1 : i16]
                ffts_base_addr = %ffts_base_addr
                tcube_pipe=#hivm.pipe<PIPE_ALL>
                tvector_pipe=#hivm.pipe<PIPE_ALL>
    return
  }
  func.func @kernel_mix_aiv() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %ffts_base_addr = arith.constant 0 : i64
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10 ffts_base_addr = %c0_i64 sync_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 11
    hivm.hir.sync_block[#hivm.sync_block_mode<ALL>, 1 : i16]
                ffts_base_addr = %ffts_base_addr
                tcube_pipe=#hivm.pipe<PIPE_ALL>
                tvector_pipe=#hivm.pipe<PIPE_ALL>
    return
  }
}


// -----
// Test SyncBlockAllCube
module {
  func.func @kernel_mix_aic() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %ffts_base_addr = arith.constant 0 : i64
    // CHECK: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 1 ffts_base_addr = %c0_i64 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block[#hivm.sync_block_mode<ALL_CUBE>, 1 : i16]
                ffts_base_addr = %ffts_base_addr
                tcube_pipe=#hivm.pipe<PIPE_FIX>
                tvector_pipe=#hivm.pipe<PIPE_FIX>
    return
  }
}

// -----
// Test SyncBlockAllCubeBarrierAll
module {
  func.func @kernel_mix_aic() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %ffts_base_addr = arith.constant 0 : i64
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1 ffts_base_addr = %c0_i64 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
    hivm.hir.sync_block[#hivm.sync_block_mode<ALL_CUBE>, 1 : i16]
                ffts_base_addr = %ffts_base_addr
                tcube_pipe=#hivm.pipe<PIPE_ALL>
                tvector_pipe=#hivm.pipe<PIPE_ALL>
    return
  }
}

// -----
// Test SyncBlockAllVector
module {
  func.func @kernel_mix_aiv() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %ffts_base_addr = arith.constant 0 : i64
    hivm.hir.sync_block[#hivm.sync_block_mode<ALL_VECTOR>, 1 : i16]
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 1 ffts_base_addr = %c0_i64 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE3>] flag = 1
                ffts_base_addr = %ffts_base_addr
                tcube_pipe=#hivm.pipe<PIPE_MTE3>
                tvector_pipe=#hivm.pipe<PIPE_MTE3>
    return
  }
}

// -----
// Test SyncBlockAllVectorBarrierAll
module {
  func.func @kernel_mix_aiv() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %ffts_base_addr = arith.constant 0 : i64
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1 ffts_base_addr = %c0_i64 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    hivm.hir.sync_block[#hivm.sync_block_mode<ALL_VECTOR>, 1 : i16]
                ffts_base_addr = %ffts_base_addr
                tcube_pipe=#hivm.pipe<PIPE_ALL>
                tvector_pipe=#hivm.pipe<PIPE_ALL>
    return
  }
}

// -----
func.func @test_cast_int8_t_to_bool_rintmode() -> tensor<1024xi1> {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ARG0:.*]] = tensor.empty() : tensor<1024xi8>
  // CHECK: %[[ARG1:.*]] = tensor.empty() : tensor<1024xi1>
  // CHECK: %[[ARG2:.*]] = tensor.empty() : tensor<1024xf16>
  // CHECK: %[[ARG3:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<1024xi8>) outs(%[[ARG2]] : tensor<1024xf16>) -> tensor<1024xf16>
  // CHECK: %[[ARG4:.*]] = tensor.empty() : tensor<1024xf16>
  // CHECK: %[[ARG5:.*]] = hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG4]] : tensor<1024xf16>) -> tensor<1024xf16>
  // CHECK: %[[ARG6:.*]] = hivm.hir.vcmp ins(%[[ARG3]], %[[ARG5]] : tensor<1024xf16>, tensor<1024xf16>) outs(%[[ARG1]] : tensor<1024xi1>) compare_mode = <ne> -> tensor<1024xi1>
  // CHECK: return %[[ARG6]] : tensor<1024xi1>
  %arg0 = tensor.empty() : tensor<1024xi8>
  %arg1 = tensor.empty() : tensor<1024xi1>
  %0 = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%arg1 : tensor<1024xi1>) -> tensor<1024xi1>
  return %0 : tensor<1024xi1>
}


// -----
func.func @test_cast_int32_to_bool() -> tensor<1024xi1> {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[ARG0:.*]] = tensor.empty() : tensor<1024xi32>
  // CHECK: %[[ARG1:.*]] = tensor.empty() : tensor<1024xi1>
  // CHECK: %[[ARG2:.*]] = tensor.empty() : tensor<1024xf32>
  // CHECK: %[[ARG3:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<1024xi32>) outs(%[[ARG2]] : tensor<1024xf32>) -> tensor<1024xf32>
  // CHECK: %[[ARG4:.*]] = tensor.empty() : tensor<1024xf32>
  // CHECK: %[[ARG5:.*]] = hivm.hir.vbrc ins(%[[CONST]] : f32) outs(%[[ARG4]] : tensor<1024xf32>) -> tensor<1024xf32>
  // CHECK: %[[ARG6:.*]] = hivm.hir.vcmp ins(%[[ARG3]], %[[ARG5]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[ARG1]] : tensor<1024xi1>) compare_mode = <ne> -> tensor<1024xi1>
  // CHECK: return %[[ARG6]] : tensor<1024xi1>
  %arg0 = tensor.empty() : tensor<1024xi32>
  %arg1 = tensor.empty() : tensor<1024xi1>
  %0 = hivm.hir.vcast ins(%arg0 : tensor<1024xi32>) outs(%arg1 : tensor<1024xi1>) -> tensor<1024xi1>
  return %0 : tensor<1024xi1>
}


// -----
func.func @test_cast_int16_to_bool() -> tensor<1024xi1> {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ARG0:.*]] = tensor.empty() : tensor<1024xi16>
  // CHECK: %[[ARG1:.*]] = tensor.empty() : tensor<1024xi1>
  // CHECK: %[[ARG2:.*]] = tensor.empty() : tensor<1024xf16>
  // CHECK: %[[ARG3:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<1024xi16>) outs(%[[ARG2]] : tensor<1024xf16>) -> tensor<1024xf16>
  // CHECK: %[[ARG4:.*]] = tensor.empty() : tensor<1024xf16>
  // CHECK: %[[ARG5:.*]] = hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG4]] : tensor<1024xf16>) -> tensor<1024xf16>
  // CHECK: %[[ARG6:.*]] = hivm.hir.vcmp ins(%[[ARG3]], %[[ARG5]] : tensor<1024xf16>, tensor<1024xf16>) outs(%[[ARG1]] : tensor<1024xi1>) compare_mode = <ne> -> tensor<1024xi1>
  // CHECK: return %[[ARG6]] : tensor<1024xi1>
  %arg0 = tensor.empty() : tensor<1024xi16>
  %arg1 = tensor.empty() : tensor<1024xi1>
  %0 = hivm.hir.vcast ins(%arg0 : tensor<1024xi16>) outs(%arg1 : tensor<1024xi1>) -> tensor<1024xi1>
  return %0 : tensor<1024xi1>
}

// -----
func.func @test_cast_bool_to_fp32(){
   %allocIn = memref.alloc() : memref<1024xi1>
   %allocOut = memref.alloc() : memref<1024xf32>
   // CHECK: hivm.hir.vcast ins({{.*}} : memref<1024xi1>) outs({{.*}} : memref<1024xf32>) round_mode = <trunc>
   hivm.hir.vcast ins(%allocIn : memref<1024xi1>) outs(%allocOut : memref<1024xf32>) round_mode = <trunc>
   return
}

// -----
func.func @test_reduce_any() -> tensor<1x1xi1> {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ARG0:.*]] = tensor.empty() : tensor<1x1024xi1>
  // CHECK: %[[ARG1:.*]] = tensor.empty() : tensor<1x1xi1>
  // CHECK: %[[ARG2:.*]] = tensor.empty() : tensor<1x1024xf16>
  // CHECK: %[[ARG3:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<1x1024xi1>) outs(%[[ARG2]] : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  // CHECK: %[[ARG4:.*]] = tensor.empty() : tensor<1x1xf16>
  // CHECK: %[[ARG5:.*]] = hivm.hir.vreduce <max> ins(%[[ARG3]] : tensor<1x1024xf16>) outs(%[[ARG4]] : tensor<1x1xf16>) reduce_dims = [1] -> tensor<1x1xf16>
  // CHECK: %[[ARG6:.*]] = tensor.empty() : tensor<1x1xi8>
  // CHECK: %[[ARG7:.*]] = hivm.hir.vcast ins(%[[ARG5]] : tensor<1x1xf16>) outs(%[[ARG6]] : tensor<1x1xi8>) -> tensor<1x1xi8>
  // CHECK: %[[ARG8:.*]] = tensor.empty() : tensor<1x1xf16>
  // CHECK: %[[ARG9:.*]] = hivm.hir.vcast ins(%[[ARG7]] : tensor<1x1xi8>) outs(%[[ARG8]] : tensor<1x1xf16>) -> tensor<1x1xf16>
  // CHECK: %[[ARG10:.*]] = tensor.empty() : tensor<1x1xf16>
  // CHECK: %[[ARG11:.*]] = hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG10]] : tensor<1x1xf16>) -> tensor<1x1xf16>
  // CHECK: %[[ARG12:.*]] = hivm.hir.vcmp ins(%[[ARG9]], %[[ARG11]] : tensor<1x1xf16>, tensor<1x1xf16>) outs(%[[ARG1]] : tensor<1x1xi1>) compare_mode = <ne> -> tensor<1x1xi1>
  // CHECK: return %[[ARG12]] : tensor<1x1xi1>
  %arg0 = tensor.empty() : tensor<1x1024xi1>
  %arg1 = tensor.empty() : tensor<1x1xi1>
  %0 = hivm.hir.vreduce <any> ins(%arg0 : tensor<1x1024xi1>) outs(%arg1 : tensor<1x1xi1>) reduce_dims = [1] -> tensor<1x1xi1>
  return %0 : tensor<1x1xi1>
}

// -----
func.func @test_reduce_any_memref() {
    // CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[alloc:.*]] = memref.alloc() : memref<1x1024xi1>
    // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<1x1xi1>
    // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<1x1024xf16>
    // CHECK: hivm.hir.vcast ins(%[[alloc]] : memref<1x1024xi1>) outs(%[[alloc_1]] : memref<1x1024xf16>)
    // CHECK: %[[alloc_2:.*]] = memref.alloc() : memref<1x1xf16>
    // CHECK: hivm.hir.vreduce <max> ins(%[[alloc_1]] : memref<1x1024xf16>) outs(%[[alloc_2]] : memref<1x1xf16>) reduce_dims = [1]
    // CHECK: %[[alloc_3:.*]] = memref.alloc() : memref<1x1xi8>
    // CHECK: hivm.hir.vcast ins(%[[alloc_2]] : memref<1x1xf16>) outs(%[[alloc_3]] : memref<1x1xi8>)
    // CHECK: %[[alloc_4:.*]] = memref.alloc() : memref<1x1xf16>
    // CHECK: hivm.hir.vcast ins(%[[alloc_3]] : memref<1x1xi8>) outs(%[[alloc_4]] : memref<1x1xf16>)
    // CHECK: %[[alloc_5:.*]] = memref.alloc() : memref<1x1xf16>
    // CHECK: hivm.hir.vbrc ins(%[[cst]] : f16) outs(%[[alloc_5]] : memref<1x1xf16>)
    // CHECK: hivm.hir.vcmp ins(%[[alloc_4]], %[[alloc_5]] : memref<1x1xf16>, memref<1x1xf16>) outs(%[[alloc_0]] : memref<1x1xi1>) compare_mode = <ne>
  %arg0 = memref.alloc() : memref<1x1024xi1>
  %arg1 = memref.alloc() : memref<1x1xi1>
  hivm.hir.vreduce <any> ins(%arg0 : memref<1x1024xi1>) outs(%arg1 : memref<1x1xi1>) reduce_dims = [1]
  return
}

// -----
func.func @test_reduce_all() -> tensor<1x1xi1> {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ARG0:.*]] = tensor.empty() : tensor<1x1024xi1>
  // CHECK: %[[ARG1:.*]] = tensor.empty() : tensor<1x1xi1>
  // CHECK: %[[ARG2:.*]] = tensor.empty() : tensor<1x1024xf16>
  // CHECK: %[[ARG3:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<1x1024xi1>) outs(%[[ARG2]] : tensor<1x1024xf16>) -> tensor<1x1024xf16>
  // CHECK: %[[ARG4:.*]] = tensor.empty() : tensor<1x1xf16>
  // CHECK: %[[ARG5:.*]] = hivm.hir.vreduce <min> ins(%[[ARG3]] : tensor<1x1024xf16>) outs(%[[ARG4]] : tensor<1x1xf16>) reduce_dims = [1] -> tensor<1x1xf16>
  // CHECK: %[[ARG6:.*]] = tensor.empty() : tensor<1x1xi8>
  // CHECK: %[[ARG7:.*]] = hivm.hir.vcast ins(%[[ARG5]] : tensor<1x1xf16>) outs(%[[ARG6]] : tensor<1x1xi8>) -> tensor<1x1xi8>
  // CHECK: %[[ARG8:.*]] = tensor.empty() : tensor<1x1xf16>
  // CHECK: %[[ARG9:.*]] = hivm.hir.vcast ins(%[[ARG7]] : tensor<1x1xi8>) outs(%[[ARG8]] : tensor<1x1xf16>) -> tensor<1x1xf16>
  // CHECK: %[[ARG10:.*]] = tensor.empty() : tensor<1x1xf16>
  // CHECK: %[[ARG11:.*]] = hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG10]] : tensor<1x1xf16>) -> tensor<1x1xf16>
  // CHECK: %[[ARG12:.*]] = hivm.hir.vcmp ins(%[[ARG9]], %[[ARG11]] : tensor<1x1xf16>, tensor<1x1xf16>) outs(%[[ARG1]] : tensor<1x1xi1>) compare_mode = <ne> -> tensor<1x1xi1>
  // CHECK: return %[[ARG12]] : tensor<1x1xi1>
  %arg0 = tensor.empty() : tensor<1x1024xi1>
  %arg1 = tensor.empty() : tensor<1x1xi1>
  %0 = hivm.hir.vreduce <all> ins(%arg0 : tensor<1x1024xi1>) outs(%arg1 : tensor<1x1xi1>) reduce_dims = [1] -> tensor<1x1xi1>
  return %0 : tensor<1x1xi1>
}

//===----------------------------------------------------------------------===//
// Test VReduce Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
// CHECK-LABEL: func @test_reduce_sum_ar_b64
func.func @test_reduce_sum_ar_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant 0 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<24x1xi64>)
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<24x1xi64>
  hivm.hir.vreduce <sum> ins(%src : memref<24x32xi64>) outs(%dst : memref<24x1xi64>) reduce_dims = [1]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_min_ar_b64
func.func @test_reduce_min_ar_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<24x1xi64>)
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<24x1xi64>
  hivm.hir.vreduce <min> ins(%src : memref<24x32xi64>) outs(%dst : memref<24x1xi64>) reduce_dims = [1]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_max_ar_b64
func.func @test_reduce_max_ar_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<24x1xi64>)
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<24x1xi64>
  hivm.hir.vreduce <max> ins(%src : memref<24x32xi64>) outs(%dst : memref<24x1xi64>) reduce_dims = [1]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_min_with_index_int64
func.func @test_reduce_min_with_index_int64() {
  // CHECK: %[[INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<1x2xi64>)
  %src = memref.alloc() : memref<2x2xi64>
  %dst1 = memref.alloc() : memref<1x2xi64>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi64>) outs(%dst1, %dst2 : memref<1x2xi64>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_min_with_index_int32
func.func @test_reduce_min_with_index_int32() {
  // CHECK: %[[INIT:.*]] = arith.constant 2147483647 : i32
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i32) outs(%[[ALLOC_0:.*]] : memref<1x2xi32>)
  %src = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi32>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi32>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_min_with_index_int16
func.func @test_reduce_min_with_index_int16() {
  // CHECK: %[[INIT:.*]] = arith.constant 32767 : i16
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i16) outs(%[[ALLOC_0:.*]] : memref<1x2xi16>
  %src = memref.alloc() : memref<2x2xi16>
  %dst1 = memref.alloc() : memref<1x2xi16>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi16>) outs(%dst1, %dst2 : memref<1x2xi16>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_max_with_index_int64
func.func @test_reduce_max_with_index_int64() {
  // CHECK: %[[INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<1x2xi64>
  %src = memref.alloc() : memref<2x2xi64>
  %dst1 = memref.alloc() : memref<1x2xi64>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi64>) outs(%dst1, %dst2 : memref<1x2xi64>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_max_with_index_int32
func.func @test_reduce_max_with_index_int32() {
  // CHECK: %[[INIT:.*]] = arith.constant -2147483648 : i32
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i32) outs(%[[ALLOC_0:.*]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi32>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi32>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_max_with_index_int16
func.func @test_reduce_max_with_index_int16() {
  // CHECK: %[[INIT:.*]] = arith.constant -32768 : i16
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i16) outs(%[[ALLOC_0:.*]] : memref<1x2xi16>
  %src = memref.alloc() : memref<2x2xi16>
  %dst1 = memref.alloc() : memref<1x2xi16>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi16>) outs(%dst1, %dst2 : memref<1x2xi16>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_sum_ra_b64
func.func @test_reduce_sum_ra_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant 0 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<1x32xi64>)
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<1x32xi64>
  hivm.hir.vreduce <sum> ins(%src : memref<24x32xi64>) outs(%dst : memref<1x32xi64>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_min_ra_b64
func.func @test_reduce_min_ra_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<1x32xi64>)
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<1x32xi64>
  hivm.hir.vreduce <min> ins(%src : memref<24x32xi64>) outs(%dst : memref<1x32xi64>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_max_ra_b64
func.func @test_reduce_max_ra_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<1x32xi64>)
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<1x32xi64>
  hivm.hir.vreduce <max> ins(%src : memref<24x32xi64>) outs(%dst : memref<1x32xi64>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_sum_r_b64
func.func @test_reduce_sum_r_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant 0 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<1xi64>
  %src = memref.alloc() : memref<32xi64>
  %dst = memref.alloc() : memref<1xi64>
  hivm.hir.vreduce <sum> ins(%src : memref<32xi64>) outs(%dst : memref<1xi64>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_min_r_b64
func.func @test_reduce_min_r_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<1xi64>)
  %src = memref.alloc() : memref<32xi64>
  %dst = memref.alloc() : memref<1xi64>
  hivm.hir.vreduce <min> ins(%src : memref<32xi64>) outs(%dst : memref<1xi64>) reduce_dims = [0]
  return
}

// -----
// CHECK-LABEL: func @test_reduce_max_r_b64
func.func @test_reduce_max_r_b64() {
  // CHECK: %[[INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK: hivm.hir.vbrc ins(%[[INIT]] : i64) outs(%[[ALLOC_0:.*]] : memref<1xi64>
  %src = memref.alloc() : memref<32xi64>
  %dst = memref.alloc() : memref<1xi64>
  hivm.hir.vreduce <max> ins(%src : memref<32xi64>) outs(%dst : memref<1xi64>) reduce_dims = [0]
  return
}

// -----
// CHECK: func.func @test_absi_decompose(
// CHECK: %[[SRC0:.*]]: memref<8x8xi32>, %[[DST0:.*]]: memref<8x8xi32>,
// CHECK-SAME: %[[SRC1:.*]]: memref<?x?xi16>, %[[DST1:.*]]: memref<?x?xi16>)
func.func @test_absi_decompose(%src0 : memref<8x8xi32>, %dst0 : memref<8x8xi32>,
                               %src1 : memref<?x?xi16>, %dst1 : memref<?x?xi16>) attributes { enable_auto_mark_buffer_size } {
  // CHECK-DAG: %[[CST_I16:.*]] = arith.constant 1 : i16
  // CHECK-DAG: %[[CST_I32:.*]] = arith.constant 1 : i32
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<8x8xi32>
  // CHECK: hivm.hir.vnot ins(%[[SRC0]] : memref<8x8xi32>) outs(%[[ALLOC0]] : memref<8x8xi32>)
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x8xi32>
  // CHECK: hivm.hir.vadd ins(%[[ALLOC0]], %[[CST_I32]] : memref<8x8xi32>, i32) outs(%[[ALLOC1]] : memref<8x8xi32>)
  // CHECK: hivm.hir.vmax ins(%[[SRC0]], %[[ALLOC1]] : memref<8x8xi32>, memref<8x8xi32>) outs(%[[DST0]] : memref<8x8xi32>)
  hivm.hir.vabs ins(%src0 : memref<8x8xi32>) outs(%dst0 : memref<8x8xi32>)
  // CHECK: %[[ALLOC2:.*]] = memref.alloc{{.*}} : memref<?x?xi16>
  // CHECK: hivm.hir.vnot ins(%[[SRC1]] : memref<?x?xi16>) outs(%[[ALLOC2]] : memref<?x?xi16>)
  // CHECK: %[[ALLOC3:.*]] = memref.alloc{{.*}} : memref<?x?xi16>
  // CHECK: hivm.hir.vadd ins(%[[ALLOC2]], %[[CST_I16]] : memref<?x?xi16>, i16) outs(%[[ALLOC3]] : memref<?x?xi16>)
  // CHECK: hivm.hir.vmax ins(%[[SRC1]], %[[ALLOC3]] : memref<?x?xi16>, memref<?x?xi16>) outs(%[[DST1]] : memref<?x?xi16>)
  hivm.hir.vabs ins(%src1 : memref<?x?xi16>) outs(%dst1 : memref<?x?xi16>)
  return
}

// -----
// CHECK: func.func @test_absi_decompose_strided(
// CHECK: %[[SRC0:.*]]: memref<8x4x2xi16, strided<[8, 2, 1]>>, %[[DST0:.*]]: memref<8x4x2xi16, strided<[8, 2, 1]>>,
// CHECK-SAME: %[[SRC1:.*]]: memref<8x4x2xi16, strided<[8, 4, 1]>>, %[[DST1:.*]]: memref<8x4x2xi16, strided<[8, 4, 1]>>)
func.func @test_absi_decompose_strided(%src0 : memref<8x4x2xi16, strided<[8, 2, 1]>>,
                                       %dst0 : memref<8x4x2xi16, strided<[8, 2, 1]>>,
                                       %src1 : memref<8x4x2xi16, strided<[8, 4, 1]>>,
                                       %dst1 : memref<8x4x2xi16, strided<[8, 4, 1]>>) {
  // CHECK: %[[CST_I16:.*]] = arith.constant 1 : i16
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<8x4x2xi16>
  // CHECK: hivm.hir.vnot ins(%[[SRC0]] : memref<8x4x2xi16, strided<[8, 2, 1]>>) outs(%[[ALLOC0]] : memref<8x4x2xi16>)
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x4x2xi16>
  // CHECK: hivm.hir.vadd ins(%[[ALLOC0]], %[[CST_I16]] : memref<8x4x2xi16>, i16) outs(%[[ALLOC1]]
  // CHECK: hivm.hir.vmax ins(%[[SRC0]], %[[ALLOC1]] : memref<8x4x2xi16, strided<[8, 2, 1]>>, memref<8x4x2xi16>) outs(%[[DST0]]
  hivm.hir.vabs ins(%src0 : memref<8x4x2xi16, strided<[8, 2, 1]>>)
                outs(%dst0 : memref<8x4x2xi16, strided<[8, 2, 1]>>)
  // CHECK: %[[ALLOC2:.*]] = memref.alloc{{.*}} : memref<8x4x2xi16>
  // CHECK: hivm.hir.vnot ins(%[[SRC1]] : memref<8x4x2xi16, strided<[8, 4, 1]>>) outs(%[[ALLOC2]] : memref<8x4x2xi16>)
  // CHECK: %[[ALLOC3:.*]] = memref.alloc{{.*}} : memref<8x4x2xi16>
  // CHECK: hivm.hir.vadd ins(%[[ALLOC2]], %[[CST_I16]] : memref<8x4x2xi16>, i16) outs(%[[ALLOC3]]
  // CHECK: hivm.hir.vmax ins(%[[SRC1]], %[[ALLOC3]] : memref<8x4x2xi16, strided<[8, 4, 1]>>, memref<8x4x2xi16>) outs(%[[DST1]]
  hivm.hir.vabs ins(%src1 : memref<8x4x2xi16, strided<[8, 4, 1]>>)
                outs(%dst1 : memref<8x4x2xi16, strided<[8, 4, 1]>>)
  return
}

// -----
// CHECK: func.func @test_absi_decompose_same_src_dst(
// CHECK: %[[SRC0:.*]]: memref<8x4x2xi16, strided<[8, 2, 1]>>, %[[SRC1:.*]]: memref<8x4x2xi16, strided<[8, 4, 1]>>
func.func @test_absi_decompose_same_src_dst(%src0 : memref<8x4x2xi16, strided<[8, 2, 1]>>,
                                            %src1 : memref<8x4x2xi16, strided<[8, 4, 1]>>) {
  // CHECK: %[[CST_I16:.*]] = arith.constant 1 : i16
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<8x4x2xi16>
  // CHECK: hivm.hir.vnot ins(%[[SRC0]] : memref<8x4x2xi16, strided<[8, 2, 1]>>) outs(%[[ALLOC0]] : memref<8x4x2xi16>)
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x4x2xi16>
  // CHECK: hivm.hir.vadd ins(%[[ALLOC0]], %[[CST_I16]] : memref<8x4x2xi16>, i16) outs(%[[ALLOC1]]
   // CHECK: hivm.hir.vmax ins(%[[SRC0]], %[[ALLOC1]] : memref<8x4x2xi16, strided<[8, 2, 1]>>, memref<8x4x2xi16>) outs(%[[SRC0]]
  hivm.hir.vabs ins(%src0 : memref<8x4x2xi16, strided<[8, 2, 1]>>)
                outs(%src0 : memref<8x4x2xi16, strided<[8, 2, 1]>>)
  // CHECK: %[[ALLOC2:.*]] = memref.alloc{{.*}} : memref<8x4x2xi16>
  // CHECK: hivm.hir.vnot ins(%[[SRC1]] : memref<8x4x2xi16, strided<[8, 4, 1]>>) outs(%[[ALLOC2]] : memref<8x4x2xi16>)
  // CHECK: %[[ALLOC3:.*]] = memref.alloc{{.*}} : memref<8x4x2xi16>
  // CHECK: hivm.hir.vadd ins(%[[ALLOC2]], %[[CST_I16]] : memref<8x4x2xi16>, i16) outs(%[[ALLOC3]]
  // CHECK: hivm.hir.vmax ins(%[[SRC1]], %[[ALLOC3]] : memref<8x4x2xi16, strided<[8, 4, 1]>>, memref<8x4x2xi16>) outs(%[[SRC1]]
  hivm.hir.vabs ins(%src1 : memref<8x4x2xi16, strided<[8, 4, 1]>>)
                outs(%src1 : memref<8x4x2xi16, strided<[8, 4, 1]>>)
  return
}

// -----
// CHECK: func.func @test_scalar_fp32_to_bf16(
// CHECK: %[[SRC0:.*]]: memref<1024xbf16>
func.func @test_scalar_fp32_to_bf16(%1 :memref<1024xbf16>) {
  // CHECK: %[[CST_ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[CST_VAL:.*]] = arith.constant 2.100000e-01 : f32
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: memref.store %[[CST_VAL]], %[[ALLOC0]][%[[CST_ZERO]]] : memref<1xf32>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1xbf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC0]] : memref<1xf32>) outs(%[[ALLOC1]] : memref<1xbf16>) round_mode = <round>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOC1]][%[[CST_ZERO]]] : memref<1xbf16>
  // CHECK: hivm.hir.vbrc ins(%[[RES]] : bf16) outs(%[[SRC0]] : memref<1024xbf16>)
  %cst = arith.constant 0.21 : f32
  %6 = "arith.truncf"(%cst) : (f32) -> bf16
  hivm.hir.vbrc ins(%6 : bf16) outs(%1 : memref<1024xbf16>)
  return
}

// -----
// CHECK-LABEL: func @test_scalar_i32_to_bf16(
// CHECK: %[[SRC0:.*]]: i32
// CHECK: %[[SRC1:.*]]: memref<1024xbf16>
func.func @test_scalar_i32_to_bf16(%0: i32, %1 :memref<1024xbf16>) {
  // CHECK: %[[CST_ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: memref.store %[[SRC0]], %[[ALLOC0]][%[[CST_ZERO]]] : memref<1xi32>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC0]] : memref<1xi32>) outs(%[[ALLOC1]] : memref<1xf32>)
  // CHECK: %[[ALLOC2:.*]] = memref.alloc() : memref<1xbf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC1]] : memref<1xf32>) outs(%[[ALLOC2]] : memref<1xbf16>)
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOC2]][%[[CST_ZERO]]] : memref<1xbf16>
  // CHECK: hivm.hir.vbrc ins(%[[RES]] : bf16) outs(%[[SRC1]] : memref<1024xbf16>)
  %6 = "arith.sitofp"(%0) : (i32) -> bf16
  hivm.hir.vbrc ins(%6 : bf16) outs(%1 : memref<1024xbf16>)
  return
}

// -----
// CHECK-LABEL: func @test_scalar_bf16_to_f32(
// CHECK: %[[SRC0:.*]]: bf16
// CHECK: %[[SRC1:.*]]: memref<1024xf32>
func.func @test_scalar_bf16_to_f32(%0: bf16, %1 :memref<1024xf32>) {
  // CHECK: %[[CST_ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1xbf16>
  // CHECK: memref.store %[[SRC0]], %[[ALLOC0]][%[[CST_ZERO]]] : memref<1xbf16>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC0]] : memref<1xbf16>) outs(%[[ALLOC1]] : memref<1xf32>)
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOC1]][%[[CST_ZERO]]] : memref<1xf32>
  // CHECK: hivm.hir.vbrc ins(%[[RES]] : f32) outs(%[[SRC1]] : memref<1024xf32>)
  %6 = arith.extf %0 : bf16 to f32
  hivm.hir.vbrc ins(%6 : f32) outs(%1 : memref<1024xf32>)
  return
}

// -----
module {
  // CHECK-LABEL: func @test_decompose_vbrc_mark_buffer_size(
  func.func @test_decompose_vbrc_mark_buffer_size(%2 : index, %3 : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<1x1x4096xf32>
  // CHECK: %[[alloc_1:.*]] = memref.alloc(%arg0, %arg1) : memref<?x?x4096xf32>
  // CHECK: %[[alloc_2:.*]] = memref.alloc(%arg1) : memref<1x?x4096xf32>
  // CHECK: hivm.hir.vbrc ins(%[[alloc_0]] : memref<1x1x4096xf32>) outs(%[[alloc_2]] : memref<1x?x4096xf32>) broadcast_dims = [1]
  // CHECK: hivm.hir.vbrc ins(%[[alloc_2]] : memref<1x?x4096xf32>) outs(%[[alloc_1]] : memref<?x?x4096xf32>) broadcast_dims = [0]
  // CHECK: annotation.mark %[[alloc_1]] {buffer_size_in_byte = 21856 : i64} : memref<?x?x4096xf32>
    %src = memref.alloc() : memref<1x1x4096xf32>
    %alloc_0 = memref.alloc(%2, %3) : memref<?x?x4096xf32>
    hivm.hir.vbrc ins(%src : memref<1x1x4096xf32>) outs(%alloc_0 : memref<?x?x4096xf32>) broadcast_dims = [0, 1]
    annotation.mark %alloc_0 {buffer_size_in_byte = 21856 : i64} : memref<?x?x4096xf32>
    return
  }
}

// -----
module {
  // CHECK-LABEL: func @test_decompose_vbrc_mark_buffer_size_static(
  func.func @test_decompose_vbrc_mark_buffer_size_static(%2 : index, %3 : index) {
  // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<32768xi8>
  // CHECK: %[[alloc_2:.*]] = memref.alloc(%arg1) : memref<1x?x4096xf32>
  // CHECK: annotation.mark %[[alloc_2]] {buffer_size_in_byte = 131072 : i64} : memref<1x?x4096xf32>
    %c0 = arith.constant 0: index
    %src = memref.alloc() : memref<1x1x4096xf32>
    %alloc_0 = memref.alloc() : memref<32768xi8>
    %buffer = memref.view %alloc_0 [%c0][%2, %3]: memref<32768xi8> to memref<?x?x4096xf32>
    hivm.hir.vbrc ins(%src : memref<1x1x4096xf32>) outs(%buffer : memref<?x?x4096xf32>) broadcast_dims = [0, 1]
    annotation.mark %buffer {buffer_size_in_byte = 21856 : i64} : memref<?x?x4096xf32>
    return
  }
}

// -----
module {
  func.func @test_decompose_vcast_mark_buffer_size(%2 : index, %3 : index) attributes { enable_auto_mark_buffer_size } {
    // CHECK: %alloc = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    // CHECK: %alloc_0 = memref.alloc(%arg0, %arg1) : memref<?x?xi8>
    // CHECK: %alloc_1 = memref.alloc(%arg0, %arg1) : memref<?x?xf16>
    // CHECK: hivm.hir.vcast ins(%alloc : memref<?x?xf32>) outs(%alloc_1 : memref<?x?xf16>)
    // CHECK: hivm.hir.vcast ins(%alloc_1 : memref<?x?xf16>) outs(%alloc_0 : memref<?x?xi8>)
    // CHECK: annotation.mark %alloc {buffer_size_in_byte = 24576 : i64} : memref<?x?xf32>
    // CHECK: annotation.mark %alloc_0 {buffer_size_in_byte = 24576 : i64} : memref<?x?xi8>

    %src1 = memref.alloc(%2, %3) : memref<?x?xf32>
    %dst1 = memref.alloc(%2, %3) : memref<?x?xi8>
    hivm.hir.vcast ins(%src1 : memref<?x?xf32>) outs(%dst1 : memref<?x?xi8>)
    annotation.mark %src1 {buffer_size_in_byte = 24576 : i64} : memref<?x?xf32>
    annotation.mark %dst1 {buffer_size_in_byte = 24576 : i64} : memref<?x?xi8>
    return
  }
}

// -----
func.func @test_decompose_vcast_mark_buffer_size_i8_to_i1(%2 : index, %3 : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %cst = arith.constant 0.000000e+00 : f16
  // CHECK: %alloc = memref.alloc(%arg0, %arg1) : memref<?x?xi8>
  // CHECK: %alloc_0 = memref.alloc(%arg0, %arg1) : memref<?x?xi1>
  // CHECK: %alloc_1 = memref.alloc(%arg0, %arg1) : memref<?x?xf16>
  // CHECK: hivm.hir.vcast ins(%alloc : memref<?x?xi8>) outs(%alloc_1 : memref<?x?xf16>)
  // CHECK: %alloc_2 = memref.alloc(%arg0, %arg1) : memref<?x?xf16>
  // CHECK: hivm.hir.vbrc ins(%cst : f16) outs(%alloc_2 : memref<?x?xf16>)
  // CHECK: hivm.hir.vcmp ins(%alloc_1, %alloc_2 : memref<?x?xf16>, memref<?x?xf16>) outs(%alloc_0 : memref<?x?xi1>) compare_mode = <ne>
  // CHECK: annotation.mark %alloc {buffer_size_in_byte = 24576 : i64} : memref<?x?xi8>
  // CHECK: annotation.mark %alloc_0 {buffer_size_in_byte = 24576 : i64} : memref<?x?xi1>

  %arg0 = memref.alloc(%2, %3) : memref<?x?xi8>
  %arg1 = memref.alloc(%2, %3) : memref<?x?xi1>
  hivm.hir.vcast ins(%arg0 : memref<?x?xi8>) outs(%arg1 : memref<?x?xi1>)

  annotation.mark %arg0 {buffer_size_in_byte = 24576 : i64} : memref<?x?xi8>
  annotation.mark %arg1 {buffer_size_in_byte = 24576 : i64} : memref<?x?xi1>
  return
}

// -----
func.func @test_decompose_vcast_mark_buffer_size_bool_to_f32(%2 : index, %3 : index) attributes { enable_auto_mark_buffer_size } {
  // CHECK: %alloc = memref.alloc(%arg0, %arg1) : memref<?x?xi1>
  // CHECK: %alloc_0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  // CHECK: hivm.hir.vcast ins(%alloc : memref<?x?xi1>) outs(%alloc_0 : memref<?x?xf32>)
  // CHECK: annotation.mark %alloc {buffer_size_in_byte = 1024 : i64} : memref<?x?xi1>
  // CHECK: annotation.mark %alloc_0 {buffer_size_in_byte = 32768 : i64} : memref<?x?xf32>

  %arg0 = memref.alloc(%2, %3) : memref<?x?xi1>
  %arg1 = memref.alloc(%2, %3) : memref<?x?xf32>

  hivm.hir.vcast ins(%arg0 : memref<?x?xi1>) outs(%arg1 : memref<?x?xf32>)
  annotation.mark %arg0 {buffer_size_in_byte = 1024 : i64} : memref<?x?xi1>
  annotation.mark %arg1 {buffer_size_in_byte = 32768 : i64} : memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL:   func.func @test_decompose_vsub_f16(
// CHECK-SAME:                                       %[[VAL_0:.*]]: memref<1024xf16>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: memref<1024xf16>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant -1.000000e+00 : f16
// CHECK:           hivm.hir.vadd ins(%[[VAL_0]], %[[VAL_2]] : memref<1024xf16>, f16) outs(%[[VAL_1]] : memref<1024xf16>)
// CHECK:           return
// CHECK:         }
func.func @test_decompose_vsub_f16(%src: memref<1024xf16>, %dst: memref<1024xf16>) {
  %cst = arith.constant 1.000000e+00 : f16
  hivm.hir.vsub ins(%src, %cst : memref<1024xf16>, f16)
                outs(%dst : memref<1024xf16>)
  return
}

// -----

// CHECK-LABEL:   func.func @test_decompose_vsub_i32(
// CHECK-SAME:                                       %[[VAL_0:.*]]: memref<1024xi32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: memref<1024xi32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant -1 : i32
// CHECK:           hivm.hir.vadd ins(%[[VAL_0]], %[[VAL_2]] : memref<1024xi32>, i32) outs(%[[VAL_1]] : memref<1024xi32>)
// CHECK:           return
// CHECK:         }
func.func @test_decompose_vsub_i32(%src: memref<1024xi32>, %dst: memref<1024xi32>) {
  %cst = arith.constant 1 : i32
  hivm.hir.vsub ins(%src, %cst : memref<1024xi32>, i32)
                outs(%dst : memref<1024xi32>)
  return
}

// -----

// CHECK-LABEL:   func.func @test_decompose_vsub_f16(
// CHECK-SAME:                                       %[[VAL_0:.*]]: memref<1024xf16>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: f16,
// CHECK-SAME:                                       %[[VAL_2:.*]]: memref<1024xf16>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:           %[[VAL_4:.*]] = arith.subf %[[VAL_3]], %[[VAL_1]] : f16
// CHECK:           hivm.hir.vadd ins(%[[VAL_0]], %[[VAL_4]] : memref<1024xf16>, f16) outs(%[[VAL_2]] : memref<1024xf16>)
// CHECK:           return
// CHECK:         }
func.func @test_decompose_vsub_f16(%src: memref<1024xf16>, %cst: f16, %dst: memref<1024xf16>) {
  hivm.hir.vsub ins(%src, %cst : memref<1024xf16>, f16)
                outs(%dst : memref<1024xf16>)
  return
}

// -----

// CHECK-LABEL:   func.func @test_decompose_vsub_i32(
// CHECK-SAME:                                       %[[VAL_0:.*]]: memref<1024xi32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                       %[[VAL_2:.*]]: memref<1024xi32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           hivm.hir.vadd ins(%[[VAL_0]], %[[VAL_4]] : memref<1024xi32>, i32) outs(%[[VAL_2]] : memref<1024xi32>)
// CHECK:           return
// CHECK:         }
func.func @test_decompose_vsub_i32(%src: memref<1024xi32>, %cst: i32, %dst: memref<1024xi32>) {
  hivm.hir.vsub ins(%src, %cst : memref<1024xi32>, i32)
                outs(%dst : memref<1024xi32>)
  return
}

// -----

// CHECK-LABEL: func @test_log_scalar_f32
func.func @test_log_scalar_f32(%src : f32, %out : memref<16xf32>) {
  // CHECK-SAME: %[[src:.*]]: f32,
  // CHECK: %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: memref.store %[[src]], {{.*}}[%[[VAL_0]]] : memref<1xf32>
  // CHECK: %[[ALLOC2:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vln ins(%[[ALLOC1]] : memref<1xf32>) outs(%[[ALLOC2]] : memref<1xf32>)
  // CHECK: memref.load {{.*}}[%[[VAL_0]]] : memref<1xf32>
  %1 = math.log %src : f32
  hivm.hir.vbrc ins(%1 : f32) outs(%out : memref<16xf32>)
  return
}

// -----

// CHECK-LABEL: func @test_log_scalar_f16
func.func @test_log_scalar_f16(%src : f16, %out : memref<16xf16>) {
  // CHECK-SAME: %[[src:.*]]: f16,
  // CHECK: %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1xf16>
  // CHECK: memref.store %[[src]], {{.*}}[%[[VAL_0]]] : memref<1xf16>
  // CHECK: %[[ALLOC2:.*]] = memref.alloc() : memref<1xf16>
  // CHECK: hivm.hir.vln ins(%[[ALLOC1]] : memref<1xf16>) outs(%[[ALLOC2]] : memref<1xf16>)
  // CHECK: memref.load {{.*}}[%[[VAL_0]]] : memref<1xf16>
  %1 = math.log %src : f16
  hivm.hir.vbrc ins(%1 : f16) outs(%out : memref<16xf16>)
  return
}

//===----------------------------------------------------------------------===//
// Test Arith_MulUIExtendedOp Decompose MemRef(I32)
//===----------------------------------------------------------------------===//
// -----
func.func @test_muluiextended_op_memref_i32() {
  // CHECK: %[[VAL_0:.*]] = arith.constant 32 : i64
  // CHECK: %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK: %[[VAL_2:.*]] = arith.constant 0 : index
  // CHECK: %[[VAL_3:.*]] = arith.constant 6 : index
  // CHECK: %[[VAL_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  // CHECK: %[[VAL_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  // CHECK: %[[VAL_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  // CHECK: %[[VAL_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  // CHECK: scf.for %[[VAL_8:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_1]] {
  // CHECK:   %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_8]]] : memref<6xi32>
  // CHECK:   %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_8]]] : memref<6xi32>
  // CHECK:   %[[VAL_11:.*]] = arith.extsi %[[VAL_9]] : i32 to i64
  // CHECK:   %[[VAL_12:.*]] = arith.extsi %[[VAL_10]] : i32 to i64
  // CHECK:   %[[VAL_13:.*]] = arith.muli %[[VAL_11]], %[[VAL_12]] : i64
  // CHECK:   %[[VAL_14:.*]] = arith.shli %[[VAL_13]], %[[VAL_0]] : i64
  // CHECK:   %[[VAL_15:.*]] = arith.shrsi %[[VAL_14]], %[[VAL_0]] : i64
  // CHECK:   %[[VAL_16:.*]] = arith.trunci %[[VAL_15]] : i64 to i32
  // CHECK:   %[[VAL_17:.*]] = arith.shrsi %[[VAL_13]], %[[VAL_0]] : i64
  // CHECK:   %[[VAL_18:.*]] = arith.trunci %[[VAL_17]] : i64 to i32
  // CHECK:   memref.store %[[VAL_16]], %[[VAL_6]]{{\[}}%[[VAL_8]]] : memref<6xi32>
  // CHECK:   memref.store %[[VAL_18]], %[[VAL_7]]{{\[}}%[[VAL_8]]] : memref<6xi32>
  // CHECK: }

  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  scf.for %arg0 = %c0 to %c6 step %c1 {
    %0 = memref.load %alloc[%arg0] : memref<6xi32>
    %1 = memref.load %alloc_0[%arg0] : memref<6xi32>
    %low, %high = arith.mului_extended %0, %1 : i32
    memref.store %low, %alloc_1[%arg0] : memref<6xi32>
    memref.store %high, %alloc_2[%arg0] : memref<6xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test VCast Decompose
//===----------------------------------------------------------------------===//
// -----
module {
  func.func @test_extui_b32_lt() {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<64x8xi32>
    %alloc_0 = memref.alloc() : memref<64x8xi32>
    %alloc_1 = memref.alloc() : memref<64x8xi1>
    %alloc_2 = memref.alloc() : memref<64x8xi8>
    // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x8xi1>
    // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<64x8xi8>
    // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK: %[[src0:.*]] = memref.load {{.*}}[%[[arg0]], %[[arg1]]]
    // CHECK: %[[src1:.*]] = memref.load {{.*}}[%[[arg0]], %[[arg1]]]
    // CHECK: %[[dst:.*]] = arith.cmpi slt, %[[src0]], %[[src1]] : i32
    // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
    // CHECK: memref.store %[[dst1]], %[[ARG1]][%[[arg0]], %[[arg1]]] : memref<64x8xi8>
    // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<64x8xf16>
    // CHECK: hivm.hir.vcast ins(%[[ARG1]] : memref<64x8xi8>) outs(%[[ARG2]] : memref<64x8xf16>)
    // CHECK: %[[ARG3:.*]] = memref.alloc() : memref<64x8xf16>
    // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG3]] : memref<64x8xf16>)
    // CHECK: hivm.hir.vcmp ins(%[[ARG2]], %[[ARG3]] : memref<64x8xf16>, memref<64x8xf16>) outs(%[[ALLOC]] : memref<64x8xi1>) compare_mode = <ne>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<64x8xi32>
        %1 = memref.load %alloc_0[%arg0, %arg1] : memref<64x8xi32>
        %2 = arith.cmpi slt, %0, %1 : i32
        %3 = arith.extui %2 : i1 to i8
        memref.store %3, %alloc_2[%arg0, %arg1] : memref<64x8xi8>
      }
    }
    hivm.hir.vcast ins(%alloc_2 : memref<64x8xi8>) outs(%alloc_1 : memref<64x8xi1>)
    return
  }
}

//===----------------------------------------------------------------------===//
// Test VCmp i1 To i8  Decompose MemRef
//===----------------------------------------------------------------------===//

// -----
func.func @test_vcmp_b32_lt() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<64x8xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<64x8xi8>
  %allocIn0 = memref.alloc() : memref<64x8xi32>
  %allocIn1 = memref.alloc() : memref<64x8xi32>
  %allocOut = memref.alloc() : memref<64x8xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<64x8xi32>, memref<64x8xi32>) outs(%[[ALLOC_DST1]] : memref<64x8xi8>) compare_mode = <lt>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<64x8xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<64x8xi8>) outs(%[[ARG1]] : memref<64x8xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<64x8xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<64x8xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<64x8xf16>, memref<64x8xf16>) outs(%[[ALLOC_DST0]] : memref<64x8xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<64x8xi32>, memref<64x8xi32>)
                outs(%allocOut : memref<64x8xi1>)
                compare_mode = #hivm.compare_mode<lt>
  return
}

// -----
func.func @test_vcmp_b32_gt() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<64x8xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<64x8xi8>
  %allocIn0 = memref.alloc() : memref<64x8xi32>
  %allocIn1 = memref.alloc() : memref<64x8xi32>
  %allocOut = memref.alloc() : memref<64x8xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<64x8xi32>, memref<64x8xi32>) outs(%[[ALLOC_DST1]] : memref<64x8xi8>) compare_mode = <gt>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<64x8xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<64x8xi8>) outs(%[[ARG1]] : memref<64x8xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<64x8xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<64x8xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<64x8xf16>, memref<64x8xf16>) outs(%[[ALLOC_DST0]] : memref<64x8xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<64x8xi32>, memref<64x8xi32>)
                outs(%allocOut : memref<64x8xi1>)
                compare_mode = #hivm.compare_mode<gt>
  return
}

// -----
func.func @test_vcmp_b32_le() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<64x8xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<64x8xi8>
  %allocIn0 = memref.alloc() : memref<64x8xi32>
  %allocIn1 = memref.alloc() : memref<64x8xi32>
  %allocOut = memref.alloc() : memref<64x8xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<64x8xi32>, memref<64x8xi32>) outs(%[[ALLOC_DST1]] : memref<64x8xi8>) compare_mode = <le>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<64x8xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<64x8xi8>) outs(%[[ARG1]] : memref<64x8xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<64x8xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<64x8xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<64x8xf16>, memref<64x8xf16>) outs(%[[ALLOC_DST0]] : memref<64x8xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<64x8xi32>, memref<64x8xi32>)
                outs(%allocOut : memref<64x8xi1>)
                compare_mode = #hivm.compare_mode<le>
  return
}

// -----
func.func @test_vcmp_b32_ge() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<64x8xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<64x8xi8>
  %allocIn0 = memref.alloc() : memref<64x8xi32>
  %allocIn1 = memref.alloc() : memref<64x8xi32>
  %allocOut = memref.alloc() : memref<64x8xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<64x8xi32>, memref<64x8xi32>) outs(%[[ALLOC_DST1]] : memref<64x8xi8>) compare_mode = <ge>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<64x8xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<64x8xi8>) outs(%[[ARG1]] : memref<64x8xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<64x8xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<64x8xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<64x8xf16>, memref<64x8xf16>) outs(%[[ALLOC_DST0]] : memref<64x8xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<64x8xi32>, memref<64x8xi32>)
                outs(%allocOut : memref<64x8xi1>)
                compare_mode = #hivm.compare_mode<ge>
  return
}

// -----
func.func @test_vcmp_b32_eq() {
  %allocIn0 = memref.alloc() : memref<64x8xi32>
  %allocIn1 = memref.alloc() : memref<64x8xi32>
  %allocOut = memref.alloc() : memref<64x8xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ARG0:.*]], %[[ARG1:.*]] : memref<64x8xi32>, memref<64x8xi32>) outs(%[[ALLOC:.*]] : memref<64x8xi1>)
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<64x8xi32>, memref<64x8xi32>)
                outs(%allocOut : memref<64x8xi1>)
                compare_mode = #hivm.compare_mode<eq>
  return
}

// -----
func.func @test_vcmp_b32_ne() {
  %allocIn0 = memref.alloc() : memref<64x8xi32>
  %allocIn1 = memref.alloc() : memref<64x8xi32>
  %allocOut = memref.alloc() : memref<64x8xi1>

  // CHECK: hivm.hir.vcmp ins(%[[ARG0:.*]], %[[ARG1:.*]] : memref<64x8xi32>, memref<64x8xi32>) outs(%[[ALLOC:.*]] : memref<64x8xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<64x8xi32>, memref<64x8xi32>)
                outs(%allocOut : memref<64x8xi1>)
                compare_mode = #hivm.compare_mode<ne>
  return
}

// -----
func.func @test_vcmp_b16_lt() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi16>
  %allocIn1 = memref.alloc() : memref<1024xi16>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi16>, memref<1024xi16>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <lt>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi16>, memref<1024xi16>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<lt>
  return
}

// -----
func.func @test_vcmp_b16_gt() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi16>
  %allocIn1 = memref.alloc() : memref<1024xi16>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi16>, memref<1024xi16>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <gt>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi16>, memref<1024xi16>)
                 outs(%allocOut : memref<1024xi1>)
                 compare_mode = #hivm.compare_mode<gt>
  return
}

// -----
func.func @test_vcmp_b16_le() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi16>
  %allocIn1 = memref.alloc() : memref<1024xi16>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi16>, memref<1024xi16>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <le>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi16>, memref<1024xi16>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<le>
  return
}

// -----
func.func @test_vcmp_b16_ge() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi16>
  %allocIn1 = memref.alloc() : memref<1024xi16>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi16>, memref<1024xi16>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <ge>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi16>, memref<1024xi16>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<ge>
  return
}

// -----
func.func @test_vcmp_b16_eq() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi16>
  %allocIn1 = memref.alloc() : memref<1024xi16>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi16>, memref<1024xi16>) outs(%[[ALLOC_DST1]] : memref<1024xi8>)
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi16>, memref<1024xi16>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<eq>
  return
}

// -----
func.func @test_vcmp_b16_ne() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi16>
  %allocIn1 = memref.alloc() : memref<1024xi16>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi16>, memref<1024xi16>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <ne>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi16>, memref<1024xi16>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<ne>
  return
}

// -----
func.func @test_vcmp_b64_lt() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi64>
  %allocIn1 = memref.alloc() : memref<1024xi64>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi64>, memref<1024xi64>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <lt>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi64>, memref<1024xi64>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<lt>
  return
}

// -----
func.func @test_vcmp_b64_gt() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi64>
  %allocIn1 = memref.alloc() : memref<1024xi64>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi64>, memref<1024xi64>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <gt>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi64>, memref<1024xi64>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<gt>
  return
}

// -----
func.func @test_vcmp_b64_le() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi64>
  %allocIn1 = memref.alloc() : memref<1024xi64>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi64>, memref<1024xi64>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <le>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi64>, memref<1024xi64>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<le>
  return
}

// -----
func.func @test_vcmp_b64_ge() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi64>
  %allocIn1 = memref.alloc() : memref<1024xi64>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi64>, memref<1024xi64>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <ge>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi64>, memref<1024xi64>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<ge>
  return
}

// -----
func.func @test_vcmp_b64_eq() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi64>
  %allocIn1 = memref.alloc() : memref<1024xi64>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi64>, memref<1024xi64>) outs(%[[ALLOC_DST1]] : memref<1024xi8>)
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi64>, memref<1024xi64>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<eq>
  return
}

// -----
func.func @test_vcmp_b64_ne() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %allocIn0 = memref.alloc() : memref<1024xi64>
  %allocIn1 = memref.alloc() : memref<1024xi64>
  %allocOut = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[ALLOC_SRC1]] : memref<1024xi64>, memref<1024xi64>) outs(%[[ALLOC_DST1]] : memref<1024xi8>) compare_mode = <ne>
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1024xi8>) outs(%[[ARG1]] : memref<1024xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1024xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1024xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1024xf16>, memref<1024xf16>) outs(%[[ALLOC_DST0]] : memref<1024xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<1024xi64>, memref<1024xi64>)
                outs(%allocOut : memref<1024xi1>)
                compare_mode = #hivm.compare_mode<ne>
  return
}

// -----
func.func @test_vcmp_b64_eq_vs() {
  // CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[CONST2:.*]] = arith.constant 2 : i64
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1xi64>
  // CHECK: %[[ALLOC_DST0:.*]] = memref.alloc() : memref<1xi1>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1xi8>
  %allocIn0 = memref.alloc() : memref<1xi64>
  %cstIn1 = arith.constant 2 : i64
  %allocOut = memref.alloc() : memref<1xi1>
  // CHECK: hivm.hir.vcmp ins(%[[ALLOC_SRC0]], %[[CONST2]] : memref<1xi64>, i64) outs(%[[ALLOC_DST1]] : memref<1xi8>)
  // CHECK: %[[ARG1:.*]] = memref.alloc() : memref<1xf16>
  // CHECK: hivm.hir.vcast ins(%[[ALLOC_DST1]] : memref<1xi8>) outs(%[[ARG1]] : memref<1xf16>)
  // CHECK: %[[ARG2:.*]] = memref.alloc() : memref<1xf16>
  // CHECK: hivm.hir.vbrc ins(%[[CONST]] : f16) outs(%[[ARG2]] : memref<1xf16>)
  // CHECK: hivm.hir.vcmp ins(%[[ARG1]], %[[ARG2]] : memref<1xf16>, memref<1xf16>) outs(%[[ALLOC_DST0]] : memref<1xi1>) compare_mode = <ne>
  hivm.hir.vcmp ins(%allocIn0, %cstIn1 : memref<1xi64>, i64)
                outs(%allocOut : memref<1xi1>)
                compare_mode = #hivm.compare_mode<eq>
  return
}


// -----
// CHECK-LABEL:   func.func @test_decompose_vdeinterleave_double_f16(
// CHECK-SAME:                                       %[[SRC:.*]]: memref<32xf16>,
// CHECK-SAME:                                       %[[EVEN:.*]]: memref<16xf16>,
// CHECK-SAME:                                       %[[ODD:.*]]: memref<16xf16>) {
// CHECK:           hivm.hir.vdeinterleave ins(%[[SRC]] : memref<32xf16>) outs(%[[EVEN]] : memref<16xf16>) channel_num = 2 index_mode = <CHANNEL_0>
// CHECK:           hivm.hir.vdeinterleave ins(%[[SRC]] : memref<32xf16>) outs(%[[ODD]] : memref<16xf16>) channel_num = 2 index_mode = <CHANNEL_1>
// CHECK:           return
// CHECK:         }
func.func @test_decompose_vdeinterleave_double_f16(%src: memref<32xf16>, %even_dst: memref<16xf16>,
                                                   %odd_dst: memref<16xf16>) {
  hivm.hir.vdeinterleave ins(%src: memref<32xf16>)
                             outs(%even_dst, %odd_dst: memref<16xf16>, memref<16xf16>)
                             index_mode = <ALL_CHANNELS>

  return
}

// -----
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16xi32>
// CHECK: hivm.hir.load ins(%arg2 : memref<16xi32>) outs(%[[ALLOC0]] : memref<16xi32>) init_out_buffer = false
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<16xi32>
// CHECK: hivm.hir.vand ins({{.*}}, %[[ALLOC0]] : memref<16xi32>, memref<16xi32>) outs(%[[ALLOC1]] : memref<16xi32>)
// CHECK: hivm.hir.store ins(%[[ALLOC1]] : memref<16xi32>) outs(%arg2 : memref<16xi32>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: return
func.func @test_decompose_atomic_and_op(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  %alloc = memref.alloc() : memref<16xi32>
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%alloc : memref<16xi32>)
  hivm.hir.store ins(%alloc : memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <and>
  return
}

// -----
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16xi32>
// CHECK: hivm.hir.load ins(%arg2 : memref<16xi32>) outs(%[[ALLOC0]] : memref<16xi32>) init_out_buffer = false
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<16xi32>
// CHECK: hivm.hir.vor ins({{.*}}, %[[ALLOC0]] : memref<16xi32>, memref<16xi32>) outs(%[[ALLOC1]] : memref<16xi32>)
// CHECK: hivm.hir.store ins(%[[ALLOC1]] : memref<16xi32>) outs(%arg2 : memref<16xi32>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: return
func.func @test_decompose_atomic_or_op(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  %alloc = memref.alloc() : memref<16xi32>
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%alloc : memref<16xi32>)
  hivm.hir.store ins(%alloc : memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <or>
  return
}

// -----
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16xi32>
// CHECK: hivm.hir.load ins(%arg2 : memref<16xi32>) outs(%[[ALLOC0]] : memref<16xi32>) init_out_buffer = false
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<16xi32>
// CHECK: hivm.hir.vxor ins({{.*}}, %[[ALLOC0]] : memref<16xi32>, memref<16xi32>) outs(%[[ALLOC1]] : memref<16xi32>)
// CHECK: hivm.hir.store ins(%[[ALLOC1]] : memref<16xi32>) outs(%arg2 : memref<16xi32>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: return
func.func @test_decompose_atomic_xor_op(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  %alloc = memref.alloc() : memref<16xi32>
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%alloc : memref<16xi32>)
  hivm.hir.store ins(%alloc : memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <xor>
  return
}

// -----
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16xi32>
// CHECK: hivm.hir.load ins(%arg2 : memref<16xi32>) outs(%[[ALLOC0]] : memref<16xi32>) init_out_buffer = false
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<16xi32>
// CHECK: hivm.hir.vxor ins({{.*}}, %[[ALLOC0]] : memref<16xi32>, memref<16xi32>) outs(%[[ALLOC1]] : memref<16xi32>)
// CHECK: hivm.hir.store ins(%[[ALLOC1]] : memref<16xi32>) outs(%arg2 : memref<16xi32>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: return
func.func @test_decompose_atomic_xor_dyn_op(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  %alloc = memref.alloc() : memref<16xi32>
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%alloc : memref<16xi32>)
  hivm.hir.store ins(%alloc : memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <xor>
  return
}

// -----
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<256xi16>
// CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<256xi16>
// CHECK: %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<256xi16>
// CHECK: hivm.hir.load ins(%[[REINTERPRET_CAST]] : memref<256xi16, strided<[1]>>) outs(%[[ALLOC_1]] : memref<256xi16>) init_out_buffer = false
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() : memref<256xi1>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<256xi8>
// CHECK: hivm.hir.vcmp ins(%[[ALLOC_1]], %[[ALLOC_0]] : memref<256xi16>, memref<256xi16>) outs(%[[ALLOC_3]] : memref<256xi8>)
// CHECK: %[[ALLOC_4:.*]] = memref.alloc() : memref<256xf16>
// CHECK: hivm.hir.vcast ins(%[[ALLOC_3]] : memref<256xi8>) outs(%[[ALLOC_4]] : memref<256xf16>)
// CHECK: %[[ALLOC_5:.*]] = memref.alloc() : memref<256xf16>
// CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_5]] : memref<256xf16>)
// CHECK: hivm.hir.vcmp ins(%[[ALLOC_4]], %[[ALLOC_5]] : memref<256xf16>, memref<256xf16>) outs(%[[ALLOC_2]] : memref<256xi1>)
// CHECK: %[[ALLOC_6:.*]] = memref.alloc() : memref<256xi16>
// CHECK: hivm.hir.vsel ins(%[[ALLOC_2]], %[[ALLOC]], %[[ALLOC_1]] : memref<256xi1>, memref<256xi16>, memref<256xi16>) outs(%[[ALLOC_6]] : memref<256xi16>)
// CHECK: hivm.hir.store ins(%[[ALLOC_6]] : memref<256xi16>) outs(%[[REINTERPRET_CAST]] : memref<256xi16, strided<[1]>>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: return
func.func @atomic_cas(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi16>) {
  %alloc = memref.alloc() : memref<256xi16>
  %alloc_1 = memref.alloc() : memref<256xi16>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
  hivm.hir.atomic_cas ins(%alloc_1, %alloc : memref<256xi16>, memref<256xi16>) outs(%reinterpret_cast_1 : memref<256xi16, strided<[1]>>)
  return
}

// -----
// CHECK-LABEL: func.func @atomic_xchg(
// CHECK: %[[VAL_2:.*]] = memref.alloc() : memref<256xi16>
// CHECK: %[[VAL_3:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
// CHECK: %[[VAL_4:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_4]] : memref<1xi64>)
// CHECK: %[[VAL_5:.*]] = memref.alloc() : memref<256xi16>
// CHECK: hivm.hir.load ins(%[[VAL_3]] : memref<256xi16, strided<[1]>>) outs(%[[VAL_5]] : memref<256xi16>)
// CHECK: hivm.hir.store ins(%[[VAL_2]] : memref<256xi16>) outs(%[[VAL_3]] : memref<256xi16, strided<[1]>>)
// CHECK: hivm.hir.copy ins(%[[VAL_5]] : memref<256xi16>) outs(%[[VAL_2]] : memref<256xi16>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_4]] : memref<1xi64>)
// CHECK: return
func.func @atomic_xchg(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi16>) {
  %alloc = memref.alloc() : memref<256xi16>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
  hivm.hir.atomic_xchg ins(%alloc : memref<256xi16>) outs(%reinterpret_cast_1 : memref<256xi16, strided<[1]>>)
  return
}

// -----
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8x4x2x4x4xi32>
// CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<8x4x2x4x4xi32>
// CHECK: %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 4, 2, 4, 4], strides: [128, 32, 16, 4, 1] : memref<?xi32> to memref<8x4x2x4x4xi32, strided<[128, 32, 16, 4, 1]>>
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<8x4x2x4x4xi32>
// CHECK: hivm.hir.load ins(%[[REINTERPRET_CAST]] : memref<8x4x2x4x4xi32, strided<[128, 32, 16, 4, 1]>>) outs(%[[ALLOC_1]] : memref<8x4x2x4x4xi32>) init_out_buffer = false
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() : memref<8x4x2x4x4xi1>
// CHECK: hivm.hir.vcmp ins(%[[ALLOC_1]], %[[ALLOC]] : memref<8x4x2x4x4xi32>, memref<8x4x2x4x4xi32>) outs(%[[ALLOC_2]] : memref<8x4x2x4x4xi1>)
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<8x4x2x4x4xi32>
// CHECK: hivm.hir.vsel ins(%[[ALLOC_2]], %[[ALLOC_0]], %[[ALLOC_1]] : memref<8x4x2x4x4xi1>, memref<8x4x2x4x4xi32>, memref<8x4x2x4x4xi32>) outs(%[[ALLOC_3]] : memref<8x4x2x4x4xi32>)
// CHECK: hivm.hir.store ins(%[[ALLOC_3]] : memref<8x4x2x4x4xi32>) outs(%[[REINTERPRET_CAST]] : memref<8x4x2x4x4xi32, strided<[128, 32, 16, 4, 1]>>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: return
func.func @atomic_cas(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi32>) {
  %alloc = memref.alloc() : memref<8x4x2x4x4xi32>
  %alloc_1 = memref.alloc() : memref<8x4x2x4x4xi32>
   %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 4, 2, 4, 4], strides: [128, 32, 16, 4, 1] : memref<?xi32> to memref<8x4x2x4x4xi32, strided<[128, 32, 16, 4, 1]>>
  hivm.hir.atomic_cas ins(%alloc, %alloc_1 : memref<8x4x2x4x4xi32>, memref<8x4x2x4x4xi32>) outs(%reinterpret_cast_2 : memref<8x4x2x4x4xi32, strided<[128, 32, 16, 4, 1]>>)
  return
}

// -----
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16xi8>
// CHECK: hivm.hir.load ins(%arg2 : memref<16xi8>) outs(%[[ALLOC0]] : memref<16xi8>) init_out_buffer = false
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<16xf16>
// CHECK: hivm.hir.vcast ins(%{{.*}} : memref<16xi8>) outs(%[[ALLOC1]] : memref<16xf16>) cast = <cast_unsigned>
// CHECK: %[[ALLOC2:.*]]  = memref.alloc() : memref<16xf16>
// CHECK: hivm.hir.vcast ins(%[[ALLOC0]] : memref<16xi8>) outs(%[[ALLOC2]] : memref<16xf16>) cast = <cast_unsigned>
// CHECK: %[[ALLOC3:.*]]  = memref.alloc() : memref<16xf16>
// CHECK: hivm.hir.vmax ins(%[[ALLOC1]], %[[ALLOC2]] : memref<16xf16>, memref<16xf16>) outs(%[[ALLOC3]] : memref<16xf16>)
// CHECK: %[[ALLOC4:.*]] = memref.alloc() : memref<16xi8>
// CHECK: hivm.hir.vcast ins(%[[ALLOC3]] : memref<16xf16>) outs(%[[ALLOC4]] : memref<16xi8>) round_mode = <trunc> cast = <cast_unsigned>
// CHECK: hivm.hir.store ins(%[[ALLOC4]] : memref<16xi8>) outs(%arg2 : memref<16xi8>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: return
func.func @test_decompose_atomic_max_ui8(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<16xi8>, %arg2: memref<16xi8>) {
  %alloc = memref.alloc() : memref<16xi8>
  hivm.hir.load ins(%arg1 : memref<16xi8>) outs(%alloc : memref<16xi8>)
  hivm.hir.store ins(%alloc : memref<16xi8>) outs(%arg2 : memref<16xi8>) atomic = <umax>
  return
}

// -----
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16xi8>
// CHECK: hivm.hir.load ins(%arg2 : memref<16xi8>) outs(%[[ALLOC0]] : memref<16xi8>) init_out_buffer = false
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<16xf16>
// CHECK: hivm.hir.vcast ins(%{{.*}} : memref<16xi8>) outs(%[[ALLOC1]] : memref<16xf16>) cast = <cast_unsigned>
// CHECK: %[[ALLOC2:.*]]  = memref.alloc() : memref<16xf16>
// CHECK: hivm.hir.vcast ins(%[[ALLOC0]] : memref<16xi8>) outs(%[[ALLOC2]] : memref<16xf16>) cast = <cast_unsigned>
// CHECK: %[[ALLOC3:.*]]  = memref.alloc() : memref<16xf16>
// CHECK: hivm.hir.vmin ins(%[[ALLOC1]], %[[ALLOC2]] : memref<16xf16>, memref<16xf16>) outs(%[[ALLOC3]] : memref<16xf16>)
// CHECK: %[[ALLOC4:.*]] = memref.alloc() : memref<16xi8>
// CHECK: hivm.hir.vcast ins(%[[ALLOC3]] : memref<16xf16>) outs(%[[ALLOC4]] : memref<16xi8>) round_mode = <trunc> cast = <cast_unsigned>
// CHECK: hivm.hir.store ins(%[[ALLOC4]] : memref<16xi8>) outs(%arg2 : memref<16xi8>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: return
func.func @test_decompose_atomic_min_ui8(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<16xi8>, %arg2: memref<16xi8>) {
  %alloc = memref.alloc() : memref<16xi8>
  hivm.hir.load ins(%arg1 : memref<16xi8>) outs(%alloc : memref<16xi8>)
  hivm.hir.store ins(%alloc : memref<16xi8>) outs(%arg2 : memref<16xi8>) atomic = <umin>
  return
}

// -----
// CHECK-LABEL: func.func @atomic_masked_xchg(
// CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<256xi16>
// CHECK: %[[VAL_4:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
// CHECK: %[[VAL_5:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_5]] : memref<1xi64>)
// CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<256xi16>
// CHECK: hivm.hir.load ins(%[[VAL_4]] : memref<256xi16, strided<[1]>>) outs(%[[VAL_6]] : memref<256xi16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
// CHECK: %[[VAL_8:.*]] = memref.alloc() : memref<256xi16>
// CHECK: hivm.hir.vsel ins(%[[ARG2:.*]], %[[VAL_3]], %[[VAL_6]] : memref<256xi1>, memref<256xi16>, memref<256xi16>) outs(%[[VAL_8]] : memref<256xi16>)
// CHECK: hivm.hir.vsel ins(%[[ARG2]], %[[VAL_6]], %[[VAL_3]] : memref<256xi1>, memref<256xi16>, memref<256xi16>) outs(%[[VAL_3]] : memref<256xi16>)
// CHECK: hivm.hir.store ins(%[[VAL_8]] : memref<256xi16>) outs(%[[VAL_4]] : memref<256xi16, strided<[1]>>)
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_5]] : memref<1xi64>)
// CHECK: return
func.func @atomic_masked_xchg(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi16>, %mask: memref<256xi1>) {
  %alloc = memref.alloc() : memref<256xi16>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
  hivm.hir.atomic_xchg ins(%alloc : memref<256xi16>) outs(%reinterpret_cast_1 : memref<256xi16, strided<[1]>>) mask(%mask : memref<256xi1>)
  return
}
