// REQUIRES: regbase
// RUN: bishengir-opt -hivm-insert-fixpipe -hivm-inline-fixpipe %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// Fractal mmadL1: all-4D inputs/output, check fixpipe insertion doesn't crash
// CHECK-LABEL: func.func @test_fractal_mmadL1_fixpipe
// CHECK: hivm.hir.mmadL1
module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">} {
  func.func @test_fractal_mmadL1_fixpipe(%a: tensor<20x10x16x16xf16>, %b: tensor<5x20x16x16xf16>) -> tensor<160x80xf32> {
    %c160 = arith.constant 160 : index
    %c320 = arith.constant 320 : index
    %c80 = arith.constant 80 : index
    %false = arith.constant false
    %empty = tensor.empty() : tensor<160x80xf32>
    %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c160, %c320, %c80 : tensor<20x10x16x16xf16>, tensor<5x20x16x16xf16>, i1, index, index, index) outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
    return %mmad : tensor<160x80xf32>
  }
}

// -----

// Fractal C output: mmadL1 result feeds convert_layout{ND->Fractal} -> inline fixpipe as NZ2NZ
// CHECK-LABEL: func.func @test_fractal_c_nz2nz_fixpipe
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: hivm.hir.fixpipe
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_fractal_c_nz2nz_fixpipe(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>, %arg2: memref<1x1x16x16xf32>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %empty = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %fractal = hivm.hir.convert_layout %mmad output_shape [1, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x16xf32>) -> tensor<1x1x16x16xf32>
    hivm.hir.store ins(%fractal : tensor<1x1x16x16xf32>) outs(%arg2 : memref<1x1x16x16xf32>)
    return
  }
}
// -----

// Batch fractal C output: batchMmadL1 result feeds convert_layout{ND->Fractal} -> inline fixpipe as NZ2NZ
// CHECK-LABEL: func.func @test_batch_fractal_c_nz2nz_fixpipe
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: hivm.hir.fixpipe
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_batch_fractal_c_nz2nz_fixpipe(%arg0: tensor<2x32x64xf16>, %arg1: tensor<2x64x32xf16>, %arg2: memref<2x2x2x16x16xf32>) {
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %empty = tensor.empty() : tensor<2x32x32xf32>
    %mmad = hivm.hir.batchMmadL1 ins(%arg0, %arg1, %true, %c32, %c64, %c32 : tensor<2x32x64xf16>, tensor<2x64x32xf16>, i1, index, index, index) outs(%empty : tensor<2x32x32xf32>) -> tensor<2x32x32xf32>
    %fractal = hivm.hir.convert_layout %mmad output_shape [2, 2, 2, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<2x32x32xf32>) -> tensor<2x2x2x16x16xf32>
    hivm.hir.store ins(%fractal : tensor<2x2x2x16x16xf32>) outs(%arg2 : memref<2x2x2x16x16xf32>)
    return
  }
}
// -----

// When NO convert_layout{ND->Fractal} follows mmadL1, fall back to NZ2ND
// CHECK-LABEL: func.func @test_nd_c_nz2nd_fixpipe_fallback
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_nd_c_nz2nd_fixpipe_fallback(%arg0: tensor<16x16xi8>, %arg1: tensor<16x16xi8>, %arg2: memref<16x16xf32>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %empty = tensor.empty() : tensor<16x16xi32>
    %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<16x16xi8>, tensor<16x16xi8>, i1, index, index, index) outs(%empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %cast_empty = tensor.empty() : tensor<16x16xf32>
    %casted = hivm.hir.vcast ins(%mmad : tensor<16x16xi32>) outs(%cast_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.store ins(%casted : tensor<16x16xf32>) outs(%arg2 : memref<16x16xf32>)
    return
  }
}
// -----

// CV scenario: mmadL1 result goes to vector ops (vadd, vcast) -> should NOT get NZ2NZ fixpipe, stays NZ2ND
// CHECK-LABEL: func.func @test_cv_mmad_vector_consumer_no_nz2nz
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_cv_mmad_vector_consumer_no_nz2nz(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>, %arg2: memref<16x16xf16>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %cst = arith.constant 1.000000e+00 : f32
    %empty = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %add_empty = tensor.empty() : tensor<16x16xf32>
    %added = hivm.hir.vadd ins(%mmad, %cst : tensor<16x16xf32>, f32) outs(%add_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %cast_empty = tensor.empty() : tensor<16x16xf16>
    %casted = hivm.hir.vcast ins(%added : tensor<16x16xf32>) outs(%cast_empty : tensor<16x16xf16>) round_mode = <rint> -> tensor<16x16xf16>
    hivm.hir.store ins(%casted : tensor<16x16xf16>) outs(%arg2 : memref<16x16xf16>)
    return
  }
}
// -----
// s_C_int8: int8 fractal C fixpipe with [16,32] block sizes.
// CHECK-LABEL: func.func @test_int8_fractal_c_nz2nz_fixpipe
// CHECK: hivm.hir.fixpipe
module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">} {
  func.func @test_int8_fractal_c_nz2nz_fixpipe(%gm: memref<2x10x16x32xi8, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %a = arith.constant dense<0> : tensor<10x10x16x32xi8>
    %b = arith.constant dense<0> : tensor<2x10x32x32xi8>
    %c = tensor.empty() : tensor<160x64xi32>
    %mmad = hivm.hir.mmadL1 ins(%a, %b, %false, %c0, %c0, %c0 : tensor<10x10x16x32xi8>, tensor<2x10x32x32xi8>, i1, index, index, index) outs(%c : tensor<160x64xi32>) -> tensor<160x64xi32>
    %fractal = hivm.hir.convert_layout %mmad output_shape [2, 10, 16, 32] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, srcLayout = #hivm.data_layout<ND>} : (tensor<160x64xi32>) -> tensor<2x10x16x32xi32>
    %strided = memref.cast %gm : memref<2x10x16x32xi8, #hivm.address_space<gm>> to memref<2x10x16x32xi8, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.fixpipe ins(%fractal : tensor<2x10x16x32xi32>) outs(%strided : memref<2x10x16x32xi8, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @dotdot
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dotdot(%4: tensor<16x16xf32>, %e4: tensor<16x16xf32>, %e5: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %7 = tensor.empty() : tensor<16x16xf32>
    %8 = hivm.hir.mmadL1 ins(%4, %e4, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    // Intermediate fixpipe feeds another mmad (MacroOp) → NZ2NZ (default, omitted).
    // CHECK: %[[ARG0:.*]] = hivm.hir.fixpipe ins(%[[input:.*]] : tensor<16x16xf32>) outs(%[[out0:.*]] : tensor<16x16xf32>) -> tensor<16x16xf32>
    %9 = tensor.empty() : tensor<16x16xf32>
    // CHECK: %[[ARG1:.*]] = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%[[ARG0]]
    %10 = hivm.hir.mmadL1 ins(%8, %e5, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%9 : tensor<16x16xf32>) -> tensor<16x16xf32>
    // Final fixpipe keeps NZ2ND.
    // CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[ARG1]] : tensor<16x16xf32>) outs(%[[out1:.*]] : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %10 : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @inline_fixpipe_fuse_i32_to_i8_with_saturate
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.fixpipe {{.*pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>}}
// CHECK-NOT: hivm.hir.store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fuse_i32_to_i8_with_saturate(
      %mmad_res: tensor<16x16xi32>,
      %fixpipe_dst: tensor<16x16xi32>,
      %cast_dst: tensor<16x16xi8>,
      %store_dst: memref<16x16xi8, strided<[16, 1]>>) {
    %fixpipe = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad_res : tensor<16x16xi32>) outs(%fixpipe_dst : tensor<16x16xi32>)
        -> tensor<16x16xi32>
    %cast = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = true,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%fixpipe : tensor<16x16xi32>) outs(%cast_dst : tensor<16x16xi8>)
        round_mode = <trunc> -> tensor<16x16xi8>
    hivm.hir.store ins(%cast : tensor<16x16xi8>)
        outs(%store_dst : memref<16x16xi8, strided<[16, 1]>>)
    return
  }
}

// -----

// Decomposed i32->i8 cast chain (i32->i16->i8) must still fuse as S322I8.
// CHECK-LABEL: func.func @inline_fixpipe_fuse_i32_to_i8_cast_chain
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.fixpipe {{.*pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>.*}} outs(%{{.*}} : memref
// CHECK-NOT: hivm.hir.store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fuse_i32_to_i8_cast_chain(
      %mmad_res: tensor<16x16xi32>,
      %fixpipe_dst: tensor<16x16xi32>,
      %cast_i16_dst: tensor<16x16xi16>,
      %cast_i8_dst: tensor<16x16xi8>,
      %store_dst: memref<16x16xi8, strided<[16, 1]>>) {
    %fixpipe = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad_res : tensor<16x16xi32>) outs(%fixpipe_dst : tensor<16x16xi32>)
        -> tensor<16x16xi32>
    %cast_i16 = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = true,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%fixpipe : tensor<16x16xi32>) outs(%cast_i16_dst : tensor<16x16xi16>)
        round_mode = <trunc> -> tensor<16x16xi16>
    %cast_i8 = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = true,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%cast_i16 : tensor<16x16xi16>) outs(%cast_i8_dst : tensor<16x16xi8>)
        round_mode = <trunc> -> tensor<16x16xi8>
    hivm.hir.store ins(%cast_i8 : tensor<16x16xi8>)
        outs(%store_dst : memref<16x16xi8, strided<[16, 1]>>)
    return
  }
}

// -----

// Decomposed i32->i8 via float (i32->f32->f16->i8). Float steps may set
// enable_saturate=false; fusion still uses overall i32->i8 as S322I8 when the
// final cast allows saturation.
// CHECK-LABEL: func.func @inline_fixpipe_fuse_i32_to_i8_via_float_cast_chain
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.fixpipe {{.*pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>.*}} outs(%{{.*}} : memref
// CHECK-NOT: hivm.hir.store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fuse_i32_to_i8_via_float_cast_chain(
      %mmad_res: tensor<32x32xi32>,
      %fixpipe_dst: tensor<32x32xi32>,
      %cast_f32_dst: tensor<32x32xf32>,
      %cast_f16_dst: tensor<32x32xf16>,
      %cast_i8_dst: tensor<32x32xi8>,
      %store_dst: memref<32x32xi8, strided<[32, 1]>>) {
    %fixpipe = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad_res : tensor<32x32xi32>) outs(%fixpipe_dst : tensor<32x32xi32>)
        -> tensor<32x32xi32>
    %cast_f32 = hivm.hir.vcast {
        enable_overflow = false, enable_saturate = false,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%fixpipe : tensor<32x32xi32>) outs(%cast_f32_dst : tensor<32x32xf32>)
        round_mode = <trunc> -> tensor<32x32xf32>
    %cast_f16 = hivm.hir.vcast {
        enable_overflow = false, enable_saturate = false,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%cast_f32 : tensor<32x32xf32>) outs(%cast_f16_dst : tensor<32x32xf16>)
        round_mode = <trunc> -> tensor<32x32xf16>
    %cast_i8 = hivm.hir.vcast {
        enable_overflow = false, enable_saturate = true,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%cast_f16 : tensor<32x32xf16>) outs(%cast_i8_dst : tensor<32x32xi8>)
        round_mode = <trunc> -> tensor<32x32xi8>
    hivm.hir.store ins(%cast_i8 : tensor<32x32xi8>)
        outs(%store_dst : memref<32x32xi8, strided<[32, 1]>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @inline_fixpipe_no_fuse_i32_to_i8_cast_chain_without_saturate
// CHECK: hivm.hir.fixpipe
// CHECK-NOT: pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_no_fuse_i32_to_i8_cast_chain_without_saturate(
      %mmad_res: tensor<16x16xi32>,
      %fixpipe_dst: tensor<16x16xi32>,
      %cast_i16_dst: tensor<16x16xi16>,
      %cast_i8_dst: tensor<16x16xi8>,
      %store_dst: memref<16x16xi8, strided<[16, 1]>>) {
    %fixpipe = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad_res : tensor<16x16xi32>) outs(%fixpipe_dst : tensor<16x16xi32>)
        -> tensor<16x16xi32>
    %cast_i16 = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = true,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%fixpipe : tensor<16x16xi32>) outs(%cast_i16_dst : tensor<16x16xi16>)
        round_mode = <trunc> -> tensor<16x16xi16>
    %cast_i8 = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = false,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%cast_i16 : tensor<16x16xi16>) outs(%cast_i8_dst : tensor<16x16xi8>)
        round_mode = <truncwithoverflow> -> tensor<16x16xi8>
    hivm.hir.store ins(%cast_i8 : tensor<16x16xi8>)
        outs(%store_dst : memref<16x16xi8, strided<[16, 1]>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @inline_fixpipe_no_fuse_i32_to_i8_without_saturate
// CHECK: hivm.hir.fixpipe
// CHECK-NOT: pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>
// CHECK: hivm.hir.vcast {enable_overflow = true, enable_saturate = false
// CHECK: hivm.hir.store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_no_fuse_i32_to_i8_without_saturate(
      %mmad_res: tensor<16x16xi32>,
      %fixpipe_dst: tensor<16x16xi32>,
      %cast_dst: tensor<16x16xi8>,
      %store_dst: memref<16x16xi8, strided<[16, 1]>>) {
    %fixpipe = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad_res : tensor<16x16xi32>) outs(%fixpipe_dst : tensor<16x16xi32>)
        -> tensor<16x16xi32>
    %cast = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = false,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%fixpipe : tensor<16x16xi32>) outs(%cast_dst : tensor<16x16xi8>)
        round_mode = <truncwithoverflow> -> tensor<16x16xi8>
    hivm.hir.store ins(%cast : tensor<16x16xi8>)
        outs(%store_dst : memref<16x16xi8, strided<[16, 1]>>)
    return
  }
}

// -----

// Chained mmad with f32->f16 vcast between them and before store:
//   intermediate fixpipe+vcast -> NZ2NZ fractal f16 feeding next mmad
//   final fixpipe+vcast+store -> NZ2ND fixpipe with F322F16 into memref
// CHECK-LABEL: func.func @chain_matmul_with_vcast
// CHECK-NOT: hivm.hir.vcast
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chain_matmul_with_vcast(
      %a: tensor<16x16xf16>,
      %b: tensor<16x16xf16>,
      %c: tensor<16x16xf16>,
      %dst: memref<16x16xf16, strided<[16, 1]>>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %empty0 = tensor.empty() : tensor<16x16xf32>
    %mmad0 = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true}
        ins(%a, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%empty0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %fp0_dst = tensor.empty() : tensor<16x16xf32>
    %fp0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad0 : tensor<16x16xf32>) outs(%fp0_dst : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    %cast_dst = tensor.empty() : tensor<16x16xf16>
    %cast0 = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = false,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%fp0 : tensor<16x16xf32>) outs(%cast_dst : tensor<16x16xf16>)
        -> tensor<16x16xf16>
    // Intermediate: NZ2NZ fractal f16 (dma_mode omitted) feeds next mmad.
    // CHECK: %[[FP0:.*]] = hivm.hir.fixpipe ins(%{{.*}} : tensor<16x16xf32>) outs(%{{.*}} : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
    // CHECK: %[[MMAD1:.*]] = hivm.hir.mmadL1 {{.*}}ins(%[[FP0]]
    %empty1 = tensor.empty() : tensor<16x16xf32>
    %mmad1 = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true}
        ins(%cast0, %c, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%empty1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %fp1_dst = tensor.empty() : tensor<16x16xf32>
    %fp1 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad1 : tensor<16x16xf32>) outs(%fp1_dst : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    %cast1 = hivm.hir.vcast {
        enable_overflow = true, enable_saturate = false,
        hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
        ins(%fp1 : tensor<16x16xf32>) outs(%cast_dst : tensor<16x16xf16>)
        -> tensor<16x16xf16>
    // Final: fuse vcast+store into NZ2ND fixpipe with F322F16.
    // CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%[[MMAD1]] : tensor<16x16xf32>) outs(%{{.*}} : memref<16x16xf16, strided<[16, 1]>>)
    // CHECK-NOT: hivm.hir.store
    hivm.hir.store ins(%cast1 : tensor<16x16xf16>)
        outs(%dst : memref<16x16xf16, strided<[16, 1]>>)
    return
  }
}
