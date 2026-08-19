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

// -----

// When mmad0's result feeds mmad1 as the A-matrix input while mmad1's outs
// (accumulator init) comes from a different mmad, insertFixpipe must replace
// only the ins use with the fixpipe result and leave the outs init on the
// prior mmad's raw L0C result.
//
// CHECK-LABEL: func.func @chained_mmad_ins_with_separate_l0c_init
// CHECK: %[[MMADPREV:.*]] = hivm.hir.mmadL1 ins(%{{.*}} : tensor<16x16xf16>
// CHECK: %[[MMAD0:.*]] = hivm.hir.mmadL1 {{.*}}ins(%{{.*}} : tensor<16x16xf16>
// CHECK: %[[FIX0:.*]] = hivm.hir.fixpipe {{.*}}ins(%[[MMAD0]]
// CHECK: %[[MMAD1:.*]] = hivm.hir.mmadL1 {{.*}}ins(%[[FIX0]]
// CHECK-SAME: outs(%[[MMADPREV]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chained_mmad_ins_with_separate_l0c_init(
      %a: tensor<16x16xf16>,
      %b: tensor<16x16xf16>,
      %a2: tensor<16x16xf16>,
      %b2: tensor<16x16xf16>,
      %c: tensor<16x16xf16>) -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %empty0 = tensor.empty() : tensor<16x16xf32>
    %mmad_prev = hivm.hir.mmadL1
        ins(%a, %b, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%empty0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %empty1 = tensor.empty() : tensor<16x16xf32>
    %mmad0 = hivm.hir.mmadL1
        ins(%a2, %b2, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%empty1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %mmad1 = hivm.hir.mmadL1
        ins(%mmad0, %c, %true, %c16, %c16, %c16
            : tensor<16x16xf32>, tensor<16x16xf16>, i1, index, index, index)
        outs(%mmad_prev : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mmad1 : tensor<16x16xf32>
  }
}

// -----

// Sink store into fallback_not_exec if, then let greedy fold Cube-side
// vcast / vtranspose / store into fixpipe (L0C -> GM, nz2dn + pre_quant).
// CHECK-LABEL: func.func @inline_fixpipe_fallback_not_exec_cast_transpose_store
// CHECK: scf.if %{{.*}} {
// CHECK:   hivm.hir.vbrc
// CHECK:   hivm.hir.vcast
// CHECK:   hivm.hir.vtranspose
// CHECK:   hivm.hir.store
// CHECK: } else {
// CHECK:   hivm.hir.fixpipe
// CHECK-SAME: dma_mode = #hivm.dma_mode<nz2dn>
// CHECK-SAME: outs(%{{.*}} : memref<16x16xf16
// CHECK-NOT: hivm.hir.store
// CHECK: } {fallback_not_exec

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fallback_not_exec_cast_transpose_store(
      %cond: i1,
      %l0c: tensor<16x16xf32>,
      %gm: memref<16x16xf16, strided<[16, 1]>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %brc_init = tensor.empty() : tensor<16x16xf32>
    %res = scf.if %cond -> (tensor<16x16xf32>) {
      %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brc_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %brc : tensor<16x16xf32>
    } else {
      %ub = tensor.empty() : tensor<16x16xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
          ins(%l0c : tensor<16x16xf32>) outs(%ub : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scf.yield %fix : tensor<16x16xf32>
    } {fallback_not_exec}
    %cast_init = tensor.empty() : tensor<16x16xf16>
    %cast = hivm.hir.vcast ins(%res : tensor<16x16xf32>) outs(%cast_init : tensor<16x16xf16>) -> tensor<16x16xf16>
    %tr_init = tensor.empty() : tensor<16x16xf16>
    %tr = hivm.hir.vtranspose ins(%cast : tensor<16x16xf16>) outs(%tr_init : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
    hivm.hir.store ins(%tr : tensor<16x16xf16>) outs(%gm : memref<16x16xf16, strided<[16, 1]>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @used_at_input_and_init
// CHECK: %[[MMAD0:.*]] = hivm.hir.mmadL1
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe {channel_split = true} ins(%[[MMAD0]] : tensor<16x16xf32>)
// CHECK: %[[MMAD1:.*]] = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%[[FIX]], %[[_:.*]] : tensor<2x1x16x8xf32>, tensor<16x16xf16>, i1, index, index, index) outs(%[[MMAD0]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[MMAD1]] : tensor<16x16xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @used_at_input_and_init(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>, %arg2: tensor<16x16xf16>, %arg3: tensor<16x16xf16>, %arg4: tensor<16x16xf16>) -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = tensor.empty() : tensor<16x16xf32>
    %2 = hivm.hir.mmadL1 ins(%arg2, %arg3, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %3 = hivm.hir.mmadL1 ins(%2, %arg4, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %3 : tensor<16x16xf32>
  }
}

// -----

  // Sink cast + transpose + extract_slice + store; greedy swaps extract_slice
  // onto L0C then folds store into fixpipe (nz2dn + pre_quant -> GM subview).
  // CHECK-LABEL: func.func @inline_fixpipe_fallback_not_exec_cast_transpose_slice_store
  // CHECK: scf.if %{{.*}} {
  // CHECK:   hivm.hir.vbrc
  // CHECK:   hivm.hir.vcast
  // CHECK:   hivm.hir.vtranspose
  // CHECK:   tensor.extract_slice
  // CHECK:   hivm.hir.store
  // CHECK: } else {
  // CHECK:   %[[SLICE:.*]] = tensor.extract_slice
  // CHECK-SAME: : tensor<16x16xf32> to tensor<8x8xf32>
  // CHECK:   hivm.hir.fixpipe
  // CHECK-SAME: dma_mode = #hivm.dma_mode<nz2dn>
  // CHECK-SAME: ins(%[[SLICE]] : tensor<8x8xf32>)
  // CHECK-SAME: outs(%{{.*}} : memref<8x8xf16
  // CHECK-NOT: hivm.hir.store
  // CHECK: } {fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fallback_not_exec_cast_transpose_slice_store(
      %cond: i1,
      %l0c: tensor<16x16xf32>,
      %gm: memref<16x16xf16, strided<[16, 1]>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %brc_init = tensor.empty() : tensor<16x16xf32>
    %res = scf.if %cond -> (tensor<16x16xf32>) {
      %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brc_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %brc : tensor<16x16xf32>
    } else {
      %ub = tensor.empty() : tensor<16x16xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
          ins(%l0c : tensor<16x16xf32>) outs(%ub : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scf.yield %fix : tensor<16x16xf32>
    } {fallback_not_exec}
    %cast_init = tensor.empty() : tensor<16x16xf16>
    %cast = hivm.hir.vcast ins(%res : tensor<16x16xf32>) outs(%cast_init : tensor<16x16xf16>) -> tensor<16x16xf16>
    %tr_init = tensor.empty() : tensor<16x16xf16>
    %tr = hivm.hir.vtranspose ins(%cast : tensor<16x16xf16>) outs(%tr_init : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
    %slice = tensor.extract_slice %tr[0, 0] [8, 8] [1, 1]
        : tensor<16x16xf16> to tensor<8x8xf16>
    %gm_subview = memref.subview %gm[0, 0] [8, 8] [1, 1]
        : memref<16x16xf16, strided<[16, 1]>> to memref<8x8xf16, strided<[16, 1]>>
    hivm.hir.store ins(%slice : tensor<8x8xf16>) outs(%gm_subview : memref<8x8xf16, strided<[16, 1]>>)
    return
  }
}

// -----

// Basic fallback_not_exec sink: store only (no cast / transpose / slice).
// CHECK-LABEL: func.func @inline_fixpipe_fallback_not_exec_store
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[BRC:.*]] = hivm.hir.vbrc
// CHECK:   hivm.hir.store ins(%[[BRC]]
// CHECK: } else {
// CHECK:   hivm.hir.fixpipe
// CHECK-SAME: outs(%{{.*}} : memref<16x16xf32
// CHECK-NOT: hivm.hir.store
// CHECK: } {fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fallback_not_exec_store(
      %cond: i1,
      %l0c: tensor<16x16xf32>,
      %gm: memref<16x16xf32, strided<[16, 1]>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %brc_init = tensor.empty() : tensor<16x16xf32>
    %res = scf.if %cond -> (tensor<16x16xf32>) {
      %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brc_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %brc : tensor<16x16xf32>
    } else {
      %ub = tensor.empty() : tensor<16x16xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
          ins(%l0c : tensor<16x16xf32>) outs(%ub : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scf.yield %fix : tensor<16x16xf32>
    } {fallback_not_exec}
    hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%gm : memref<16x16xf32, strided<[16, 1]>>)
    return
  }
}

// -----

// Without fallback_not_exec the sink must not apply: store stays outside the if.
// CHECK-LABEL: func.func @inline_fixpipe_no_fallback_not_exec_no_sink
// CHECK: %[[RES:.*]] = scf.if %{{.*}} -> (tensor<16x16xf32>) {
// CHECK:   hivm.hir.vbrc
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   hivm.hir.fixpipe
// CHECK:   scf.yield
// CHECK: }
// CHECK-NOT: fallback_not_exec
// CHECK: hivm.hir.store ins(%[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_no_fallback_not_exec_no_sink(
      %cond: i1,
      %l0c: tensor<16x16xf32>,
      %gm: memref<16x16xf32, strided<[16, 1]>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %brc_init = tensor.empty() : tensor<16x16xf32>
    %res = scf.if %cond -> (tensor<16x16xf32>) {
      %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brc_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %brc : tensor<16x16xf32>
    } else {
      %ub = tensor.empty() : tensor<16x16xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
          ins(%l0c : tensor<16x16xf32>) outs(%ub : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scf.yield %fix : tensor<16x16xf32>
    }
    hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%gm : memref<16x16xf32, strided<[16, 1]>>)
    return
  }
}

// -----

// fallback_not_exec if + store nested inside scf.for.
// CHECK-LABEL: func.func @inline_fixpipe_fallback_not_exec_inside_for
// CHECK: scf.for
// CHECK:   scf.if %{{.*}} {
// CHECK:     %[[BRC:.*]] = hivm.hir.vbrc
// CHECK:     hivm.hir.store ins(%[[BRC]]
// CHECK:   } else {
// CHECK:     hivm.hir.fixpipe
// CHECK-SAME: outs(%{{.*}} : memref<16x16xf32
// CHECK-NOT: hivm.hir.store
// CHECK:   } {fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fallback_not_exec_inside_for(
      %lb: index, %ub: index, %step: index,
      %cond: i1,
      %l0c: tensor<16x16xf32>,
      %gm: memref<16x16xf32, strided<[16, 1]>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %brc_init = tensor.empty() : tensor<16x16xf32>
    scf.for %i = %lb to %ub step %step {
      %res = scf.if %cond -> (tensor<16x16xf32>) {
        %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brc_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %brc : tensor<16x16xf32>
      } else {
        %empty = tensor.empty() : tensor<16x16xf32>
        %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
            ins(%l0c : tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>)
            -> tensor<16x16xf32>
        scf.yield %fix : tensor<16x16xf32>
      } {fallback_not_exec}
      hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%gm : memref<16x16xf32, strided<[16, 1]>>)
    }
    return
  }
}

// -----

// annotation.mark on the fallback_not_exec if-result is ignored when walking
// the store chain (same as hivm.debug / tensor.dim): it does not count as a
// second user, so the sink still applies. After the rewrite the if is
// resultless, so the mark on the old if-result is erased. Marks on values
// that still exist (e.g. the L0C source) are left in place.
// CHECK-LABEL: func.func @inline_fixpipe_fallback_not_exec_markop
// CHECK: annotation.mark %{{.*}} : tensor<16x16xf32>
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[BRC:.*]] = hivm.hir.vbrc
// CHECK:   hivm.hir.store ins(%[[BRC]]
// CHECK: } else {
// CHECK:   hivm.hir.fixpipe
// CHECK-SAME: outs(%{{.*}} : memref<16x16xf32
// CHECK-NOT: hivm.hir.store
// CHECK: } {fallback_not_exec
// CHECK-NOT: annotation.mark %{{.*}} : tensor<16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fallback_not_exec_markop(
      %cond: i1,
      %l0c: tensor<16x16xf32>,
      %gm: memref<16x16xf32, strided<[16, 1]>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %brc_init = tensor.empty() : tensor<16x16xf32>
    annotation.mark %l0c : tensor<16x16xf32>
    %res = scf.if %cond -> (tensor<16x16xf32>) {
      %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brc_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %brc : tensor<16x16xf32>
    } else {
      %ub = tensor.empty() : tensor<16x16xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
          ins(%l0c : tensor<16x16xf32>) outs(%ub : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scf.yield %fix : tensor<16x16xf32>
    } {fallback_not_exec}
    // expected-warning @below {{dropping annotation.mark on fallback_not_exec if result; the tensor no longer exists after InlineFixpipe sink}}
    annotation.mark %res : tensor<16x16xf32>
    hivm.hir.store ins(%res : tensor<16x16xf32>) outs(%gm : memref<16x16xf32, strided<[16, 1]>>)
    return
  }
}

// -----

// Dynamic extract_slice offsets/sizes (and the matching GM subview) sit after
// the if; the sink hoists those effect-free defs so they dominate uses inside
// both branches.
// CHECK-LABEL: func.func @inline_fixpipe_fallback_not_exec_dynamic_slice_store
// CHECK: scf.if %{{.*}} {
// CHECK:   hivm.hir.vbrc
// CHECK:   tensor.extract_slice
// CHECK:   hivm.hir.store
// CHECK: } else {
// CHECK:   tensor.extract_slice
// CHECK: } {fallback_not_exec
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fallback_not_exec_dynamic_slice_store(
      %cond: i1,
      %l0c: tensor<16x16xf32>,
      %gm: memref<16x16xf16, strided<[16, 1]>>,
      %off: index,
      %sz: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %brc_init = tensor.empty() : tensor<16x16xf32>
    %res = scf.if %cond -> (tensor<16x16xf32>) {
      %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brc_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %brc : tensor<16x16xf32>
    } else {
      %ub = tensor.empty() : tensor<16x16xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
          ins(%l0c : tensor<16x16xf32>) outs(%ub : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scf.yield %fix : tensor<16x16xf32>
    } {fallback_not_exec}
    %cast_init = tensor.empty() : tensor<16x16xf16>
    %cast = hivm.hir.vcast ins(%res : tensor<16x16xf32>) outs(%cast_init : tensor<16x16xf16>) -> tensor<16x16xf16>
    %tr_init = tensor.empty() : tensor<16x16xf16>
    %tr = hivm.hir.vtranspose ins(%cast : tensor<16x16xf16>) outs(%tr_init : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
    %off0 = arith.addi %off, %c0 : index
    %slice = tensor.extract_slice %tr[%off0, %off0] [%sz, %sz] [1, 1]
        : tensor<16x16xf16> to tensor<?x?xf16>
    %gm_subview = memref.subview %gm[%off0, %off0] [%sz, %sz] [1, 1]
        : memref<16x16xf16, strided<[16, 1]>> to memref<?x?xf16, strided<[16, 1], offset: ?>>
    hivm.hir.store ins(%slice : tensor<?x?xf16>) outs(%gm_subview : memref<?x?xf16, strided<[16, 1], offset: ?>>)
    return
  }
}

// -----

// The fallback_not_exec consumer chain is nested inside a second scf.if, so
// its dynamic slice indices / store dst are defined in that inner region and
// cannot dominate a sunk if (and sinking would drop the inner guard). The
// sink must not apply and must not crash: the fallback if keeps its result
// and the store stays nested.
// CHECK-LABEL: func.func @inline_fixpipe_fallback_not_exec_nested_if_no_sink
// CHECK: %[[RES:.*]] = scf.if %{{.*}} -> (tensor<16x16xf32>) {
// CHECK:   hivm.hir.vbrc
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   hivm.hir.fixpipe
// CHECK:   scf.yield
// CHECK: } {fallback_not_exec}
// CHECK: scf.if %{{.*}} {
// CHECK:   tensor.extract_slice
// CHECK:   hivm.hir.store
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inline_fixpipe_fallback_not_exec_nested_if_no_sink(
      %cond: i1,
      %cond2: i1,
      %l0c: tensor<16x16xf32>,
      %gm: memref<16x16xf16, strided<[16, 1]>>,
      %off: index,
      %sz: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %brc_init = tensor.empty() : tensor<16x16xf32>
    %res = scf.if %cond -> (tensor<16x16xf32>) {
      %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brc_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %brc : tensor<16x16xf32>
    } else {
      %ub = tensor.empty() : tensor<16x16xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
          ins(%l0c : tensor<16x16xf32>) outs(%ub : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scf.yield %fix : tensor<16x16xf32>
    } {fallback_not_exec}
    scf.if %cond2 {
      %cast_init = tensor.empty() : tensor<16x16xf16>
      %cast = hivm.hir.vcast ins(%res : tensor<16x16xf32>) outs(%cast_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %o = arith.addi %off, %off : index
      %slice = tensor.extract_slice %cast[%o, %o] [%sz, %sz] [1, 1]
          : tensor<16x16xf16> to tensor<?x?xf16>
      %sub = memref.subview %gm[%o, %o] [%sz, %sz] [1, 1]
          : memref<16x16xf16, strided<[16, 1]>> to memref<?x?xf16, strided<[16, 1], offset: ?>>
      hivm.hir.store ins(%slice : tensor<?x?xf16>) outs(%sub : memref<?x?xf16, strided<[16, 1], offset: ?>>)
    }
    return
  }
}
