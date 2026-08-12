// RUN: bishengir-opt %s --hivm-insert-convert-layout --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @insert_for_mmad_basic(
// CHECK-SAME: %[[A:.*]]: tensor<64x16xf16>, %[[B:.*]]: tensor<16x32xf16>)
// CHECK: %[[OUT_INIT:.*]] = tensor.empty() : tensor<64x32xf32>
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[A]] output_shape [1, 4, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[B]] output_shape [2, 1, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[C_FR:.*]] = hivm.hir.convert_layout %[[OUT_INIT]] output_shape [2, 4, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 ins(%[[A_FR]], %[[B_FR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x16x16xf16>, tensor<2x1x16x16xf16>, i1, index, index, index) outs(%[[C_FR]] : tensor<2x4x16x16xf32>) -> tensor<2x4x16x16xf32>
// CHECK: %[[RES_ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [64, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: return %[[RES_ND]] : tensor<64x32xf32>
func.func @insert_for_mmad_basic(
    %arg0: tensor<64x16xf16>, %arg1: tensor<16x32xf16>) -> tensor<64x32xf32> {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %out = tensor.empty() : tensor<64x32xf32>
  %res = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c64, %c16, %c32 : tensor<64x16xf16>, tensor<16x32xf16>, i1, index, index, index)
                        outs(%out : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %res : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: func.func @insert_for_mmad_transpose_b(
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[B:.+]] output_shape [1, 2, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 {b_transpose} ins(%{{.*}}, %[[B_FR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x4x16x16xf16>, tensor<1x2x16x16xf16>, i1, index, index, index) outs(%{{.*}} : tensor<2x4x16x16xf32>) -> tensor<2x4x16x16xf32>
// CHECK: %[[RES_ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [64, 32]
// CHECK: return %[[RES_ND]] : tensor<64x32xf32>
func.func @insert_for_mmad_transpose_b(
    %arg0: tensor<64x16xf16>, %arg1: tensor<32x16xf16>) -> tensor<64x32xf32> {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %out = tensor.empty() : tensor<64x32xf32>
  %res = hivm.hir.mmadL1 {b_transpose} ins(%arg0, %arg1, %true, %c64, %c16, %c32 : tensor<64x16xf16>, tensor<32x16xf16>, i1, index, index, index)
                                      outs(%out : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %res : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: func.func @skip_when_mmad_is_already_fractal(
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1 ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<4x1x16x16xf16>, tensor<1x2x16x16xf16>, i1, index, index, index) outs(%{{.*}} : tensor<4x2x16x16xf32>) -> tensor<4x2x16x16xf32>
// CHECK: return %[[MMAD]] : tensor<4x2x16x16xf32>
func.func @skip_when_mmad_is_already_fractal(
    %arg0: tensor<4x1x16x16xf16>, %arg1: tensor<1x2x16x16xf16>) -> tensor<4x2x16x16xf32> {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %out = tensor.empty() : tensor<4x2x16x16xf32>
  %res = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c64, %c16, %c32 : tensor<4x1x16x16xf16>, tensor<1x2x16x16xf16>, i1, index, index, index)
                        outs(%out : tensor<4x2x16x16xf32>) -> tensor<4x2x16x16xf32>
  return %res : tensor<4x2x16x16xf32>
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @insert_for_mmadmx_A5(
// CHECK-SAME: %[[A:.*]]: tensor<208x64xf8E4M3FN>, %[[B:.*]]: tensor<64x224xf8E4M3FN>, %[[SA:.*]]: tensor<208x2xi8>, %[[SB:.*]]: tensor<224x2xi8>)
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[A]] output_shape [2, 13, 16, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[B_FR:.*]] = hivm.hir.convert_layout %[[B]] output_shape [7, 2, 32, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[SA_FR:.*]] = hivm.hir.convert_layout %[[SA]] output_shape [13, 1, 16, 2]
// CHECK-SAME: {dstLayout = #hivm.data_layout<SCALEA_zZ, fractalSizes = [16, 2]>, srcLayout = #hivm.data_layout<SCALEA_ND>}
// CHECK: %[[SB_FR:.*]] = hivm.hir.convert_layout %[[SB]] output_shape [14, 1, 16, 2]
// CHECK-SAME: {dstLayout = #hivm.data_layout<SCALEB_nN, fractalSizes = [16, 2]>, srcLayout = #hivm.data_layout<SCALEB_DN>}
// CHECK: %[[C_FR:.*]] = hivm.hir.convert_layout %{{.*}} output_shape [14, 13, 16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadmxL1 {already_set_real_mkn} ins(%[[A_FR]], %[[B_FR]], %[[SA_FR]], %[[SB_FR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x13x16x32xf8E4M3FN>, tensor<7x2x32x32xf8E4M3FN>, tensor<13x1x16x2xi8>, tensor<14x1x16x2xi8>, i1, index, index, index) outs(%[[C_FR]] : tensor<14x13x16x16xf32>) -> tensor<14x13x16x16xf32>
// CHECK: %[[RES_ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [208, 224]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: return %[[RES_ND]] : tensor<208x224xf32>
  func.func @insert_for_mmadmx_A5(
      %arg0: tensor<208x64xf8E4M3FN>, %arg1: tensor<64x224xf8E4M3FN>,
      %arg2: tensor<208x2xi8>, %arg3: tensor<224x2xi8>) -> tensor<208x224xf32> {
    %true = arith.constant true
    %c208 = arith.constant 208 : index
    %c64 = arith.constant 64 : index
    %c224 = arith.constant 224 : index
    %out = tensor.empty() : tensor<208x224xf32>
    %res = hivm.hir.mmadmxL1 {already_set_real_mkn}
        ins(%arg0, %arg1, %arg2, %arg3, %true, %c208, %c64, %c224
            : tensor<208x64xf8E4M3FN>, tensor<64x224xf8E4M3FN>,
              tensor<208x2xi8>, tensor<224x2xi8>, i1, index, index, index)
        outs(%out : tensor<208x224xf32>) -> tensor<208x224xf32>
    return %res : tensor<208x224xf32>
  }
}

// -----

// When mmadL1 result is only consumed by fixpipe writing to L1 (cbuf), skip
// Fractal->ND convert_layout on the mmad result so fixpipe can stay NZ2NZ.
// CHECK-LABEL: func.func @mmad_result_to_fixpipe_cbuf_no_nd_convert(
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.convert_layout %[[MMAD]]
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[MMAD]] : tensor<1x1x16x16xf32>) outs(%{{.*}} : memref<16x16xf32, #hivm.address_space<cbuf>>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @mmad_result_to_fixpipe_cbuf_no_nd_convert(
      %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %out = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1
        ins(%a, %b, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad : tensor<16x16xf32>)
        outs(%alloc : memref<16x16xf32, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

// Contrast: fixpipe writing to UB still needs Fractal->ND convert_layout.
// CHECK-LABEL: func.func @mmad_result_to_fixpipe_ub_keeps_nd_convert(
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK: %[[ND:.*]] = hivm.hir.convert_layout %[[MMAD]] output_shape [16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[ND]] : tensor<16x16xf32>) outs(%{{.*}} : memref<16x16xf32, #hivm.address_space<ub>>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @mmad_result_to_fixpipe_ub_keeps_nd_convert(
      %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %out = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1
        ins(%a, %b, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%mmad : tensor<16x16xf32>)
        outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Rank-2 NZ2NZ Fixpipe carrying the previous mmad's accumulator to the next
// mmad is already physically fractal. Retarget the Fixpipe to a rank-4
// fractal type; do not insert ND→Fractal on it.
// f32 A uses fractalSizes [16, 8] → 16x16 becomes 2x1x16x8.
// CHECK-LABEL: func.func @fixpipe_nz2nz_chain_retargets_to_fractal(
// CHECK: %[[MMAD0:.*]] = hivm.hir.mmadL1
// CHECK: %[[ACC_ND:.*]] = hivm.hir.convert_layout %[[MMAD0]] output_shape [16, 16]
// CHECK-SAME: {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe ins(%[[ACC_ND]] : tensor<16x16xf32>){{.*}}-> tensor<2x1x16x8xf32>
// CHECK-NOT: hivm.hir.convert_layout %[[FIX]]
// CHECK: %[[MMAD1:.*]] = hivm.hir.mmadL1 {{.*}} ins(%[[FIX]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x1x16x8xf32>, tensor<1x1x16x16xf16>, i1, index, index, index)
// CHECK: %[[RES:.*]] = hivm.hir.convert_layout %[[MMAD1]] output_shape [16, 16]
// CHECK: return %[[RES]] : tensor<16x16xf32>
func.func @fixpipe_nz2nz_chain_retargets_to_fractal(
    %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) -> tensor<16x16xf32> {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %acc_out = tensor.empty() : tensor<16x16xf32>
  %acc = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
      ins(%a, %b, %true, %c16, %c16, %c16
          : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
      outs(%acc_out : tensor<16x16xf32>) -> tensor<16x16xf32>
  %fix_out = tensor.empty() : tensor<16x16xf32>
  // dma_mode omitted defaults to NZ2NZ.
  %fix = hivm.hir.fixpipe
      ins(%acc : tensor<16x16xf32>) outs(%fix_out : tensor<16x16xf32>)
      -> tensor<16x16xf32>
  %out = tensor.empty() : tensor<16x16xf32>
  %mmad = hivm.hir.mmadL1 {already_set_real_mkn}
      ins(%fix, %b, %true, %c16, %c16, %c16
          : tensor<16x16xf32>, tensor<16x16xf16>, i1, index, index, index)
      outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mmad : tensor<16x16xf32>
}

// -----

// Contrast: NZ2ND Fixpipe result is true ND and still needs ND→Fractal.
// CHECK-LABEL: func.func @fixpipe_nz2nd_still_gets_nd_to_fractal(
// CHECK: %[[MMAD0:.*]] = hivm.hir.mmadL1
// CHECK: %[[ACC_ND:.*]] = hivm.hir.convert_layout %[[MMAD0]] output_shape [16, 16]
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[ACC_ND]] : tensor<16x16xf32>){{.*}}-> tensor<16x16xf32>
// CHECK: %[[A_FR:.*]] = hivm.hir.convert_layout %[[FIX]] output_shape [2, 1, 16, 8]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: hivm.hir.mmadL1 {{.*}} ins(%[[A_FR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x1x16x8xf32>, tensor<1x1x16x16xf16>, i1, index, index, index)
func.func @fixpipe_nz2nd_still_gets_nd_to_fractal(
    %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) -> tensor<16x16xf32> {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %acc_out = tensor.empty() : tensor<16x16xf32>
  %acc = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
      ins(%a, %b, %true, %c16, %c16, %c16
          : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
      outs(%acc_out : tensor<16x16xf32>) -> tensor<16x16xf32>
  %fix_out = tensor.empty() : tensor<16x16xf32>
  %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%acc : tensor<16x16xf32>) outs(%fix_out : tensor<16x16xf32>)
      -> tensor<16x16xf32>
  %out = tensor.empty() : tensor<16x16xf32>
  %mmad = hivm.hir.mmadL1 {already_set_real_mkn}
      ins(%fix, %b, %true, %c16, %c16, %c16
          : tensor<16x16xf32>, tensor<16x16xf16>, i1, index, index, index)
      outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mmad : tensor<16x16xf32>
}

// -----

// Channel-merge style chain (simplified from E2E):
//   mmadL1(i8,i8)->i32  --NZ2NZ fixpipe + S322I8-->  i8  -->  mmadL1(i8,i8)->i32
// On Ascend950, i8 A uses fractalSizes [16, 32] so 32x32 → 1x2x16x32.
// The NZ2NZ i8 Fixpipe must be retargeted to fractal; no ND→Fractal on it.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @channel_merge_fixpipe_nz2nz_i8_i32(
// CHECK: %[[MMAD0:.*]] = hivm.hir.mmadL1
// CHECK: %[[ACC_ND:.*]] = hivm.hir.convert_layout %[[MMAD0]] output_shape [32, 32]
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>} ins(%[[ACC_ND]] : tensor<32x32xi32>){{.*}}-> tensor<1x2x16x32xi8>
// CHECK-NOT: hivm.hir.convert_layout %[[FIX]]
// CHECK: %[[MMAD1:.*]] = hivm.hir.mmadL1 {{.*}} ins(%[[FIX]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x16x32xi8>, tensor<1x2x16x32xi8>, i1, index, index, index)
// CHECK: %[[RES:.*]] = hivm.hir.convert_layout %[[MMAD1]]
// CHECK: return %[[RES]] : tensor<32x32xi32>
  func.func @channel_merge_fixpipe_nz2nz_i8_i32(
      %a: tensor<32x32xi8>, %b: tensor<32x32xi8>) -> tensor<32x32xi32> {
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %out0 = tensor.empty() : tensor<32x32xi32>
    %mmad0 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
        ins(%a, %b, %true, %c32, %c32, %c32
            : tensor<32x32xi8>, tensor<32x32xi8>, i1, index, index, index)
        outs(%out0 : tensor<32x32xi32>) -> tensor<32x32xi32>
    %fix_out = tensor.empty() : tensor<32x32xi8>
    // dma_mode omitted defaults to NZ2NZ (channel-merge / keep NZ for next cube).
    %fix = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>}
        ins(%mmad0 : tensor<32x32xi32>) outs(%fix_out : tensor<32x32xi8>)
        -> tensor<32x32xi8>
    %out1 = tensor.empty() : tensor<32x32xi32>
    %mmad1 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
        ins(%fix, %b, %true, %c32, %c32, %c32
            : tensor<32x32xi8>, tensor<32x32xi8>, i1, index, index, index)
        outs(%out1 : tensor<32x32xi32>) -> tensor<32x32xi32>
    return %mmad1 : tensor<32x32xi32>
  }
}

// -----

// Multi-use NZ2NZ Fixpipe cannot be retargeted: its result feeds two mmads.
// Fall back to a Fractal→Fractal convert_layout type marker per consumer;
// Combine folds each marker into a 4D Fixpipe later.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @fixpipe_nz2nz_multi_use_inserts_fractal_marker(
// CHECK: %[[MMAD0:.*]] = hivm.hir.mmadL1
// CHECK: %[[ACC_ND:.*]] = hivm.hir.convert_layout %[[MMAD0]] output_shape [32, 32]
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>} ins(%[[ACC_ND]] : tensor<32x32xi32>){{.*}}-> tensor<32x32xi8>
// CHECK: %[[FIX_FR0:.*]] = hivm.hir.convert_layout %[[FIX]] output_shape [1, 2, 16, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>}
// CHECK: %[[MMAD1:.*]] = hivm.hir.mmadL1 {{.*}} ins(%[[FIX_FR0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x16x32xi8>, tensor<1x2x16x32xi8>, i1, index, index, index)
// CHECK: %[[FIX_FR1:.*]] = hivm.hir.convert_layout %[[FIX]] output_shape [1, 2, 16, 32]
// CHECK-SAME: {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>}
// CHECK: %[[MMAD2:.*]] = hivm.hir.mmadL1 {{.*}} ins(%[[FIX_FR1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x16x32xi8>, tensor<1x2x16x32xi8>, i1, index, index, index)
  func.func @fixpipe_nz2nz_multi_use_inserts_fractal_marker(
      %a: tensor<32x32xi8>, %b0: tensor<32x32xi8>, %b1: tensor<32x32xi8>)
      -> (tensor<32x32xi32>, tensor<32x32xi32>) {
    %true = arith.constant true
    %c32 = arith.constant 32 : index
    %acc_out = tensor.empty() : tensor<32x32xi32>
    %acc = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
        ins(%a, %b0, %true, %c32, %c32, %c32
            : tensor<32x32xi8>, tensor<32x32xi8>, i1, index, index, index)
        outs(%acc_out : tensor<32x32xi32>) -> tensor<32x32xi32>
    %fix_out = tensor.empty() : tensor<32x32xi8>
    // dma_mode omitted defaults to NZ2NZ (channel-merge / keep NZ for next cube).
    %fix = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>}
        ins(%acc : tensor<32x32xi32>) outs(%fix_out : tensor<32x32xi8>)
        -> tensor<32x32xi8>
    %out0 = tensor.empty() : tensor<32x32xi32>
    %mmad0 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
        ins(%fix, %b0, %true, %c32, %c32, %c32
            : tensor<32x32xi8>, tensor<32x32xi8>, i1, index, index, index)
        outs(%out0 : tensor<32x32xi32>) -> tensor<32x32xi32>
    %out1 = tensor.empty() : tensor<32x32xi32>
    %mmad1 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
        ins(%fix, %b1, %true, %c32, %c32, %c32
            : tensor<32x32xi8>, tensor<32x32xi8>, i1, index, index, index)
        outs(%out1 : tensor<32x32xi32>) -> tensor<32x32xi32>
    return %mmad0, %mmad1 : tensor<32x32xi32>, tensor<32x32xi32>
  }
}
