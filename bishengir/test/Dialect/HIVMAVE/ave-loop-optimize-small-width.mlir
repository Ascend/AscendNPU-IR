// RUN: bishengir-opt -ave-loop-optimize="max-small-width-merge-factor=4" %s --split-input-file | FileCheck %s
// RUN: bishengir-opt -ave-loop-optimize="max-small-width-merge-factor=2" %s --split-input-file | FileCheck %s --check-prefix=MAX2
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 -ave-loop-optimize="max-small-width-merge-factor=4" %s --split-input-file | FileCheck %s --check-prefix=TARGET
// RUN: bishengir-opt -ave-process-vsstb -ave-loop-optimize="max-small-width-merge-factor=4" %s --split-input-file | FileCheck %s --check-prefix=PIPE
// RUN: bishengir-opt -ave-process-vsstb -ave-loop-optimize="max-small-width-merge-factor=4" -hacc-append-device-spec=target=Ascend910_9589 -analyze-vector-layout -canonicalize -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin %s --split-input-file | FileCheck %s --check-prefix=LOWER

// The short full chain is execute-bound. Saving one load/store does not cover
// the added vintlv/vdintlv work, so the loop must remain unchanged.
// CHECK-LABEL: func.func @factor2_full_chain
// CHECK: scf.for {{.*}} step %c64
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<64xf16>
// CHECK-NOT: ave.hir.vintlv
// CHECK-NOT: ave.hir.vdintlv
// CHECK: ave.hir.masked_store <NORM_B16> {{.*}} vector<64xi1>, vector<64xf16>
// CHECK-NOT: __ave_small_width
// MAX2-LABEL: func.func @factor2_full_chain
// MAX2: ave.hir.vload <NORM> {{.*}} into vector<64xf16>
// MAX2-NOT: ave.hir.vintlv
// MAX2-NOT: ave.hir.vdintlv
// LOWER-LABEL: func.func @factor2_full_chain
// LOWER-NOT: hivm_regbaseintrins.intr.hivm.vintlv
// LOWER-NOT: hivm_regbaseintrins.intr.hivm.vdintlv
// LOWER: hivm_regbaseintrins.intr.hivm.vcvtff.f162f32
func.func @factor2_full_chain(
    %src: memref<256xf16, #hivm.address_space<ub>>,
    %dst: memref<256xf16, #hivm.address_space<ub>>,
    %wideBias: f32,
    %bias: f16)
    attributes {
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      hivm.vector_function,
      no_inline
    } {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %iv = %c0 to %c256 step %c64 {
    %loaded = ave.hir.vload <NORM> %src[%iv]
        : memref<256xf16, #hivm.address_space<ub>> into vector<64xf16>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %wide = ave.hir.vextf %loaded, <part_even>, %mask
        : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %wideBiased = ave.hir.vadds %wide, %wideBias, %mask
        : vector<64xf32>, f32, vector<64xi1>
    %narrow = ave.hir.vtruncf %wideBiased, <rint>, false, <part_even>, %mask
        : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %biased = ave.hir.vadds %narrow, %bias, %mask
        : vector<64xf16>, f16, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%iv], %mask, %biased
        : memref<256xf16, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf16>
  }
  return
}

// -----

// CHECK-LABEL: func.func @factor4_load_i8_to_i32
// CHECK: %[[PACKED:.*]] = memref.reinterpret_cast {{.*}} to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK: ave.hir.vload <NORM> %[[PACKED]][{{.*}}] : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>
// CHECK-COUNT-3: ave.hir.vintlv
// CHECK-COUNT-4: ave.hir.vextsi
// CHECK: ave.hir.plt
// CHECK-NOT: __ave_small_width
// MAX2-LABEL: func.func @factor4_load_i8_to_i32
// MAX2: scf.for
// MAX2: ave.hir.vload <NORM> {{.*}} into vector<64xi8>
// MAX2-NOT: ave.hir.vintlv
// LOWER-LABEL: func.func @factor4_load_i8_to_i32
// LOWER: hivm_regbaseintrins.intr.hivm.vldsx1.v256s8
// LOWER-COUNT-3: hivm_regbaseintrins.intr.hivm.vintlv
// LOWER-COUNT-4: hivm_regbaseintrins.intr.hivm.vcvtii.s82s32
func.func @factor4_load_i8_to_i32(
    %src: memref<256xi8, #hivm.address_space<ub>>,
    %dst: memref<256xi32, #hivm.address_space<ub>>,
    %aux0: memref<256xi32, #hivm.address_space<ub>>,
    %aux1: memref<256xi32, #hivm.address_space<ub>>,
    %auxDst: memref<256xi32, #hivm.address_space<ub>>)
    attributes {
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      hivm.vector_function,
      no_inline
    } {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %iv = %c0 to %c256 step %c64 {
    %loaded = ave.hir.vload <NORM> %src[%iv]
        : memref<256xi8, #hivm.address_space<ub>> into vector<64xi8>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %wide = ave.hir.vextsi %loaded, %mask
        {pp = #ave.vcvt_pp_type<pp0>}
        : vector<64xi8>, vector<64xi32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %dst[%iv], %mask, %wide
        : memref<256xi32, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xi32>
    // Plt is intentionally outside the cost table. It must not prevent the
    // remaining known pipeline costs from approving this load merge.
    %auxMask, %trueShape = ave.hir.plt %iv : vector<64xi1>, index
    %auxLoaded0 = ave.hir.vload <NORM> %aux0[%iv]
        : memref<256xi32, #hivm.address_space<ub>> into vector<64xi32>
    %auxLoaded1 = ave.hir.vload <NORM> %aux1[%iv]
        : memref<256xi32, #hivm.address_space<ub>> into vector<64xi32>
    %auxSum = ave.hir.vadd %auxLoaded0, %auxLoaded1, %auxMask
        : vector<64xi32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %auxDst[%iv], %auxMask, %auxSum
        : memref<256xi32, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xi32>
  }
  return
}

// -----

// Without a target cost table, the fallback makes the layout tree too
// expensive. The AVE target table uses the measured i8 throughput and makes
// the four narrow chains profitable.
// CHECK-LABEL: func.func @factor4_narrow_i32_to_i8
// CHECK: scf.for {{.*}} step %c64
// CHECK: ave.hir.vtrunci {{.*}} : vector<64xi32>, vector<64xi8>, vector<64xi1>
// CHECK-NOT: ave.hir.vdintlv
// CHECK: ave.hir.masked_store <NORM_B8> {{.*}} vector<64xi1>, vector<64xi8>
// CHECK-NOT: __ave_small_width
// MAX2-LABEL: func.func @factor4_narrow_i32_to_i8
// MAX2: scf.for
// MAX2-NOT: ave.hir.vdintlv
// MAX2: ave.hir.masked_store <NORM_B8> {{.*}} vector<64xi1>, vector<64xi8>
// TARGET-LABEL: func.func @factor4_narrow_i32_to_i8
// TARGET-COUNT-4: ave.hir.vtrunci
// TARGET-COUNT-3: ave.hir.vdintlv
// TARGET-COUNT-1: ave.hir.vadds {{.*}} : vector<256xi8>, i8, vector<256xi1>
// TARGET-COUNT-1: ave.hir.masked_store <NORM_B8> {{.*}} vector<256xi1>, vector<256xi8>
// LOWER-LABEL: func.func @factor4_narrow_i32_to_i8
// LOWER-COUNT-1: hivm_regbaseintrins.intr.hivm.vcvtii.s322u8.x
// LOWER-NOT: hivm_regbaseintrins.intr.hivm.vdintlv
// LOWER: hivm_regbaseintrins.intr.hivm.vadds.s.x
// LOWER: hivm_regbaseintrins.intr.hivm.vstsx1
func.func @factor4_narrow_i32_to_i8(
    %src: memref<256xi32, #hivm.address_space<ub>>,
    %dst: memref<256xi8, #hivm.address_space<ub>>,
    %bias: i8)
    attributes {
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      hivm.vector_function,
      no_inline
    } {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %iv = %c0 to %c256 step %c64 {
    %loaded = ave.hir.vload <NORM> %src[%iv]
        : memref<256xi32, #hivm.address_space<ub>> into vector<64xi32>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %narrow = ave.hir.vtrunci %loaded, false, %mask
        {pp = #ave.vcvt_pp_type<pp0>}
        : vector<64xi32>, vector<64xi8>, vector<64xi1>
    %biased = ave.hir.vadds %narrow, %bias, %mask
        : vector<64xi8>, i8, vector<64xi1>
    ave.hir.masked_store <NORM_B8> %dst[%iv], %mask, %biased
        : memref<256xi8, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xi8>
  }
  return
}

// -----

// CHECK-LABEL: func.func @factor2_store_with_stride
// CHECK: scf.for
// CHECK-COUNT-1: ave.hir.vtruncf
// CHECK-NOT: ave.hir.vdintlv
// CHECK: ave.hir.store_with_stride {{.*}} vector<64xi1>, vector<64xf16>
// CHECK: } {unroll_for_vsstb}
// CHECK-NOT: __ave_small_width
// MAX2-LABEL: func.func @factor2_store_with_stride
// MAX2: scf.for
// MAX2-NOT: ave.hir.vdintlv
// MAX2: ave.hir.store_with_stride {{.*}} vector<64xi1>, vector<64xf16>
// PIPE-LABEL: func.func @factor2_store_with_stride
// PIPE-COUNT-2: ave.hir.vtruncf
// PIPE-COUNT-1: ave.hir.vdintlv
// PIPE-COUNT-1: ave.hir.store_with_stride {{.*}} vector<128xi1>, vector<128xf16>
// PIPE-NOT: unroll_for_vsstb
// LOWER-LABEL: func.func @factor2_store_with_stride
// LOWER: hivm_regbaseintrins.intr.hivm.vdintlv
// LOWER: hivm_regbaseintrins.intr.hivm.vsstb.v128f16
func.func @factor2_store_with_stride(
    %src: memref<128xf32, #hivm.address_space<ub>>,
    %dst: memref<2x64xf16, #hivm.address_space<ub>>)
    attributes {
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      hivm.vector_function,
      no_inline
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  scf.for %outer = %c0 to %c1 step %c1 {
    scf.for %iv = %c0 to %c2 step %c1 {
      %offset = affine.apply affine_map<(d0) -> (d0 * 64)>(%iv)
      %loaded = ave.hir.vload <NORM> %src[%offset]
          : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
      %mask = ave.hir.pge <ALL> : vector<64xi1>
      %narrow = ave.hir.vtruncf %loaded, <rint>, false, <part_even>, %mask
          : vector<64xf32>, vector<64xf16>, vector<64xi1>
      ave.hir.store_with_stride %dst[%iv, %c0], %c64, %mask, %narrow
          : memref<2x64xf16, #hivm.address_space<ub>>,
            vector<64xi1>, vector<64xf16>
    } {unroll_for_vsstb}
  }
  return
}

// -----

// CHECK-LABEL: func.func @mixed_narrowing_types
// CHECK: scf.for {{.*}} step %c64
// CHECK-NOT: ave.hir.vdintlv
// CHECK: return
// CHECK-NOT: __ave_small_width
// TARGET-LABEL: func.func @mixed_narrowing_types
// TARGET-DAG: ave.hir.vdintlv {{.*}} : vector<64xbf16>, vector<128xbf16>
// TARGET-DAG: ave.hir.vdintlv {{.*}} : vector<128xi8>, vector<256xi8>
// TARGET-DAG: ave.hir.vdintlv {{.*}} : vector<64xi16>, vector<128xi16>
// TARGET-DAG: ave.hir.vdintlv {{.*}} : vector<128xf8E4M3FN>, vector<256xf8E4M3FN>
// TARGET-DAG: ave.hir.vadds {{.*}} : vector<256xi8>, i8, vector<256xi1>
// TARGET-DAG: ave.hir.masked_store <NORM_B16> {{.*}} vector<128xi1>, vector<128xbf16>
// TARGET-DAG: ave.hir.masked_store <NORM_B8> {{.*}} vector<256xi1>, vector<256xi8>
// TARGET-DAG: ave.hir.masked_store <NORM_B16> {{.*}} vector<128xi1>, vector<128xi16>
// TARGET-DAG: ave.hir.masked_store <NORM_B8> {{.*}} vector<256xi1>, vector<256xf8E4M3FN>
func.func @mixed_narrowing_types(
    %f32Src0: memref<256xf32, #hivm.address_space<ub>>,
    %i16Src: memref<512xi16, #hivm.address_space<ub>>,
    %i32Src: memref<256xi32, #hivm.address_space<ub>>,
    %f32Src1: memref<256xf32, #hivm.address_space<ub>>,
    %bf16Dst: memref<256xbf16, #hivm.address_space<ub>>,
    %i8Dst: memref<512xi8, #hivm.address_space<ub>>,
    %i16Dst: memref<256xi16, #hivm.address_space<ub>>,
    %f8Dst: memref<256xf8E4M3FN, #hivm.address_space<ub>>,
    %i8Bias: i8,
    %i16Bias: i16)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %iv = %c0 to %c256 step %c64 {
    %offset128 = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
    %mask64 = ave.hir.pge <ALL> : vector<64xi1>
    %mask128 = ave.hir.pge <ALL> : vector<128xi1>
    %f32ForBf16 = ave.hir.vload <NORM> %f32Src0[%iv]
        : memref<256xf32, #hivm.address_space<ub>> into vector<64xf32>
    %bf16 = ave.hir.vtruncf %f32ForBf16, <rint>, false, <part_even>, %mask64
        : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    %bf16Biased0 = ave.hir.vabs %bf16, %mask64
        : vector<64xbf16>, vector<64xi1>
    %bf16Biased1 = ave.hir.vabs %bf16Biased0, %mask64
        : vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %bf16Dst[%iv], %mask64, %bf16Biased1
        : memref<256xbf16, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xbf16>
    %i16Input = ave.hir.vload <NORM> %i16Src[%offset128]
        : memref<512xi16, #hivm.address_space<ub>> into vector<128xi16>
    %i8 = ave.hir.vtrunci %i16Input, false, %mask128
        {part = #ave.vcvt_part_type<part_even>}
        : vector<128xi16>, vector<128xi8>, vector<128xi1>
    %biased0 = ave.hir.vadds %i8, %i8Bias, %mask128
        : vector<128xi8>, i8, vector<128xi1>
    %biased1 = ave.hir.vadds %biased0, %i8Bias, %mask128
        : vector<128xi8>, i8, vector<128xi1>
    ave.hir.masked_store <NORM_B8> %i8Dst[%offset128], %mask128, %biased1
        : memref<512xi8, #hivm.address_space<ub>>,
          vector<128xi1>, vector<128xi8>
    %i32Input = ave.hir.vload <NORM> %i32Src[%iv]
        : memref<256xi32, #hivm.address_space<ub>> into vector<64xi32>
    %i16 = ave.hir.vtrunci %i32Input, false, %mask64
        {part = #ave.vcvt_part_type<part_even>}
        : vector<64xi32>, vector<64xi16>, vector<64xi1>
    %i16Biased0 = ave.hir.vadds %i16, %i16Bias, %mask64
        : vector<64xi16>, i16, vector<64xi1>
    %i16Biased1 = ave.hir.vadds %i16Biased0, %i16Bias, %mask64
        : vector<64xi16>, i16, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %i16Dst[%iv], %mask64, %i16Biased1
        : memref<256xi16, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xi16>
    %f32ForF8 = ave.hir.vload <NORM> %f32Src1[%iv]
        : memref<256xf32, #hivm.address_space<ub>> into vector<64xf32>
    %f8 = ave.hir.vtruncf %f32ForF8, <round>, false, <part_even>, %mask64
        : vector<64xf32>, vector<64xf8E4M3FN>, vector<64xi1>
    ave.hir.masked_store <NORM_B8> %f8Dst[%iv], %mask64, %f8
        : memref<256xf8E4M3FN, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf8E4M3FN>
  }
  return
}

// -----

// CHECK-LABEL: func.func @reject_partial_row_widening_load
// CHECK: scf.for {{.*}} step %[[STEP:.*]]
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<64xbf16>
// CHECK-NOT: vector<128xbf16>
// CHECK-NOT: ave.hir.vintlv
func.func @reject_partial_row_widening_load(
    %src: memref<8x16xbf16, #hivm.address_space<ub>>,
    %dst: memref<8x16xf32, #hivm.address_space<ub>>)
    attributes {
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      hivm.vector_function,
      no_inline
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
    %srcRow = memref.subview %src[%iv, 0] [1, 16] [1, 1]
        : memref<8x16xbf16, #hivm.address_space<ub>>
          to memref<1x16xbf16, strided<[16, 1], offset: ?>,
                    #hivm.address_space<ub>>
    %srcFlat = memref.subview %srcRow[0, 0] [1, 16] [1, 1]
        : memref<1x16xbf16, strided<[16, 1], offset: ?>,
                 #hivm.address_space<ub>>
          to memref<16xbf16, strided<[1], offset: ?>,
                    #hivm.address_space<ub>>
    %loaded = ave.hir.vload <NORM> %srcFlat[%c0]
        : memref<16xbf16, strided<[1], offset: ?>,
                 #hivm.address_space<ub>> into vector<64xbf16>
    %all = ave.hir.pge <ALL> : vector<64xi1>
    %wide = ave.hir.vextf %loaded, <part_even>, %all
        : vector<64xbf16>, vector<64xf32>, vector<64xi1>
    %dstRow = memref.subview %dst[%iv, 0] [1, 16] [1, 1]
        : memref<8x16xf32, #hivm.address_space<ub>>
          to memref<1x16xf32, strided<[16, 1], offset: ?>,
                    #hivm.address_space<ub>>
    %mask = ave.hir.pge <VL16> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %dstRow[%c0, %c0], %mask, %wide
        : memref<1x16xf32, strided<[16, 1], offset: ?>,
                 #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @reject_loop_invariant_widening_load
// CHECK: scf.for {{.*}} step %[[STEP:.*]]
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<64xf16>
// CHECK-NOT: vector<128xf16>
// CHECK-NOT: ave.hir.vintlv
func.func @reject_loop_invariant_widening_load(
    %weights: memref<128xf16, #hivm.address_space<ub>>,
    %dst: memref<32x64xf32, #hivm.address_space<ub>>)
    attributes {
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      hivm.vector_function,
      no_inline
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %iv = %c0 to %c32 step %c1 {
    %loaded = ave.hir.vload <NORM> %weights[%c0]
        : memref<128xf16, #hivm.address_space<ub>> into vector<64xf16>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %wide = ave.hir.vextf %loaded, <part_even>, %mask
        : vector<64xf16>, vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %dst[%iv, %c0], %mask, %wide
        : memref<32x64xf32, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @reject_partial_row_narrow_store
// CHECK: scf.for {{.*}} step %[[STEP:.*]]
// CHECK: ave.hir.vtruncf
// CHECK-NOT: ave.hir.vdintlv
// CHECK-NOT: vector<128xf16>
func.func @reject_partial_row_narrow_store(
    %src: memref<8x64xf32, #hivm.address_space<ub>>,
    %dst: memref<8x16xf16, #hivm.address_space<ub>>)
    attributes {
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      hivm.vector_function,
      no_inline
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
    %loaded = ave.hir.vload <NORM> %src[%iv, %c0]
        : memref<8x64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %mask = ave.hir.pge <VL16> : vector<64xi1>
    %narrow = ave.hir.vtruncf %loaded, <rint>, false, <part_even>, %mask
        : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %dstRow = memref.subview %dst[%iv, 0] [1, 16] [1, 1]
        : memref<8x16xf16, #hivm.address_space<ub>>
          to memref<1x16xf16, strided<[16, 1], offset: ?>,
                    #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B16> %dstRow[%c0, %c0], %mask, %narrow
        : memref<1x16xf16, strided<[16, 1], offset: ?>,
                 #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf16>
  }
  return
}

// -----

// Gate regression: a loop-varying scalar cannot be shared after merging.
// CHECK-LABEL: func.func @reject_loop_varying_scalar
// CHECK: scf.for {{.*}} step %[[STEP:.*]]
// CHECK-COUNT-1: ave.hir.vtruncf
// CHECK-COUNT-1: memref.load
// CHECK-COUNT-1: ave.hir.vadds
// CHECK-COUNT-1: ave.hir.masked_store
// CHECK-NOT: ave.hir.vdintlv
func.func @reject_loop_varying_scalar(
    %src: memref<256xf32, #hivm.address_space<ub>>,
    %bias: memref<256xf16, #hivm.address_space<ub>>,
    %dst: memref<256xf16, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %iv = %c0 to %c256 step %c64 {
    %loaded = ave.hir.vload <NORM> %src[%iv]
        : memref<256xf32, #hivm.address_space<ub>> into vector<64xf32>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %narrow = ave.hir.vtruncf %loaded, <rint>, false, <part_even>, %mask
        : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %biasValue = memref.load %bias[%iv]
        : memref<256xf16, #hivm.address_space<ub>>
    %biased = ave.hir.vadds %narrow, %biasValue, %mask
        : vector<64xf16>, f16, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%iv], %mask, %biased
        : memref<256xf16, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf16>
  }
  return
}

// -----

// Gate regression: keep continuous ONEPT loops for HoistVstas.
// CHECK-LABEL: func.func @reject_continuous_onepoint_store
// CHECK: scf.for {{.*}} step %[[STEP:.*]]
// CHECK-COUNT-1: ave.hir.vload <NORM> {{.*}} into vector<64xf16>
// CHECK-NOT: vector<128xf16>
// CHECK-NOT: ave.hir.vintlv
// CHECK: ave.hir.masked_store <ONEPT_B32> {{.*}} {hivm.is_continuous}
func.func @reject_continuous_onepoint_store(
    %src: memref<256xf16, #hivm.address_space<ub>>,
    %dst: memref<4xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %offset = affine.apply affine_map<(d0) -> (d0 * 64)>(%iv)
    %loaded = ave.hir.vload <NORM> %src[%offset]
        : memref<256xf16, #hivm.address_space<ub>> into vector<64xf16>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %wide = ave.hir.vextf %loaded, <part_even>, %mask
        : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %one = builtin.unrealized_conversion_cast %wide
        : vector<64xf32> to vector<1xf32>
    %oneMask = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B32> %dst[%iv], %oneMask, %one
        {hivm.is_continuous}
        : memref<4xf32, #hivm.address_space<ub>>,
          vector<1xi1>, vector<1xf32>
  }
  return
}

// -----

// Gate regression: unmarked strided stores must not be merged.
// CHECK-LABEL: func.func @reject_unmarked_strided_column_store
// CHECK: scf.for
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<64xf16>
// CHECK-NOT: ave.hir.vintlv
// CHECK-COUNT-1: ave.hir.store_with_stride {{.*}} vector<64xi1>, vector<64xf16>
// CHECK-NOT: ave.hir.store_with_stride {{.*}} vector<128xf16>
// PIPE-LABEL: func.func @reject_unmarked_strided_column_store
// PIPE: ave.hir.vload <NORM> {{.*}} into vector<64xf16>
// PIPE-NOT: ave.hir.vintlv
// PIPE-COUNT-1: ave.hir.store_with_stride {{.*}} vector<128xi1>, vector<64xf16>
// PIPE-NOT: ave.hir.store_with_stride {{.*}} vector<128xf16>
func.func @reject_unmarked_strided_column_store(
    %src: memref<16x64xf16, #hivm.address_space<ub>>,
    %dst: memref<4x16x16xf16, strided<[272, 16, 1]>,
                 #hivm.address_space<ub>>,
    %scale: f32)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c272 = arith.constant 272 : index
  scf.for %iv = %c0 to %c16 step %c1 {
    %srcView = memref.subview %src[%iv, 0] [1, 64] [1, 1]
        : memref<16x64xf16, #hivm.address_space<ub>>
          to memref<1x64xf16, strided<[64, 1], offset: ?>,
                    #hivm.address_space<ub>>
    %srcFlat = memref.subview %srcView[0, 0] [1, 64] [1, 1]
        : memref<1x64xf16, strided<[64, 1], offset: ?>,
                 #hivm.address_space<ub>>
          to memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>,
                    #hivm.address_space<ub>>
    %loaded = ave.hir.vload <NORM> %srcFlat[%c0]
        : memref<64xf16, affine_map<(d0)[s0] -> (d0 + s0)>,
                 #hivm.address_space<ub>> into vector<64xf16>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %wide = ave.hir.vextf %loaded, <part_even>, %mask
        : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %scaled = ave.hir.vmuls %wide, %scale, %mask
        : vector<64xf32>, f32, vector<64xi1>
    %narrow = ave.hir.vtruncf %scaled, <rint>, false, <part_even>, %mask
        : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %dstView = memref.subview %dst[0, %iv, 0] [4, 1, 16] [1, 1, 1]
        : memref<4x16x16xf16, strided<[272, 16, 1]>,
                 #hivm.address_space<ub>>
          to memref<4x1x16xf16, strided<[272, 16, 1], offset: ?>,
                    #hivm.address_space<ub>>
    %dstFlat = memref.subview %dstView[0, 0, 0] [4, 1, 16] [1, 1, 1]
        : memref<4x1x16xf16, strided<[272, 16, 1], offset: ?>,
                 #hivm.address_space<ub>>
          to memref<4x16xf16,
                    affine_map<(d0, d1)[s0] -> (d0 * 272 + d1 + s0)>,
                    #hivm.address_space<ub>>
    ave.hir.store_with_stride %dstFlat[%c0, %c0], %c272, %mask, %narrow
        : memref<4x16xf16,
                 affine_map<(d0, d1)[s0] -> (d0 * 272 + d1 + s0)>,
                 #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf16>
  }
  return
}

// -----

// Extra loads make the loop load-bound, so one packed f16 load plus vintlv is
// cheaper than two original loads.
// CHECK-LABEL: func.func @accept_factor2_load_bound
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<128xf16>
// CHECK: ave.hir.vintlv
// TARGET-LABEL: func.func @accept_factor2_load_bound
// TARGET: ave.hir.vload <NORM> {{.*}} into vector<128xf16>
// TARGET: ave.hir.vintlv
func.func @accept_factor2_load_bound(
    %src: memref<128xf16, #hivm.address_space<ub>>,
    %aux0: memref<128xf32, #hivm.address_space<ub>>,
    %aux1: memref<128xf32, #hivm.address_space<ub>>,
    %aux2: memref<128xf32, #hivm.address_space<ub>>,
    %aux3: memref<128xf32, #hivm.address_space<ub>>,
    %dst: memref<128xf32, #hivm.address_space<ub>>)
    attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %iv = %c0 to %c128 step %c64 {
    %loaded = ave.hir.vload <NORM> %src[%iv]
        : memref<128xf16, #hivm.address_space<ub>> into vector<64xf16>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %wide = ave.hir.vextf %loaded, <part_even>, %mask
        : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %a0 = ave.hir.vload <NORM> %aux0[%iv]
        : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
    %a1 = ave.hir.vload <NORM> %aux1[%iv]
        : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
    %a2 = ave.hir.vload <NORM> %aux2[%iv]
        : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
    %a3 = ave.hir.vload <NORM> %aux3[%iv]
        : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
    %fused0 = ave.hir.vmula %wide, %a0, %a1, %mask
        : vector<64xf32>, vector<64xi1>
    %fused1 = ave.hir.vmula %fused0, %a2, %a3, %mask
        : vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %dst[%iv], %mask, %fused1
        : memref<128xf32, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf32>
  }
  return
}

// -----

// Vexp already dominates Vector execute. Adding vintlv would increase the
// bottleneck, so the widening load is not merged.
// CHECK-LABEL: func.func @reject_factor2_execute_bound
// CHECK: ave.hir.vload <NORM> {{.*}} into vector<64xf16>
// CHECK-NOT: ave.hir.vintlv
// TARGET-LABEL: func.func @reject_factor2_execute_bound
// TARGET: ave.hir.vload <NORM> {{.*}} into vector<64xf16>
// TARGET-NOT: ave.hir.vintlv
func.func @reject_factor2_execute_bound(
    %src: memref<128xf16, #hivm.address_space<ub>>,
    %dst: memref<128xf32, #hivm.address_space<ub>>)
    attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %iv = %c0 to %c128 step %c64 {
    %loaded = ave.hir.vload <NORM> %src[%iv]
        : memref<128xf16, #hivm.address_space<ub>> into vector<64xf16>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %wide = ave.hir.vextf %loaded, <part_even>, %mask
        : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %exp = ave.hir.vexp %wide, %mask : vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %dst[%iv], %mask, %exp
        : memref<128xf32, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf32>
  }
  return
}

// -----

// Two narrow elementwise ops cover one vdintlv tree and one merged store.
// CHECK-LABEL: func.func @accept_factor2_narrow_chain
// CHECK: ave.hir.vdintlv
// CHECK: ave.hir.vabs {{.*}} : vector<128xf16>, vector<128xi1>
// CHECK: ave.hir.masked_store <NORM_B16> {{.*}} vector<128xi1>, vector<128xf16>
// TARGET-LABEL: func.func @accept_factor2_narrow_chain
// TARGET: ave.hir.vdintlv
// TARGET: ave.hir.vabs {{.*}} : vector<128xf16>, vector<128xi1>
func.func @accept_factor2_narrow_chain(
    %src: memref<128xf32, #hivm.address_space<ub>>,
    %dst: memref<128xf16, #hivm.address_space<ub>>)
    attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %iv = %c0 to %c128 step %c64 {
    %loaded = ave.hir.vload <NORM> %src[%iv]
        : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %narrow = ave.hir.vtruncf %loaded, <rint>, false, <part_even>, %mask
        : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %abs0 = ave.hir.vabs %narrow, %mask : vector<64xf16>, vector<64xi1>
    %abs1 = ave.hir.vabs %abs0, %mask : vector<64xf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%iv], %mask, %abs1
        : memref<128xf16, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf16>
  }
  return
}

// -----

// Reduced from the layer_norm_gated regression. Wide calculation already
// dominates, so adding vdintlv for the final f16 store is unprofitable.
// CHECK-LABEL: func.func @reject_compute_bound_store_merge
// CHECK: ave.hir.vtruncf {{.*}} vector<64xf16>
// CHECK-NOT: ave.hir.vdintlv
// CHECK: ave.hir.masked_store <NORM_B16> {{.*}} vector<64xi1>, vector<64xf16>
// TARGET-LABEL: func.func @reject_compute_bound_store_merge
// TARGET: ave.hir.vtruncf {{.*}} vector<64xf16>
// TARGET-NOT: ave.hir.vdintlv
func.func @reject_compute_bound_store_merge(
    %src0: memref<128xf32, #hivm.address_space<ub>>,
    %src1: memref<128xf32, #hivm.address_space<ub>>,
    %src2: memref<128xf32, #hivm.address_space<ub>>,
    %dst: memref<128xf16, #hivm.address_space<ub>>)
    attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %iv = %c0 to %c128 step %c64 {
    %lhs = ave.hir.vload <NORM> %src0[%iv]
        : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
    %rhs = ave.hir.vload <NORM> %src1[%iv]
        : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
    %factor = ave.hir.vload <NORM> %src2[%iv]
        : memref<128xf32, #hivm.address_space<ub>> into vector<64xf32>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %div = ave.hir.vdiv %lhs, %rhs, %mask
        : vector<64xf32>, vector<64xi1>
    %sum = ave.hir.vadd %div, %lhs, %mask
        : vector<64xf32>, vector<64xi1>
    %product = ave.hir.vmul %sum, %factor, %mask
        : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %product, <rint>, false, <part_even>, %mask
        : vector<64xf32>, vector<64xf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%iv], %mask, %narrow
        : memref<128xf16, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf16>
  }
  return
}

// -----

// Both truncations lead to the same vsub/store tail. That tail is one merge
// group: a second narrow operand is packed by the first group's auxiliary
// vdintlv tree. Charging a group from each truncation would subtract the one
// store twice and make the cost model's IO count negative.
// CHECK-LABEL: func.func @reject_overlapping_narrow_roots
// CHECK: scf.for {{.*}} step %c64
// CHECK-COUNT-2: ave.hir.vtrunci
// CHECK-NOT: ave.hir.vdintlv
// CHECK-COUNT-1: ave.hir.vsub
// CHECK-COUNT-1: ave.hir.masked_store <NORM_B8> {{.*}} vector<64xi1>, vector<64xi8>
// CHECK-NOT: __ave_small_width
func.func @reject_overlapping_narrow_roots(
    %src0: memref<256xi32, #hivm.address_space<ub>>,
    %src1: memref<256xi32, #hivm.address_space<ub>>,
    %dst: memref<256xi8, #hivm.address_space<ub>>)
    attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %iv = %c0 to %c256 step %c64 {
    %lhs = ave.hir.vload <NORM> %src0[%iv]
        : memref<256xi32, #hivm.address_space<ub>> into vector<64xi32>
    %rhs = ave.hir.vload <NORM> %src1[%iv]
        : memref<256xi32, #hivm.address_space<ub>> into vector<64xi32>
    %mask = ave.hir.pge <ALL> : vector<64xi1>
    %narrowLhs = ave.hir.vtrunci %lhs, false, %mask
        {pp = #ave.vcvt_pp_type<pp0>}
        : vector<64xi32>, vector<64xi8>, vector<64xi1>
    %narrowRhs = ave.hir.vtrunci %rhs, false, %mask
        {pp = #ave.vcvt_pp_type<pp0>}
        : vector<64xi32>, vector<64xi8>, vector<64xi1>
    %difference = ave.hir.vsub %narrowLhs, %narrowRhs, %mask
        : vector<64xi8>, vector<64xi1>
    ave.hir.masked_store <NORM_B8> %dst[%iv], %mask, %difference
        : memref<256xi8, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xi8>
  }
  return
}
