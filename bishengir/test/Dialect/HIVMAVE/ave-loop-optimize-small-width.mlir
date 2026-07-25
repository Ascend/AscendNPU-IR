// RUN: bishengir-opt -ave-loop-optimize="max-small-width-merge-factor=4" %s --split-input-file | FileCheck %s
// RUN: bishengir-opt -ave-loop-optimize="max-small-width-merge-factor=2" %s --split-input-file | FileCheck %s --check-prefix=MAX2
// RUN: bishengir-opt -ave-process-vsstb -ave-loop-optimize="max-small-width-merge-factor=4" %s --split-input-file | FileCheck %s --check-prefix=PIPE
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 -ave-process-vsstb -ave-loop-optimize="max-small-width-merge-factor=4" -analyze-vector-layout -canonicalize -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin %s --split-input-file | FileCheck %s --check-prefix=LOWER

// CHECK-LABEL: func.func @factor2_full_chain
// CHECK: scf.for {{.*}} step %[[STEP:.*]]
// CHECK: %[[LOAD:.*]] = ave.hir.vload <NORM> {{.*}} into vector<128xf16>
// CHECK-NEXT: %[[GROUP0:.*]], %[[GROUP1:.*]] = ave.hir.vintlv %[[LOAD]], %[[LOAD]] {layout_change = #ave<layout_change SPARSE>} : vector<128xf16>, vector<64xf16>
// CHECK: %[[WIDE0:.*]] = ave.hir.vextf %[[GROUP0]], <part_even>
// CHECK: %[[WIDE_ADD0:.*]] = ave.hir.vadds %[[WIDE0]], %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<64xi1>
// CHECK: %[[NARROW0:.*]] = ave.hir.vtruncf %[[WIDE_ADD0]]
// CHECK: %[[WIDE1:.*]] = ave.hir.vextf %[[GROUP1]], <part_even>
// CHECK: %[[WIDE_ADD1:.*]] = ave.hir.vadds %[[WIDE1]], %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<64xi1>
// CHECK: %[[NARROW1:.*]] = ave.hir.vtruncf %[[WIDE_ADD1]]
// CHECK: %[[PACKED:.*]], %{{.*}} = ave.hir.vdintlv %[[NARROW0]], %[[NARROW1]] {layout_change = #ave<layout_change DENSE>} : vector<64xf16>, vector<128xf16>
// CHECK: %[[NARROW_ADD:.*]] = ave.hir.vadds %[[PACKED]], %{{.*}}, %{{.*}} : vector<128xf16>, f16, vector<128xi1>
// CHECK: %[[STORE_VIEW:.*]] = memref.reinterpret_cast {{.*}} to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK: ave.hir.masked_store <NORM_B16> %[[STORE_VIEW]][{{.*}}], %{{.*}}, %[[NARROW_ADD]] : {{.*}}, vector<128xi1>, vector<128xf16>
// CHECK-NOT: __ave_small_width
// MAX2-LABEL: func.func @factor2_full_chain
// MAX2: ave.hir.vload <NORM> {{.*}} into vector<128xf16>
// MAX2: ave.hir.vdintlv {{.*}} : vector<64xf16>, vector<128xf16>
// LOWER-LABEL: func.func @factor2_full_chain
// LOWER: hivm_regbaseintrins.intr.hivm.vldsx1.v128f16
// LOWER: hivm_regbaseintrins.intr.hivm.vintlv
// LOWER: hivm_regbaseintrins.intr.hivm.vdintlv
// LOWER: hivm_regbaseintrins.intr.hivm.vadds
// LOWER: hivm_regbaseintrins.intr.hivm.vstsx1.v128f16
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
    %dst: memref<256xi32, #hivm.address_space<ub>>)
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
  }
  return
}

// -----

// CHECK-LABEL: func.func @factor4_narrow_i32_to_i8
// CHECK-COUNT-4: ave.hir.vtrunci
// CHECK-COUNT-3: ave.hir.vdintlv
// CHECK-COUNT-1: ave.hir.vadds {{.*}} : vector<256xi8>, i8, vector<256xi1>
// CHECK-COUNT-1: ave.hir.masked_store <NORM_B8> {{.*}} vector<256xi1>, vector<256xi8>
// CHECK-NOT: __ave_small_width
// MAX2-LABEL: func.func @factor4_narrow_i32_to_i8
// MAX2: scf.for
// MAX2-NOT: ave.hir.vdintlv
// MAX2: ave.hir.masked_store <NORM_B8> {{.*}} vector<64xi1>, vector<64xi8>
// LOWER-LABEL: func.func @factor4_narrow_i32_to_i8
// LOWER-COUNT-4: hivm_regbaseintrins.intr.hivm.vcvtii.s322u8
// LOWER-COUNT-3: hivm_regbaseintrins.intr.hivm.vdintlv
// LOWER: hivm_regbaseintrins.intr.hivm.vadds
// LOWER: hivm_regbaseintrins.intr.hivm.vstsx1.v256s8
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

// One factor-4 group owns unrolling; factor-2 groups merge in pairs.
// CHECK-LABEL: func.func @mixed_widening_types
// CHECK-DAG: ave.hir.vload <NORM> {{.*}} into vector<128xbf16>
// CHECK-DAG: ave.hir.vload <NORM> {{.*}} into vector<256xi8>
// CHECK-DAG: ave.hir.vload <NORM> {{.*}} into vector<128xi16>
// CHECK-DAG: ave.hir.vload <NORM> {{.*}} into vector<256xf8E5M2>
// CHECK-DAG: ave.hir.vintlv {{.*}} : vector<128xbf16>, vector<64xbf16>
// CHECK-DAG: ave.hir.vintlv {{.*}} : vector<256xi8>, vector<128xi8>
// CHECK-DAG: ave.hir.vintlv {{.*}} : vector<128xi16>, vector<64xi16>
// CHECK-DAG: ave.hir.vintlv {{.*}} : vector<128xf8E5M2>, vector<64xf8E5M2>
// CHECK-NOT: __ave_small_width
func.func @mixed_widening_types(
    %bf16Src: memref<256xbf16, #hivm.address_space<ub>>,
    %u8Src: memref<512xi8, #hivm.address_space<ub>>,
    %i16Src: memref<256xi16, #hivm.address_space<ub>>,
    %f8Src: memref<256xf8E5M2, #hivm.address_space<ub>>,
    %f32Dst0: memref<256xf32, #hivm.address_space<ub>>,
    %i16Dst: memref<512xi16, #hivm.address_space<ub>>,
    %i32Dst: memref<256xi32, #hivm.address_space<ub>>,
    %f32Dst1: memref<256xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %iv = %c0 to %c256 step %c64 {
    %offset128 = affine.apply affine_map<(d0) -> (d0 * 2)>(%iv)
    %mask64 = ave.hir.pge <ALL> : vector<64xi1>
    %mask128 = ave.hir.pge <ALL> : vector<128xi1>
    %bf16 = ave.hir.vload <NORM> %bf16Src[%iv]
        : memref<256xbf16, #hivm.address_space<ub>> into vector<64xbf16>
    %f32FromBf16 = ave.hir.vextf %bf16, <part_even>, %mask64
        : vector<64xbf16>, vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %f32Dst0[%iv], %mask64, %f32FromBf16
        : memref<256xf32, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf32>
    %u8 = ave.hir.vload <NORM> %u8Src[%offset128]
        : memref<512xi8, #hivm.address_space<ub>> into vector<128xi8>
    %i16 = ave.hir.vextui %u8, %mask128
        {part = #ave.vcvt_part_type<part_even>}
        : vector<128xi8>, vector<128xi16>, vector<128xi1>
    ave.hir.masked_store <NORM_B16> %i16Dst[%offset128], %mask128, %i16
        : memref<512xi16, #hivm.address_space<ub>>,
          vector<128xi1>, vector<128xi16>
    %i16Input = ave.hir.vload <NORM> %i16Src[%iv]
        : memref<256xi16, #hivm.address_space<ub>> into vector<64xi16>
    %i32 = ave.hir.vextsi %i16Input, %mask64
        {part = #ave.vcvt_part_type<part_even>}
        : vector<64xi16>, vector<64xi32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %i32Dst[%iv], %mask64, %i32
        : memref<256xi32, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xi32>
    %f8 = ave.hir.vload <NORM> %f8Src[%iv]
        : memref<256xf8E5M2, #hivm.address_space<ub>>
          into vector<64xf8E5M2>
    %f32FromF8 = ave.hir.vextf %f8, <part_even>, %mask64
        : vector<64xf8E5M2>, vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %f32Dst1[%iv], %mask64, %f32FromF8
        : memref<256xf32, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xf32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @mixed_narrowing_types
// CHECK-DAG: ave.hir.vdintlv {{.*}} : vector<64xbf16>, vector<128xbf16>
// CHECK-DAG: ave.hir.vdintlv {{.*}} : vector<128xi8>, vector<256xi8>
// CHECK-DAG: ave.hir.vdintlv {{.*}} : vector<64xi16>, vector<128xi16>
// CHECK-DAG: ave.hir.vdintlv {{.*}} : vector<128xf8E4M3FN>, vector<256xf8E4M3FN>
// CHECK-DAG: ave.hir.vadds {{.*}} : vector<256xi8>, i8, vector<256xi1>
// CHECK-DAG: ave.hir.masked_store <NORM_B16> {{.*}} vector<128xi1>, vector<128xbf16>
// CHECK-DAG: ave.hir.masked_store <NORM_B8> {{.*}} vector<256xi1>, vector<256xi8>
// CHECK-DAG: ave.hir.masked_store <NORM_B16> {{.*}} vector<128xi1>, vector<128xi16>
// CHECK-DAG: ave.hir.masked_store <NORM_B8> {{.*}} vector<256xi1>, vector<256xf8E4M3FN>
// CHECK-NOT: __ave_small_width
func.func @mixed_narrowing_types(
    %f32Src0: memref<256xf32, #hivm.address_space<ub>>,
    %i16Src: memref<512xi16, #hivm.address_space<ub>>,
    %i32Src: memref<256xi32, #hivm.address_space<ub>>,
    %f32Src1: memref<256xf32, #hivm.address_space<ub>>,
    %bf16Dst: memref<256xbf16, #hivm.address_space<ub>>,
    %i8Dst: memref<512xi8, #hivm.address_space<ub>>,
    %i16Dst: memref<256xi16, #hivm.address_space<ub>>,
    %f8Dst: memref<256xf8E4M3FN, #hivm.address_space<ub>>,
    %bias: i8)
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
    ave.hir.masked_store <NORM_B16> %bf16Dst[%iv], %mask64, %bf16
        : memref<256xbf16, #hivm.address_space<ub>>,
          vector<64xi1>, vector<64xbf16>
    %i16Input = ave.hir.vload <NORM> %i16Src[%offset128]
        : memref<512xi16, #hivm.address_space<ub>> into vector<128xi16>
    %i8 = ave.hir.vtrunci %i16Input, false, %mask128
        {part = #ave.vcvt_part_type<part_even>}
        : vector<128xi16>, vector<128xi8>, vector<128xi1>
    %biased = ave.hir.vadds %i8, %bias, %mask128
        : vector<128xi8>, i8, vector<128xi1>
    ave.hir.masked_store <NORM_B8> %i8Dst[%offset128], %mask128, %biased
        : memref<512xi8, #hivm.address_space<ub>>,
          vector<128xi1>, vector<128xi8>
    %i32Input = ave.hir.vload <NORM> %i32Src[%iv]
        : memref<256xi32, #hivm.address_space<ub>> into vector<64xi32>
    %i16 = ave.hir.vtrunci %i32Input, false, %mask64
        {part = #ave.vcvt_part_type<part_even>}
        : vector<64xi32>, vector<64xi16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %i16Dst[%iv], %mask64, %i16
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
// CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata
// CHECK: %[[PACKED:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[OFFSET]]], sizes: [128], strides: [1] : {{.*}} to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK: ave.hir.vload <NORM> %[[PACKED]][{{.*}}] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xf16>
// CHECK: ave.hir.vintlv
// CHECK-COUNT-2: ave.hir.store_with_stride {{.*}} vector<64xi1>, vector<64xf16>
// CHECK-NOT: ave.hir.store_with_stride {{.*}} vector<128xf16>
// PIPE-LABEL: func.func @reject_unmarked_strided_column_store
// PIPE: ave.hir.vload <NORM> {{.*}} into vector<128xf16>
// PIPE-COUNT-2: ave.hir.store_with_stride {{.*}} vector<128xi1>, vector<64xf16>
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
