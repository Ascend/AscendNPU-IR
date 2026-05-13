// RUN: bishengir-opt --arith-vector-mask-analyze --convert-vector-to-hivmave \
// RUN: --convert-arith-to-hivmave --annotation-lowering -split-input-file %s | FileCheck %s

func.func @arith_vector_mask_infer_maskedload_maskedstore_1mask(%base0: memref<64xf32>, %base1: memref<64xf32>, %pass_thru: vector<64xf32>, %dst: memref<64xf32>) {
  %mask0 = vector.constant_mask [64] : vector<64xi1>
  %0 = arith.constant 0 : index
  %1 = vector.maskedload %base0[%0], %mask0, %pass_thru : memref<64xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  %2 = vector.maskedload %base1[%0], %mask0, %pass_thru : memref<64xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  %3 = arith.negf %1 : vector<64xf32>
  %4 = math.exp %2 : vector<64xf32>
  %5 = math.round %4 : vector<64xf32>
  %6 = arith.addf %3, %5 : vector<64xf32>
  %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xf16>
  %8 = arith.extf %7 : vector<64xf16> to vector<64xf32>
  vector.maskedstore %dst[%0], %mask0, %8 : memref<64xf32>, vector<64xi1>, vector<64xf32>
  return
}

// CHECK-LABEL: func.func @arith_vector_mask_infer_maskedload_maskedstore_1mask
// CHECK:	        %[[CONSTANT:.+]] = arith.constant 0
// CHECK:       	%[[PGE:.+]] = ave.hir.pge <ALL> {mask_op_idx = 0 : i32} : vector<64xi1>
// CHECK:	        %[[LOAD0:.+]] = ave.hir.vload
// CHECK:	        %[[LOAD1:.+]] = ave.hir.vload
// CHECK:		%[[NEG:.+]] = ave.hir.vneg %[[LOAD0]], %[[PGE]]
// CHECK:		%[[EXP:.+]] = ave.hir.vexp %[[LOAD1]], %[[PGE]]
// CHECK:		%[[VTRC:.+]] = ave.hir.vtrc %[[EXP]], <round>, %[[PGE]]
// CHECK:		%[[ADD:.+]] = ave.hir.vadd %[[NEG]], %[[VTRC]], %[[PGE]]
// CHECK:		%[[TRUNCF:.+]] = ave.hir.vtruncf %[[ADD]], <rint>, false, <part_even>, %[[PGE]]
// CHECK:		%[[EXTF:.+]] = ave.hir.vextf %[[TRUNCF]], <part_even>, %[[PGE]]
// CHECK:               ave.hir.masked_store <NORM_B32> {{.*}}, %[[PGE]], %[[EXTF]]

func.func @arith_vector_mask_infer_maskedload_maskedstore_2mask(%base0: memref<64xf32>, %base1: memref<64xf32>, %pass_thru: vector<64xf32>, %dst0: memref<64xf32>, %dst1: memref<64xf32>) {
  %mask0 = vector.constant_mask [64] : vector<64xi1>
  %mask1 = vector.constant_mask [64] : vector<64xi1>
  %0 = arith.constant 0 : index
  %1 = vector.maskedload %base0[%0], %mask0, %pass_thru : memref<64xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  %2 = vector.maskedload %base1[%0], %mask1, %pass_thru : memref<64xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  %3 = arith.negf %1 : vector<64xf32>
  %4 = math.exp %2 : vector<64xf32>
  %5 = math.round %4 : vector<64xf32>
  %6 = arith.addf %3, %5 : vector<64xf32>
  %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xf16>
  %8 = arith.extf %7 : vector<64xf16> to vector<64xf32>
  vector.maskedstore %dst0[%0], %mask0, %8 : memref<64xf32>, vector<64xi1>, vector<64xf32>
  vector.maskedstore %dst1[%0], %mask1, %4 : memref<64xf32>, vector<64xi1>, vector<64xf32>
  return
}

// CHECK-LABEL: func.func @arith_vector_mask_infer_maskedload_maskedstore_2mask
// CHECK:               %[[CONSTANT:.+]] = arith.constant 0
// CHECK:               %[[PGE0:.+]] = ave.hir.pge <ALL> {mask_op_idx = 0 : i32} : vector<64xi1>
// CHECK:               %[[PGE1:.+]] = ave.hir.pge <ALL> {mask_op_idx = 1 : i32} : vector<64xi1>
// CHECK:               %[[LOAD0:.+]] = ave.hir.vload
// CHECK:               %[[LOAD1:.+]] = ave.hir.vload
// CHECK:               %[[SELECT:.+]] = ave.hir.vsel %[[PGE1]], %[[LOAD1]]
// CHECK:               %[[NEG:.+]] = ave.hir.vneg %[[LOAD0]], %[[PGE0]]
// CHECK:               %[[PGE2:.+]] = ave.hir.pge <ALL>
// CHECK:               %[[EXP:.+]] = ave.hir.vexp %[[SELECT]], %[[PGE2]]
// CHECK:               %[[VTRC:.+]] = ave.hir.vtrc %[[EXP]], <round>, %[[PGE0]]
// CHECK:               %[[ADD:.+]] = ave.hir.vadd %[[NEG]], %[[VTRC]], %[[PGE0]]
// CHECK:               %[[TRUNCF:.+]] = ave.hir.vtruncf %[[ADD]], <rint>, false, <part_even>, %[[PGE0]]
// CHECK:               %[[EXTF:.+]] = ave.hir.vextf %[[TRUNCF]], <part_even>, %[[PGE0]]
// CHECK:               ave.hir.masked_store <NORM_B32> {{.*}}, %[[PGE0]], %[[EXTF]]
// CHECK:               ave.hir.masked_store <NORM_B32> {{.*}}, %[[PGE1]], %[[EXP]]
