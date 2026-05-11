// RUN: bishengir-opt --arith-vector-mask-analyze --convert-vector-to-hivmave \
// RUN: --convert-arith-to-hivmave --annotation-lowering -split-input-file %s | FileCheck %s

func.func @arith_vector_mask_infer_transferread_transferwriter_1mask(%arg0: memref<8xf16>, %arg1: memref<8xf16>, %arg2: memref<8xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [8] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<8xf16>, vector<128xf16>
  %2 = vector.transfer_read %arg1[%c0], %cst, %0 {in_bounds = [true]} : memref<8xf16>, vector<128xf16>
  %3 = math.absf %1 : vector<128xf16>
  %4 = math.log %3 : vector<128xf16>
  %5 = arith.mulf %3, %4 : vector<128xf16>
  %6 = math.sqrt %2 : vector<128xf16>
  %7 = math.rsqrt %6 : vector<128xf16>
  %8 = arith.divf %6, %7 : vector<128xf16>
  %9 = arith.subf %5, %8 : vector<128xf16>
  vector.transfer_write %9, %arg2[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<8xf16>
  return
}

// CHECK-LABEL: func.func @arith_vector_mask_infer_transferread_transferwriter_1mask
// CHECK:               %[[CONSTANT:.+]] = arith.constant 0.000000e+00
// CHECK:               %[[IDX:.+]] = arith.constant 0
// CHECK:		%[[PGE:.+]] = ave.hir.pge <VL8> {mask_op_idx = 0 : i32} : vector<128xi1>
// CHECK:               %[[LOAD0:.+]] = ave.hir.vload
// CHECK:               %[[LOAD1:.+]] = ave.hir.vload
// CHECK:               %[[ABS:.+]] = ave.hir.vabs %[[LOAD0]], %[[PGE]]
// CHECK:               %[[LN:.+]] = ave.hir.vln %[[ABS]], %[[PGE]]
// CHECK:               %[[MUL:.+]] = ave.hir.vmul %[[ABS]], %[[LN]], %[[PGE]]
// CHECK:               %[[SQRT:.+]] = ave.hir.vsqrt %[[LOAD1]], %[[PGE]]
// CHECK:               %[[RSQRT:.+]] = ave.hir.vrsqrt %[[SQRT]], %[[PGE]]
// CHECK:               %[[DIV:.+]] = ave.hir.vdiv %[[SQRT]], %[[RSQRT]], %[[PGE]]
// CHECK:               %[[SUB:.+]] = ave.hir.vsub %[[MUL]], %[[DIV]], %[[PGE]]
// CHECK:               ave.hir.masked_store <NORM_B16>

func.func @arith_vector_mask_infer_transferread_transferwriter_2mask(%arg0: memref<8xf16>, %arg1: memref<8xf16>, %arg2: memref<8xf16>, %arg3: memref<8xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %mask0 = vector.constant_mask [8] : vector<128xi1>
  %mask1 = vector.constant_mask [8] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %mask0 {in_bounds = [true]} : memref<8xf16>, vector<128xf16>
  %2 = vector.transfer_read %arg1[%c0], %cst, %mask1 {in_bounds = [true]} : memref<8xf16>, vector<128xf16>
  %3 = math.absf %1 : vector<128xf16>
  %4 = math.log %3 : vector<128xf16>
  %5 = arith.mulf %3, %4 : vector<128xf16>
  %6 = math.sqrt %2 : vector<128xf16>
  %7 = math.rsqrt %6 : vector<128xf16>
  %8 = arith.divf %6, %7 : vector<128xf16>
  %9 = arith.subf %5, %8 : vector<128xf16>
  vector.transfer_write %9, %arg2[%c0], %mask0 {in_bounds = [true]} : vector<128xf16>, memref<8xf16>
  vector.transfer_write %7, %arg3[%c0], %mask1 {in_bounds = [true]} : vector<128xf16>, memref<8xf16>
  return
}

// CHECK-LABEL: func.func @arith_vector_mask_infer_transferread_transferwriter_2mask
// CHECK:               %[[CONSTANT:.+]] = arith.constant 0.000000e+00
// CHECK:               %[[IDX:.+]] = arith.constant 0
// CHECK:		%[[PGE0:.+]] = ave.hir.pge <VL8> {mask_op_idx = 0 : i32} : vector<128xi1>
// CHECK:		%[[PGE1:.+]] = ave.hir.pge <VL8> {mask_op_idx = 1 : i32} : vector<128xi1>
// CHECK:               %[[LOAD0:.+]] = ave.hir.vload
// CHECK:               %[[LOAD1:.+]] = ave.hir.vload
// CHECK:               %[[ABS:.+]] = ave.hir.vabs %[[LOAD0]], %[[PGE0]]
// CHECK:               %[[LN:.+]] = ave.hir.vln %[[ABS]], %[[PGE0]]
// CHECK:               %[[MUL:.+]] = ave.hir.vmul %[[ABS]], %[[LN]], %[[PGE0]]
// CHECK:		%[[PGE2:.+]] = ave.hir.pge <ALL>
// CHECK:               %[[SQRT:.+]] = ave.hir.vsqrt %[[LOAD1]], %[[PGE2]]
// CHECK:		%[[PGE3:.+]] = ave.hir.pge <ALL>
// CHECK:               %[[RSQRT:.+]] = ave.hir.vrsqrt %[[SQRT]], %[[PGE3]]
// CHECK:               %[[DIV:.+]] = ave.hir.vdiv %[[SQRT]], %[[RSQRT]], %[[PGE0]]
// CHECK:               %[[SUB:.+]] = ave.hir.vsub %[[MUL]], %[[DIV]], %[[PGE0]]
// CHECK:               ave.hir.masked_store <NORM_B16>
// CHECK:               ave.hir.masked_store <NORM_B16>
