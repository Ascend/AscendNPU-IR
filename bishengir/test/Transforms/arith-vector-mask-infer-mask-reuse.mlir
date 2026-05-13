// RUN: bishengir-opt --cse --arith-vector-mask-analyze --convert-vector-to-hivmave  \
// RUN: --convert-arith-to-hivmave --annotation-lowering -split-input-file %s | FileCheck %s

//----- Case From #2778

func.func @triton_mul_4d_5d_outlined_vf_0(%arg0: memref<2x3x4x5x6xf32, #hivm.address_space<ub>>, %arg1: memref<2x3x4x5x6xf32, #hivm.address_space<ub>>, %arg2: memref<2x4x5x6xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  scf.for %arg3 = %c0 to %c2 step %c1 {
    scf.for %arg4 = %c0 to %c3 step %c1 {
      scf.for %arg5 = %c0 to %c4 step %c1 {
        scf.for %arg6 = %c0 to %c5 step %c1 {
          %subview = memref.subview %arg0[%arg3, %arg4, %arg5, %arg6, 0] [1, 1, 1, 1, 6] [1, 1, 1, 1, 1] : memref<2x3x4x5x6xf32, #hivm.address_space<ub>> to memref<1x1x1x1x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_0 = memref.subview %arg1[%arg3, %arg4, %arg5, %arg6, 0] [1, 1, 1, 1, 6] [1, 1, 1, 1, 1] : memref<2x3x4x5x6xf32, #hivm.address_space<ub>> to memref<1x1x1x1x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg2[%arg3, %arg5, %arg6, 0] [1, 1, 1, 6] [1, 1, 1, 1] : memref<2x4x5x6xf32, #hivm.address_space<ub>> to memref<1x1x1x6xf32, strided<[120, 30, 6, 1], offset: ?>, #hivm.address_space<ub>>
          %0 = vector.constant_mask [6] : vector<64xi1>
          %subview_2 = memref.subview %subview[0, 0, 0, 0, 0] [1, 1, 1, 1, 6] [1, 1, 1, 1, 1] : memref<1x1x1x1x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>, #hivm.address_space<ub>> to memref<6xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
          %1 = vector.transfer_read %subview_2[%c0], %cst, %0 {in_bounds = [true]} : memref<6xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
          %2 = vector.constant_mask [6] : vector<64xi1>
          %subview_3 = memref.subview %subview_0[0, 0, 0, 0, 0] [1, 1, 1, 1, 6] [1, 1, 1, 1, 1] : memref<1x1x1x1x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>, #hivm.address_space<ub>> to memref<6xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
          %3 = vector.transfer_read %subview_3[%c0], %cst, %2 {in_bounds = [true]} : memref<6xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
          %4 = vector.constant_mask [6] : vector<64xi1>
          %subview_4 = memref.subview %subview_1[0, 0, 0, 0] [1, 1, 1, 6] [1, 1, 1, 1] : memref<1x1x1x6xf32, strided<[120, 30, 6, 1], offset: ?>, #hivm.address_space<ub>> to memref<6xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview_4[%c0], %cst, %4 {in_bounds = [true]} : memref<6xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
          %6 = arith.mulf %1, %3 : vector<64xf32>
          %7 = arith.addf %5, %6 : vector<64xf32>
          %8 = vector.constant_mask [6] : vector<64xi1>
          %9 = arith.select %8, %7, %6 : vector<64xi1>, vector<64xf32>
          %10 = vector.constant_mask [6] : vector<64xi1>
          %subview_5 = memref.subview %subview_1[0, 0, 0, 0] [1, 1, 1, 6] [1, 1, 1, 1] : memref<1x1x1x6xf32, strided<[120, 30, 6, 1], offset: ?>, #hivm.address_space<ub>> to memref<6xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
          vector.transfer_write %9, %subview_5[%c0], %10 {in_bounds = [true]} : vector<64xf32>, memref<6xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        }
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @triton_mul_4d_5d_outlined_vf_0
// CHECK:      %[[CST:.+]] = arith.constant 6 : index
// CHECK:      %[[PLT:.+]], %[[TRUE_SHAPE:.+]] = ave.hir.plt %[[CST]]
// CHECK:      %[[LD0:.+]] = ave.hir.vload <NORM>
// CHECK:      %[[LD1:.+]] = ave.hir.vload <NORM>
// CHECK:      %[[LD2:.+]] = ave.hir.vload <NORM>
// CHECK:      %[[MUL:.+]] = ave.hir.vmul %[[LD0]], %[[LD1]], %[[PLT]]
// CHECK:      %[[ADD:.+]] = ave.hir.vadd %[[LD2]], %[[MUL]], %[[PLT]]
// CHECK:      %[[SEL:.+]] = ave.hir.vsel %[[PLT]], %[[ADD]], %[[MUL]]
// CHECK:      ave.hir.masked_store <NORM_B32>
