// RUN: bishengir-opt %s -normalize-vector -convert-arith-to-hivmave  -split-input-file | FileCheck %s

module {
  func.func @test_arith_constant_regbase_broadcast(%arg0: memref<1024xf32, #hivm.address_space<gm>>, %arg1: memref<1024xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c4096_i64 = arith.constant 4096 : i64
    %0 = hivm.hir.pointer_cast(%c4096_i64) : memref<1024xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<1024xf32, #hivm.address_space<gm>>) outs(%1 : memref<1024xf32, #hivm.address_space<ub>>)
    scf.for %arg2 = %c0 to %c1024 step %c64 {
      %subview = memref.subview %0[%arg2] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %1[%arg2] [64] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %3 = ave.hir.pge <ALL> : vector<64xi1>
      %4 = ave.hir.vneg %2, %3 : vector<64xf32>, vector<64xi1>
      %5 = ave.hir.pge <ALL> : vector<64xi1>
      %6 = ave.hir.vexp %4, %5 : vector<64xf32>, vector<64xi1>
      %7 = ave.hir.pge <ALL> : vector<64xi1>
      %8 = ave.hir.vadd %6, %cst, %7 : vector<64xf32>, vector<64xi1>
      %9 = ave.hir.pge <ALL> : vector<64xi1>
      %10 = ave.hir.vdiv %cst, %8, %9 : vector<64xf32>, vector<64xi1>
      %11 = ave.hir.pge <ALL> : vector<64xi1>
      %12 = ave.hir.vmul %2, %10, %11 : vector<64xf32>, vector<64xi1>
      %13 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview[%c0], %13, %12 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    hivm.hir.store ins(%0 : memref<1024xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<1024xf32, #hivm.address_space<gm>>)
    return
  }
}

// CHECK-LABEL:       func.func @test_arith_constant_regbase_broadcast
// CHECK:               %[[CONSTANT:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:               %[[PGE1:.+]] = ave.hir.pge <ALL>
// CHECK:               %[[BROADCAST:.+]] = ave.hir.broadcast %[[CONSTANT]], %[[PGE1]]
// CHECK:               %[[EXP:.+]] = ave.hir.vexp
// CHECK:               %[[PGE0:.+]] = ave.hir.pge <ALL>
// CHECK:               %[[ADD:.+]] = ave.hir.vadd %[[EXP]], %[[BROADCAST]], %[[PGE0]]
// CHECK:               %[[PGE2:.+]] = ave.hir.pge <ALL>
// CHECK:               %[[DIV:.+]] = ave.hir.vdiv %[[BROADCAST]], %[[ADD]], %[[PGE2]]

// -----


func.func @test_regbase_broadcast_different_constant(%arg0: memref<128x64xf32, #hivm.address_space<gm>>, %arg1: memref<64xf32, #hivm.address_space<gm>>, %arg2: memref<128x64xf32, #hivm.address_space<gm>>, %arg3: memref<128xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<1.562500e-02> : vector<64xf32>
  %cst_0 = arith.constant dense<1.000000e-01> : vector<64xf32>
  %cst_1 = arith.constant dense<1.000000e+00> : vector<64xf32>
  %c33024_i64 = arith.constant 33024 : i64
  %0 = hivm.hir.pointer_cast(%c33024_i64) : memref<128xf32, #hivm.address_space<ub>>
  %1 = hivm.hir.pointer_cast(%c33024_i64) : memref<128x1xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %1 [[0, 1]] : memref<128x1xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  scf.for %arg4 = %c0 to %c128 step %c64 {
    %subview = memref.subview %0[%arg4] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %collapse_shape[%arg4] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %2 = ave.hir.vload <NORM> %subview_2[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %3 = ave.hir.pge <ALL> : vector<64xi1>
    %4 = ave.hir.vmul %2, %cst, %3 : vector<64xf32>, vector<64xi1>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.vadd %4, %cst_0, %5 : vector<64xf32>, vector<64xi1>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vsqrt %6, %7 : vector<64xf32>, vector<64xi1>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    %10 = ave.hir.vdiv %cst_1, %8, %9 : vector<64xf32>, vector<64xi1>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview[%c0], %11, %10 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  return
}

// CHECK-LABEL:       func.func @test_regbase_broadcast_different_constant
// CHECK:               %[[CONST1:.+]] = arith.constant 1.562500e-02 : f32
// CHECK:                 %[[PGE1:.+]] = ave.hir.pge <ALL>
// CHECK:                 %[[BRC1:.+]] = ave.hir.broadcast %[[CONST1]], %[[PGE1]]
// CHECK:               %[[CONST0:.+]] = arith.constant 1.000000e-01 : f32
// CHECK:                 %[[PGE3:.+]] = ave.hir.pge <ALL>
// CHECK:                 %[[BRC2:.+]] = ave.hir.broadcast %[[CONST0]], %[[PGE3]]
// CHECK:               %[[CONST:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:                 %[[PGE6:.+]] = ave.hir.pge <ALL>
// CHECK:                 %[[BRC3:.+]] = ave.hir.broadcast %[[CONST]], %[[PGE6]]
// CHECK:               scf.for
// CHECK:                 %[[LOAD:.+]] = ave.hir.vload
// CHECK:                 %[[PGE0:.+]] = ave.hir.pge <ALL>
// CHECK:                 %[[MUL:.+]] = ave.hir.vmul %[[LOAD]], %[[BRC1]], %[[PGE0]]
// CHECK:                 %[[PGE2:.+]] = ave.hir.pge <ALL>
// CHECK:                 %[[ADD:.+]] = ave.hir.vadd %[[MUL]], %[[BRC2]], %[[PGE2]]
// CHECK:                 %[[PGE4:.+]] = ave.hir.pge <ALL>
// CHECK:                 %[[SQRT:.+]] = ave.hir.vsqrt %[[ADD]], %[[PGE4]]
// CHECK:                 %[[PGE5:.+]] = ave.hir.pge <ALL>
// CHECK:                 %[[DIV:.+]] = ave.hir.vdiv %[[BRC3]], %[[SQRT]], %[[PGE5]]
