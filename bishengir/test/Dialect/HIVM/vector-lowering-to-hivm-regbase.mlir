// RUN: bishengir-opt %s -convert-arith-to-hivmave -convert-vector-to-hivmave -cse -split-input-file | FileCheck %s
// The test case have random failures in convert-arith-to-hivmave, in function findPredicateMapOp. issue2498

func.func @test_load_store_eltwiseop(%arg0: memref<1024xf32, #hivm.address_space<gm>>, %arg1: memref<1024xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
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
    %2 = vector.load %subview_0[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
    %3 = arith.negf %2 : vector<64xf32>
    %4 = math.exp %3 : vector<64xf32>
    %5 = arith.addf %4, %cst : vector<64xf32>
    %6 = arith.divf %cst, %5 : vector<64xf32>
    %7 = arith.mulf %2, %6 : vector<64xf32>
    vector.store %7, %subview[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
  }
  hivm.hir.store ins(%0 : memref<1024xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<1024xf32, #hivm.address_space<gm>>)
  return
}

// CHECK-LABEL:       func.func @test_load_store_eltwiseop
// CHECK:               %[[PGE:.+]] = ave.hir.pge <ALL>
// CHECK:               %[[CONSTANT0:.+]] = ave.hir.broadcast
// CHECK:               scf.for
// CHECK:                 %[[LOAD:.+]] = ave.hir.vload
// CHECK:                 %[[NEG:.+]] = ave.hir.vneg %[[LOAD]], %[[PGE]]
// CHECK:                 %[[EXP:.+]] = ave.hir.vexp %[[NEG]], %[[PGE]]
// CHECK:                 %[[ADD:.+]] =  ave.hir.vadd %[[EXP]], %[[CONSTANT0]], %[[PGE]]
// CHECK:                 %[[DIV:.+]] = ave.hir.vdiv %[[CONSTANT0]], %[[ADD]], %[[PGE]]
// CHECK:                 %[[MUL:.+]] = ave.hir.vmul %[[LOAD]], %[[DIV]], %[[PGE]]
// CHECK:                 ave.hir.masked_store <NORM_B32>

// -----
func.func @test_maskedload_maskedstore_createmask(%arg0: memref<1023xf32, #hivm.address_space<gm>>, %arg1: memref<1023xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  %c0_i64 = arith.constant 0 : i64
  %cst_0 = arith.constant dense<1.000000e+00> : vector<64xf32>
  %c64 = arith.constant 64 : index
  %c1023 = arith.constant 1023 : index
  %c0 = arith.constant 0 : index
  %c4096_i64 = arith.constant 4096 : i64
  %0 = hivm.hir.pointer_cast(%c4096_i64) : memref<1023xf32, #hivm.address_space<ub>>
  %1 = hivm.hir.pointer_cast(%c0_i64) : memref<1023xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<1023xf32, #hivm.address_space<gm>>) outs(%1 : memref<1023xf32, #hivm.address_space<ub>>)
  scf.for %arg2 = %c0 to %c1023 step %c64 {
    %2 = affine.min affine_map<(d0) -> (-d0 + 1023, 64)>(%arg2)
    %3 = vector.create_mask %2 : vector<64xi1>
    %subview = memref.subview %0[%arg2] [%2] [1] : memref<1023xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %1[%arg2] [%2] [1] : memref<1023xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %4 = vector.maskedload %subview_1[%c0], %3, %cst : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
    %5 = arith.negf %4 : vector<64xf32>
    %6 = math.exp %5 : vector<64xf32>
    %7 = arith.addf %6, %cst_0 : vector<64xf32>
    %8 = arith.divf %cst_0, %7 : vector<64xf32>
    %9 = arith.mulf %4, %8 : vector<64xf32>
    vector.maskedstore %subview[%c0], %3, %9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  hivm.hir.store ins(%0 : memref<1023xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<1023xf32, #hivm.address_space<gm>>)
  return
}

// CHECK-LABEL:       func.func @test_maskedload_maskedstore_createmask
// CHECK:               %[[PGE:.+]] = ave.hir.pge <ALL>
// CHECK:               %[[CONSTANT0:.+]] = ave.hir.broadcast
// CHECK:               %[[CONSTANT1:.+]] = ave.hir.broadcast
// CHECK:               scf.for
// CHECK:                  %[[PLT:.+]], %[[OUT:.+]] = ave.hir.plt
// CHECK:                  %[[LOAD:.+]] = ave.hir.vload
// CHECK:                  %[[SELECT:.+]] = ave.hir.vsel %[[PLT]], %[[LOAD]], %[[CONSTANT0]]
// CHECK:                  %[[NEG:.+]] = ave.hir.vneg %[[SELECT]], %[[PGE]]
// CHECK:                  %[[EXP:.+]] = ave.hir.vexp %[[NEG]], %[[PGE]]
// CHECK:                  %[[ADD:.+]] = ave.hir.vadd %[[EXP]], %[[CONSTANT1]], %[[PGE]]
// CHECK:                  %[[DIV:.+]] = ave.hir.vdiv %[[CONSTANT1]], %[[ADD]], %[[PGE]]
// CHECK:                  %[[MUL:.+]] = ave.hir.vmul %[[SELECT]], %[[DIV]], %[[PGE]]
// CHECK:                  ave.hir.masked_store <NORM_B32>
