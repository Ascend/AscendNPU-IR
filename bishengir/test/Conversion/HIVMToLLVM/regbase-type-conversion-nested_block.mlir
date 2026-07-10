// RUN: bishengir-opt %s -ave-normalize-ops -convert-hivmave-to-ave-intrin -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

func.func @test_store_in_nested_for(%arg0: vector<64xf32>, %arg1: memref<128xf16, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[VAL:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"
  // CHECK: %[[CASTED_VAL_1:.*]] = builtin.unrealized_conversion_cast %[[VAL]]
  %1 = ave.hir.vtruncf %arg0, <ceil>, true, <part_even>, %0  : vector<64xf32>, vector<64xf16>, vector<64xi1>
  scf.for %arg2 = %c0 to %c128 step %c64 {
    // CHECK: %[[MASK:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"
    // CHECK: %[[CASTED_MASK_1:.*]] = builtin.unrealized_conversion_cast %[[MASK]]
    // CHECK: %[[CASTED_VAL_2:.*]] = builtin.unrealized_conversion_cast %[[CASTED_VAL_1]]
    // CHECK: builtin.unrealized_conversion_cast %[[CASTED_MASK_1]]
    // CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v128f16"
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B16> %arg1[%arg2], %2, %1  : memref<128xf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  }
  return
}
