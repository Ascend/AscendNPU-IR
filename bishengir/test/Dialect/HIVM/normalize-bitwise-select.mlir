// RUN: bishengir-opt --hivm-normalize-bitwise-select %s -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_i8_bitwise_mask(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<16xi8>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<16xf32>,
// CHECK-SAME:                                    %[[VAL_2:.*]]: tensor<16xf32>) -> tensor<16xf32> {
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<16xf32>
// CHECK:           %[[VAL_4:.*]] = hivm.hir.vsel ins(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : tensor<16xi8>, tensor<16xf32>, tensor<16xf32>) outs(%[[VAL_3]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK:           return %[[VAL_4]] : tensor<16xf32>
// CHECK:         }
func.func @test_i8_bitwise_mask(%arg0: tensor<16xi8>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>) -> tensor<16xf32> {
  %zero = arith.constant dense<0.> : tensor<16xf16>
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hivm.hir.vcast ins(%arg0: tensor<16xi8>) outs(%0: tensor<16xf16>) -> tensor<16xf16>
  %2 = tensor.empty() : tensor<16xi1>
  %3 = hivm.hir.vcmp ins(%1, %zero: tensor<16xf16>, tensor<16xf16>) outs(%2: tensor<16xi1>) -> tensor<16xi1>
  %4 = tensor.empty() : tensor<16xi1>
  %5 = hivm.hir.vnot ins(%3: tensor<16xi1>) outs(%3: tensor<16xi1>) -> tensor<16xi1>
  %6 = tensor.empty() : tensor<16xf32>
  %7 = hivm.hir.vsel ins(%5, %arg1, %arg2: tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs(%6: tensor<16xf32>) -> tensor<16xf32>
  annotation.mark %7 {bitwise_mask} : tensor<16xf32>
  return %7: tensor<16xf32>
}
