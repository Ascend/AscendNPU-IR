// REQUIRES: regbase
// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy register-based=true" 2>&1 | FileCheck %s

// -----


// CHECK-LABEL: func.func private @where_mask_no_collapse(
// CHECK-NOT: into tensor<32xi1>
// CHECK: hfusion.select ins({{.*}} : tensor<8x4xi1>, f16, f16)
func.func private @where_mask_no_collapse(
    %mask: tensor<8x4xi1>, %f0: tensor<8x4xf16>,
    %f1: tensor<8x4x2xf16>, %m1: tensor<8x4x2xi1>,
    %c2: tensor<2xi1>, %f2: tensor<2xf16>,
    %f3: tensor<8x4x2x3xf16>, %m3: tensor<8x4x2x3xi1>,
    %c3: tensor<3xi1>, %f4: tensor<3xf16>,
    %out: tensor<8x4x2x3xi8>) -> tensor<8x4x2x3xi8> {
  %cst = arith.constant -1.000000e+00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = hfusion.select ins(%mask, %cst, %cst_0 : tensor<8x4xi1>, f16, f16) outs(%f0 : tensor<8x4xf16>) -> tensor<8x4xf16>
  %broadcasted = linalg.broadcast ins(%0 : tensor<8x4xf16>) outs(%f1 : tensor<8x4x2xf16>) dimensions = [2]
  %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted, %cst_0 : tensor<8x4x2xf16>, f16) outs(%m1 : tensor<8x4x2xi1>) -> tensor<8x4x2xi1>
  %2 = hfusion.select ins(%c2, %cst, %cst_0 : tensor<2xi1>, f16, f16) outs(%f2 : tensor<2xf16>) -> tensor<2xf16>
  %broadcasted_1 = linalg.broadcast ins(%2 : tensor<2xf16>) outs(%f1 : tensor<8x4x2xf16>) dimensions = [0, 1]
  %3 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_1, %cst_0 : tensor<8x4x2xf16>, f16) outs(%m1 : tensor<8x4x2xi1>) -> tensor<8x4x2xi1>
  %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%1, %3 : tensor<8x4x2xi1>, tensor<8x4x2xi1>) outs(%m1 : tensor<8x4x2xi1>) -> tensor<8x4x2xi1>
  %5 = hfusion.select ins(%4, %cst, %cst_0 : tensor<8x4x2xi1>, f16, f16) outs(%f1 : tensor<8x4x2xf16>) -> tensor<8x4x2xf16>
  %broadcasted_2 = linalg.broadcast ins(%5 : tensor<8x4x2xf16>) outs(%f3 : tensor<8x4x2x3xf16>) dimensions = [3]
  %6 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_2, %cst_0 : tensor<8x4x2x3xf16>, f16) outs(%m3 : tensor<8x4x2x3xi1>) -> tensor<8x4x2x3xi1>
  %7 = hfusion.select ins(%c3, %cst, %cst_0 : tensor<3xi1>, f16, f16) outs(%f4 : tensor<3xf16>) -> tensor<3xf16>
  %broadcasted_3 = linalg.broadcast ins(%7 : tensor<3xf16>) outs(%f3 : tensor<8x4x2x3xf16>) dimensions = [0, 1, 2]
  %8 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_3, %cst_0 : tensor<8x4x2x3xf16>, f16) outs(%m3 : tensor<8x4x2x3xi1>) -> tensor<8x4x2x3xi1>
  %9 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%6, %8 : tensor<8x4x2x3xi1>, tensor<8x4x2x3xi1>) outs(%m3 : tensor<8x4x2x3xi1>) -> tensor<8x4x2x3xi1>
  %10 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<8x4x2x3xi1>) outs(%out : tensor<8x4x2x3xi8>) -> tensor<8x4x2x3xi8>
  return %10 : tensor<8x4x2x3xi8>
}


// -----


// CHECK-LABEL: func.func private @where_mask_byte_no_collapse(
// CHECK-NOT: into tensor<64xi1>
// CHECK: hfusion.select ins({{.*}} : tensor<8x8xi1>, f16, f16)
func.func private @where_mask_byte_no_collapse(
    %mask: tensor<8x8xi1>, %f0: tensor<8x8xf16>,
    %f1: tensor<8x8x2xf16>, %m1: tensor<8x8x2xi1>,
    %c2: tensor<2xi1>, %f2: tensor<2xf16>,
    %f3: tensor<8x8x2x3xf16>, %m3: tensor<8x8x2x3xi1>,
    %c3: tensor<3xi1>, %f4: tensor<3xf16>,
    %out: tensor<8x8x2x3xi8>) -> tensor<8x8x2x3xi8> {
  %cst = arith.constant -1.000000e+00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = hfusion.select ins(%mask, %cst, %cst_0 : tensor<8x8xi1>, f16, f16) outs(%f0 : tensor<8x8xf16>) -> tensor<8x8xf16>
  %broadcasted = linalg.broadcast ins(%0 : tensor<8x8xf16>) outs(%f1 : tensor<8x8x2xf16>) dimensions = [2]
  %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted, %cst_0 : tensor<8x8x2xf16>, f16) outs(%m1 : tensor<8x8x2xi1>) -> tensor<8x8x2xi1>
  %2 = hfusion.select ins(%c2, %cst, %cst_0 : tensor<2xi1>, f16, f16) outs(%f2 : tensor<2xf16>) -> tensor<2xf16>
  %broadcasted_1 = linalg.broadcast ins(%2 : tensor<2xf16>) outs(%f1 : tensor<8x8x2xf16>) dimensions = [0, 1]
  %3 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_1, %cst_0 : tensor<8x8x2xf16>, f16) outs(%m1 : tensor<8x8x2xi1>) -> tensor<8x8x2xi1>
  %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%1, %3 : tensor<8x8x2xi1>, tensor<8x8x2xi1>) outs(%m1 : tensor<8x8x2xi1>) -> tensor<8x8x2xi1>
  %5 = hfusion.select ins(%4, %cst, %cst_0 : tensor<8x8x2xi1>, f16, f16) outs(%f1 : tensor<8x8x2xf16>) -> tensor<8x8x2xf16>
  %broadcasted_2 = linalg.broadcast ins(%5 : tensor<8x8x2xf16>) outs(%f3 : tensor<8x8x2x3xf16>) dimensions = [3]
  %6 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_2, %cst_0 : tensor<8x8x2x3xf16>, f16) outs(%m3 : tensor<8x8x2x3xi1>) -> tensor<8x8x2x3xi1>
  %7 = hfusion.select ins(%c3, %cst, %cst_0 : tensor<3xi1>, f16, f16) outs(%f4 : tensor<3xf16>) -> tensor<3xf16>
  %broadcasted_3 = linalg.broadcast ins(%7 : tensor<3xf16>) outs(%f3 : tensor<8x8x2x3xf16>) dimensions = [0, 1, 2]
  %8 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_3, %cst_0 : tensor<8x8x2x3xf16>, f16) outs(%m3 : tensor<8x8x2x3xi1>) -> tensor<8x8x2x3xi1>
  %9 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%6, %8 : tensor<8x8x2x3xi1>, tensor<8x8x2x3xi1>) outs(%m3 : tensor<8x8x2x3xi1>) -> tensor<8x8x2x3xi1>
  %10 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<8x8x2x3xi1>) outs(%out : tensor<8x8x2x3xi8>) -> tensor<8x8x2x3xi8>
  return %10 : tensor<8x8x2x3xi8>
}
