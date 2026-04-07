// RUN: bishengir-opt --hfusion-normalize-ops="enable-high-precision=false" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeSin_hfusion_sin_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<5x1xf32>) -> tensor<5x1xf32> {
// CHECK: %[[VAL_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[VAL_2:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[VAL_3:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAL_5:.*]] = arith.constant -0.166666672 : f32
// CHECK: %[[VAL_6:.*]] = arith.constant 0.00833333377 : f32
// CHECK: %[[VAL_7:.*]] = arith.constant -1.98412701E-4 : f32
// CHECK: %[[VAL_8:.*]] = arith.constant 2.75573188E-6 : f32
// CHECK: %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VAL_10:.*]] = arith.constant -1.02906229E-13 : f32
// CHECK: %[[VAL_11:.*]] = arith.constant 1.21644916E-10 : f32
// CHECK: %[[VAL_12:.*]] = arith.constant 6.27711415E-7 : f32
// CHECK: %[[VAL_13:.*]] = arith.constant 9.67025756E-4 : f32
// CHECK: %[[VAL_14:.*]] = arith.constant 3.140625 : f32
// CHECK: %[[VAL_15:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_16:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_0]], %[[VAL_15]] : tensor<5x1xf32>, f32) outs(%[[VAL_16]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_18:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_19:.*]] = hfusion.cast {{.*}} ins(%[[VAL_17]] : tensor<5x1xf32>) outs(%[[VAL_18]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_20:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_21:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_14]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_22:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_0]], %[[VAL_21]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_23:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_13]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_22]], %[[VAL_23]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_12]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_24]], %[[VAL_25]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_11]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_26]], %[[VAL_27]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_10]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_28]], %[[VAL_29]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_30]], %[[VAL_9]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_32:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_31]], %[[VAL_31]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_33]], %[[VAL_8]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_34]], %[[VAL_7]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_35]], %[[VAL_33]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_36]], %[[VAL_6]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_37]], %[[VAL_33]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_38]], %[[VAL_5]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_39]], %[[VAL_33]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_40]], %[[VAL_4]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_42:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_41]], %[[VAL_31]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_43:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_3]] : tensor<5x1xf32>, f32) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_45:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_46:.*]] = hfusion.cast {{.*}} ins(%[[VAL_44]] : tensor<5x1xf32>) outs(%[[VAL_45]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_46]], %[[VAL_2]] : tensor<5x1xf32>, f32) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_48:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_1]] : tensor<5x1xf32>, f32) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_49:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_47]], %[[VAL_48]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_49]], %[[VAL_4]] : tensor<5x1xf32>, f32) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_42]], %[[VAL_50]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_16]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK  return %[[VAL_51]] : tensor<5x1xf32>
// CHECK: }
func.func @test_NormalizeSin_hfusion_sin_ops(%arg0 : tensor<5x1xf32>) ->  tensor<5x1xf32> {
  %0 = tensor.empty() : tensor<5x1xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %ret : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCos_hfusion_cos_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<5x1xf16>) -> tensor<5x1xf16> {
// CHECK: %[[VAL_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[VAL_2:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAL_4:.*]] = arith.constant -0.166666672 : f32
// CHECK: %[[VAL_5:.*]] = arith.constant 0.00833333377 : f32
// CHECK: %[[VAL_6:.*]] = arith.constant -1.98412701E-4 : f32
// CHECK: %[[VAL_7:.*]] = arith.constant 2.75573188E-6 : f32
// CHECK: %[[VAL_8:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[VAL_9:.*]] = arith.constant -1.02906229E-13 : f32
// CHECK: %[[VAL_10:.*]] = arith.constant 1.21644916E-10 : f32
// CHECK: %[[VAL_11:.*]] = arith.constant 6.27711415E-7 : f32
// CHECK: %[[VAL_12:.*]] = arith.constant 9.67025756E-4 : f32
// CHECK: %[[VAL_13:.*]] = arith.constant 3.140625 : f32
// CHECK: %[[VAL_14:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[VAL_15:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_16:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_17:.*]] = hfusion.cast {{.*}} ins(%[[VAL_0]] : tensor<5x1xf16>) outs(%[[VAL_16]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_18:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_17]], %[[VAL_15]] : tensor<5x1xf32>, f32) outs(%[[VAL_18]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_20:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_19]], %[[VAL_14]] : tensor<5x1xf32>, f32) outs(%[[VAL_18]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_21:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_22:.*]] = hfusion.cast {{.*}} ins(%[[VAL_20]] : tensor<5x1xf32>) outs(%[[VAL_21]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_23:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_13]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_17]], %[[VAL_24]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_12]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_25]], %[[VAL_26]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_11]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_27]], %[[VAL_28]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_10]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_29]], %[[VAL_30]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_9]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_31]], %[[VAL_32]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_33]], %[[VAL_8]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_35:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_34]], %[[VAL_34]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_36]], %[[VAL_7]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_37]], %[[VAL_6]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_38]], %[[VAL_36]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_39]], %[[VAL_5]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_40]], %[[VAL_36]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_42:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_41]], %[[VAL_4]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_42]], %[[VAL_36]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43]], %[[VAL_3]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_45:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_44]], %[[VAL_34]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_46:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_14]] : tensor<5x1xf32>, f32) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_48:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_49:.*]] = hfusion.cast {{.*}} ins(%[[VAL_47]] : tensor<5x1xf32>) outs(%[[VAL_48]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_49]], %[[VAL_2]] : tensor<5x1xf32>, f32) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_1]] : tensor<5x1xf32>, f32) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_52:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_50]], %[[VAL_51]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_53:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_52]], %[[VAL_3]] : tensor<5x1xf32>, f32) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_54:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_55:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_45]], %[[VAL_53]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_54]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_56:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[VAL_57:.*]] = hfusion.cast {{.*}} ins(%[[VAL_55]] : tensor<5x1xf32>) outs(%[[VAL_56]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[VAL_57]] : tensor<5x1xf16>
// CHECK: }
func.func @test_NormalizeCos_hfusion_cos_ops(%arg0 : tensor<5x1xf16>) ->  tensor<5x1xf16> {
  %0 = tensor.empty() : tensor<5x1xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}
