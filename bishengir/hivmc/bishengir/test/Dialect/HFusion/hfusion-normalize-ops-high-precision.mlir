// RUN: bishengir-opt --hfusion-normalize-ops="enable-high-precision" %s -split-input-file -verify-diagnostics | FileCheck %s
// -----
// CHECK-LABEL: func.func @test_hfusion_sin_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<5x1xf32>) -> tensor<5x1xf32> {
// CHECK: %[[CST:.*]] = arith.constant -0.166666672 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 0.00833333377 : f32
// CHECK: %[[CST_1:.*]] = arith.constant -1.98412701E-4 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 2.75573188E-6 : f32
// CHECK: %[[CST_NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST_POS1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[CNEG1_I32:.*]] = arith.constant -1 : i32
// CHECK: %[[C8388607_I32:.*]] = arith.constant 8388607 : i32
// CHECK: %[[C255_I32:.*]] = arith.constant 255 : i32
// CHECK: %[[C23_I32:.*]] = arith.constant 23 : i32
// CHECK: %[[C8388608_I32:.*]] = arith.constant 8388608 : i32
// CHECK: %[[C65535_I32:.*]] = arith.constant 65535 : i32
// CHECK: %[[C32_I32:.*]] = arith.constant 32 : i32
// CHECK: %[[CST_5:.*]] = arith.constant 4.65661287E-10 : f32
// CHECK: %[[C2147483647_I32:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[C31_I32:.*]] = arith.constant 31 : i32
// CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
// CHECK: %[[C8_I32:.*]] = arith.constant 8 : i32
// CHECK: %[[PI:.*]] = arith.constant 3.14159274 : f32
// CHECK: %[[QNAN_F32:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: %[[TBL0:.*]] = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: %[[TBL_RC:.*]] = memref.reinterpret_cast %[[TBL0]] to offset: [0], sizes: [320], strides: [1] : memref<320xi32, #hivm.address_space<gm>> to memref<320xi32, strided<[1]>, #hivm.address_space<gm>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<320xi32>
// CHECK: memref.copy %[[TBL_RC]], %[[ALLOC]] : memref<320xi32, strided<[1]>, #hivm.address_space<gm>> to memref<320xi32>
// CHECK: %[[TBL_T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<320xi32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} : tensor<5x1xf32> into tensor<5xf32>
// CHECK: %[[E_I32_0:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[BITCAST:.*]] = hfusion.bitcast ins(%[[COLLAPSED]] : tensor<5xf32>) outs(%[[E_I32_0]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_1:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[SHR_SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[BITCAST]], %[[C31_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_1]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_2:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[SHR_SIGN]], %[[C1_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_2]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_3:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[SHR_EXP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[BITCAST]], %[[C23_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_3]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_4:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[EXP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[C255_I32]], %[[SHR_EXP]] : i32, tensor<5xi32>) outs(%[[E_I32_4]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I1_0:.*]] = tensor.empty() : tensor<5xi1>
// CHECK: %[[IS_NAN_INF:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[EXP]], %[[C255_I32]] : tensor<5xi32>, i32) outs(%[[E_I1_0]] : tensor<5xi1>) -> tensor<5xi1>
// CHECK: %[[E_I32_5:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[MANT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[C8388607_I32]], %[[BITCAST]] : i32, tensor<5xi32>) outs(%[[E_I32_5]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_6:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[MANT_ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[MANT]], %[[C8388608_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_6]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_7:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[EXP_P8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP]], %[[C8_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_7]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_8:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[EXP_P8_P32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP_P8]], %[[C32_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_8]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_9:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G0:.*]] = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%[[TBL_T]], %[[EXP_P8]] : tensor<320xi32>, tensor<5xi32>) outs(%[[E_I32_9]] : tensor<5xi32>) axis = 0 -> tensor<5xi32>
// CHECK: %[[E_I32_10:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G1:.*]] = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%[[TBL_T]], %[[EXP_P8_P32]] : tensor<320xi32>, tensor<5xi32>) outs(%[[E_I32_10]] : tensor<5xi32>) axis = 0 -> tensor<5xi32>
// CHECK: %[[E_I32_11:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[MHI:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MANT_ADD]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_11]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_12:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[MLO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MANT_ADD]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_12]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_13:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G0_HI:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[G0]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_13]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_14:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G0_LO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[G0]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_14]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_15:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G1_HI:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[G1]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_15]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_16:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G1_LO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[G1]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_16]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_17:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MLO]], %[[G1_HI]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_17]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_18:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P0_SHR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[P0]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_18]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_19:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MLO]], %[[G0_LO]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_19]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_20:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MLO]], %[[G0_HI]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_20]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_21:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P2_LO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[P2]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_21]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_22:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P2_LO_SHL:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[P2_LO]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_22]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_23:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MHI]], %[[G1_LO]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_23]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_24:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P3_SHR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[P3]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_24]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_25:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MHI]], %[[G1_HI]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_25]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_26:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MHI]], %[[G0_LO]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_26]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_27:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P5_LO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[P5]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_27]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_28:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P5_LO_SHL:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[P5_LO]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_28]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_29:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P0_SHR]], %[[P1]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_29]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_30:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P2_LO_SHL]], %[[P3_SHR]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_30]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_31:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P4]], %[[P5_LO_SHL]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_31]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_32:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[S0]], %[[S1]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_32]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_33:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[S3]], %[[S2]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_33]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_34:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[ABS_SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[S4]], %[[C31_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_34]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_35:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[ABS_BITS:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[S4]], %[[C2147483647_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_35]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_F32_0:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[ABS_F:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[ABS_BITS]] : tensor<5xi32>) outs(%[[E_F32_0]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_1:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[X:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ABS_F]], %[[CST_5]] : tensor<5xf32>, f32) outs(%[[E_F32_1]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_I32_36:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[OR_QNAN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[ABS_SIGN]], %{{.*}} : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_36]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_37:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[AND_QNAN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[ABS_SIGN]], %{{.*}} : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_37]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[NOT_AND:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[AND_QNAN]] : tensor<5xi32>) outs(%[[AND_QNAN]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_38:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[AND2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[NOT_AND]], %[[OR_QNAN]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_38]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_39:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[LSB:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[AND2]], %[[C1_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_39]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I1_1:.*]] = tensor.empty() : tensor<5xi1>
// CHECK: %[[LSB_IS0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[LSB]], %[[C0_I32]] : tensor<5xi32>, i32) outs(%[[E_I1_1]] : tensor<5xi1>) -> tensor<5xi1>
// CHECK: %[[E_F32_2:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[SIGN_F:.*]] = hfusion.select ins(%[[LSB_IS0]], %[[CST_POS1]], %[[CST_NEG1]] : tensor<5xi1>, f32, f32) outs(%[[E_F32_2]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_3:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[XPI:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X]], %[[PI]] : tensor<5xf32>, f32) outs(%[[E_F32_3]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_4:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[E_F32_5:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[NEG_XPI:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[XPI]], %[[CST_NEG1]] : tensor<5xf32>, f32) outs(%[[E_F32_5]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[PI_MINUS:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[PI]], %[[NEG_XPI]] : f32, tensor<5xf32>) outs(%[[E_F32_4]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_6:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[R:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[PI_MINUS]], %[[XPI]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_7:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[R2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[R]], %[[R]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[T0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[R2]], %[[CST_2]] : tensor<5xf32>, f32) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[T1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[T0]], %[[CST_1]] : tensor<5xf32>, f32) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[T2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[T1]], %[[R2]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[T3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[T2]], %[[CST_0]] : tensor<5xf32>, f32) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[T4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[T3]], %[[R2]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[T5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[T4]], %[[CST]] : tensor<5xf32>, f32) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[T6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[T5]], %[[R2]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[T7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[T6]], %[[CST_POS1]] : tensor<5xf32>, f32) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[POLY:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[T7]], %[[R]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_8:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[SIGNED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[POLY]], %[[SIGN_F]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_8]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_9:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[OUT:.*]] = hfusion.select ins(%[[IS_NAN_INF]], %[[QNAN_F32]], %[[SIGNED]] : tensor<5xi1>, f32, tensor<5xf32>) outs(%[[E_F32_9]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[EXPAND:.*]] = tensor.expand_shape %[[OUT]] {{\[}}[0, 1]{{\]}} output_shape {{\[}}5, 1{{\]}} : tensor<5xf32> into tensor<5x1xf32>
// CHECK: return %[[EXPAND]] : tensor<5x1xf32>
// CHECK: }

func.func @test_hfusion_sin_ops(%arg0 : tensor<5x1xf32>) ->  tensor<5x1xf32> {
  %0 = tensor.empty() : tensor<5x1xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %ret : tensor<5x1xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_cos_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<5x1xf16>) -> tensor<5x1xf16> {
// CHECK: %[[CST:.*]] = arith.constant -0.166666672 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 0.00833333377 : f32
// CHECK: %[[CST_1:.*]] = arith.constant -1.98412701E-4 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 2.75573188E-6 : f32
// CHECK: %[[CST_NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST_POS1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[C8388607_I32:.*]] = arith.constant 8388607 : i32
// CHECK: %[[C255_I32:.*]] = arith.constant 255 : i32
// CHECK: %[[C23_I32:.*]] = arith.constant 23 : i32
// CHECK: %[[C8388608_I32:.*]] = arith.constant 8388608 : i32
// CHECK: %[[C65535_I32:.*]] = arith.constant 65535 : i32
// CHECK: %[[C32_I32:.*]] = arith.constant 32 : i32
// CHECK: %[[CST_5:.*]] = arith.constant 4.65661287E-10 : f32
// CHECK: %[[C2147483647_I32:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[C31_I32:.*]] = arith.constant 31 : i32
// CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
// CHECK: %[[C8_I32:.*]] = arith.constant 8 : i32
// CHECK: %[[PI:.*]] = arith.constant 3.14159274 : f32
// CHECK: %[[PIO2:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[QNAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: %[[E_F32_IN:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[IN_F32:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[ARG0]] : tensor<5x1xf16>) outs(%[[E_F32_IN]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[TBL0:.*]] = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: %[[TBL_RC:.*]] = memref.reinterpret_cast %[[TBL0]] to offset: [0], sizes: [320], strides: [1] : memref<320xi32, #hivm.address_space<gm>> to memref<320xi32, strided<[1]>, #hivm.address_space<gm>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<320xi32>
// CHECK: memref.copy %[[TBL_RC]], %[[ALLOC]] : memref<320xi32, strided<[1]>, #hivm.address_space<gm>> to memref<320xi32>
// CHECK: %[[TBL_T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<320xi32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[IN_F32]] {{\[}}[0, 1]{{\]}} : tensor<5x1xf32> into tensor<5xf32>
// CHECK: %[[E_I32_0:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[BITCAST:.*]] = hfusion.bitcast ins(%[[COLLAPSED]] : tensor<5xf32>) outs(%[[E_I32_0]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_1:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[SHR_EXP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[BITCAST]], %[[C23_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_1]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_2:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[EXP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[C255_I32]], %[[SHR_EXP]] : i32, tensor<5xi32>) outs(%[[E_I32_2]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I1_0:.*]] = tensor.empty() : tensor<5xi1>
// CHECK: %[[IS_NAN_INF:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[EXP]], %[[C255_I32]] : tensor<5xi32>, i32) outs(%[[E_I1_0]] : tensor<5xi1>) -> tensor<5xi1>
// CHECK: %[[E_I32_3:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[MANT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[C8388607_I32]], %[[BITCAST]] : i32, tensor<5xi32>) outs(%[[E_I32_3]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_4:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[MANT_ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[MANT]], %[[C8388608_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_4]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_5:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[EXP_P8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP]], %[[C8_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_5]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_6:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[EXP_P8_P32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP_P8]], %[[C32_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_6]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_7:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G0:.*]] = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%[[TBL_T]], %[[EXP_P8]] : tensor<320xi32>, tensor<5xi32>) outs(%[[E_I32_7]] : tensor<5xi32>) axis = 0 -> tensor<5xi32>
// CHECK: %[[E_I32_8:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G1:.*]] = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%[[TBL_T]], %[[EXP_P8_P32]] : tensor<320xi32>, tensor<5xi32>) outs(%[[E_I32_8]] : tensor<5xi32>) axis = 0 -> tensor<5xi32>
// CHECK: %[[E_I32_9:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[MHI:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[MANT_ADD]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_9]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_10:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[MLO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MANT_ADD]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_10]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_11:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G0_HI:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[G0]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_11]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_12:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G0_LO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[G0]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_12]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_13:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G1_HI:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[G1]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_13]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_14:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[G1_LO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[G1]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_14]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_15:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MLO]], %[[G1_HI]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_15]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_16:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P0_SHR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[P0]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_16]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_17:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MLO]], %[[G0_LO]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_17]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_18:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MLO]], %[[G0_HI]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_18]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_19:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P2_LO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[P2]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_19]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_20:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P2_LO_SHL:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[P2_LO]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_20]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_21:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MHI]], %[[G1_LO]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_21]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_22:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P3_SHR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[P3]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_22]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_23:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MHI]], %[[G1_HI]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_23]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_24:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MHI]], %[[G0_LO]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_24]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_25:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P5_LO:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[P5]], %[[C65535_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_25]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_26:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[P5_LO_SHL:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[P5_LO]], %[[C16_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_26]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_27:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P0_SHR]], %[[P1]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_27]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_28:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P2_LO_SHL]], %[[P3_SHR]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_28]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_29:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P4]], %[[P5_LO_SHL]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_29]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_30:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[S0]], %[[S1]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_30]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_31:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[S4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[S3]], %[[S2]] : tensor<5xi32>, tensor<5xi32>) outs(%[[E_I32_31]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_32:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[ABS_SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[S4]], %[[C31_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_32]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I32_33:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[ABS_BITS:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[S4]], %[[C2147483647_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_33]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_F32_0:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[ABS_F:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[ABS_BITS]] : tensor<5xi32>) outs(%[[E_F32_0]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_1:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[X:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ABS_F]], %[[CST_5]] : tensor<5xf32>, f32) outs(%[[E_F32_1]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_I32_34:.*]] = tensor.empty() : tensor<5xi32>
// CHECK: %[[LSB:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[ABS_SIGN]], %[[C1_I32]] : tensor<5xi32>, i32) outs(%[[E_I32_34]] : tensor<5xi32>) -> tensor<5xi32>
// CHECK: %[[E_I1_1:.*]] = tensor.empty() : tensor<5xi1>
// CHECK: %[[LSB_IS0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[LSB]], %[[C0_I32]] : tensor<5xi32>, i32) outs(%[[E_I1_1]] : tensor<5xi1>) -> tensor<5xi1>
// CHECK: %[[E_F32_2:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[SIGN_F:.*]] = hfusion.select ins(%[[LSB_IS0]], %[[CST_POS1]], %[[CST_NEG1]] : tensor<5xi1>, f32, f32) outs(%[[E_F32_2]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_3:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[XPI:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X]], %[[PI]] : tensor<5xf32>, f32) outs(%[[E_F32_3]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_4:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[E_F32_5:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[NEG_XPI:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[XPI]], %[[CST_NEG1]] : tensor<5xf32>, f32) outs(%[[E_F32_5]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[SHIFT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[PIO2]], %[[NEG_XPI]] : f32, tensor<5xf32>) outs(%[[E_F32_4]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_6:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[R2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SHIFT]], %[[SHIFT]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[P0F:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[R2]], %[[CST_2]] : tensor<5xf32>, f32) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[P1F:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P0F]], %[[CST_1]] : tensor<5xf32>, f32) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[P2F:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[P1F]], %[[R2]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[P3F:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P2F]], %[[CST_0]] : tensor<5xf32>, f32) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[P4F:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[P3F]], %[[R2]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[P5F:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P4F]], %[[CST]] : tensor<5xf32>, f32) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[P6F:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[P5F]], %[[R2]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[P7F:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[P6F]], %[[CST_POS1]] : tensor<5xf32>, f32) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[POLY:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[P7F]], %[[SHIFT]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_6]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_7:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[SIGNED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[POLY]], %[[SIGN_F]] : tensor<5xf32>, tensor<5xf32>) outs(%[[E_F32_7]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[E_F32_8:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[OUT_F32:.*]] = hfusion.select ins(%[[IS_NAN_INF]], %[[QNAN]], %[[SIGNED]] : tensor<5xi1>, f32, tensor<5xf32>) outs(%[[E_F32_8]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[EXPAND:.*]] = tensor.expand_shape %[[OUT_F32]] {{\[}}[0, 1]{{\]}} output_shape {{\[}}5, 1{{\]}} : tensor<5xf32> into tensor<5x1xf32>
// CHECK: %[[E_F16:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[OUT_F16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[EXPAND]] : tensor<5x1xf32>) outs(%[[E_F16]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[OUT_F16]] : tensor<5x1xf16>
// CHECK: }

func.func @test_hfusion_cos_ops(%arg0 : tensor<5x1xf16>) ->  tensor<5x1xf16> {
  %0 = tensor.empty() : tensor<5x1xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}
