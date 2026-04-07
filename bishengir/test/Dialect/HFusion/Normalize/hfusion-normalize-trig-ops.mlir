// RUN: bishengir-opt --hfusion-normalize-ops="enable-high-precision=true" %s -split-input-file -verify-diagnostics | FileCheck %s
// High-precision sin/cos coverage lives in this file.
// Low-precision sin/cos coverage is in hfusion-normalize-trig-low-precision-ops.mlir.

// CHECK-LABEL: func.func @test_NormalizeSin_hfusion_sin_ops(
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
func.func @test_NormalizeSin_hfusion_sin_ops(%arg0 : tensor<5x1xf32>) ->  tensor<5x1xf32> {
  %0 = tensor.empty() : tensor<5x1xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %ret : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSin_hfusion_sin_ops_rank1_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7xf16>) -> tensor<7xf16> {
// CHECK: %[[EMPTY_IN:.*]] = tensor.empty() : tensor<7xf32>
// CHECK: %[[IN_F32:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[ARG0]] : tensor<7xf16>) outs(%[[EMPTY_IN]] : tensor<7xf32>) -> tensor<7xf32>
// CHECK: %[[TBL0:.*]] = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: %[[BITCAST:.*]] = hfusion.bitcast ins(%[[IN_F32]] : tensor<7xf32>) outs(%{{.*}} : tensor<7xi32>) -> tensor<7xi32>
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<7xf16>
// CHECK: %[[OUT:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%{{.*}} : tensor<7xf32>) outs(%[[EMPTY_OUT]] : tensor<7xf16>) -> tensor<7xf16>
// CHECK: return %[[OUT]] : tensor<7xf16>
// CHECK: }
func.func @test_NormalizeSin_hfusion_sin_ops_rank1_f16(%arg0 : tensor<7xf16>) -> tensor<7xf16> {
  %0 = tensor.empty() : tensor<7xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%arg0 : tensor<7xf16>) outs(%0 : tensor<7xf16>) -> tensor<7xf16>
  return %ret : tensor<7xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCos_hfusion_cos_ops(
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
func.func @test_NormalizeCos_hfusion_cos_ops(%arg0 : tensor<5x1xf16>) ->  tensor<5x1xf16> {
  %0 = tensor.empty() : tensor<5x1xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCos_hfusion_cos_ops_rank1_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<7xf32>) -> tensor<7xf32> {
// CHECK: %[[TBL0:.*]] = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: %[[BITCAST:.*]] = hfusion.bitcast ins(%[[ARG0]] : tensor<7xf32>) outs(%{{.*}} : tensor<7xi32>) -> tensor<7xi32>
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: return %{{.*}} : tensor<7xf32>
// CHECK: }
func.func @test_NormalizeCos_hfusion_cos_ops_rank1_f32(%arg0 : tensor<7xf32>) -> tensor<7xf32> {
  %0 = tensor.empty() : tensor<7xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%arg0 : tensor<7xf32>) outs(%0 : tensor<7xf32>) -> tensor<7xf32>
  return %ret : tensor<7xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtan_hfusion_atan_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CONST_0:.*]] = arith.constant 2.16840434E-19 : f32
// CHECK: %[[CONST_1:.*]] = arith.constant 4.61168602E+18 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 0.785398185 : f32
// CHECK: %[[NEGONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST_1:.*]] = arith.constant 0.392699093 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 0.414213568 : f32
// CHECK: %[[CST_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST_4:.*]] = arith.constant -0.333333343 : f32
// CHECK: %[[CST_5:.*]] = arith.constant 2.000000e-01 : f32
// CHECK: %[[CST_6:.*]] = arith.constant -0.142857149 : f32
// CHECK: %[[CST_7:.*]] = arith.constant 0.111111112 : f32
// CHECK: %[[CST_8:.*]] = arith.constant -0.0909090936 : f32
// CHECK: %[[CST_9:.*]] = arith.constant 0.0769230798 : f32
// CHECK: %[[LOWER_BOUND:.*]] = arith.constant -1.000000e+04 : f32
// CHECK: %[[UPPER_BOUND:.*]] = arith.constant 1.000000e+04 : f32
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_0]], %[[UPPER_BOUND]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_2]], %[[LOWER_BOUND]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_3]] : tensor<32xf32>) outs(%[[EMPTY1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_4]], %[[VAL_4]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_7]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_8]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_9]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_10]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_11]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_12]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_13]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_14]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_16:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_15]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_16]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_17]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_18]], %[[VAL_4]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_20:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_21:.*]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[VAL_20]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_22:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_23:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_4]], %[[VAL_21]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_22]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_23]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_22]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_25:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_4]], %[[VAL_21]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_26]], %[[VAL_24]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_27]] : tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_29:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_28]], %[[VAL_28]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_30]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_31]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_32]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_33]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_34]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_35]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_36]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_37]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_38]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_39]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_40]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_42:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_41]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_42]], %[[VAL_28]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_45:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_19]], %[[VAL_44]] : tensor<32xf32>, tensor<32xf32>)
// CHECK: %[[VAL_46:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_4]], %[[NEGONE]] : tensor<32xf32>, f32)
// CHECK: %[[VAL_48:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_4]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_48]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_52:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_53:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_47]], %[[VAL_51]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_52]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_54:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_53]] : tensor<32xf32>) outs(%[[VAL_52]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_55:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_56:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_54]], %[[VAL_54]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_57:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_56]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_58:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_57]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_59:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_58]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_60:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_59]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_61:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_60]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_62:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_61]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_63:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_62]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_64:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_63]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_65:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_64]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_66:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_65]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_67:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_66]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_68:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_67]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_69:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_68]], %[[VAL_54]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_70:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_71:.*]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[VAL_70]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_72:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_73:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_54]], %[[VAL_71]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_72]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_74:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_73]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_72]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_75:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_76:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_54]], %[[VAL_71]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_77:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_76]], %[[VAL_74]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_78:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_77]] : tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_79:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_80:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_78]], %[[VAL_78]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_81:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_80]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_82:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_81]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_83:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_82]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_84:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_83]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_85:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_84]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_86:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_85]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_87:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_86]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_88:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_87]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_89:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_88]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_90:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_89]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_91:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_90]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_92:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_91]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_93:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_92]], %[[VAL_78]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_94:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_93]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_95:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_69]], %[[VAL_94]] : tensor<32xf32>, tensor<32xf32>)
// CHECK: %[[VAL_96:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_97:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_95]], %[[CST_0]] : tensor<32xf32>, f32) outs(%[[VAL_96]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_98:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_45]], %[[VAL_97]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY2]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_99:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_100:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_0]], %[[CONST_1]] : tensor<32xf32>, f32) outs(%[[VAL_99]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_101:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_102:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_100]] : tensor<32xf32>) outs(%[[VAL_101]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_103:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_102]], %[[CONST_0]] : tensor<32xf32>, f32) outs(%[[VAL_101]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_104:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_100]], %[[VAL_103]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_99]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_105:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_98]], %[[VAL_104]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY3]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK  return %[[VAL_105]] : tensor<32xf32>
// CHECK: }
func.func @test_NormalizeAtan_hfusion_atan_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<atan>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtan_hfusion_atan_ops_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[ARG0]] : tensor<8xf16>) outs(%[[EMPTY0]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[MINF:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[CAST0]], %{{.*}} : tensor<8xf32>, f32) outs(%[[EMPTY1]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[MAXF:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MINF]], %{{.*}} : tensor<8xf32>, f32) outs(%[[EMPTY1]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[ABS:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[MAXF]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<8xf16>
// CHECK: %[[OUT:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%{{.*}} : tensor<8xf32>) outs(%[[EMPTY_OUT]] : tensor<8xf16>) -> tensor<8xf16>
// CHECK: return %[[OUT]] : tensor<8xf16>
// CHECK: }
func.func @test_NormalizeAtan_hfusion_atan_ops_f16(%arg0 : tensor<8xf16>) -> tensor<8xf16> {
  %0 = tensor.empty() : tensor<8xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<atan>} ins(%arg0 : tensor<8xf16>) outs(%0 : tensor<8xf16>) -> tensor<8xf16>
  return %ret : tensor<8xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeTan_hfusion_tan_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CST:.*]] = arith.constant -24.8048935 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 61.2036247 : f32
// CHECK: %[[CST_1:.*]] = arith.constant -6.87115717 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 0.0698520839 : f32
// CHECK: %[[CST_3:.*]] = arith.constant -1.02906229E-13 : f32
// CHECK: %[[CST_4:.*]] = arith.constant 1.21644916E-10 : f32
// CHECK: %[[CST_5:.*]] = arith.constant 4.37113883E-8 : f32
// CHECK: %[[CST_6:.*]] = arith.constant -4.37113883E-8 : f32
// CHECK: %[[CST_7:.*]] = arith.constant 6.27711415E-7 : f32
// CHECK: %[[CST_8:.*]] = arith.constant -1.57079637 : f32
// CHECK: %[[CST_9:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[CST_10:.*]] = arith.constant 9.67025756E-4 : f32
// CHECK: %[[CST_11:.*]] = arith.constant 3.140625 : f32
// CHECK: %[[CST_12:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[CST_12]] : tensor<32xf32>, f32) outs(%[[VAL_0]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {{.*}} ins(%[[VAL_1]] : tensor<32xf32>) outs(%[[VAL_2]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_11]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[ARG0]], %[[VAL_5]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_10]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_6]], %[[VAL_7]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_8]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_9]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_11:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_8]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_11]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_13:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_13]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_10]], %[[VAL_14]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_13]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_16:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_16]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_12]], %[[VAL_17]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_16]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_19:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_20:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_15]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_19]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_21:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_22:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_18]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_21]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_23:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_20]], %[[VAL_24]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_25]], %[[VAL_26]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_28:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_22]], %[[VAL_29]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_30]], %[[VAL_31]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_33:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_8]], %[[VAL_34]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_35]], %[[VAL_36]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_37]], %[[VAL_38]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_40:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_39]], %[[VAL_39]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_40]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_42:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_41]], %[[CST_2]] : tensor<32xf32>, f32) outs(%[[VAL_42]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_44:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_45:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_44]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_46:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_45]], %[[VAL_41]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_44]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_46]], %[[CST_0]] : tensor<32xf32>, f32) outs(%[[VAL_46]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_48:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_47]], %[[VAL_39]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_46]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_49:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_41]], %[[CST]] : tensor<32xf32>, f32) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_50]], %[[VAL_27]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_52:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_51]], %[[VAL_32]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_53:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_54:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_48]], %[[VAL_52]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_53]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: return %[[VAL_54]] : tensor<32xf32>
// CHECK: }
func.func @test_NormalizeTan_hfusion_tan_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeTan_hfusion_tan_ops_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[ARG0]] : tensor<8xf16>) outs(%[[EMPTY0]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[MUL_PI_REC:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[CAST0]], %{{.*}} : tensor<8xf32>, f32) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[XROUND:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[MUL_PI_REC]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[POS_PIO2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%{{.*}}, %{{.*}} : tensor<8xf32>, f32) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[NEG_PIO2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%{{.*}}, %{{.*}} : tensor<8xf32>, f32) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[DIV:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<8xf16>
// CHECK: %[[OUT:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<round>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[DIV]] : tensor<8xf32>) outs(%[[EMPTY_OUT]] : tensor<8xf16>) -> tensor<8xf16>
// CHECK: return %[[OUT]] : tensor<8xf16>
// CHECK: }
func.func @test_NormalizeTan_hfusion_tan_ops_f16(%arg0 : tensor<8xf16>) -> tensor<8xf16> {
  %0 = tensor.empty() : tensor<8xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>} ins(%arg0 : tensor<8xf16>) outs(%0 : tensor<8xf16>) -> tensor<8xf16>
  return %ret : tensor<8xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeTanh_hfusion_tanh_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST0:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CST2:.*]] = arith.constant -8.800000e+00 : f32
// CHECK: %[[CST3:.*]] = arith.constant 8.800000e+00 : f32
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_0]], %[[CST3]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_2]], %[[CST2]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST1]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_5]] : tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_6]], %[[CST0]] : tensor<32xf32>, f32) outs(%[[VAL_7]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_6]], %[[CST]] : tensor<32xf32>, f32) outs(%[[VAL_9]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_8]], %[[VAL_10]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_7]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: return %[[VAL_11]] : tensor<32xf32>
// CHECK: }
func.func @test_NormalizeTanh_hfusion_tanh_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeTanh_hfusion_tanh_ops_f16
// CHECK: %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST_NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CST_NEG8:.*]] = arith.constant -8.800000e+00 : f32
// CHECK: %[[CST_8DOT8:.*]] = arith.constant 8.800000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {{.*}} ins(%[[ARG0:.*]] : tensor<32xf16>) outs(%[[EMPTY0:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[MINF:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[CAST0:.*]], %[[CST_8DOT8:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[MAXF:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MINF:.*]], %[[CST_NEG8:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MAXF:.*]], %[[CST_2:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY2:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EXP:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL:.*]] : tensor<32xf32>) outs(%[[EMPTY2:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[ADD0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP:.*]], %[[CST_NEG1:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY3:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[ADD1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP:.*]], %[[CST_1:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY4:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[DIV:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[ADD0:.*]], %[[ADD1:.*]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY3:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[DIV:.*]] : tensor<32xf32>) outs(%[[EMPTY5:.*]] : tensor<32xf16>) -> tensor<32xf16>
func.func @test_NormalizeTanh_hfusion_tanh_ops_f16(%arg0 : tensor<32xf16>) ->  tensor<32xf16> {
  %0 = tensor.empty() : tensor<32xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%arg0 : tensor<32xf16>) outs(%0 : tensor<32xf16>) -> tensor<32xf16>
  return %ret : tensor<32xf16>
}
