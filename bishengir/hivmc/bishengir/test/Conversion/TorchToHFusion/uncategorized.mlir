// RUN: bishengir-opt %s --split-input-file -convert-torch-to-hfusion | FileCheck %s

// CHECK-LABEL:   func.func @aten_gelu_f32(
// CHECK-SAME:                             %[[VAL_0:.*]]: !torch.vtensor<[1024],f32>) -> !torch.vtensor<[1024],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1024],f32> -> tensor<1024xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.str "none"
// CHECK:           %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 4.471500e-02 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant -1.59576917 : f32
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_1]], %[[VAL_1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_1]], %[[VAL_7]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_9:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_8]], %[[VAL_4]] : tensor<1024xf32>, f32) outs(%[[VAL_6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_9]], %[[VAL_1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_10]], %[[VAL_5]] : tensor<1024xf32>, f32) outs(%[[VAL_6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_12:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_11]] : tensor<1024xf32>) outs(%[[VAL_6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_12]], %[[VAL_3]] : tensor<1024xf32>, f32) outs(%[[VAL_6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_1]], %[[VAL_13]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_15:.*]] = torch_c.from_builtin_tensor %[[VAL_14]] : tensor<1024xf32> -> !torch.vtensor<[1024],f32>
// CHECK:           return %[[VAL_15]] : !torch.vtensor<[1024],f32>
// CHECK:         }
func.func @aten_gelu_f32(%arg0: !torch.vtensor<[1024],f32>) -> (!torch.vtensor<[1024],f32>) {
  %str = torch.constant.str "none"
  %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[1024],f32>, !torch.str -> !torch.vtensor<[1024],f32>
  return %0 : !torch.vtensor<[1024],f32>
}

// -----

// CHECK-LABEL:   func.func @aten_gelu_f16(
// CHECK-SAME:                             %[[VAL_0:.*]]: !torch.vtensor<[1024],f16>) -> !torch.vtensor<[1024],f16> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1024],f16> -> tensor<1024xf16>
// CHECK:           %[[VAL_2:.*]] = torch.constant.str "none"
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_6:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_1]] : tensor<1024xf16>) outs(%[[VAL_5]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = arith.constant 4.471500e-02 : f32
// CHECK:           %[[VAL_9:.*]] = arith.constant -1.59576917 : f32
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[VAL_6]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[VAL_11]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_12]], %[[VAL_8]] : tensor<1024xf32>, f32) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_13]], %[[VAL_6]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_14]], %[[VAL_9]] : tensor<1024xf32>, f32) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_16:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_15]] : tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_16]], %[[VAL_7]] : tensor<1024xf32>, f32) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_6]], %[[VAL_17]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_21:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK:           %[[VAL_22:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_18]] : tensor<1024xf32>) outs(%[[VAL_21]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK:           %[[VAL_23:.*]] = torch_c.from_builtin_tensor %[[VAL_22]] : tensor<1024xf16> -> !torch.vtensor<[1024],f16>
// CHECK:           return %[[VAL_23]] : !torch.vtensor<[1024],f16>
// CHECK:         }
func.func @aten_gelu_f16(%arg0: !torch.vtensor<[1024],f16>) -> (!torch.vtensor<[1024],f16>) {
  %str = torch.constant.str "none"
  %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[1024],f16>, !torch.str -> !torch.vtensor<[1024],f16>
  return %0 : !torch.vtensor<[1024],f16>
}

// -----

// CHECK-LABEL:   func.func @aten_gelu_bf16(
// CHECK-SAME:                              %[[VAL_0:.*]]: !torch.vtensor<[1024],bf16>) -> !torch.vtensor<[1024],bf16> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1024],bf16> -> tensor<1024xbf16>
// CHECK:           %[[VAL_2:.*]] = torch.constant.str "none"
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_6:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_1]] : tensor<1024xbf16>) outs(%[[VAL_5]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = arith.constant 4.471500e-02 : f32
// CHECK:           %[[VAL_9:.*]] = arith.constant -1.59576917 : f32
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[VAL_6]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[VAL_11]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_12]], %[[VAL_8]] : tensor<1024xf32>, f32) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_13]], %[[VAL_6]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_14]], %[[VAL_9]] : tensor<1024xf32>, f32) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_16:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_15]] : tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_16]], %[[VAL_7]] : tensor<1024xf32>, f32) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_6]], %[[VAL_17]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_21:.*]] = tensor.empty() : tensor<1024xbf16>
// CHECK:           %[[VAL_22:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_18]] : tensor<1024xf32>) outs(%[[VAL_21]] : tensor<1024xbf16>) -> tensor<1024xbf16>
// CHECK:           %[[VAL_23:.*]] = torch_c.from_builtin_tensor %[[VAL_22]] : tensor<1024xbf16> -> !torch.vtensor<[1024],bf16>
// CHECK:           return %[[VAL_23]] : !torch.vtensor<[1024],bf16>
// CHECK:         }
func.func @aten_gelu_bf16(%arg0: !torch.vtensor<[1024],bf16>) -> (!torch.vtensor<[1024],bf16>) {
  %str = torch.constant.str "none"
  %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[1024],bf16>, !torch.str -> !torch.vtensor<[1024],bf16>
  return %0 : !torch.vtensor<[1024],bf16>
}

// -----

// CHECK-LABEL:   func.func @aten_gelu_f16_dynamic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?],f16>) -> !torch.vtensor<[?],f16> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?],f16> -> tensor<?xf16>
// CHECK:           %[[VAL_2:.*]] = torch.constant.str "none"
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?xf16>
// CHECK:           %[[VAL_5:.*]] = tensor.empty(%[[VAL_4]]) : tensor<?xf32>
// CHECK:           %[[VAL_6:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_1]] : tensor<?xf16>) outs(%[[VAL_5]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = arith.constant 4.471500e-02 : f32
// CHECK:           %[[VAL_9:.*]] = arith.constant -1.59576917 : f32
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_6]], %[[VAL_10]] : tensor<?xf32>
// CHECK:           %[[VAL_12:.*]] = tensor.empty(%[[VAL_11]]) : tensor<?xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[VAL_6]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_12]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[VAL_13]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_12]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_14]], %[[VAL_8]] : tensor<?xf32>, f32) outs(%[[VAL_12]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_16:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_15]], %[[VAL_6]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_12]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_16]], %[[VAL_9]] : tensor<?xf32>, f32) outs(%[[VAL_12]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_18:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_17]] : tensor<?xf32>) outs(%[[VAL_12]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_18]], %[[VAL_7]] : tensor<?xf32>, f32) outs(%[[VAL_12]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_20:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_6]], %[[VAL_19]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_12]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_22:.*]] = tensor.dim %[[VAL_20]], %[[VAL_21]] : tensor<?xf32>
// CHECK:           %[[VAL_23:.*]] = tensor.empty(%[[VAL_22]]) : tensor<?xf16>
// CHECK:           %[[VAL_24:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_20]] : tensor<?xf32>) outs(%[[VAL_23]] : tensor<?xf16>) -> tensor<?xf16>
// CHECK:           %[[VAL_25:.*]] = torch_c.from_builtin_tensor %[[VAL_24]] : tensor<?xf16> -> !torch.vtensor<[?],f16>
// CHECK:           return %[[VAL_25]] : !torch.vtensor<[?],f16>
// CHECK:         }
func.func @aten_gelu_f16_dynamic(%arg0: !torch.vtensor<[?],f16>) -> (!torch.vtensor<[?],f16>) {
  %str = torch.constant.str "none"
  %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[?],f16>, !torch.str -> !torch.vtensor<[?],f16>
  return %0 : !torch.vtensor<[?],f16>
}
