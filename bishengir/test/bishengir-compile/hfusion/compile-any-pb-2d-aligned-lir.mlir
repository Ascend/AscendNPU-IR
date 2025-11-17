// RUN: bishengir-compile -enable-hfusion-compile=true  -block-dim=1 %s

module {
  func.func @model_2(%arg0: tensor<391x1xf16>, %arg1: tensor<1x288xf16>, %arg2: tensor<391x288xf16>) -> tensor<391x288xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<391x288xf16>
    %1 = tensor.empty() : tensor<391x1xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<391x1xf16> into tensor<391xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<391xf16>) outs(%0 : tensor<391x288xf16>) dimensions = [1]
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x288xf16> into tensor<288xf16>
    %broadcasted_1 = linalg.broadcast ins(%collapsed_0 : tensor<288xf16>) outs(%0 : tensor<391x288xf16>) dimensions = [0]
    %2 = linalg.elemwise_binary {__a__, fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_1 : tensor<391x288xf16>, tensor<391x288xf16>) outs(%arg2 : tensor<391x288xf16>) -> tensor<391x288xf16>
    return %2 : tensor<391x288xf16>
  }
}