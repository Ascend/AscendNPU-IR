// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true %s | FileCheck %s

// CHECK: LLVMDialectModule
// CHECK: define dso_local void @model_21(ptr addrspace(1) %0, ptr addrspace(1) %1)
module {
  func.func @model_21(%arg0: tensor<24x192x192xbf16>) -> tensor<24x192x192xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [24, 192, 1, 192] : tensor<24x192x192xbf16> into tensor<24x192x1x192xbf16>
    %0 = tensor.empty() : tensor<24x192x192x1xbf16>
    %transposed = linalg.transpose ins(%expanded : tensor<24x192x1x192xbf16>) outs(%0 : tensor<24x192x192x1xbf16>) permutation = [0, 1, 3, 2] 
    %collapsed = tensor.collapse_shape %transposed [[0], [1], [2, 3]] : tensor<24x192x192x1xbf16> into tensor<24x192x192xbf16>
    %1 = tensor.empty() : tensor<24x192x192xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<24x192x192xbf16>) outs(%1 : tensor<24x192x192xf32>) -> tensor<24x192x192xf32>
    return %2 : tensor<24x192x192xf32>
  }
}