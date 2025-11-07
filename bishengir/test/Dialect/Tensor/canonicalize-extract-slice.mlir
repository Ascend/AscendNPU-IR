// RUN: bishengir-opt %s -canonicalize-ext -cse -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @test_extract_from_concat_0
// CHECK-SAME: %[[arg0:.*]]: tensor<24x512x48xbf16>
// CHECK-NOT: tensor.concat
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<24x512x48xbf16>)
func.func @test_extract_from_concat_0(%arg0: tensor<24x512x48xbf16>, %arg1: tensor<24x1024x48xbf16>) -> tensor<24x512x48xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %cst = arith.constant 0.210224107 : f32
  %0 = tensor.empty() : tensor<24x512x48xbf16>
  %1 = tensor.empty() : tensor<24x512x48xf32>
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<24x512x48xbf16>, tensor<24x1024x48xbf16>) -> tensor<24x1536x48xbf16>
  %extracted_slice = tensor.extract_slice %concat[0, 0, 0] [24, 512, 48] [1, 1, 1] : tensor<24x1536x48xbf16> to tensor<24x512x48xbf16>
  %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice : tensor<24x512x48xbf16>) outs(%1 : tensor<24x512x48xf32>) -> tensor<24x512x48xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %cst : tensor<24x512x48xf32>, f32) outs(%1 : tensor<24x512x48xf32>) -> tensor<24x512x48xf32>
  %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<24x512x48xf32>) outs(%0 : tensor<24x512x48xbf16>) -> tensor<24x512x48xbf16>
  return %4 : tensor<24x512x48xbf16>
}

// -----

// CHECK-LABEL: func.func @test_extract_from_concat_1
// CHECK-SAME: , %[[arg1:.*]]: tensor<24x1024x192xbf16>
// CHECK-NOT: tensor.concat
// CHECK: tensor.extract_slice %[[arg1]]{{\[}}0, 0, 0] {{\[}}24, 512, 192] {{\[}}1, 1, 1]
func.func @test_extract_from_concat_1(%arg0: tensor<24x512x192xbf16>, %arg1: tensor<24x1024x192xbf16>) -> tensor<24x512x192xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.21022410381342865 : f32
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<24x512x192xbf16>, tensor<24x1024x192xbf16>) -> tensor<24x1536x192xbf16>
  %extracted_slice = tensor.extract_slice %concat[0, 512, 0] [24, 512, 192] [1, 1, 1] : tensor<24x1536x192xbf16> to tensor<24x512x192xbf16>
  %0 = arith.truncf %cst : f32 to bf16
  %1 = tensor.empty() : tensor<24x512x192xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<24x512x192xbf16>) -> tensor<24x512x192xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %2 : tensor<24x512x192xbf16>, tensor<24x512x192xbf16>) outs(%1 : tensor<24x512x192xbf16>) -> tensor<24x512x192xbf16>
  return %3 : tensor<24x512x192xbf16>
}

// -----

// CHECK-LABEL: func.func @test_extract_from_concat_2
// CHECK-SAME: , %[[arg1:.*]]: tensor<24x1024x192xbf16>
// CHECK-NOT: tensor.concat
// CHECK: tensor.extract_slice %[[arg1]]{{\[}}0, 512, 0] {{\[}}24, 512, 192] {{\[}}1, 1, 1]
func.func @test_extract_from_concat_2(%arg0: tensor<24x512x192xbf16>, %arg1: tensor<24x1024x192xbf16>) -> tensor<24x512x192xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.21022410381342865 : f32
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<24x512x192xbf16>, tensor<24x1024x192xbf16>) -> tensor<24x1536x192xbf16>
  %extracted_slice = tensor.extract_slice %concat[0, 1024, 0] [24, 512, 192] [1, 1, 1] : tensor<24x1536x192xbf16> to tensor<24x512x192xbf16>
  %0 = arith.truncf %cst : f32 to bf16
  %1 = tensor.empty() : tensor<24x512x192xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<24x512x192xbf16>) -> tensor<24x512x192xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %2 : tensor<24x512x192xbf16>, tensor<24x512x192xbf16>) outs(%1 : tensor<24x512x192xbf16>) -> tensor<24x512x192xbf16>
  return %3 : tensor<24x512x192xbf16>
}

// -----

// CHECK-LABEL: func.func @test_extract_from_concat_fail
// CHECK-SAME: , %[[arg1:.*]]: tensor<24x1024x192xbf16>
// CHECK: %[[concat:.*]] = tensor.concat
// CHECK: tensor.extract_slice %[[concat]]{{\[}}0, 511, 0] {{\[}}24, 512, 192] {{\[}}1, 1, 1]
func.func @test_extract_from_concat_fail(%arg0: tensor<24x512x192xbf16>, %arg1: tensor<24x1024x192xbf16>) -> tensor<24x512x192xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.21022410381342865 : f32
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<24x512x192xbf16>, tensor<24x1024x192xbf16>) -> tensor<24x1536x192xbf16>
  %extracted_slice = tensor.extract_slice %concat[0, 511, 0] [24, 512, 192] [1, 1, 1] : tensor<24x1536x192xbf16> to tensor<24x512x192xbf16>
  %0 = arith.truncf %cst : f32 to bf16
  %1 = tensor.empty() : tensor<24x512x192xbf16>
  %2 = linalg.fill ins(%0 : bf16) outs(%1 : tensor<24x512x192xbf16>) -> tensor<24x512x192xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %2 : tensor<24x512x192xbf16>, tensor<24x512x192xbf16>) outs(%1 : tensor<24x512x192xbf16>) -> tensor<24x512x192xbf16>
  return %3 : tensor<24x512x192xbf16>
}

// -----

// CHECK-LABEL: func.func @test_extract_from_concat_dyn_size
// CHECK: tensor.concat
func.func @test_extract_from_concat_dyn_size(%arg0: tensor<24x?x192xbf16>, %arg1: tensor<24x1024x192xbf16>) -> tensor<24x512x192xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.21022410381342865 : f32
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<24x?x192xbf16>, tensor<24x1024x192xbf16>) -> tensor<24x?x192xbf16>
  %extracted_slice = tensor.extract_slice %concat[0, 0, 0] [24, 512, 192] [1, 1, 1] : tensor<24x?x192xbf16> to tensor<24x512x192xbf16>
  return %extracted_slice : tensor<24x512x192xbf16>
}