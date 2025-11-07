// RUN: bishengir-opt %s --trickle-concat-down --cse --canonicalize-ext --split-input-file | FileCheck %s

// CHECK-LABEL: test_concat
// CHECK: elemwise_unary
// CHECK: elemwise_unary
// CHECK: concat
// CHECK: return
module {
  func.func @test_concat(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x6xf32> {
    %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
    %empty1 = tensor.empty() : tensor<2x6xf32>
    %unary  = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%concat : tensor<2x6xf32>) outs(%empty1 : tensor<2x6xf32>) -> tensor<2x6xf32>
    return %unary : tensor<2x6xf32>
  }
}

// -----

// CHECK-LABEL: test_concat_cast
// CHECK: hfusion.cast
// CHECK: hfusion.cast
// CHECK: concat
// CHECK: return
module {
  func.func @test_concat_cast(%arg0: tensor<2x3xbf16>, %arg1: tensor<2x3xbf16>) -> tensor<2x6xf32> {
    %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<2x6xbf16>
    %empty1 = tensor.empty() : tensor<2x6xf32>
    %unary  = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%concat : tensor<2x6xbf16>) outs(%empty1 : tensor<2x6xf32>) -> tensor<2x6xf32>
    return %unary : tensor<2x6xf32>
  }
}

// -----

// CHECK-LABEL: test_concat_aligned
// CHECK: concat
// CHECK: elemwise_unary
// CHECK: return
module {
  func.func @test_concat_aligned(%arg0: tensor<2x64xf32>, %arg1: tensor<2x64xf32>) -> tensor<2x128xf32> {
    %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x128xf32>
    %empty1 = tensor.empty() : tensor<2x128xf32>
    %unary  = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%concat : tensor<2x128xf32>) outs(%empty1 : tensor<2x128xf32>) -> tensor<2x128xf32>
    return %unary : tensor<2x128xf32>
  }
}

// -----

// CHECK-LABEL: func @concat_down_on_extract_slice
// CHECK: %[[source:.*]] = hfusion.cast
// [Z] = extract_slice [A B C D .. E Z]
// CHECK: %[[extractBack:.*]] = tensor.extract_slice %[[source]][2046] [1] [1]
// [B C D .. E Z] = extract_slice [A B C D .. E Z]
// CHECK: %[[extractFront:.*]] = tensor.extract_slice %[[source]][1] [2046] [1]
// [B C D .. E Z Z] = concat [B C D .. E Z] + [Z]
// CHECK: %[[concat:.*]] = tensor.concat dim(0) %[[extractFront]], %[[extractBack]]
// CHECK: return

module {
  func.func @concat_down_on_extract_slice(%arg0: tensor<1x2047xi64>) -> tensor<2047x2047xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<2047xi32>
    %1 = tensor.empty() : tensor<2047x2047xi32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2047xi64> into tensor<2047xi64>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<2047xi64>) outs(%0 : tensor<2047xi32>) -> tensor<2047xi32>
    // [Z] = extract_slice [A B C D .. E Z]
    %extracted_slice = tensor.extract_slice %2[2046] [1] [1] : tensor<2047xi32> to tensor<1xi32>
    // [2047] + [1] -> [2048]
    // [A B C D .. E Z Z] = concat [A B C D .. E Z] + [Z]
    %concat = tensor.concat dim(0) %2, %extracted_slice : (tensor<2047xi32>, tensor<1xi32>) -> tensor<2048xi32>
    // [B C D .. E Z Z] = extract_slice [A B C D .. E Z Z]
    %extracted_slice_0 = tensor.extract_slice %concat[1] [2047] [1] : tensor<2048xi32> to tensor<2047xi32>
    %broadcasted = linalg.broadcast ins(%extracted_slice_0 : tensor<2047xi32>) outs(%1 : tensor<2047x2047xi32>) dimensions = [1]
    return %broadcasted : tensor<2047x2047xi32>
  }
}

// -----

// CHECK-LABEL: func @concat_down_on_extract_slice_multi_dim
// CHECK: %[[source:.*]] = hfusion.cast
// [Z] = extract_slice [A B C D .. E Z]
// CHECK: %[[extractFront:.*]] = tensor.extract_slice %[[a1:.*]][3, 1, 0] [2, 2046, 2] [2, 1, 3]
// [B C D .. E Z] = extract_slice [A B C D .. E Z]
// CHECK: %[[extractBack:.*]] = tensor.extract_slice %[[a2:.*]][3, 0, 0] [2, 1, 2] [2, 1, 3]
// [B C D .. E Z Z] = concat [B C D .. E Z] + [Z]
// CHECK: %[[concat:.*]] = tensor.concat dim(1) %[[extractFront]], %[[extractBack]]
// CHECK: return

module {
  func.func @concat_down_on_extract_slice_multi_dim(%arg0: tensor<40x1x2047x30xi64>, %arg1: tensor<8x2047x9xi32>) -> tensor<2x2047x2047x2xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<40x2047x30xi32>
    %1 = tensor.empty() : tensor<2x2047x2047x2xi32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<40x1x2047x30xi64> into tensor<40x2047x30xi64>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<40x2047x30xi64>) outs(%0 : tensor<40x2047x30xi32>) -> tensor<40x2047x30xi32>
    %extracted_slice = tensor.extract_slice %2[4, 2046, 1] [8, 1, 9] [2, 1, 3] : tensor<40x2047x30xi32> to tensor<8x1x9xi32>
    %concat = tensor.concat dim(1) %arg1, %extracted_slice : ( tensor<8x2047x9xi32>, tensor<8x1x9xi32>) -> tensor<8x2048x9xi32>
    %extracted_slice_0 = tensor.extract_slice %concat[3, 1, 0] [2, 2047, 2] [2, 1, 3] : tensor<8x2048x9xi32> to tensor<2x2047x2xi32>
    %broadcasted = linalg.broadcast ins(%extracted_slice_0 : tensor<2x2047x2xi32>) outs(%1 : tensor<2x2047x2047x2xi32>) dimensions = [1]
    return %broadcasted : tensor<2x2047x2047x2xi32>
  }
}

// -----

// CHECK-LABEL: func @complicated
// CHECK: %[[extractFront:.*]] = tensor.extract_slice %[[a1:.*]][3, 100, 0] [2, 217, 2] [2, 9, 3]
// CHECK: %[[extractBack:.*]] = tensor.extract_slice %[[a2:.*]][3, 5, 0] [2, 83, 2] [2, 9, 3]
// CHECK: %[[concat:.*]] = tensor.concat dim(1) %[[extractFront]], %[[extractBack]]
// CHECK: return

module {
  func.func @complicated(%arg0: tensor<40x1x2047x30xi64>, %arg1: tensor<8x2047x9xi32>, %arg2: tensor<8x3400x9xi32>) -> tensor<2x2047x300x2xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<40x2047x30xi32>
    %1 = tensor.empty() : tensor<2x2047x300x2xi32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<40x1x2047x30xi64> into tensor<40x2047x30xi64>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<40x2047x30xi64>) outs(%0 : tensor<40x2047x30xi32>) -> tensor<40x2047x30xi32>
    %extracted_slice = tensor.extract_slice %2[4, 2046, 1] [8, 1, 9] [2, 1, 3] : tensor<40x2047x30xi32> to tensor<8x1x9xi32>
    // Last element is 2044 from extracted_slice
    // extracted_slice is banished
    // [.. 2044 2045 2046] [0] [0 1 2 3 4 5 6 ...]
    //     0    1    2      3   4 5 6 7 8 0
    // Next offset should be 5
    // Ends at 743 (last one, so arg2 size should at least be 744)
    %concat = tensor.concat dim(1) %arg1, %extracted_slice, %arg2 : ( tensor<8x2047x9xi32>, tensor<8x1x9xi32>, tensor<8x3400x9xi32>) -> tensor<8x5448x9xi32>
    %extracted_slice_0 = tensor.extract_slice %concat[3, 100, 0] [2, 300, 2] [2, 9, 3] : tensor<8x5448x9xi32> to tensor<2x300x2xi32>
    %broadcasted = linalg.broadcast ins(%extracted_slice_0 : tensor<2x300x2xi32>) outs(%1 : tensor<2x2047x300x2xi32>) dimensions = [1]
    return %broadcasted : tensor<2x2047x300x2xi32>
  }
}

// -----

// no fold on this one, because unverified
// CHECK-LABEL: func @breaker
// CHECK: extract_slice
// CHECK: concat
// CHECK: extract_slice
// CHECK: return

module {
  func.func @breaker(%arg0: tensor<40x1x2047x30xi64>, %arg1: tensor<8x2047x9xi32>, %arg2: tensor<8x743x9xi32>) -> tensor<2x2047x300x2xi32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<40x2047x30xi32>
    %1 = tensor.empty() : tensor<2x2047x300x2xi32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<40x1x2047x30xi64> into tensor<40x2047x30xi64>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<40x2047x30xi64>) outs(%0 : tensor<40x2047x30xi32>) -> tensor<40x2047x30xi32>
    %extracted_slice = tensor.extract_slice %2[4, 2046, 1] [8, 1, 9] [2, 1, 3] : tensor<40x2047x30xi32> to tensor<8x1x9xi32>
    // Last element is 2044 from extracted_slice
    // extracted_slice is banished
    // ??
    // [.. 2044 2045 2046] [0] [0 1 2 3 4 5 6 ...]
    //     0    1    2      3   4 5 6 7 8 0
    // Next offset should be 5
    // Ends at 743 (last one, so arg2 size should at least be 744)
    %concat = tensor.concat dim(1) %arg1, %extracted_slice, %arg2 : ( tensor<8x2047x9xi32>, tensor<8x1x9xi32>, tensor<8x743x9xi32>) -> tensor<8x2791x9xi32>
    %extracted_slice_0 = tensor.extract_slice %concat[3, 100, 0] [2, 300, 2] [2, 9, 3] : tensor<8x2791x9xi32> to tensor<2x300x2xi32>
    %broadcasted = linalg.broadcast ins(%extracted_slice_0 : tensor<2x300x2xi32>) outs(%1 : tensor<2x2047x300x2xi32>) dimensions = [1]
    return %broadcasted : tensor<2x2047x300x2xi32>
  }
}
