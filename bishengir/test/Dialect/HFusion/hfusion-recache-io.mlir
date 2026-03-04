// RUN: bishengir-opt %s -hfusion-recache-io -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: @test_recache_unaligned_access
// CHECK-SAME: %[[arg0:.*]]: tensor<1x2048xi32>, %[[arg1:.*]]: tensor<1x2047x2047xf32>)
// CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[arg0]]
// CHECK-DAG: %[[load0:.*]] = hfusion.load ins(%[[collapsed]] : tensor<2048xi32>)
// CHECK-DAG: %[[extracted0:.*]] = tensor.extract_slice %[[collapsed]]{{\[}}1] {{\[}}2047] {{\[}}1]
// CHECK-DAG: %[[load1:.*]] = hfusion.load ins(%[[extracted0]] : tensor<2047xi32>)
// CHECK-DAG: %[[extracted1:.*]] = tensor.extract_slice %[[load0]]{{\[}}0] {{\[}}2047] {{\[}}1]
func.func @test_recache_unaligned_access(%arg0: tensor<1x2048xi32>, %arg1: tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %0 = tensor.empty() : tensor<2047x2047xi32>
  %1 = tensor.empty() : tensor<2047x2047xf32>
  %2 = tensor.empty() : tensor<2048xi32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2048xi32> into tensor<2048xi32>
  %3 = hfusion.load ins(%collapsed : tensor<2048xi32>) outs(%2 : tensor<2048xi32>) -> tensor<2048xi32>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x2047x2047xf32> into tensor<2047x2047xf32>
  %extracted_slice = tensor.extract_slice %3[1] [2047] [1] : tensor<2048xi32> to tensor<2047xi32>
  %broadcasted = linalg.broadcast ins(%extracted_slice : tensor<2047xi32>) outs(%0 : tensor<2047x2047xi32>) dimensions = [1] 
  %extracted_slice_1 = tensor.extract_slice %3[0] [2047] [1] : tensor<2048xi32> to tensor<2047xi32>
  %broadcasted_2 = linalg.broadcast ins(%extracted_slice_1 : tensor<2047xi32>) outs(%0 : tensor<2047x2047xi32>) dimensions = [0] 
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted, %broadcasted_2 : tensor<2047x2047xi32>, tensor<2047x2047xi32>) outs(%0 : tensor<2047x2047xi32>) -> tensor<2047x2047xi32>
  %5 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%4 : tensor<2047x2047xi32>) outs(%1 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
  %6 = hfusion.store ins(%5 : tensor<2047x2047xf32>) outs(%collapsed_0 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
  %expanded = tensor.expand_shape %6 [[0, 1], [2]] output_shape [1, 2047, 2047] : tensor<2047x2047xf32> into tensor<1x2047x2047xf32>
  return %expanded : tensor<1x2047x2047xf32>
}

// -----
// CHECK-LABEL: @recache_equal_split_two_slices
// CHECK: hfusion.load
// CHECK: hfusion.load
func.func @recache_equal_split_two_slices(%arg0: tensor<4096x36864xbf16>, %arg1: tensor<4096x18432xbf16>) -> tensor<4096x18432xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %0 = tensor.empty() : tensor<4096x36864xbf16>
  %1 = tensor.empty() : tensor<4096x18432xf32>
  %2 = tensor.empty() : tensor<4096x18432xbf16>
  %3 = hfusion.load ins(%arg0 : tensor<4096x36864xbf16>) outs(%0 : tensor<4096x36864xbf16>) -> tensor<4096x36864xbf16>
  %extracted_slice = tensor.extract_slice %3[0, 0] [4096, 18432] [1, 1] : tensor<4096x36864xbf16> to tensor<4096x18432xbf16>
  %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice : tensor<4096x18432xbf16>) outs(%1 : tensor<4096x18432xf32>) -> tensor<4096x18432xf32>
  %extracted_slice_0 = tensor.extract_slice %3[0, 18432] [4096, 18432] [1, 1] : tensor<4096x36864xbf16> to tensor<4096x18432xbf16>
  %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice_0 : tensor<4096x18432xbf16>) outs(%1 : tensor<4096x18432xf32>) -> tensor<4096x18432xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %5 : tensor<4096x18432xf32>, tensor<4096x18432xf32>) outs(%1 : tensor<4096x18432xf32>) -> tensor<4096x18432xf32>
  %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%6 : tensor<4096x18432xf32>) outs(%2 : tensor<4096x18432xbf16>) -> tensor<4096x18432xbf16>
  %8 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%7 : tensor<4096x18432xbf16>) outs(%arg1 : tensor<4096x18432xbf16>) -> tensor<4096x18432xbf16>
  return %8 : tensor<4096x18432xbf16>
}

// -----
// CHECK-LABEL: @recache_not_equal_split_two_slices
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @recache_not_equal_split_two_slices(%arg0: tensor<4096x36864xbf16>, %arg1: tensor<4096x18432xbf16>) -> (tensor<4096x10000xf32>, tensor<4096x26864xf32>)
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %0 = tensor.empty() : tensor<4096x36864xbf16>
  %1 = tensor.empty() : tensor<4096x10000xf32>
  %2 = tensor.empty() : tensor<4096x26864xf32>
  %3 = hfusion.load ins(%arg0 : tensor<4096x36864xbf16>) outs(%0 : tensor<4096x36864xbf16>) -> tensor<4096x36864xbf16>
  %extracted_slice = tensor.extract_slice %3[0, 0] [4096, 10000] [1, 1] : tensor<4096x36864xbf16> to tensor<4096x10000xbf16>
  %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice : tensor<4096x10000xbf16>) outs(%1 : tensor<4096x10000xf32>) -> tensor<4096x10000xf32>
  %extracted_slice_0 = tensor.extract_slice %3[0, 10000] [4096, 26864] [1, 1] : tensor<4096x36864xbf16> to tensor<4096x26864xbf16>
  %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice_0 : tensor<4096x26864xbf16>) outs(%2 : tensor<4096x26864xf32>) -> tensor<4096x26864xf32>
  return %4, %5 : tensor<4096x10000xf32>, tensor<4096x26864xf32>
}

// -----
// CHECK-LABEL: @recache_equal_split_three_slices
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK: hfusion.load
func.func @recache_equal_split_three_slices(%arg0: tensor<30x64xbf16>, %arg1: tensor<10x64xbf16>) -> tensor<10x64xbf16>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %0 = tensor.empty() : tensor<30x64xbf16>
  %1 = tensor.empty() : tensor<10x64xbf16>
  %2 = hfusion.load ins(%arg0 : tensor<30x64xbf16>) outs(%0 : tensor<30x64xbf16>) -> tensor<30x64xbf16>
  %extracted_slice = tensor.extract_slice %2[0, 0] [10, 64] [1, 1] : tensor<30x64xbf16> to tensor<10x64xbf16>
  %extracted_slice_0 = tensor.extract_slice %2[10, 0] [10, 64] [1, 1] : tensor<30x64xbf16> to tensor<10x64xbf16>
  %extracted_slice_1 = tensor.extract_slice %2[20, 0] [10, 64] [1, 1] : tensor<30x64xbf16> to tensor<10x64xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %extracted_slice_0 : tensor<10x64xbf16>, tensor<10x64xbf16>) outs(%1 : tensor<10x64xbf16>) -> tensor<10x64xbf16>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice_1, %3 : tensor<10x64xbf16>, tensor<10x64xbf16>) outs(%1 : tensor<10x64xbf16>) -> tensor<10x64xbf16>
  %5 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%4 : tensor<10x64xbf16>) outs(%arg1 : tensor<10x64xbf16>) -> tensor<10x64xbf16>
  return %5 : tensor<10x64xbf16>
}

// -----
// CHECK-LABEL: @recache_not_equal_split_three_slices
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @recache_not_equal_split_three_slices(%arg0: tensor<30x64xbf16>, %arg1: tensor<20x64xbf16>, %arg2: tensor<5x64xbf16>) -> (tensor<20x64xbf16>, tensor<5x64xbf16>)
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %0 = tensor.empty() : tensor<30x64xbf16>
  %1 = tensor.empty() : tensor<5x64xbf16>
  %2 = hfusion.load ins(%arg0 : tensor<30x64xbf16>) outs(%0 : tensor<30x64xbf16>) -> tensor<30x64xbf16>
  %extracted_slice = tensor.extract_slice %2[0, 0] [20, 64] [1, 1] : tensor<30x64xbf16> to tensor<20x64xbf16>
  %extracted_slice_0 = tensor.extract_slice %2[20, 0] [5, 64] [1, 1] : tensor<30x64xbf16> to tensor<5x64xbf16>
  %extracted_slice_1 = tensor.extract_slice %2[25, 0] [5, 64] [1, 1] : tensor<30x64xbf16> to tensor<5x64xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice_0, %extracted_slice_1 : tensor<5x64xbf16>, tensor<5x64xbf16>) outs(%1 : tensor<5x64xbf16>) -> tensor<5x64xbf16>
  %4 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%extracted_slice : tensor<20x64xbf16>) outs(%arg1 : tensor<20x64xbf16>) -> tensor<20x64xbf16>
  %5 = hfusion.store {hfusion.return_operand_num = 1 : i64} ins(%3 : tensor<5x64xbf16>) outs(%arg2 : tensor<5x64xbf16>) -> tensor<5x64xbf16>
  return %4, %5 : tensor<20x64xbf16>, tensor<5x64xbf16>
}

// -----
// CHECK-LABEL: @recache_multiple_split_dims
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @recache_multiple_split_dims(%arg0: tensor<30x64xbf16>, %arg1: tensor<10x32xbf16>) -> tensor<10x32xbf16>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %0 = tensor.empty() : tensor<30x64xbf16>
  %1 = tensor.empty() : tensor<10x32xbf16>
  %2 = hfusion.load ins(%arg0 : tensor<30x64xbf16>) outs(%0 : tensor<30x64xbf16>) -> tensor<30x64xbf16>
  %extracted_slice = tensor.extract_slice %2[0, 32] [10, 32] [1, 1] : tensor<30x64xbf16> to tensor<10x32xbf16>
  %extracted_slice_0 = tensor.extract_slice %2[10, 32] [10, 32] [1, 1] : tensor<30x64xbf16> to tensor<10x32xbf16>
  %extracted_slice_1 = tensor.extract_slice %2[20, 32] [10, 32] [1, 1] : tensor<30x64xbf16> to tensor<10x32xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %extracted_slice_0 : tensor<10x32xbf16>, tensor<10x32xbf16>) outs(%1 : tensor<10x32xbf16>) -> tensor<10x32xbf16>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice_1, %3 : tensor<10x32xbf16>, tensor<10x32xbf16>) outs(%1 : tensor<10x32xbf16>) -> tensor<10x32xbf16>
  %5 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%4 : tensor<10x32xbf16>) outs(%arg1 : tensor<10x32xbf16>) -> tensor<10x32xbf16>
  return %5 : tensor<10x32xbf16>
}

// -----
// CHECK-LABEL: @recache_partial_split_multiple_dims_0(
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @recache_partial_split_multiple_dims_0(%arg0: tensor<16x18432xf32>, %arg1: tensor<16x77x3072xf32>) -> tensor<16x77x3072xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %0 = tensor.empty() : tensor<16x18432xf32>
  %1 = tensor.empty() : tensor<16x77x3072xf32>
  %2 = hfusion.load {__intermediate_producer__} ins(%arg0 : tensor<16x18432xf32>) outs(%0 : tensor<16x18432xf32>) -> tensor<16x18432xf32>
  %extracted_slice = tensor.extract_slice %2[0, 6144] [16, 3072] [1, 1] : tensor<16x18432xf32> to tensor<16x3072xf32>
  %broadcasted = linalg.broadcast ins(%extracted_slice : tensor<16x3072xf32>) outs(%1 : tensor<16x77x3072xf32>) dimensions = [1]  {__intermediate_producer__}
  %extracted_slice_0 = tensor.extract_slice %2[0, 12288] [16, 3072] [1, 1] : tensor<16x18432xf32> to tensor<16x3072xf32>
  %broadcasted_1 = linalg.broadcast ins(%extracted_slice_0 : tensor<16x3072xf32>) outs(%1 : tensor<16x77x3072xf32>) dimensions = [1]  {__intermediate_producer__}
  %extracted_slice_2 = tensor.extract_slice %2[0, 9216] [16, 3072] [1, 1] : tensor<16x18432xf32> to tensor<16x3072xf32>
  %broadcasted_3 = linalg.broadcast ins(%extracted_slice_2 : tensor<16x3072xf32>) outs(%1 : tensor<16x77x3072xf32>) dimensions = [1]  {__intermediate_producer__}
  %3 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<add>} ins(%broadcasted, %broadcasted_1 : tensor<16x77x3072xf32>, tensor<16x77x3072xf32>) outs(%1 : tensor<16x77x3072xf32>) -> tensor<16x77x3072xf32>
  %4 = linalg.elemwise_binary {__intermediate_producer__, fun = #linalg.binary_fn<add>} ins(%3, %broadcasted_3 : tensor<16x77x3072xf32>, tensor<16x77x3072xf32>) outs(%1 : tensor<16x77x3072xf32>) -> tensor<16x77x3072xf32>
  %5 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%4 : tensor<16x77x3072xf32>) outs(%arg1 : tensor<16x77x3072xf32>) -> tensor<16x77x3072xf32>
  return %5 : tensor<16x77x3072xf32>
}

// -----

// CHECK-LABEL: @recache_exclusive_slices_0
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @recache_exclusive_slices_0(%arg0: tensor<100x4096xbf16>, %arg1: tensor<20x4096xbf16>) -> tensor<20x4096xbf16> {
  %0 = tensor.empty() : tensor<100x4096xbf16>
  %1 = tensor.empty() : tensor<20x4096xbf16>
  %2 = hfusion.load ins(%arg0 : tensor<100x4096xbf16>) outs(%0 : tensor<100x4096xbf16>) -> tensor<100x4096xbf16>
  %extracted_slice = tensor.extract_slice %2[0, 0] [20, 4096] [1, 1] : tensor<100x4096xbf16> to tensor<20x4096xbf16>
  %extracted_slice_0 = tensor.extract_slice %2[20, 0] [20, 4096] [1, 1] : tensor<100x4096xbf16> to tensor<20x4096xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_0 : tensor<20x4096xbf16>, tensor<20x4096xbf16>) outs(%1 : tensor<20x4096xbf16>) -> tensor<20x4096xbf16>
  %4 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%3 : tensor<20x4096xbf16>) outs(%arg1 : tensor<20x4096xbf16>) -> tensor<20x4096xbf16>
  return %4 : tensor<20x4096xbf16>
}

// -----

// CHECK-LABEL: @recache_exclusive_slices_1
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @recache_exclusive_slices_1(%arg0: tensor<100x4096xbf16>, %arg1: tensor<20x2048xbf16>) -> tensor<20x2048xbf16> {
  %0 = tensor.empty() : tensor<100x4096xbf16>
  %1 = tensor.empty() : tensor<20x2048xbf16>
  %2 = hfusion.load ins(%arg0 : tensor<100x4096xbf16>) outs(%0 : tensor<100x4096xbf16>) -> tensor<100x4096xbf16>
  %extracted_slice = tensor.extract_slice %2[0, 0] [20, 2048] [1, 1] : tensor<100x4096xbf16> to tensor<20x2048xbf16>
  %extracted_slice_0 = tensor.extract_slice %2[20, 2048] [20, 2048] [1, 1] : tensor<100x4096xbf16> to tensor<20x2048xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_0 : tensor<20x2048xbf16>, tensor<20x2048xbf16>) outs(%1 : tensor<20x2048xbf16>) -> tensor<20x2048xbf16>
  %4 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%3 : tensor<20x2048xbf16>) outs(%arg1 : tensor<20x2048xbf16>) -> tensor<20x2048xbf16>
  return %4 : tensor<20x2048xbf16>
}

// -----

// CHECK-LABEL: @recache_partial_overlap_slices_1
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @recache_partial_overlap_slices_1(%arg0: tensor<100x4096xbf16>, %arg1: tensor<20x4096xbf16>) -> tensor<20x4096xbf16> {
  %0 = tensor.empty() : tensor<100x4096xbf16>
  %1 = tensor.empty() : tensor<20x4096xbf16>
  %2 = hfusion.load ins(%arg0 : tensor<100x4096xbf16>) outs(%0 : tensor<100x4096xbf16>) -> tensor<100x4096xbf16>
  %extracted_slice = tensor.extract_slice %2[0, 0] [20, 4096] [1, 1] : tensor<100x4096xbf16> to tensor<20x4096xbf16>
  %extracted_slice_0 = tensor.extract_slice %2[10, 0] [20, 4096] [1, 1] : tensor<100x4096xbf16> to tensor<20x4096xbf16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_0 : tensor<20x4096xbf16>, tensor<20x4096xbf16>) outs(%1 : tensor<20x4096xbf16>) -> tensor<20x4096xbf16>
  %4 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%3 : tensor<20x4096xbf16>) outs(%arg1 : tensor<20x4096xbf16>) -> tensor<20x4096xbf16>
  return %4 : tensor<20x4096xbf16>
}


// -----

// CHECK-LABEL: @not_recache_full_overlap_slices_0
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @not_recache_full_overlap_slices_0(%arg0: tensor<100x4096xbf16>, %arg1: tensor<20x4096xbf16>, %arg2: tensor<5x4096xbf16>) -> (tensor<20x4096xbf16>, tensor<5x4096xbf16>) {
  %0 = tensor.empty() : tensor<100x4096xbf16>
  %1 = hfusion.load ins(%arg0 : tensor<100x4096xbf16>) outs(%0 : tensor<100x4096xbf16>) -> tensor<100x4096xbf16>
  %extracted_slice = tensor.extract_slice %1[0, 0] [20, 4096] [1, 1] : tensor<100x4096xbf16> to tensor<20x4096xbf16>
  %extracted_slice_0 = tensor.extract_slice %1[5, 0] [5, 4096] [1, 1] : tensor<100x4096xbf16> to tensor<5x4096xbf16>
  %2 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%extracted_slice : tensor<20x4096xbf16>) outs(%arg1 : tensor<20x4096xbf16>) -> tensor<20x4096xbf16>
  %3 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%extracted_slice_0 : tensor<5x4096xbf16>) outs(%arg2 : tensor<5x4096xbf16>) -> tensor<5x4096xbf16>
  return %2, %3 : tensor<20x4096xbf16>, tensor<5x4096xbf16>
}


// -----

// CHECK-LABEL: @not_recache_slices_with_all_same_ranges_0
// CHECK: hfusion.load
// CHECK-NOT: hfusion.load
func.func @not_recache_slices_with_all_same_ranges_0(%arg0: tensor<100x4096xbf16>, %arg1: tensor<20x4096xbf16>, %arg2: tensor<20x4096xbf16>) -> (tensor<20x4096xbf16>, tensor<20x4096xbf16>) {
  %0 = tensor.empty() : tensor<100x4096xbf16>
  %1 = hfusion.load ins(%arg0 : tensor<100x4096xbf16>) outs(%0 : tensor<100x4096xbf16>) -> tensor<100x4096xbf16>
  %extracted_slice = tensor.extract_slice %1[0, 0] [20, 4096] [1, 1] : tensor<100x4096xbf16> to tensor<20x4096xbf16>
  %extracted_slice_0 = tensor.extract_slice %1[0, 0] [20, 4096] [1, 1] : tensor<100x4096xbf16> to tensor<20x4096xbf16>
  %2 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%extracted_slice : tensor<20x4096xbf16>) outs(%arg1 : tensor<20x4096xbf16>) -> tensor<20x4096xbf16>
  %3 = hfusion.store {hfusion.return_operand_num = 0 : i64} ins(%extracted_slice_0 : tensor<20x4096xbf16>) outs(%arg2 : tensor<20x4096xbf16>) -> tensor<20x4096xbf16>
  return %2, %3 : tensor<20x4096xbf16>, tensor<20x4096xbf16>
}