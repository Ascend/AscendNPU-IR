// RUN: bishengir-opt %s --normalize-last-dim-unaligned-tensor-op --split-input-file | FileCheck %s

// CHECK-LABEL: test_basic_concat_2d
func.func @test_basic_concat_2d(%A: tensor<16x31xf32>, %B: tensor<16x63xf32>) -> tensor<16x94xf32> {
  // CHECK: tensor.concat dim(0)
  %0 = tensor.concat dim(1) %A, %B : (tensor<16x31xf32>, tensor<16x63xf32>) ->  tensor<16x94xf32>
  func.return %0 : tensor<16x94xf32>
}

// -----

// CHECK-LABEL: test_basic_concat_unchanged
func.func @test_basic_concat_unchanged(%A: tensor<32x16xf32>, %B: tensor<63x16xf32>) -> tensor<95x16xf32> {
  // CHECK: tensor.concat dim(0)
  %0 = tensor.concat dim(0) %A, %B : (tensor<32x16xf32>, tensor<63x16xf32>) ->  tensor<95x16xf32>
  func.return %0 : tensor<95x16xf32>
}

// -----

// CHECK-LABEL: test_rank_1_concat
// CHECK-DAG: %[[brc0:.*]] = linalg.broadcast {{.*}} dimensions = [0]
// CHECK-DAG: %[[brc1:.*]] = linalg.broadcast {{.*}} dimensions = [0]
// CHECK-DAG: %[[transposed0:.*]] = linalg.transpose ins(%[[brc0]]
// CHECK-DAG: %[[transposed1:.*]] = linalg.transpose ins(%[[brc1]]
// CHECK: %[[concat:.*]] = tensor.concat dim(0) %[[transposed0]], %[[transposed1]]
// CHECK: %[[transposed2:.*]] = linalg.transpose ins(%[[concat]]
// CHECK: %[[slice:.*]] = tensor.extract_slice %[[transposed2]][0, 0] [1, 2047] [1, 1]
// CHECK: return %[[slice]]
func.func @test_rank_1_concat(%arg0: tensor<2046xf32>, %arg1: tensor<1xf32>) -> tensor<2047xf32> {
  %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<2046xf32>, tensor<1xf32>) -> tensor<2047xf32>
  return %concat : tensor<2047xf32>
}

// -----

// CHECK-LABEL: test_basic_pad_2d
func.func @test_basic_pad_2d(%A: tensor<16x27xf32>) -> tensor<16x32xf32> {
  %pad_value = arith.constant 0.0 : f32
  // CHECK: low[3, 0] high[2, 0]
  %0 = tensor.pad %A low[0, 3] high[0, 2] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %pad_value : f32
  } : tensor<16x27xf32> to tensor<16x32xf32>

  func.return %0: tensor<16x32xf32>
}

// -----

// CHECK-LABEL: test_basic_pad_unchanged
func.func @test_basic_pad_unchanged(%A: tensor<27x16xf32>) -> tensor<32x16xf32> {
  %pad_value = arith.constant 0.0 : f32
  // CHECK: low[3, 0] high[2, 0]
  %0 = tensor.pad %A low[3, 0] high[2, 0] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %pad_value : f32
  } : tensor<27x16xf32> to tensor<32x16xf32>

  func.return %0: tensor<32x16xf32>
}

// -----

// CHECK-LABEL: test_complex_pad_2d
func.func @test_complex_pad_2d(%A: tensor<16x27xf32>) -> tensor<16x32xf32> {
  %c3 = arith.constant 3 : index
  %pad_value = arith.constant 0.0 : f32
  %other_pad_value = arith.constant 1.0 : f32
  // CHECK: low[3, 0] high[2, 0]
  %0 = tensor.pad %A low[0, 3] high[0, 2] {
  ^bb0(%arg0: index, %arg1: index):
    %cmp = arith.cmpi sle, %arg1, %c3 : index
    %res = scf.if %cmp -> (f32) {
      scf.yield %other_pad_value : f32
    } else {
      scf.yield %pad_value : f32
    }
    tensor.yield %res : f32
  } : tensor<16x27xf32> to tensor<16x32xf32>

  func.return %0: tensor<16x32xf32>
}

// -----

// CHECK-LABEL: test_rank_1_pad
// CHECK: %[[brc:.*]] = linalg.broadcast {{.*}} dimensions = [0]
// CHECK: %[[transposed:.*]] = linalg.transpose ins(%[[brc]]
// CHECK: %[[pad:.*]] = tensor.pad %[[transposed]] low[2046, 0] high[0, 0]
// CHECK: %[[transposed0:.*]] = linalg.transpose ins(%[[pad]]
// CHECK: %[[slice:.*]] = tensor.extract_slice %[[transposed0]][0, 0] [1, 4093] [1, 1]
// CHECK: return %[[slice]]
func.func @test_rank_1_pad(%arg0: tensor<2047xf32>) -> tensor<4093xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %arg0 low[2046] high[0] {
  ^bb0(%arg1: index):
    tensor.yield %cst : f32
  } : tensor<2047xf32> to tensor<4093xf32>
  return %padded : tensor<4093xf32>
}

// -----

// CHECK-LABEL: test_dynamic_concat_0
// CHECK: linalg.transpose
// CHECK: linalg.transpose
// CHECK: tensor.concat
// CHECK: linalg.transpose
func.func @test_dynamic_concat_0(%arg0: tensor<32x?x32x1xf32>, %arg1: tensor<32x?x32x1xf32>) -> tensor<32x?x32x2xf32> {
  %concat = tensor.concat dim(3) %arg0, %arg1 : (tensor<32x?x32x1xf32>, tensor<32x?x32x1xf32>) -> tensor<32x?x32x2xf32>
  return %concat : tensor<32x?x32x2xf32>
}

// -----

// CHECK-LABEL: test_dynamic_concat_1
// CHECK: linalg.transpose
// CHECK: linalg.transpose
// CHECK: tensor.concat
// CHECK: linalg.transpose
func.func @test_dynamic_concat_1(%arg0: tensor<32x?x32x?xf32>, %arg1: tensor<32x?x32x?xf32>) -> tensor<32x?x32x?xf32> {
  %concat = tensor.concat dim(3) %arg0, %arg1 : (tensor<32x?x32x?xf32>, tensor<32x?x32x?xf32>) -> tensor<32x?x32x?xf32>
  return %concat : tensor<32x?x32x?xf32>
}

// -----

// CHECK-LABEL: test_dynamic_concat_2
// CHECK: linalg.transpose
// CHECK: linalg.transpose
// CHECK: tensor.concat dim(0) {{.*}} (tensor<?x1xf32>, tensor<?x1xf32>)
// CHECK: linalg.transpose
func.func @test_dynamic_concat_2(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<1024xf32> {
  %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<1024xf32>
  return %concat : tensor<1024xf32>
}

// -----

// CHECK-LABEL: test_dynamic_concat_with_annotation
// This test verifies that annotation.mark with buffer_size_in_byte is generated
// when operand shapes are dynamic but concat result shape is fully static.
// CHECK: %[[EMPTY0:.*]] = tensor.empty
// CHECK: annotation.mark %[[EMPTY0]] {buffer_size_in_byte = 6016 : i64}
// CHECK: linalg.transpose
// CHECK: %[[EMPTY1:.*]] = tensor.empty
// CHECK: annotation.mark %[[EMPTY1]] {buffer_size_in_byte = 6016 : i64}
// CHECK: linalg.transpose
// CHECK: tensor.concat
func.func @test_dynamic_concat_with_annotation(%arg0: tensor<?x31xf32>, %arg1: tensor<?x63xf32>) -> tensor<16x94xf32> {
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<?x31xf32>, tensor<?x63xf32>) -> tensor<16x94xf32>
  return %concat : tensor<16x94xf32>
}

// -----

// CHECK-LABEL: test_dynamic_concat_result_dynamic_no_annotation
// When concat result shape is also dynamic, no annotation.mark is generated
// because buffer_size_in_byte cannot be computed.
// CHECK: %[[EMPTY0:.*]] = tensor.empty
// CHECK-NOT: annotation.mark %[[EMPTY0]]
// CHECK: linalg.transpose
// CHECK: %[[EMPTY1:.*]] = tensor.empty
// CHECK-NOT: annotation.mark %[[EMPTY1]]
// CHECK: linalg.transpose
// CHECK: tensor.concat
func.func @test_dynamic_concat_result_dynamic_no_annotation(%arg0: tensor<?x31xf32>, %arg1: tensor<?x63xf32>) -> tensor<?x94xf32> {
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<?x31xf32>, tensor<?x63xf32>) -> tensor<?x94xf32>
  return %concat : tensor<?x94xf32>
}

// -----

// CHECK-LABEL: test_dynamic_concat_fully_static_no_annotation
// CHECK: %[[EMPTY0:.*]] = tensor.empty
// CHECK-NOT: annotation.mark %[[EMPTY0]]
// CHECK: linalg.transpose
// CHECK: %[[EMPTY1:.*]] = tensor.empty
// CHECK-NOT: annotation.mark %[[EMPTY1]]
// CHECK: linalg.transpose
// CHECK: tensor.concat
func.func @test_dynamic_concat_fully_static_no_annotation(%arg0: tensor<16x31xf32>, %arg1: tensor<16x63xf32>) -> tensor<16x94xf32> {
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<16x31xf32>, tensor<16x63xf32>) -> tensor<16x94xf32>
  return %concat : tensor<16x94xf32>
}
