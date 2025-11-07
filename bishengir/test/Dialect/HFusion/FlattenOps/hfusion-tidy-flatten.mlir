// RUN: bishengir-opt %s                              \
// RUN:   -pass-pipeline="builtin.module(func.func(   \
// RUN:      hfusion-flatten-ops{flatten-mode=tidy}), \
// RUN:      cse, canonicalize-ext)"                      \
// RUN:   -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @broadcast_mul_reduce(
// CHECK: linalg.broadcast
// CHECK-NOT: tensor.collapse_shape
func.func @broadcast_mul_reduce(%arg0: tensor<1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1024xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<1024x1024xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = linalg.broadcast ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024x1024xf32>) dimensions = [1]
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %4 = linalg.reduce ins(%3 : tensor<1024x1024xf32>) outs(%1 : tensor<1024xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
  return %4 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_mul_onebc(
// CHECK: %[[COLLAPSED0:.*]] = tensor.collapse_shape 
// CHECK-SAME{LITERAL}: [[0, 1]]
// CHECK-SAME: tensor<1024x1024xf32> into tensor<1048576xf32>
// CHECK: %[[OUT1:.*]] = tensor.collapse_shape %0 {{\[\[}}0, 1], {{\[}}2]] : tensor<1024x1024x1024xf32> into tensor<1048576x1024xf32>
// CHECK: %[[BC1:.*]] = linalg.broadcast ins(%[[COLLAPSED0]] : tensor<1048576xf32>) outs(%[[OUT1]] : tensor<1048576x1024xf32>)
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[BC1]]
// CHECK-SAME{LITERAL}: [[0, 1], [2]]
// CHECK-SAME{LITERAL}: tensor<1048576x1024xf32> into tensor<1024x1024x1024xf32>
// return %[[EXPANDED]] : tensor<1024x1024x1024xf32>
func.func @broadcast_mul_onebc(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024x1024xf32>) -> tensor<1024x1024x1024xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<1024x1024x1024xf32>
  %1 = tensor.empty() : tensor<1024x1024xf32>
  %2 = linalg.broadcast ins(%arg0 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024x1024xf32>) dimensions = [2]
  return %2 : tensor<1024x1024x1024xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_1(
// CHECK: tensor.collapse_shape %arg2
// CHECK-SAME: tensor<5x6x6xf32> into tensor<5x36xf32>
// CHECK: tensor.collapse_shape %arg1
// CHECK-SAME: tensor<6x6xf32> into tensor<36xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<5x6x6xf32> into tensor<5x36xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<5x36xf32> into tensor<5x6x6xf32>
// CHECK: %[[EXPANDED2:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<5x36xf32> into tensor<5x6x6xf32>
// return %[[EXPANDED]], %[[EXPANDED2:.*]] : tensor<5x6x6xf32>, tensor<5x6x6xf32>
func.func @complex_ops_1(%arg0: tensor<5x6x6xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<5x6x6xf32>) -> (tensor<5x6x6xf32>, tensor<5x6x6xf32>)
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<5x6x6xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg2 : tensor<5x6x6xf32>, tensor<5x6x6xf32>) outs(%0 : tensor<5x6x6xf32>) -> tensor<5x6x6xf32>
  
  %2 = tensor.empty() : tensor<6x6xf32>
  %3 = linalg.reduce ins(%arg0 : tensor<5x6x6xf32>) outs(%2 : tensor<6x6xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  %4 = tensor.empty() : tensor<6x6xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%4 : tensor<6x6xf32>) -> tensor<6x6xf32>
  
  %6 = tensor.empty() : tensor<5x6x6xf32>
  %7 = linalg.broadcast ins(%5 : tensor<6x6xf32>) outs(%6 : tensor<5x6x6xf32>) dimensions = [0]
  
  return %1, %7 : tensor<5x6x6xf32>, tensor<5x6x6xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_2(
// CHECK: tensor.collapse_shape %arg2
// CHECK-SAME: tensor<8x7x6x9xf32> into tensor<8x42x9xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<7x6xf32> into tensor<42xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<42xf32> into tensor<7x6xf32>
// CHECK: %[[EXPANDED2:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<42xf32> into tensor<7x6xf32>
// CHECK: return %[[EXPANDED]], %[[EXPANDED2]] : tensor<7x6xf32>, tensor<7x6xf32>
func.func @complex_ops_2(%arg0: tensor<7x6xf32>, %arg1: tensor<f32>, %arg2: tensor<8x7x6x9xf32>) -> (tensor<7x6xf32>, tensor<7x6xf32>)
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<7x6xf32>
  %1 = linalg.reduce ins(%arg2 : tensor<8x7x6x9xf32>) outs(%0 : tensor<7x6xf32>) dimensions = [0, 3]
      (%in: f32, %init: f32) {
        %7 = arith.addf %in, %init : f32
        linalg.yield %7 : f32
      }
  
  %2 = tensor.empty() : tensor<7x6xf32>
  %3 = linalg.broadcast ins(%arg1 : tensor<f32>) outs(%2 : tensor<7x6xf32>) dimensions = [0, 1]
  
  %4 = tensor.empty() : tensor<7x6xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %1 : tensor<7x6xf32>, tensor<7x6xf32>) outs(%4 : tensor<7x6xf32>) -> tensor<7x6xf32>
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %3 : tensor<7x6xf32>, tensor<7x6xf32>) outs(%4 : tensor<7x6xf32>) -> tensor<7x6xf32>
  
  return %5, %6 : tensor<7x6xf32>, tensor<7x6xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_3(
// CHECK: tensor.collapse_shape %arg2
// CHECK-SAME: tensor<9x8xf32> into tensor<72xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<10x9x8x7xf32> into tensor<10x72x7xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<10x72x7xf32> into tensor<10x9x8x7xf32>
// CHECK: return %[[EXPANDED]] : tensor<10x9x8x7xf32>
func.func @complex_ops_3(%arg0: tensor<10x9x8x7xf32>, %arg1: tensor<f32>, %arg2: tensor<9x8xf32>) -> tensor<10x9x8x7xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<10x9x8x7xf32>
  %1 = linalg.broadcast ins(%arg2 : tensor<9x8xf32>) outs(%0 : tensor<10x9x8x7xf32>) dimensions = [0, 3]
  
  %2 = tensor.empty() : tensor<10x9x8x7xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %1 : tensor<10x9x8x7xf32>, tensor<10x9x8x7xf32>) outs(%2 : tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32>
  
  %4 = tensor.empty() : tensor<10x9x8x7xf32>
  %5 = linalg.broadcast ins(%arg1 : tensor<f32>) outs(%4 : tensor<10x9x8x7xf32>) dimensions = [0, 1, 2, 3]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<10x9x8x7xf32>, tensor<10x9x8x7xf32>) outs(%4 : tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32>
  
  %7 = tensor.empty() : tensor<10x9x8xf32>
  %8 = linalg.reduce ins(%6 : tensor<10x9x8x7xf32>) outs(%7 : tensor<10x9x8xf32>) dimensions = [3]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  %10 = linalg.broadcast ins(%8 : tensor<10x9x8xf32>) outs(%4 : tensor<10x9x8x7xf32>) dimensions = [3]
  
  return %10 : tensor<10x9x8x7xf32>
}


// -----

// CHECK-LABEL: func.func @complex_ops_4(
// CHECK: tensor.collapse_shape %arg2
// CHECK-SAME: tensor<9x8xf32> into tensor<72xf32>
// CHECK: tensor.collapse_shape %arg1
// CHECK-SAME: tensor<1x2x4x9x8xf32> into tensor<8x72xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<10x9x8x7xf32> into tensor<10x72x7xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<10x72x7xf32> into tensor<10x9x8x7xf32>
// CHECK: return %[[EXPANDED]] : tensor<10x9x8x7xf32>
func.func @complex_ops_4(%arg0: tensor<10x9x8x7xf32>, %arg1: tensor<1x2x4x9x8xf32>, %arg2: tensor<9x8xf32>) -> tensor<10x9x8x7xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<10x9x8x7xf32>
  %1 = linalg.broadcast ins(%arg2 : tensor<9x8xf32>) outs(%0 : tensor<10x9x8x7xf32>) dimensions = [0, 3]
  
  %2 = tensor.empty() : tensor<10x9x8x7xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %1 : tensor<10x9x8x7xf32>, tensor<10x9x8x7xf32>) outs(%2 : tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32>
  
  %4 = tensor.empty() : tensor<1x2x4x6x9x8xf32>
  %tmp1 = linalg.broadcast ins(%arg1 : tensor<1x2x4x9x8xf32>) outs(%4 : tensor<1x2x4x6x9x8xf32>) dimensions = [3]

  %empty0 = tensor.empty() : tensor<9x8xf32>
  %tmp2 = linalg.reduce ins(%tmp1 : tensor<1x2x4x6x9x8xf32>) outs(%empty0 : tensor<9x8xf32>) dimensions = [0, 1, 2, 3]
      (%in: f32, %init: f32) {
        %xx = arith.addf %in, %init : f32
        linalg.yield %xx : f32
      }

  %empty1 = tensor.empty() : tensor<10x9x8x7xf32>
  %5 = linalg.broadcast ins(%tmp2 : tensor<9x8xf32>) outs(%empty1 : tensor<10x9x8x7xf32>) dimensions = [0, 3]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<10x9x8x7xf32>, tensor<10x9x8x7xf32>) outs(%empty1 : tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32>
  
  %7 = tensor.empty() : tensor<10x9x8xf32>
  %8 = linalg.reduce ins(%6 : tensor<10x9x8x7xf32>) outs(%7 : tensor<10x9x8xf32>) dimensions = [3]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  %10 = linalg.broadcast ins(%8 : tensor<10x9x8xf32>) outs(%empty1 : tensor<10x9x8x7xf32>) dimensions = [3]
  
  return %10 : tensor<10x9x8x7xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_5(
// CHECK: tensor.collapse_shape %arg1
// CHECK-SAME: tensor<12x10xf32> into tensor<120xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<15x12x10x8xf32> into tensor<15x120x8xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<15x120xf32> into tensor<15x12x10xf32>
// CHECK: return %[[EXPANDED]] : tensor<15x12x10xf32>
func.func @complex_ops_5(%arg0: tensor<15x12x10x8xf32>, %arg1: tensor<12x10xf32>, %arg2: tensor<f32>) -> tensor<15x12x10xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<15x12x10x8xf32>
  %1 = linalg.broadcast ins(%arg1 : tensor<12x10xf32>) outs(%0 : tensor<15x12x10x8xf32>) dimensions = [0, 3]
  
  %2 = tensor.empty() : tensor<15x12x10x8xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %1 : tensor<15x12x10x8xf32>, tensor<15x12x10x8xf32>) outs(%2 : tensor<15x12x10x8xf32>) -> tensor<15x12x10x8xf32>
  
  %4 = tensor.empty() : tensor<15x12x10x8xf32>
  %5 = linalg.broadcast ins(%arg2 : tensor<f32>) outs(%4 : tensor<15x12x10x8xf32>) dimensions = [0, 1, 2, 3]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<15x12x10x8xf32>, tensor<15x12x10x8xf32>) outs(%4 : tensor<15x12x10x8xf32>) -> tensor<15x12x10x8xf32>
  
  %7 = tensor.empty() : tensor<15x12x10xf32>
  %8 = linalg.reduce ins(%6 : tensor<15x12x10x8xf32>) outs(%7 : tensor<15x12x10xf32>) dimensions = [3]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  return %8 : tensor<15x12x10xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_6(
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @complex_ops_6(%arg0: tensor<5x4x3x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<5x3x2xf32>) -> tensor<5x4x3x2xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<5x4x3x2xf32>
  %1 = linalg.broadcast ins(%arg1 : tensor<4x3xf32>) outs(%0 : tensor<5x4x3x2xf32>) dimensions = [0, 3]
  
  %2 = tensor.empty() : tensor<5x4x3x2xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %1 : tensor<5x4x3x2xf32>, tensor<5x4x3x2xf32>) outs(%2 : tensor<5x4x3x2xf32>) -> tensor<5x4x3x2xf32>
  
  %4 = tensor.empty() : tensor<5x4x3x2xf32>
  %5 = linalg.broadcast ins(%arg2 : tensor<5x3x2xf32>) outs(%4 : tensor<5x4x3x2xf32>) dimensions = [1]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<5x4x3x2xf32>, tensor<5x4x3x2xf32>) outs(%4 : tensor<5x4x3x2xf32>) -> tensor<5x4x3x2xf32>
  
  %7 = tensor.empty() : tensor<5x4x3xf32>
  %8 = linalg.reduce ins(%6 : tensor<5x4x3x2xf32>) outs(%7 : tensor<5x4x3xf32>) dimensions = [3]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  %10 = linalg.broadcast ins(%8 : tensor<5x4x3xf32>) outs(%4 : tensor<5x4x3x2xf32>) dimensions = [3]
  
  return %10 : tensor<5x4x3x2xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_7(
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @complex_ops_7(%arg0: tensor<8x7x6x5x4xf32>, %arg1: tensor<7x6xf32>, %arg2: tensor<8x7x5x4xf32>) -> tensor<8x7x6x5xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<8x7x6x5x4xf32>
  %1 = linalg.broadcast ins(%arg1 : tensor<7x6xf32>) outs(%0 : tensor<8x7x6x5x4xf32>) dimensions = [0, 3, 4]
  
  %2 = tensor.empty() : tensor<8x7x6x5x4xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %1 : tensor<8x7x6x5x4xf32>, tensor<8x7x6x5x4xf32>) outs(%2 : tensor<8x7x6x5x4xf32>) -> tensor<8x7x6x5x4xf32>
  
  %4 = tensor.empty() : tensor<8x7x6x5x4xf32>
  %5 = linalg.broadcast ins(%arg2 : tensor<8x7x5x4xf32>) outs(%4 : tensor<8x7x6x5x4xf32>) dimensions = [2]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %5 : tensor<8x7x6x5x4xf32>, tensor<8x7x6x5x4xf32>) outs(%4 : tensor<8x7x6x5x4xf32>) -> tensor<8x7x6x5x4xf32>
  
  %7 = tensor.empty() : tensor<8x7x6x5xf32>
  %8 = linalg.reduce ins(%6 : tensor<8x7x6x5x4xf32>) outs(%7 : tensor<8x7x6x5xf32>) dimensions = [4]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  return %8 : tensor<8x7x6x5xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_8(
// CHECK: tensor.collapse_shape %arg2
// CHECK-SAME: tensor<6x5x4x2xf32> into tensor<30x4x2xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<6x5x4x3x2xf32> into tensor<30x4x3x2xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<30x4x3xf32> into tensor<6x5x4x3xf32>
// CHECK: return %[[EXPANDED]] : tensor<6x5x4x3xf32>
func.func @complex_ops_8(%arg0: tensor<6x5x4x3x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<6x5x4x2xf32>) -> tensor<6x5x4x3xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<6x5x4x3x2xf32>
  %1 = linalg.broadcast ins(%arg1 : tensor<4x3xf32>) outs(%0 : tensor<6x5x4x3x2xf32>) dimensions = [0, 1, 4]
  
  %2 = tensor.empty() : tensor<6x5x4x3x2xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %1 : tensor<6x5x4x3x2xf32>, tensor<6x5x4x3x2xf32>) outs(%2 : tensor<6x5x4x3x2xf32>) -> tensor<6x5x4x3x2xf32>
  
  %4 = tensor.empty() : tensor<6x5x4x3x2xf32>
  %5 = linalg.broadcast ins(%arg2 : tensor<6x5x4x2xf32>) outs(%4 : tensor<6x5x4x3x2xf32>) dimensions = [3]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<6x5x4x3x2xf32>, tensor<6x5x4x3x2xf32>) outs(%4 : tensor<6x5x4x3x2xf32>) -> tensor<6x5x4x3x2xf32>
  
  %7 = tensor.empty() : tensor<6x5x4x3xf32>
  %8 = linalg.reduce ins(%6 : tensor<6x5x4x3x2xf32>) outs(%7 : tensor<6x5x4x3xf32>) dimensions = [4]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  return %8 : tensor<6x5x4x3xf32>
}


// -----

// CHECK-LABEL: func.func @complex_ops_9(
// CHECK: tensor.collapse_shape %arg1
// CHECK-SAME: tensor<12x10xf32> into tensor<120xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<15x12x10x8xf32> into tensor<15x120x8xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<15x120xf32> into tensor<15x12x10xf32>
// CHECK: return %[[EXPANDED]] : tensor<15x12x10xf32>
func.func @complex_ops_9(%arg0: tensor<15x12x10x8xf32>, %arg1: tensor<12x10xf32>, %arg2: tensor<15x8xf32>) -> tensor<15x12x10xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<15x12x10x8xf32>
  %1 = linalg.broadcast ins(%arg1 : tensor<12x10xf32>) outs(%0 : tensor<15x12x10x8xf32>) dimensions = [0, 3]
  
  %2 = tensor.empty() : tensor<15x12x10x8xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %1 : tensor<15x12x10x8xf32>, tensor<15x12x10x8xf32>) outs(%2 : tensor<15x12x10x8xf32>) -> tensor<15x12x10x8xf32>
  
  %4 = tensor.empty() : tensor<15x12x10x8xf32>
  %5 = linalg.broadcast ins(%arg2 : tensor<15x8xf32>) outs(%4 : tensor<15x12x10x8xf32>) dimensions = [1, 2]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<15x12x10x8xf32>, tensor<15x12x10x8xf32>) outs(%4 : tensor<15x12x10x8xf32>) -> tensor<15x12x10x8xf32>
  
  %7 = tensor.empty() : tensor<15x12x10xf32>
  %8 = linalg.reduce ins(%6 : tensor<15x12x10x8xf32>) outs(%7 : tensor<15x12x10xf32>) dimensions = [3]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  return %8 : tensor<15x12x10xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_10(
// CHECK: tensor.collapse_shape %arg2
// CHECK-SAME: tensor<6x5x4x2xf32> into tensor<6x20x2xf32>
// CHECK: tensor.collapse_shape %arg1
// CHECK-SAME: tensor<5x4x3xf32> into tensor<20x3xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<6x5x4x3x2xf32> into tensor<6x20x3x2xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<6x20x3xf32> into tensor<6x5x4x3xf32>
// CHECK: return %[[EXPANDED]] : tensor<6x5x4x3xf32>
func.func @complex_ops_10(%arg0: tensor<6x5x4x3x2xf32>, %arg1: tensor<5x4x3xf32>, %arg2: tensor<6x5x4x2xf32>) -> tensor<6x5x4x3xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<6x5x4x3x2xf32>
  %1 = linalg.broadcast ins(%arg1 : tensor<5x4x3xf32>) outs(%0 : tensor<6x5x4x3x2xf32>) dimensions = [0, 4]
  
  %2 = tensor.empty() : tensor<6x5x4x3x2xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %1 : tensor<6x5x4x3x2xf32>, tensor<6x5x4x3x2xf32>) outs(%2 : tensor<6x5x4x3x2xf32>) -> tensor<6x5x4x3x2xf32>
  
  %4 = tensor.empty() : tensor<6x5x4x3x2xf32>
  %5 = linalg.broadcast ins(%arg2 : tensor<6x5x4x2xf32>) outs(%4 : tensor<6x5x4x3x2xf32>) dimensions = [3]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<6x5x4x3x2xf32>, tensor<6x5x4x3x2xf32>) outs(%4 : tensor<6x5x4x3x2xf32>) -> tensor<6x5x4x3x2xf32>
  
  %7 = tensor.empty() : tensor<6x5x4x3xf32>
  %8 = linalg.reduce ins(%6 : tensor<6x5x4x3x2xf32>) outs(%7 : tensor<6x5x4x3xf32>) dimensions = [4]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  return %8 : tensor<6x5x4x3xf32>
}

// -----

// CHECK-LABEL: func.func @complex_ops_11(
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @complex_ops_11(%arg0: tensor<8x7x6x5x4xf32>, %arg1: tensor<7x6xf32>, %arg2: tensor<8x7x5x4xf32>) -> tensor<8x7x6x5xf32>
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<8x7x6x5x4xf32>
  %1 = linalg.broadcast ins(%arg1 : tensor<7x6xf32>) outs(%0 : tensor<8x7x6x5x4xf32>) dimensions = [0, 3, 4]
  
  %2 = tensor.empty() : tensor<8x7x6x5x4xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %1 : tensor<8x7x6x5x4xf32>, tensor<8x7x6x5x4xf32>) outs(%2 : tensor<8x7x6x5x4xf32>) -> tensor<8x7x6x5x4xf32>
  
  %4 = tensor.empty() : tensor<8x7x6x5x4xf32>
  %5 = linalg.broadcast ins(%arg2 : tensor<8x7x5x4xf32>) outs(%4 : tensor<8x7x6x5x4xf32>) dimensions = [2]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %5 : tensor<8x7x6x5x4xf32>, tensor<8x7x6x5x4xf32>) outs(%4 : tensor<8x7x6x5x4xf32>) -> tensor<8x7x6x5x4xf32>
  
  %7 = tensor.empty() : tensor<8x7x6x5xf32>
  %8 = linalg.reduce ins(%6 : tensor<8x7x6x5x4xf32>) outs(%7 : tensor<8x7x6x5xf32>) dimensions = [4]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
  
  return %8 : tensor<8x7x6x5xf32>
}

// -----
// CHECK-LABEL: func.func @complex_ops_12(
// CHECK: tensor.collapse_shape %arg1
// CHECK-SAME: tensor<12x10xf32> into tensor<120xf32>
// CHECK: tensor.collapse_shape %arg0
// CHECK-SAME: tensor<15x12x10x8x9x6xf32> into tensor<15x120x8x54xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<15x120xf32> into tensor<15x12x10xf32>
// CHECK: %[[EXPANDED2:.*]] = tensor.expand_shape
// CHECK-SAME: tensor<120x8x54xf32> into tensor<12x10x8x9x6xf32>
// CHECK: return %[[EXPANDED]], %[[EXPANDED2]] : tensor<15x12x10xf32>, tensor<12x10x8x9x6xf32>
func.func @complex_ops_12(%arg0: tensor<15x12x10x8x9x6xf32>, %arg1: tensor<12x10xf32>, %arg2: tensor<15x8xf32>) -> (tensor<15x12x10xf32>, tensor<12x10x8x9x6xf32>)
attributes {hivm.entry, hivm.dso_local, hivm.spir_kernel} {
  %0 = tensor.empty() : tensor<15x12x10x8x9x6xf32>
  %1 = linalg.broadcast ins(%arg1 : tensor<12x10xf32>) outs(%0 : tensor<15x12x10x8x9x6xf32>) dimensions = [0, 3, 4, 5]
  
  %2 = tensor.empty() : tensor<15x12x10x8x9x6xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %1 : tensor<15x12x10x8x9x6xf32>, tensor<15x12x10x8x9x6xf32>) outs(%2 : tensor<15x12x10x8x9x6xf32>) -> tensor<15x12x10x8x9x6xf32>
  
  %4 = tensor.empty() : tensor<15x12x10x8x9x6xf32>
  %5 = linalg.broadcast ins(%arg2 : tensor<15x8xf32>) outs(%4 : tensor<15x12x10x8x9x6xf32>) dimensions = [1, 2, 4, 5]
  
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %5 : tensor<15x12x10x8x9x6xf32>, tensor<15x12x10x8x9x6xf32>) outs(%4 : tensor<15x12x10x8x9x6xf32>) -> tensor<15x12x10x8x9x6xf32>
  
  %7 = tensor.empty() : tensor<15x12x10xf32>
  %8 = linalg.reduce ins(%6 : tensor<15x12x10x8x9x6xf32>) outs(%7 : tensor<15x12x10xf32>) dimensions = [3, 4, 5]
      (%in: f32, %init: f32) {
        %inside = arith.addf %in, %init : f32
        linalg.yield %inside : f32
      }

  %9 = tensor.empty() : tensor<12x10x8x9x6xf32>
  %10 = linalg.reduce ins(%6 : tensor<15x12x10x8x9x6xf32>) outs(%9 : tensor<12x10x8x9x6xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %inside = arith.addf %in, %init : f32
        linalg.yield %inside : f32
      }
  
  return %8, %10 : tensor<15x12x10xf32>, tensor<12x10x8x9x6xf32>
}

// -----
// CHECK-LABEL: func.func @fused_with_constant(
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1, 2]]
// CHECK-SAME: tensor<4x3072x3072xf32> into tensor<37748736xf32>
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1, 2]]
// CHECK-SAME: tensor<4x3072x3072xf32> into tensor<37748736xf32>
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1, 2]] output_shape {{\[}}4, 3072, 3072]
// CHECK-SAME: tensor<37748736xf32> into tensor<4x3072x3072xf32>
// CHECK: return
func.func @fused_with_constant(%arg0: tensor<4x3072x3072xf32>, %arg1: tensor<4x3072x3072xf32>) -> tensor<4x3072x3072xf32> attributes {OperatorType = "Default", compute_capability = "", frontend_symbol = {input_0 = ["4", "3072", "3072"], output_0 = ["4", "3072", "3072"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant dense<0.0441941731> : tensor<f32>
  %0 = tensor.empty() : tensor<4x3072x3072xf32>
  %broadcasted = linalg.broadcast ins(%cst : tensor<f32>) outs(%0 : tensor<4x3072x3072xf32>) dimensions = [0, 1, 2]
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %broadcasted : tensor<4x3072x3072xf32>, tensor<4x3072x3072xf32>) outs(%arg1 : tensor<4x3072x3072xf32>) -> tensor<4x3072x3072xf32>
  return %1 : tensor<4x3072x3072xf32>
}

// -----

// CHECK-LABEL: func.func @fused_with_real_constant2(
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1]]
// CHECK-SAME: tensor<200x200xf32> into tensor<40000xf32>
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1]]
// CHECK-SAME: tensor<200x200xf32> into tensor<40000xf32>
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1]] output_shape {{\[}}200, 200]
// CHECK-SAME: tensor<40000xf32> into tensor<200x200xf32>
// CHECK: return
func.func @fused_with_real_constant2(%arg0: tensor<200x200xf32>, %arg1: tensor<200x200xf32>) -> tensor<200x200xf32> attributes {OperatorType = "Broadcast", compute_capability = "", frontend_symbol = {input_0 = ["200", "200"], output_0 = ["200", "200"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant dense<8.000000e+00> : tensor<f32>
  %cst_0 = arith.constant dense<2.885390e+00> : tensor<f32>
  %cst_1 = arith.constant dense<1.250000e-01> : tensor<f32>
  %0 = tensor.empty() : tensor<200x200xf32>
  %broadcasted = linalg.broadcast ins(%cst_1 : tensor<f32>) outs(%0 : tensor<200x200xf32>) dimensions = [0, 1]
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %broadcasted : tensor<200x200xf32>, tensor<200x200xf32>) outs(%0 : tensor<200x200xf32>) -> tensor<200x200xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%1 : tensor<200x200xf32>) outs(%0 : tensor<200x200xf32>) -> tensor<200x200xf32>
  %broadcasted_2 = linalg.broadcast ins(%cst_0 : tensor<f32>) outs(%0 : tensor<200x200xf32>) dimensions = [0, 1]
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %broadcasted_2 : tensor<200x200xf32>, tensor<200x200xf32>) outs(%0 : tensor<200x200xf32>) -> tensor<200x200xf32>
  %broadcasted_3 = linalg.broadcast ins(%cst : tensor<f32>) outs(%0 : tensor<200x200xf32>) dimensions = [0, 1]
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted_3, %3 : tensor<200x200xf32>, tensor<200x200xf32>) outs(%arg1 : tensor<200x200xf32>) -> tensor<200x200xf32>
  return %4 : tensor<200x200xf32>
}

// -----

// CHECK-LABEL: func.func @check_without_collapse(
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1], {{\[}}2, 3]]
// CHECK-SAME: tensor<2x4x768x1152xf32> into tensor<8x884736xf32>
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1]]
// CHECK-SAME: tensor<768x1152xf32> into tensor<884736xf32>
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1], {{\[}}2, 3]]
// CHECK-SAME: tensor<2x4x768x1152xf32> into tensor<8x884736xf32>
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1], {{\[}}2, 3]] output_shape {{\[}}2, 4, 768, 1152]
// CHECK-SAME: tensor<8x884736xf32> into tensor<2x4x768x1152xf32>
// CHECK: return
func.func @check_without_collapse (%arg0: tensor<2x4x768x1152xf32>, %arg1: tensor<768x1152xf32>, %arg2: tensor<2x4x768x1152xf32>) -> (tensor<2x4x768x1152xf32>) {
  %0 = tensor.empty() : tensor<2x4x768x1152xf32>
  %broadcasted = linalg.broadcast ins(%arg1 : tensor<768x1152xf32>) outs(%0 : tensor<2x4x768x1152xf32>) dimensions = [0, 1]
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %broadcasted : tensor<2x4x768x1152xf32>, tensor<2x4x768x1152xf32>) outs (%arg2: tensor<2x4x768x1152xf32>) -> tensor<2x4x768x1152xf32>
  return %1 : tensor<2x4x768x1152xf32>
}


// -----
// CHECK-LABEL: func.func @check_with_collapse(
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1], {{\[}}2, 3]]
// CHECK-SAME: tensor<2x4x768x1152xf32> into tensor<8x884736xf32>
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1], {{\[}}2, 3]]
// CHECK-SAME: tensor<2x4x768x1152xf32> into tensor<8x884736xf32>
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1, 2]]
// CHECK-SAME: tensor<1x768x1152xf32> into tensor<884736xf32>
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1], {{\[}}2, 3]] output_shape {{\[}}2, 4, 768, 1152]
// CHECK-SAME: tensor<8x884736xf32> into tensor<2x4x768x1152xf32>
// CHECK: return
func.func @check_with_collapse (%arg0: tensor<2x4x768x1152xf32>, %arg1: tensor<1x768x1152xf32>, %arg2: tensor<2x4x768x1152xf32>) -> (tensor<2x4x768x1152xf32>) {
  %collapsed = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor <1x768x1152xf32> into tensor <768x1152xf32>
  %0 = tensor.empty() : tensor<2x4x768x1152xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<768x1152xf32>) outs(%0 : tensor<2x4x768x1152xf32>) dimensions = [0, 1]
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %broadcasted : tensor<2x4x768x1152xf32>, tensor<2x4x768x1152xf32>) outs (%arg2: tensor<2x4x768x1152xf32>) -> tensor<2x4x768x1152xf32>
  return %1 : tensor<2x4x768x1152xf32>
}

// -----
// CHECK-LABEL: func.func @elemwise_constant(
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1]] : tensor<5x1xf32> into tensor<5xf32>
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1]] : tensor<5x1xf32> into tensor<5xf32>
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1]] output_shape {{\[}}5, 1] : tensor<5xf32> into tensor<5x1xf32>
// CHECK: return
func.func @elemwise_constant(%arg0: tensor<5x1xf32>, %arg1: tensor<5x1xf32>) -> tensor<5x1xf32> attributes {hacc.entry} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %cst : tensor<5x1xf32>, f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %3 : f32, tensor<5x1xf32>) outs(%arg1 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %4 : tensor<5x1xf32>
}


// -----
// CHECK-LABEL: func.func @triton_cast_compare(
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @triton_cast_compare(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {global_kernel = "local"} {
  %cst = arith.constant 1.000000e-01 : f32
  %0 = tensor.empty() : tensor<2xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2xf32>) -> tensor<2xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2], strides: [1] : memref<?xf32> to memref<2xf32, strided<[1]>>
  %alloc = memref.alloc() : memref<2xf32>
  memref.copy %reinterpret_cast, %alloc : memref<2xf32, strided<[1]>> to memref<2xf32>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<2xf32>
  %3 = tensor.empty() : tensor<2xi1>
  %4 = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%2, %1 : tensor<2xf32>, tensor<2xf32>) outs(%3 : tensor<2xi1>) -> tensor<2xi1>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [2], strides: [1] : memref<?xi8> to memref<2xi8, strided<[1]>>
  %5 = tensor.empty() : tensor<2xi8>
  %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<2xi1>) outs(%5 : tensor<2xi8>) -> tensor<2xi8>
  bufferization.materialize_in_destination %6 in writable %reinterpret_cast_0 : (tensor<2xi8>, memref<2xi8, strided<[1]>>) -> ()
  return
}

// -----
// CHECK: func.func @fn_cache_slice(
// CHECK: return
func.func @fn_cache_slice(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {global_kernel = "local"} {
  %cst = arith.constant 2.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x8xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16x8xf32>) -> tensor<16x8xf32>
  %2 = tensor.empty() : tensor<16x8xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16x8xf32>) -> tensor<16x8xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 8], strides: [8, 1] : memref<?xf32> to memref<16x8xf32, strided<[8, 1]>>
  %alloc = memref.alloc() : memref<16x8xf32>
  memref.copy %reinterpret_cast, %alloc : memref<16x8xf32, strided<[8, 1]>> to memref<16x8xf32>
  %4 = bufferization.to_tensor %alloc restrict writable : memref<16x8xf32>
  %5 = tensor.empty() : tensor<16x8xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %3 : tensor<16x8xf32>, tensor<16x8xf32>) outs(%5 : tensor<16x8xf32>) -> tensor<16x8xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 8], strides: [8, 1] : memref<?xf32> to memref<16x8xf32, strided<[8, 1]>>
  bufferization.materialize_in_destination %6 in writable %reinterpret_cast_1 : (tensor<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) -> ()
  %extracted_slice = tensor.extract_slice %6[4, 0] [1, 8] [1, 1] : tensor<16x8xf32> to tensor<1x8xf32>
  %inserted_slice = tensor.insert_slice %extracted_slice into %1[4, 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<16x8xf32>
  %7 = tensor.empty() : tensor<8xf32>
  %reduced = linalg.reduce ins(%inserted_slice : tensor<16x8xf32>) outs(%7 : tensor<8xf32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %8 = arith.addf %in, %init : f32
      linalg.yield %8 : f32
    }
  %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
  bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_2 : (tensor<8xf32>, memref<8xf32, strided<[1]>>) -> ()
  return
}

// -----
// CHECK-LABEL: func.func @triton_test_fn_min_with_index_inner(
// CHECK-NOT: tensor<2x16x128xf16> into tensor<32x128xf16>
// CHECK-NOT: tensor<2x16x128xi32> into tensor<32x128xi32>
// CHECK-NOT: tensor<2x16xf16> into tensor<32xf16>
// CHECK-NOT: tensor<2x16xi32> into tensor<32xi32>
// CHECK-NOT: tensor<32xf16> into tensor<2x16xf16>
// CHECK: return
func.func @triton_test_fn_min_with_index_inner(%arg0: memref<?x?xf16> {tt.divisibility = 16 : i32}, %arg1: memref<?xi32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xi32>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {global_kernel = "local"} {
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<[2, 16, 128]> : tensor<3xi64>
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c4096 = arith.constant 4096 : index
  %true = arith.constant true
  %c4096_i32 = arith.constant 4096 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = arith.muli %arg9, %c4096_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [4096], strides: [1] : memref<?xf16> to memref<4096xf16, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<4096xf16>
  %2 = arith.addi %1, %c4096 : index
  %3 = arith.index_cast %arg4 : i32 to index
  %4 = arith.maxsi %1, %3 : index
  %5 = arith.minsi %2, %4 : index
  %6 = arith.subi %5, %1 : index
  %7 = arith.cmpi slt, %6, %c4096 : index
  scf.if %7 {
    linalg.fill ins(%cst_0 : f16) outs(%alloc : memref<4096xf16>)
  }
  %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
  %subview_1 = memref.subview %alloc[0] [%6] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
  memref.copy %subview, %subview_1 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
  %8 = bufferization.to_tensor %alloc restrict writable : memref<4096xf16>
  %reshape = tensor.reshape %8(%cst) : (tensor<4096xf16>, tensor<3xi64>) -> tensor<2x16x128xf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [4096], strides: [1] : memref<?xi32> to memref<4096xi32, strided<[1], offset: ?>>
  %alloc_3 = memref.alloc() : memref<4096xi32>
  scf.if %7 {
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_3 : memref<4096xi32>)
  }
  %subview_4 = memref.subview %reinterpret_cast_2[0] [%6] [1] : memref<4096xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
  %subview_5 = memref.subview %alloc_3[0] [%6] [1] : memref<4096xi32> to memref<?xi32, strided<[1]>>
  memref.copy %subview_4, %subview_5 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
  %9 = bufferization.to_tensor %alloc_3 restrict writable : memref<4096xi32>
  %reshape_6 = tensor.reshape %9(%cst) : (tensor<4096xi32>, tensor<3xi64>) -> tensor<2x16x128xi32>
  %10 = tensor.empty() : tensor<2x16xf16>
  %11 = tensor.empty() : tensor<2x16xi32>
  %reduced:2 = linalg.reduce ins(%reshape, %reshape_6 : tensor<2x16x128xf16>, tensor<2x16x128xi32>) outs(%10, %11 : tensor<2x16xf16>, tensor<2x16xi32>) dimensions = [2]
    (%in: f16, %in_12: i32, %init: f16, %init_13: i32) {
      %19 = arith.cmpf olt, %in, %init : f16
      %20 = arith.cmpf oeq, %in, %init : f16
      %21 = arith.cmpf une, %in, %in : f16
      %22 = arith.cmpf une, %init, %init : f16
      %23 = arith.xori %22, %true : i1
      %24 = arith.andi %21, %23 : i1
      %25 = arith.ori %19, %24 : i1
      %26 = arith.andi %21, %22 : i1
      %27 = arith.ori %20, %26 : i1
      %28 = arith.cmpi slt, %in_12, %init_13 : i32
      %29 = arith.andi %27, %28 : i1
      %30 = arith.ori %25, %29 : i1
      %31 = arith.select %30, %in, %init : f16
      %32 = arith.select %30, %in_12, %init_13 : i32
      linalg.yield %31, %32 : f16, i32
    }
  %12 = arith.muli %arg9, %c32_i32 : i32
  %13 = arith.index_cast %12 : i32 to index
  %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [2, 16], strides: [1, 1] : memref<?x?xf16> to memref<2x16xf16, strided<[1, 1], offset: ?>>
  %14 = arith.addi %13, %c32 : index
  %15 = arith.index_cast %arg5 : i32 to index
  %16 = arith.maxsi %13, %15 : index
  %17 = arith.minsi %14, %16 : index
  %18 = arith.subi %17, %13 : index
  %extracted_slice = tensor.extract_slice %reduced#0[0, 0] [%18, %18] [1, 1] : tensor<2x16xf16> to tensor<?x?xf16>
  %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [%18, %18] [1, 1] : memref<2x16xf16, strided<[1, 1], offset: ?>> to memref<?x?xf16, strided<[1, 1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_8 : (tensor<?x?xf16>, memref<?x?xf16, strided<[1, 1], offset: ?>>) -> ()
  %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [2, 16], strides: [1, 1] : memref<?xi32> to memref<2x16xi32, strided<[1, 1], offset: ?>>
  %extracted_slice_10 = tensor.extract_slice %reduced#1[0, 0] [%18, %18] [1, 1] : tensor<2x16xi32> to tensor<?x?xi32>
  %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%18, %18] [1, 1] : memref<2x16xi32, strided<[1, 1], offset: ?>> to memref<?x?xi32, strided<[1, 1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice_10 in writable %subview_11 : (tensor<?x?xi32>, memref<?x?xi32, strided<[1, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @permutation_linalg(
// CHECK: tensor<2x16x8x4x3xf32> into tensor<32x8x4x3xf32>
// CHECK: tensor<2x16x4x8x3xf32> into tensor<32x4x8x3xf32>
// CHECK: tensor<32x4x8x3xf32> into tensor<2x16x4x8x3xf32>
func.func @permutation_linalg(%arg0: tensor<2x16x8x4x3xf32>) -> tensor<2x16x4x8x3xf32> {
  %0 = tensor.empty() : tensor<2x16x4x8x3xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<2x16x8x4x3xf32>) outs(%0 : tensor<2x16x4x8x3xf32>) permutation = [0, 1, 3, 2, 4]
  return %1 : tensor<2x16x4x8x3xf32>
}

// -----

// CHECK-LABEL: func.func @permutation_consecutives(
// CHECK-NOT: collapse_shape
func.func @permutation_consecutives(%arg0: tensor<1x16x8xf32>) -> tensor<1x16x8xf32> {
  %0 = tensor.empty() : tensor<16x1x8xf32>
  %1 = tensor.empty() : tensor<8x1x16xf32>
  %2 = tensor.empty() : tensor<1x8x16xf32>
  %3 = tensor.empty() : tensor<1x16x8xf32>
  %4 = linalg.transpose ins(%arg0 : tensor<1x16x8xf32>) outs(%0 : tensor<16x1x8xf32>) permutation = [1, 0, 2]
  %5 = linalg.transpose ins(%4 : tensor<16x1x8xf32>) outs(%1 : tensor<8x1x16xf32>) permutation = [2, 1, 0]
  %6 = linalg.transpose ins(%5 : tensor<8x1x16xf32>) outs(%2 : tensor<1x8x16xf32>) permutation = [1, 0, 2] 
  %7 = linalg.transpose ins(%6 : tensor<1x8x16xf32>) outs(%3 : tensor<1x16x8xf32>) permutation = [0, 2, 1]
  return %7 : tensor<1x16x8xf32>
}


// -----

// CHECK-LABEL: func.func @permutation_linalg_much(
// CHECK: tensor<2x16x8x4x3xf32> into tensor<32x8x4x3xf32>
// CHECK: tensor<2x16x3x4x8xf32> into tensor<32x3x4x8xf32>
// CHECK: tensor<32x3x4x8xf32> into tensor<2x16x3x4x8xf32>
func.func @permutation_linalg_much(%arg0: tensor<2x16x8x4x3xf32>) -> tensor<2x16x3x4x8xf32> {
  %0 = tensor.empty() : tensor<2x16x3x4x8xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<2x16x8x4x3xf32>) outs(%0 : tensor<2x16x3x4x8xf32>) permutation = [0, 1, 4, 3, 2]
  return %1 : tensor<2x16x3x4x8xf32>
}

// -----

// CHECK-LABEL: func.func @fn_2d_trans(
// CHECK: linalg.transpose
// CHECK-NOT: expand_shape
// CHECK: return
func.func @fn_2d_trans(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {global_kernel = "local"} {
  %c768_i64 = arith.constant 768 : i64
  %cst = arith.constant dense<[48, 16]> : tensor<2xi64>
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [768], strides: [1] : memref<?xf32> to memref<768xf32, strided<[1]>>
  %alloc = memref.alloc() : memref<768xf32>
  memref.copy %reinterpret_cast, %alloc : memref<768xf32, strided<[1]>> to memref<768xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<768xf32>
  %reshape = tensor.reshape %0(%cst) : (tensor<768xf32>, tensor<2xi64>) -> tensor<48x16xf32>
  %1 = tensor.empty() : tensor<16x48xf32>
  %transposed = linalg.transpose ins(%reshape : tensor<48x16xf32>) outs(%1 : tensor<16x48xf32>) permutation = [1, 0]
  %2 = tensor.empty() : tensor<1xi64>
  %3 = linalg.fill ins(%c768_i64 : i64) outs(%2 : tensor<1xi64>) -> tensor<1xi64>
  %reshape_0 = tensor.reshape %transposed(%3) : (tensor<16x48xf32>, tensor<1xi64>) -> tensor<768xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [768], strides: [1] : memref<?xf32> to memref<768xf32, strided<[1]>>
  bufferization.materialize_in_destination %reshape_0 in writable %reinterpret_cast_1 : (tensor<768xf32>, memref<768xf32, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @fused_constant_args(
// CHECK: tensor<1xf32> into tensor<f32>
// CHECK: return
func.func @fused_constant_args(%arg0: tensor<1xf32>, %arg1: tensor<24xf32>, %arg2: tensor<24xf32>) -> tensor<24xf32> attributes {OperatorType = "Broadcast", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant dense<1.16666663> : tensor<f32>
  %collapsed = tensor.collapse_shape %arg0 [] : tensor<1xf32> into tensor<f32>
  %0 = tensor.empty() : tensor<24xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<f32>) outs(%0 : tensor<24xf32>) dimensions = [0]
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg1, %broadcasted : tensor<24xf32>, tensor<24xf32>) outs(%0 : tensor<24xf32>) -> tensor<24xf32>
  %broadcasted_0 = linalg.broadcast ins(%cst : tensor<f32>) outs(%0 : tensor<24xf32>) dimensions = [0]
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %broadcasted_0 : tensor<24xf32>, tensor<24xf32>) outs(%arg2 : tensor<24xf32>) -> tensor<24xf32>
  return %2 : tensor<24xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_add_mul(
// CHECK: tensor<1024x1024x2048x20xf16> into tensor<1024x1024x40960xf16>
// CHECK: return
func.func @matmul_add_mul(%arg0: tensor<1024x1024x2048x20xf16>, %tmp: tensor<1024x1024xf16>, %arg1: tensor<1024x1024xf16>, %arg2: tensor<1024x1024xf16>, %arg3: tensor<1024x1024xf16>, %arg4: tensor<1024x1024xf16>) -> tensor<1024x1024xf16> {
  %0 = tensor.empty() : tensor<1024x1024xf16>
  %reduced = linalg.reduce ins(%arg0 : tensor<1024x1024x2048x20xf16>) outs(%tmp : tensor<1024x1024xf16>) dimensions = [2, 3]
      (%in: f16, %init: f16) {
        %5 = arith.addf %in, %init : f16
        linalg.yield %5 : f16
      }
  %1 = linalg.matmul ins(%reduced, %arg1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %2 = tensor.empty() : tensor<1024x1024xf16>
  %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%2 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, sub} ins(%3, %arg3 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%arg4 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  return %4 : tensor<1024x1024xf16>
}

// -----
// CHECK-LABEL: func.func @mlir_fused_cast
// CHECK: hfusion.cast
// CHECK-SAME: -> tensor<24xf32>
func.func @mlir_fused_cast(%arg0: tensor<24xi1>, %arg1: tensor<24x3x256x192xf32>, %arg2: tensor<24x3x256x192xf32>) -> tensor<24x3x256x192xf32> attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [24, 1, 1, 1] : tensor<24xi1> into tensor<24x1x1x1xi1>
  %0 = tensor.empty() : tensor<24x1x1x1xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded : tensor<24x1x1x1xi1>) outs(%0 : tensor<24x1x1x1xf32>) -> tensor<24x1x1x1xf32>
  %2 = tensor.empty() : tensor<24x3x256x192xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1, 2, 3]] : tensor<24x1x1x1xf32> into tensor<24xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<24xf32>) outs(%2 : tensor<24x3x256x192xf32>) dimensions = [1, 2, 3]
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %arg1 : tensor<24x3x256x192xf32>, tensor<24x3x256x192xf32>) outs(%arg2 : tensor<24x3x256x192xf32>) -> tensor<24x3x256x192xf32>
  return %3 : tensor<24x3x256x192xf32>
}


// -----
// CHECK-LABEL: extract_slice
// CHECK: tensor.extract_slice
// CHECK-SAME: tensor<24x256xbf16> to tensor<24x128xbf16>
// CHECK: linalg.fill
// CHECK-SAME: tensor<24x128xbf16>) -> tensor<24x128xbf16>
func.func @extract_slice(%arg0: tensor<24x256xbf16>, %arg1: tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 1.000000e+00 : bf16
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3]] output_shape [24, 256, 1, 1] : tensor<24x256xbf16> into tensor<24x256x1x1xbf16>
  %extracted_slice = tensor.extract_slice %expanded[0, 0, 0, 0] [24, 128, 1, 1] [1, 1, 1, 1] : tensor<24x256x1x1xbf16> to tensor<24x128x1x1xbf16>
  %0 = tensor.empty() : tensor<24x128x1x1xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
  %2 = tensor.empty() : tensor<24x128x1x1xf32>
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<24x128x1x1xbf16>) outs(%2 : tensor<24x128x1x1xf32>) -> tensor<24x128x1x1xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %3 : tensor<24x128x1x1xf32>, tensor<24x128x1x1xf32>) outs(%2 : tensor<24x128x1x1xf32>) -> tensor<24x128x1x1xf32>
  %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice : tensor<24x128x1x1xbf16>) outs(%2 : tensor<24x128x1x1xf32>) -> tensor<24x128x1x1xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %4 : tensor<24x128x1x1xf32>, tensor<24x128x1x1xf32>) outs(%2 : tensor<24x128x1x1xf32>) -> tensor<24x128x1x1xf32>
  %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%6 : tensor<24x128x1x1xf32>) outs(%arg1 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
  return %7 : tensor<24x128x1x1xbf16>
}

// -----

// CHECK-LABEL: func.func @test_extract_collapse
// CHECK: arith.constant 1109
// This 1109 is from 3 + (7 * 5) + (7 * 9 * 1) + (7 * 9 * 8 * 2)
//                   ^        ^             ^                 ^
// Dimensions checking is from grid into a single integer for each reassociation group
func.func @test_extract_collapse(%arg0: tensor<3x8x9x7xf32>) -> f32 {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  // Original extract with indices [2, 1, 5, 3]
  %extracted = tensor.extract %arg0[%c2, %c1, %c5, %c3] : tensor<3x8x9x7xf32>

  return %extracted : f32
}

// -----
// CHECK-LABEL: func.func @test_extract_collapse2
// CHECK: arith.constant 1109
// CHECK: %extracted = tensor.extract
// CHECK-SAME: c5
// CHECK-SAME: c1109
// CHECK-SAME: tensor<10x1512xf32>
func.func @test_extract_collapse2(%arg0: tensor<3x8x9x7xf32>, %arg1: tensor<2x5x3x8x9x7xf32>) -> f32 {
  %c1_i = arith.constant 1 : index
  %c0_i = arith.constant 0 : index
  %c2_i = arith.constant 2 : index
  %c3_i = arith.constant 3 : index
  %c5_i = arith.constant 5 : index

  %0 = tensor.empty() : tensor<3x8x9x7xf32>
  %1 = linalg.elemwise_unary ins(%arg0 : tensor<3x8x9x7xf32>)
                             outs(%0 : tensor<3x8x9x7xf32>) -> tensor<3x8x9x7xf32>

  %2 = tensor.empty() : tensor<2x5x3x8x9x7xf32>
  %3 = linalg.broadcast ins(%1 : tensor<3x8x9x7xf32>)
                        outs(%2 : tensor<2x5x3x8x9x7xf32>)
                        dimensions = [0, 1]

  %4 = tensor.empty() : tensor<2x5x3x8x9x7xf32>
  %5 = linalg.elemwise_binary ins(%3, %arg1 : tensor<2x5x3x8x9x7xf32>, tensor<2x5x3x8x9x7xf32>)
                              outs(%4 : tensor<2x5x3x8x9x7xf32>) -> tensor<2x5x3x8x9x7xf32>

  %extracted = tensor.extract %5[%c1_i, %c0_i, %c2_i, %c1_i, %c5_i, %c3_i] : tensor<2x5x3x8x9x7xf32>

  return %extracted : f32
}

// -----
// CHECK-LABEL: @mlir_fused_add_div_npu_dtype_cast_pow_rsqrt_sub_sum_0(
// CHECK: hfusion.bitcast
// CHECK-SAME: -> tensor<150994944xi32>
// CHECK: return
module {
  func.func @mlir_fused_add_div_npu_dtype_cast_pow_rsqrt_sub_sum_0(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xbf16>, %arg2: tensor<24x128x256x192xbf16>) -> (tensor<24x128x256x192xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 1.966080e+05 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c31_i32 = arith.constant 31 : i32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %cst_4 = arith.constant -2.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x1x1x4x256x192xbf16>
    %expanded_6 = tensor.expand_shape %arg1 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x1x1x4x256x192xbf16>
    %expanded_7 = tensor.expand_shape %arg2 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x1x1x4x256x192xbf16>
    %0 = tensor.empty() : tensor<24x128x256x192xf32>
    %expanded_8 = tensor.expand_shape %0 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x128x256x192xf32> into tensor<24x32x1x1x4x256x192xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded_6 : tensor<24x32x1x1x4x256x192xbf16>) outs(%expanded_8 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded : tensor<24x32x1x1x4x256x192xbf16>) outs(%expanded_8 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %1 : tensor<24x32x1x1x4x256x192xf32>, tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_8 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded_7 : tensor<24x32x1x1x4x256x192xbf16>) outs(%expanded_8 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %4 : tensor<24x32x1x1x4x256x192xf32>, tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_8 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %collapsed = tensor.collapse_shape %5 [[0], [1, 2, 3, 4], [5], [6]] : tensor<24x32x1x1x4x256x192xf32> into tensor<24x128x256x192xf32>
    %6 = tensor.empty() : tensor<24x32xf32>
    %expanded_9 = tensor.expand_shape %6 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%expanded_9 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %reduced = linalg.reduce ins(%5 : tensor<24x32x1x1x4x256x192xf32>) outs(%7 : tensor<24x32x1x1xf32>) dimensions = [4, 5, 6]
      (%in: f32, %init: f32) {
        %43 = arith.addf %in, %init : f32
        linalg.yield %43 : f32
      }
    %8 = tensor.empty() : tensor<24x32x1x1xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %9 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%8 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %11 = tensor.empty() : tensor<24x32x4x49152xf32>
    %expanded_10 = tensor.expand_shape %11 [[0], [1, 2, 3], [4], [5, 6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x32x4x49152xf32> into tensor<24x32x1x1x4x256x192xf32>
    %broadcasted = linalg.broadcast ins(%10 : tensor<24x32x1x1xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) dimensions = [4, 5, 6]
    %12 = tensor.empty() : tensor<24x32x4x49152xi64>
    %expanded_11 = tensor.expand_shape %12 [[0], [1, 2, 3], [4], [5, 6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x32x4x49152xi64> into tensor<24x32x1x1x4x256x192xi64>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%5, %broadcasted : tensor<24x32x1x1x4x256x192xf32>, tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %collapsed_12 = tensor.collapse_shape %13 [[0], [1, 2, 3], [4], [5, 6]] : tensor<24x32x1x1x4x256x192xf32> into tensor<24x32x4x49152xf32>
    %14 = linalg.fill ins(%c2_i64 : i64) outs(%expanded_11 : tensor<24x32x1x1x4x256x192xi64>) -> tensor<24x32x1x1x4x256x192xi64>
    %15 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%14 : tensor<24x32x1x1x4x256x192xi64>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %16 = tensor.empty() : tensor<24x32x1x1x4x256x192xi32>
    %collapsed_13 = tensor.collapse_shape %16 [[0], [1, 2, 3], [4], [5, 6]] : tensor<24x32x1x1x4x256x192xi32> into tensor<24x32x4x49152xi32>
    %17 = hfusion.bitcast ins(%collapsed_12 : tensor<24x32x4x49152xf32>) outs(%collapsed_13 : tensor<24x32x4x49152xi32>) -> tensor<24x32x4x49152xi32>
    %expanded_13 = tensor.expand_shape %17 [[0], [1, 2, 3], [4], [5, 6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x32x4x49152xi32> into tensor<24x32x1x1x4x256x192xi32>
    %18 = tensor.empty() : tensor<24x32x4x49152xi32>
    %expanded_14 = tensor.expand_shape %18 [[0], [1, 2, 3], [4], [5, 6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x32x4x49152xi32> into tensor<24x32x1x1x4x256x192xi32>
    %19 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%expanded_13, %c31_i32 : tensor<24x32x1x1x4x256x192xi32>, i32) outs(%expanded_14 : tensor<24x32x1x1x4x256x192xi32>) -> tensor<24x32x1x1x4x256x192xi32>
    %20 = tensor.empty() : tensor<24x32x4x49152xi1>
    %expanded_15 = tensor.expand_shape %20 [[0], [1, 2, 3], [4], [5, 6]] output_shape [24, 32, 1, 1, 4, 256, 192] : tensor<24x32x4x49152xi1> into tensor<24x32x1x1x4x256x192xi1>
    %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%19 : tensor<24x32x1x1x4x256x192xi32>) outs(%expanded_11 : tensor<24x32x1x1x4x256x192xi64>) -> tensor<24x32x1x1x4x256x192xi64>
    %22 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%21, %c1_i64 : tensor<24x32x1x1x4x256x192xi64>, i64) outs(%expanded_15 : tensor<24x32x1x1x4x256x192xi1>) -> tensor<24x32x1x1x4x256x192xi1>
    %23 = hfusion.cast {round_mode = #hfusion.round_mode<floor>} ins(%15 : tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %24 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%23, %15 : tensor<24x32x1x1x4x256x192xf32>, tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_15 : tensor<24x32x1x1x4x256x192xi1>) -> tensor<24x32x1x1x4x256x192xi1>
    %25 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%22, %24 : tensor<24x32x1x1x4x256x192xi1>, tensor<24x32x1x1x4x256x192xi1>) outs(%expanded_15 : tensor<24x32x1x1x4x256x192xi1>) -> tensor<24x32x1x1x4x256x192xi1>
    %26 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%15 : tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %cst_5 : tensor<24x32x1x1x4x256x192xf32>, f32) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %28 = hfusion.cast {round_mode = #hfusion.round_mode<floor>} ins(%27 : tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%28, %cst_3 : tensor<24x32x1x1x4x256x192xf32>, f32) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%26, %28 : tensor<24x32x1x1x4x256x192xf32>, tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %cst_4 : tensor<24x32x1x1x4x256x192xf32>, f32) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%31, %cst_2 : tensor<24x32x1x1x4x256x192xf32>, f32) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %33 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%13 : tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %34 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%33 : tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%34, %cst_3 : tensor<24x32x1x1x4x256x192xf32>, f32) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %36 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%35 : tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%36, %32 : tensor<24x32x1x1x4x256x192xf32>, tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %38 = hfusion.select ins(%25, %37, %36 : tensor<24x32x1x1x4x256x192xi1>, tensor<24x32x1x1x4x256x192xf32>, tensor<24x32x1x1x4x256x192xf32>) outs(%expanded_10 : tensor<24x32x1x1x4x256x192xf32>) -> tensor<24x32x1x1x4x256x192xf32>
    %reduced_16 = linalg.reduce ins(%38 : tensor<24x32x1x1x4x256x192xf32>) outs(%7 : tensor<24x32x1x1xf32>) dimensions = [4, 5, 6]
      (%in: f32, %init: f32) {
        %44 = arith.addf %in, %init : f32
        linalg.yield %44 : f32
      }
    %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced_16, %9 : tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) outs(%8 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %40 = arith.truncf %cst_0 : f64 to f32
    %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%39, %40 : tensor<24x32x1x1xf32>, f32) outs(%8 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %42 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%41 : tensor<24x32x1x1xf32>) outs(%8 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    %43 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%42 : tensor<24x32x1x1xf32>) outs(%8 : tensor<24x32x1x1xf32>) -> tensor<24x32x1x1xf32>
    return %collapsed, %reduced, %10, %reduced_16, %43 : tensor<24x128x256x192xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>
  }
}

// -----
// CHECK-LABEL: mlir_fused_abs_clamp_div_log_npu_dtype_cast_sub_7
// CHECK: %[[cast:.*]] = hfusion.cast
// CHECK: tensor.expand_shape %[[cast]] {{\[}}{{\[}}0, 1]] output_shape {{\[}}1, 2047]
// CHECK: tensor.concat {{.*}} %[[cast]],
// CHECK: tensor.extract_slice {{\%}}concat{{\[}}1] {{\[}}2047] {{\[}}1]
// CHECK: return
module {
  func.func @mlir_fused_abs_clamp_div_log_npu_dtype_cast_sub_7(%arg0: tensor<1x2047xi64>) -> (tensor<1x2047x2047xf32>, tensor<1x2047x2047xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 3.3222591362126246 : f64
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x2047xi32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<1x2047xi64>) outs(%0 : tensor<1x2047xi32>) -> tensor<1x2047xi32>
    %extracted_slice = tensor.extract_slice %1[0, 2046] [1, 1] [1, 1] : tensor<1x2047xi32> to tensor<1x1xi32>
    %concat = tensor.concat dim(1) %1, %extracted_slice : (tensor<1x2047xi32>, tensor<1x1xi32>) -> tensor<1x2048xi32>
    %extracted_slice_1 = tensor.extract_slice %concat[0, 1] [1, 2047] [1, 1] : tensor<1x2048xi32> to tensor<1x2047xi32>
    %expanded = tensor.expand_shape %extracted_slice_1 [[0], [1, 2]] output_shape [1, 2047, 1] : tensor<1x2047xi32> into tensor<1x2047x1xi32>
    %2 = tensor.empty() : tensor<1x2047x2047xi32>
    %collapsed = tensor.collapse_shape %expanded [[0], [1, 2]] : tensor<1x2047x1xi32> into tensor<1x2047xi32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<1x2047xi32>) outs(%2 : tensor<1x2047x2047xi32>) dimensions = [2]
    %extracted_slice_2 = tensor.extract_slice %concat[0, 0] [1, 2047] [1, 1] : tensor<1x2048xi32> to tensor<1x2047xi32>
    %expanded_3 = tensor.expand_shape %extracted_slice_2 [[0], [1, 2]] output_shape [1, 1, 2047] : tensor<1x2047xi32> into tensor<1x1x2047xi32>
    %collapsed_4 = tensor.collapse_shape %expanded_3 [[0, 1], [2]] : tensor<1x1x2047xi32> into tensor<1x2047xi32>
    %broadcasted_5 = linalg.broadcast ins(%collapsed_4 : tensor<1x2047xi32>) outs(%2 : tensor<1x2047x2047xi32>) dimensions = [1]
    %3 = linalg.fill ins(%c1_i32 : i32) outs(%2 : tensor<1x2047x2047xi32>) -> tensor<1x2047x2047xi32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_5, %3 : tensor<1x2047x2047xi32>, tensor<1x2047x2047xi32>) outs(%2 : tensor<1x2047x2047xi32>) -> tensor<1x2047x2047xi32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted, %4 : tensor<1x2047x2047xi32>, tensor<1x2047x2047xi32>) outs(%2 : tensor<1x2047x2047xi32>) -> tensor<1x2047x2047xi32>
    %6 = tensor.empty() : tensor<1x2047x2047xf32>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%5 : tensor<1x2047x2047xi32>) outs(%6 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%7 : tensor<1x2047x2047xf32>) outs(%6 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%8, %9 : tensor<1x2047x2047xf32>, tensor<1x2047x2047xf32>) outs(%6 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%10 : tensor<1x2047x2047xf32>) outs(%6 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %12 = arith.truncf %cst : f64 to f32
    %13 = linalg.fill ins(%12 : f32) outs(%6 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%11, %13 : tensor<1x2047x2047xf32>, tensor<1x2047x2047xf32>) outs(%6 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %15 = tensor.empty() : tensor<1x2047x2047xi64>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%14 : tensor<1x2047x2047xf32>) outs(%15 : tensor<1x2047x2047xi64>) -> tensor<1x2047x2047xi64>
    %17 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<1x2047x2047xi64>) -> tensor<1x2047x2047xi64>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%16, %17 : tensor<1x2047x2047xi64>, tensor<1x2047x2047xi64>) outs(%15 : tensor<1x2047x2047xi64>) -> tensor<1x2047x2047xi64>
    %19 = linalg.fill ins(%c128_i64 : i64) outs(%15 : tensor<1x2047x2047xi64>) -> tensor<1x2047x2047xi64>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%18, %19 : tensor<1x2047x2047xi64>, tensor<1x2047x2047xi64>) outs(%15 : tensor<1x2047x2047xi64>) -> tensor<1x2047x2047xi64>
    return %10, %20 : tensor<1x2047x2047xf32>, tensor<1x2047x2047xi64>
  }
}
// -----
// CHECK-LABEL: mlir_fused_add_div_fill_mul_silu_sub_14
// CHECK: 1x4x4190209xf32
// CHECK: return
module {
  func.func @mlir_fused_add_div_fill_mul_silu_sub_14(%arg0: tensor<4x2047x2047xf32>, %arg1: tensor<4093xf32>, %arg2: tensor<4190209xf32>) -> (tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant dense<true> : tensor<2047x2047xi1>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 4.8851978505129456E-4 : f64
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant dense<1> : tensor<i64>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3, 4]] output_shape [4, 2047, 1, 1, 2047] : tensor<4x2047x2047xf32> into tensor<4x2047x1x1x2047xf32>
    %0 = tensor.empty() : tensor<1x4x2047x2047x1xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<4x2047x1x1x2047xf32>) outs(%0 : tensor<1x4x2047x2047x1xf32>) permutation = [3, 0, 1, 4, 2]
    %collapsed = tensor.collapse_shape %transposed [[0], [1], [2], [3, 4]] : tensor<1x4x2047x2047x1xf32> into tensor<1x4x2047x2047xf32>
    %padded = tensor.pad %arg1 low[0] high[2047] {
    ^bb0(%arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<4093xf32> to tensor<6140xf32>
    %expanded_4 = tensor.expand_shape %padded [[0, 1]] output_shape [1, 6140] : tensor<6140xf32> into tensor<1x6140xf32>
    %extracted_slice = tensor.extract_slice %expanded_4[0, 2046] [1, 2047] [1, 1] : tensor<1x6140xf32> to tensor<1x2047xf32>
    %expanded_5 = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [1, 1, 2047] : tensor<1x2047xf32> into tensor<1x1x2047xf32>
    %1 = tensor.empty() : tensor<1x2047x2047xf32>
    %collapsed_6 = tensor.collapse_shape %expanded_5 [[0, 1], [2]] : tensor<1x1x2047xf32> into tensor<1x2047xf32>
    %broadcasted = linalg.broadcast ins(%collapsed_6 : tensor<1x2047xf32>) outs(%1 : tensor<1x2047x2047xf32>) dimensions = [1]
    %expanded_7 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 2047, 2047] : tensor<4190209xf32> into tensor<1x2047x2047xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_7, %2 : tensor<1x2047x2047xf32>, tensor<1x2047x2047xf32>) outs(%1 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted, %3 : tensor<1x2047x2047xf32>, tensor<1x2047x2047xf32>) outs(%1 : tensor<1x2047x2047xf32>) -> tensor<1x2047x2047xf32>
    %expanded_8 = tensor.expand_shape %4 [[0], [1, 2], [3]] output_shape [1, 1, 2047, 2047] : tensor<1x2047x2047xf32> into tensor<1x1x2047x2047xf32>
    %5 = tensor.empty() : tensor<1x4x2047x2047xf32>
    %collapsed_9 = tensor.collapse_shape %expanded_8 [[0, 1], [2], [3]] : tensor<1x1x2047x2047xf32> into tensor<1x2047x2047xf32>
    %broadcasted_10 = linalg.broadcast ins(%collapsed_9 : tensor<1x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) dimensions = [1]
    %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_10, %6 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%collapsed, %7 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%8 : tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %10 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%9 : tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %cst_0 : tensor<1x4x2047x2047xf32>, f32) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst_0, %11 : f32, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %12 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %14 = arith.truncf %cst_1 : f64 to f32
    %15 = linalg.fill ins(%14 : f32) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %15 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %expanded_11 = tensor.expand_shape %cst [[0, 1, 2], [3]] output_shape [1, 1, 2047, 2047] : tensor<2047x2047xi1> into tensor<1x1x2047x2047xi1>
    %17 = tensor.empty() : tensor<1x1x2047x2047xf32>
    %18 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%expanded_11 : tensor<1x1x2047x2047xi1>) outs(%17 : tensor<1x1x2047x2047xf32>) -> tensor<1x1x2047x2047xf32>
    %collapsed_12 = tensor.collapse_shape %18 [[0, 1], [2], [3]] : tensor<1x1x2047x2047xf32> into tensor<1x2047x2047xf32>
    %broadcasted_13 = linalg.broadcast ins(%collapsed_12 : tensor<1x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) dimensions = [1]
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%16, %broadcasted_13 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %20 = arith.sitofp %cst_3 : tensor<i64> to tensor<f32>
    %broadcasted_14 = linalg.broadcast ins(%20 : tensor<f32>) outs(%5 : tensor<1x4x2047x2047xf32>) dimensions = [0, 1, 2, 3]
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %6 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted_14, %21 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %22 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %6 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%23, %24 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %25 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%5 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
    return %19, %26 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>
  }
}

// -----
// CHECK-LABEL: @flatten_empty(
// CHECK: return
module {
  func.func @flatten_empty(%arg0: tensor<1xf16>, %arg1: tensor<f16>) -> tensor<1xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<1xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<f16>) outs(%0 : tensor<1xf16>) dimensions = [0]
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %broadcasted : tensor<1xf16>, tensor<1xf16>) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%1, %2 : tensor<1xf16>, tensor<1xf16>) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %broadcasted : tensor<1xf16>, tensor<1xf16>) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %4 : tensor<1xf16>, tensor<1xf16>) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
    return %5 : tensor<1xf16>
  }
}

// -----

// CHECK-LABEL: @flatten_with_reduce_index(
// CHECK: dimensions = [1]
// CHECK: dimensions = [1]
// CHECK: return
func.func @flatten_with_reduce_index(%arg0: tensor<24x48x48xbf16>, %arg1: tensor<24x48x48xi64>) -> (tensor<24x48x1xf32>, tensor<24x48x1xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<24x48x48xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<24x48x48xbf16>) outs(%0 : tensor<24x48x48xf32>) -> tensor<24x48x48xf32>
  %2 = tensor.empty() : tensor<24x48xi64>
  %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<24x48xi64>) -> tensor<24x48xi64>
  %4 = tensor.empty() : tensor<24x48xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<24x48xf32>) -> tensor<24x48xf32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <max> ins(%1 : tensor<24x48x48xf32>) outs(%5, %3 : tensor<24x48xf32>, tensor<24x48xi64>) dimensions = [2] -> tensor<24x48xf32>, tensor<24x48xi64>
  %reduced_1:2 = hfusion.reduce_with_index {tie_break_left = true} <min> ins(%1, %arg1 : tensor<24x48x48xf32>, tensor<24x48x48xi64>) outs(%5, %3 : tensor<24x48xf32>, tensor<24x48xi64>) dimensions = [2] -> tensor<24x48xf32>, tensor<24x48xi64>
  %expanded = tensor.expand_shape %reduced#0 [[0], [1, 2]] output_shape [24, 48, 1] : tensor<24x48xf32> into tensor<24x48x1xf32>
  %expanded_1 = tensor.expand_shape %reduced_1#1 [[0], [1, 2]] output_shape [24, 48, 1] : tensor<24x48xi64> into tensor<24x48x1xi64>
  return %expanded, %expanded_1 : tensor<24x48x1xf32>, tensor<24x48x1xi64>
}

// -----

// CHECK-LABEL: @collapse_to_empty(
// CHECK: return
func.func @collapse_to_empty(%arg0: tensor<768xbf16>, %arg1: tensor<1xbf16>) -> tensor<768xbf16> attributes {OperatorType = "Default", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %collapsed = tensor.collapse_shape %arg1 [] : tensor<1xbf16> into tensor<bf16>
  %0 = tensor.empty() : tensor<768xbf16>
  %1 = tensor.empty() : tensor<f32>
  %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<bf16>) outs(%1 : tensor<f32>) -> tensor<f32>
  %3 = tensor.empty() : tensor<768xf32>
  %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<768xbf16>) outs(%3 : tensor<768xf32>) -> tensor<768xf32>
  %extracted = tensor.extract %2[] : tensor<f32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %extracted : tensor<768xf32>, f32) outs(%3 : tensor<768xf32>) -> tensor<768xf32>
  %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%5 : tensor<768xf32>) outs(%0 : tensor<768xbf16>) -> tensor<768xbf16>
  return %6 : tensor<768xbf16>
}

// -----

// CHECK-LABEL: func.func @matmul_transpose_b_static_dimension(
// CHECK: tensor<1024x20x2048x20xf16> into tensor<1024x20x40960xf16>
// CHECK: return
func.func @matmul_transpose_b_static_dimension(%arg0: tensor<1024x20x2048x20xf16>, %tmp: tensor<20x20xf16>, %tmp2: tensor<1024x20xf16>, %arg1: tensor<20x20xf16>, %arg2: tensor<1024x20xf16>) -> tensor<1024x20xf16> {
  %0 = tensor.empty() : tensor<1024x20xf16>
  %reduced = linalg.reduce ins(%arg0 : tensor<1024x20x2048x20xf16>) outs(%tmp2 : tensor<1024x20xf16>) dimensions = [2, 3]
      (%in: f16, %init: f16) {
        %5 = arith.addf %in, %init : f16
        linalg.yield %5 : f16
      }
  %1 = linalg.matmul_transpose_b ins(%reduced, %arg1 : tensor<1024x20xf16>, tensor<20x20xf16>) outs(%0 : tensor<1024x20xf16>) -> tensor<1024x20xf16>
  %2 = tensor.empty() : tensor<1024x20xf16>
  %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<1024x20xf16>, tensor<1024x20xf16>) outs(%2 : tensor<1024x20xf16>) -> tensor<1024x20xf16>
  return %3 : tensor<1024x20xf16>
}

// -----

// CHECK-LABEL: func.func @check_rank0_reduce(
// CHECK: linalg.reduce ins(%{{.*}} : tensor<393216xf32>) outs(%{{.*}} : tensor<f32>) dimensions = [0] 
// CHECK: return
func.func @check_rank0_reduce(%arg0: tensor<768x512xbf16>) -> tensor<1xbf16> attributes {OperatorType = "Reduce", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<768x512xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<768x512xbf16>) outs(%0 : tensor<768x512xf32>) -> tensor<768x512xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<768x512xf32>, tensor<768x512xf32>) outs(%0 : tensor<768x512xf32>) -> tensor<768x512xf32>
  %3 = tensor.empty() : tensor<f32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
  %reduced = linalg.reduce ins(%2 : tensor<768x512xf32>) outs(%4 : tensor<f32>) dimensions = [0, 1]
    (%in: f32, %init: f32) {
      %7 = arith.addf %in, %init : f32
      linalg.yield %7 : f32
    }
  %5 = tensor.empty() : tensor<bf16>
  %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<f32>) outs(%5 : tensor<bf16>) -> tensor<bf16>
  %expanded = tensor.expand_shape %6 [] output_shape [1] : tensor<bf16> into tensor<1xbf16>
  return %expanded : tensor<1xbf16>
}

// -----
// CHECK-LABEL: func.func @emptyArgToExtract(
// CHECK: return
func.func @emptyArgToExtract(%arg0: tensor<32xf32>, %arg1: tensor<f32>) -> tensor<32xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<32xf32>
  %extracted = tensor.extract %arg1[] : tensor<f32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %extracted : tensor<32xf32>, f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %cst : tensor<32xf32>, f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst_0, %2 : f32, tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %extracted : tensor<32xf32>, f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %4 : tensor<32xf32>
}

// -----

// CHECK-LABEL: test_deinterleave(
// CHECK: tensor<4x2x128xf16> into tensor<1024xf16>
// CHECK: hfusion.deinterleave
// CHECK-SAME: tensor<1024xf16> -> tensor<512xf16>
func.func @test_deinterleave() -> tensor<4x2x64xf16> {
  %0 = tensor.empty() : tensor<4x2x128xf16>
  %1 = hfusion.deinterleave %0 channel<0> : tensor<4x2x128xf16> -> tensor<4x2x64xf16>
  return %1 : tensor<4x2x64xf16>
}

// -----
// CHECK-LABEL: test_interleave(
// CHECK: tensor<4x2x64xf16> into tensor<512xf16>
// CHECK: hfusion.interleave
// CHECK-SAME: tensor<512xf16> -> tensor<1024xf16>
func.func @test_interleave(%arg0: tensor<4x2x64xf16>, %arg1: tensor<4x2x64xf16>) -> tensor<4x2x128xf16> {
  %2 = hfusion.interleave %arg0, %arg1 : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
  return %2 : tensor<4x2x128xf16>
}

// -----

// CHECK-LABEL: rank_reducing_extract(
// CHECK: extract_slice
// CHECK-SAME: tensor<24x1024xbf16> to tensor<24x512xbf16>
// CHECK: return
func.func @rank_reducing_extract(%arg0: tensor<24x1024xbf16>, %arg1: tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %cst = arith.constant 1.000000e+00 : bf16
    %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2, 3]] : tensor<24x512x1x1xbf16> into tensor<12288xbf16>
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3]] output_shape [24, 1024, 1, 1] : tensor<24x1024xbf16> into tensor<24x1024x1x1xbf16>
    %extracted_slice = tensor.extract_slice %expanded[0, 0, 0, 0] [24, 512, 1, 1] [1, 1, 1, 1] : tensor<24x1024x1x1xbf16> to tensor<24x512x1x1xbf16>
    %collapsed_0 = tensor.collapse_shape %extracted_slice [[0, 1, 2, 3]] : tensor<24x512x1x1xbf16> into tensor<12288xbf16>
    %0 = tensor.empty() : tensor<12288xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<12288xbf16>) -> tensor<12288xbf16>
    %2 = tensor.empty() : tensor<12288xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<12288xbf16>) outs(%2 : tensor<12288xf32>) -> tensor<12288xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %3 : tensor<12288xf32>, tensor<12288xf32>) outs(%2 : tensor<12288xf32>) -> tensor<12288xf32>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_0 : tensor<12288xbf16>) outs(%2 : tensor<12288xf32>) -> tensor<12288xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %4 : tensor<12288xf32>, tensor<12288xf32>) outs(%2 : tensor<12288xf32>) -> tensor<12288xf32>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%6 : tensor<12288xf32>) outs(%collapsed : tensor<12288xbf16>) -> tensor<12288xbf16>
    %expanded_1 = tensor.expand_shape %7 [[0, 1, 2, 3]] output_shape [24, 512, 1, 1] : tensor<12288xbf16> into tensor<24x512x1x1xbf16>
    return %expanded_1 : tensor<24x512x1x1xbf16>
}

// -----

// CHECK-LABEL: test_gather_do_not_flatten_last_axis
func.func @test_gather_do_not_flatten_last_axis(%src: tensor<4x64xf32>, %idx: tensor<4x32xi64>) -> tensor<4x32xf32> {
  %init = tensor.empty() : tensor<4x32xf32>
  // CHECK:  axis = 1
  // CHECK-SAME: tensor<4x32xf32>
  %res = hfusion.gather ins(%src, %idx : tensor<4x64xf32>, tensor<4x32xi64>) outs(%init : tensor<4x32xf32>) axis = 1 -> tensor<4x32xf32>
  return %res : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: test_gather_flatten
func.func @test_gather_flatten(%src: tensor<2x4x64xf32>, %idx: tensor<2x4x32xi32>) -> tensor<2x4x32xf32> {
  %init = tensor.empty() : tensor<2x4x32xf32>
  // CHECK:  axis = 1
  // CHECK-SAME: tensor<8x32xf32>
  %res = hfusion.gather ins(%src, %idx : tensor<2x4x64xf32>, tensor<2x4x32xi32>) outs(%init : tensor<2x4x32xf32>) axis = 2 -> tensor<2x4x32xf32>
  return %res : tensor<2x4x32xf32>
}

// -----

// CHECK-LABEL: test_gather_flatten
func.func @test_gather_flatten(%src: tensor<2x4x32xf32>, %idx: tensor<2x1x32xi32>) -> tensor<2x1x32xf32> {
  %init = tensor.empty() : tensor<2x1x32xf32>
  // CHECK:  axis = 1
  // CHECK-SAME: tensor<2x1x32xf32>
  %res = hfusion.gather ins(%src, %idx : tensor<2x4x32xf32>, tensor<2x1x32xi32>) outs(%init : tensor<2x1x32xf32>) axis = 1 -> tensor<2x1x32xf32>
  return %res : tensor<2x1x32xf32>
}

// -----
// CHECK-LABEL: test_cumsum_mid(
// CHECK: tensor<4x2x6x4x8xf16> into tensor<8x6x32xf16>
func.func @test_cumsum_mid(%arg0: tensor<4x2x6x4x8xf16>) -> tensor<4x2x6x4x8xf16> {
  %0 = hfusion.cumsum %arg0 : tensor<4x2x6x4x8xf16> cum_dims = [2] -> tensor<4x2x6x4x8xf16>
  return %0 : tensor<4x2x6x4x8xf16>
}

// -----
// CHECK-LABEL: test_cumsum_first(
// CHECK: tensor<4x2x6x4x8xf16> into tensor<4x384xf16>
func.func @test_cumsum_first(%arg0: tensor<4x2x6x4x8xf16>) -> tensor<4x2x6x4x8xf16> {
  %0 = hfusion.cumsum %arg0 : tensor<4x2x6x4x8xf16> cum_dims = [0] -> tensor<4x2x6x4x8xf16>
  return %0 : tensor<4x2x6x4x8xf16>
}

// -----
// CHECK-LABEL: test_cumprod_mid(
// CHECK: tensor<4x2x6x4x8xf16> into tensor<48x4x8xf16>
func.func @test_cumprod_mid(%arg0: tensor<4x2x6x4x8xf16>) -> tensor<4x2x6x4x8xf16> {
  %0 = hfusion.cumprod %arg0 : tensor<4x2x6x4x8xf16> cum_dims = [3] -> tensor<4x2x6x4x8xf16>
  return %0 : tensor<4x2x6x4x8xf16>
}

// -----
// CHECK-LABEL: test_cumprod_last(
// CHECK: tensor<4x2x6x4x8xf16> into tensor<192x8xf16>
func.func @test_cumprod_last(%arg0: tensor<4x2x6x4x8xf16>) -> tensor<4x2x6x4x8xf16> {
  %0 = hfusion.cumprod %arg0 : tensor<4x2x6x4x8xf16> cum_dims = [4] -> tensor<4x2x6x4x8xf16>
  return %0 : tensor<4x2x6x4x8xf16>
}

// -----
// CHECK-LABEL: test_cumprod_0(
// CHECK: tensor<1x1x4x2x6x4x8xf16> into tensor<1x1536xf16>
func.func @test_cumprod_0(%arg0: tensor<1x1x4x2x6x4x8xf16>) -> tensor<1x1x4x2x6x4x8xf16> {
  %0 = hfusion.cumprod %arg0 : tensor<1x1x4x2x6x4x8xf16> cum_dims = [0] -> tensor<1x1x4x2x6x4x8xf16>
  return %0 : tensor<1x1x4x2x6x4x8xf16>
}

// -----
// CHECK-LABEL: test_cumprod_2(
// CHECK: tensor<1x1x4x1xf16> into tensor<4xf16>
func.func @test_cumprod_2(%arg0: tensor<1x1x4x1xf16>) -> tensor<1x1x4x1xf16> {
  %0 = hfusion.cumprod %arg0 : tensor<1x1x4x1xf16> cum_dims = [2] -> tensor<1x1x4x1xf16>
  return %0 : tensor<1x1x4x1xf16>
}
