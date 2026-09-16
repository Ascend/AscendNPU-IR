// RUN: bishengir-opt -test-buffer-utils %s -split-input-file -test-buffer-utils-var="enable-dma-opt" | FileCheck %s --check-prefix=BUFFER
// RUN: bishengir-opt -test-buffer-utils %s -split-input-file -test-buffer-utils-var="pass-double-to-all-args,enable-dma-opt" | FileCheck %s --check-prefix=BUFFERDOUBLE

// BUFFER: two_el_binary: 6
// BUFFERDOUBLE: two_el_binary: 11
module {
  func.func @two_el_binary(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = hfusion.load ins(%arg0 : tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = hfusion.load ins(%arg1 : tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %3 = hfusion.load ins(%arg2 : tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %3 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %6 = hfusion.store ins(%4 : tensor<8xf32>) outs(%arg3 : tensor<8xf32>) -> tensor<8xf32>
    %7 = hfusion.store ins(%5 : tensor<8xf32>) outs(%arg4 : tensor<8xf32>) -> tensor<8xf32>
    return %6, %7 : tensor<8xf32>, tensor<8xf32>
  }
}

// -----
// BUFFER: two_el_binary_2: 4
// BUFFERDOUBLE: two_el_binary_2: 7
module {
  func.func @two_el_binary_2(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = hfusion.load ins(%arg0 : tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = hfusion.load ins(%arg1 : tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %5 = hfusion.store ins(%4 : tensor<8xf32>) outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
    return %5 : tensor<8xf32>
  }
}

// -----
// BUFFER: reduce: 5
// BUFFERDOUBLE: reduce: 7
module {
  func.func @reduce(%arg0: tensor<8x9xf32>, %arg1: tensor<8x9xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    %0 = tensor.empty() : tensor<8x9xf32>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hfusion.load ins(%arg0 : tensor<8x9xf32>) outs(%0 : tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = hfusion.load ins(%arg1 : tensor<8x9xf32>) outs(%0 : tensor<8x9xf32>) -> tensor<8x9xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %3 : tensor<8x9xf32>, tensor<8x9xf32>) outs(%0 : tensor<8x9xf32>) -> tensor<8x9xf32>
    %reduced = linalg.reduce { arith.addf } ins(%4 : tensor<8x9xf32>) outs(%1 : tensor<8xf32>) dimensions = [1]
    %5 = hfusion.store ins(%reduced : tensor<8xf32>) outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
    return %5 : tensor<8xf32>
  }
}

// -----
// BUFFER: reduce: 4
// BUFFERDOUBLE: reduce: 7
module {
  func.func @reduce(%arg0: tensor<8x9xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    %0 = tensor.empty() : tensor<8x9xf32>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hfusion.load ins(%arg0 : tensor<8x9xf32>) outs(%0 : tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = hfusion.load ins(%arg1 : tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
    %reduced = linalg.reduce { arith.addf } ins(%2 : tensor<8x9xf32>) outs(%1 : tensor<8xf32>) dimensions = [1]
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced, %3 : tensor<8xf32>, tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
    %5 = hfusion.store ins(%4 : tensor<8xf32>) outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
    return %5 : tensor<8xf32>
  }
}


// -----
// BUFFERDOUBLE: two_el_binary: 7
module {
  func.func @two_el_binary(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = hfusion.load ins(%arg0 : tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = hfusion.load ins(%arg1 : tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%1, %2 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %4 = hfusion.store ins(%3 : tensor<8xf32>) outs(%arg2 : tensor<8xf32>) -> tensor<8xf32>
    return %4 : tensor<8xf32>
  }
}


// -----
// Considering 5 and 0 extra Live Range:
// BUFFER: 0 7 0
// BUFFER: 1 11 0
//         Copy In live range extended to fullest
// BUFFER: 6 13 1
// BUFFER: 8 11 4
//         Copy Out live range extended to fullest
// BUFFER: 10 13 4
// BUFFER: mlir_fused_mul: 9

// BUFFERDOUBLE: mlir_fused_mul: 14
module {
  func.func @mlir_fused_mul(%arg0: tensor<24xi8>, %arg1: tensor<24xf32>) -> tensor<24xf32> {
    %0 = tensor.empty() : tensor<24xi8>
    %1 = tensor.empty() : tensor<24xf32>
    %2 = hfusion.load ins(%arg0 : tensor<24xi8>) outs(%0 : tensor<24xi8>) -> tensor<24xi8>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<24xi8>) outs(%1 : tensor<24xf32>) -> tensor<24xf32>
    %4 = hfusion.store ins(%3 : tensor<24xf32>) outs(%arg1 : tensor<24xf32>) -> tensor<24xf32>
    return %4 : tensor<24xf32>
  }
}

// -----

// BUFFER: 0 3 0
// BUFFER: 4 4 0
// BUFFER: 4 9 1
// BUFFER: 8 10 1
// BUFFER: 9 9 1
//                 0 1 2 3 4 5 6 7 8 9 0
// Arg0 (Double) : 1 1 1 1 1 1       |
// Op empty      :   0               |
// cst           :     0             |
// Op0           :       0           |
// Un1           :         1 1 1 1 1 1 ( extended by collapse alias)
// Op1           :           0       |
// Col2          :             0     |
// Op2           :               0   |
// Red3          :                 1 1 1 (new buffer)
// Op3           :                   1   (extra usage for reduce)
// return        :                     0
// BUFFER: collapse_reduce_3: 3
module {
  func.func @collapse_reduce_3(%arg0: tensor<2x3x4x5x6x7x8xf32>) -> tensor<6x5x8xf32> {
    %0 = tensor.empty() : tensor<6x5x8xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<2x3x4x5x6x7x8xf32>) outs(%arg0 : tensor<2x3x4x5x6x7x8xf32>) -> tensor<2x3x4x5x6x7x8xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3], [4, 5], [6]] : tensor<2x3x4x5x6x7x8xf32> into tensor<6x4x5x42x8xf32>
    %reduced = linalg.reduce ins(%collapsed : tensor<6x4x5x42x8xf32>) outs(%0 : tensor<6x5x8xf32>) dimensions = [1, 3]
      (%in: f32, %init: f32) {
        %2 = arith.addf %in, %init : f32
        linalg.yield %2 : f32
      }
    return %reduced : tensor<6x5x8xf32>
  }
}


// -----
// BUFFER: 0 4 0
// BUFFER: 1 1 0
// BUFFER: 1 8 0
// BUFFER: 2 2 0
// BUFFER: 2 13 0
// BUFFER: 3 3 0
// BUFFER: 3 13 1
// BUFFER: 12 14 1
// BUFFER: collapse_elemwise: 2
module {
  func.func @collapse_elemwise(%arg0: tensor<24x32x8x9xf32>, %arg1: tensor<24x32x8x9xf32>, %arg2: tensor<24x32x8x9xf32>) -> tensor<24x256x9xf32> {
    %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<24x32x8x9xf32>) outs(%arg1 : tensor<24x32x8x9xf32>) -> tensor<24x32x8x9xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1, 2], [3]] : tensor<24x32x8x9xf32> into tensor<24x256x9xf32>
    %collapsed_0 = tensor.collapse_shape %arg1 [[0], [1, 2], [3]] : tensor<24x32x8x9xf32> into tensor<24x256x9xf32>
    %collapsed_1 = tensor.collapse_shape %arg2 [[0], [1, 2], [3]] : tensor<24x32x8x9xf32> into tensor<24x256x9xf32>
    %cst = arith.constant 1.000000e+00 : f32
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed, %cst : tensor<24x256x9xf32>, f32) outs(%collapsed_1 : tensor<24x256x9xf32>) -> tensor<24x256x9xf32>
    return %1 : tensor<24x256x9xf32>
  }
}
