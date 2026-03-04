// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20 enable-count-buffer-dma-opt" -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20 enable-auto-multi-buffer=true enable-count-buffer-dma-opt" -split-input-file | FileCheck --check-prefix=CHECK-DB %s
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20 max-buffer-count-tuning=1 enable-count-buffer-dma-opt" -split-input-file | FileCheck --check-prefix=CHECK-MT %s

// CHECK: @add_mul_fusion
func.func @add_mul_fusion(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32>
 attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
// CHECK: hacc.block_dim = 20
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>

  // CHECK: scf.for
  // CHECK: buffer_size_in_byte = 39296
  // CHECK-DB: buffer_size_in_byte = 21824
  // CHECK-MT: buffer_size_in_byte = 32736
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  
  %1 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %arg2 : tensor<?xf32>, tensor<?xf32>) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

// -----

func.func @add_mul_fusion_multi_output(%arg0: tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}
{
  // CHECK: add_mul_fusion_multi_output
  // CHECK: hacc.block_dim = 20
  // CHECK: buffer_size_in_byte = 32736
  // CHECK: return
  // CHECK-DB: buffer_size_in_byte = 21824
  // CHECK-MT: buffer_size_in_byte = 28064
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?xf32>
  %2 = tensor.empty(%0) : tensor<?xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4 = tensor.empty(%0) : tensor<?xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg2 : tensor<?xf32>, tensor<?xf32>) outs(%4 : tensor<?xf32>) -> tensor<?xf32>
  %6 = tensor.empty(%0) : tensor<?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%5 : tensor<?xf32>) outs(%6 : tensor<?xf32>) -> tensor<?xf32>
  return %5, %7 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @test_fusing_intermediate_producers(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
// CHECK: hacc.block_dim = 20
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: linalg.elemwise_binary
  // CHECK: buffer_size_in_byte = 39296
  // CHECK-DB: buffer_size_in_byte = 21824
  // CHECK-MT: buffer_size_in_byte = 28064
  // CHECK: hivm.block
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg2 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xf32>) -> tensor<1024xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %4 : tensor<1024xf32>, tensor<1024xf32>) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  return %5 : tensor<1024xf32>
}

// -----

func.func @func_arg_as_init(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>) -> tensor<1024xf32>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
// CHECK: hacc.block_dim = 20
  // CHECK: buffer_size_in_byte = 39296
  // CHECK-DB: buffer_size_in_byte = 21824
  // CHECK-MT: buffer_size_in_byte = 32736
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg2 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg3 : tensor<1024xf32>) -> tensor<1024xf32>
  return %2 : tensor<1024xf32>
}


// -----

// CHECK: linalg.fill
// CHECK: hfusion.store
module {
  func.func @single_outlined() -> tensor<2x128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x128xf32>) -> tensor<2x128xf32>
    return %1 : tensor<2x128xf32>
  }
}
