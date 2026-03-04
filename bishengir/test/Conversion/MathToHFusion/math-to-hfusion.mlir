// RUN: bishengir-opt -convert-math-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_exp
func.func @test_exp(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
  %ret = math.exp %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_log
func.func @test_log(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
  %ret = math.log %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_absf
func.func @test_absf(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
  %ret = math.absf %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_sqrt
func.func @test_sqrt(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
  %ret = math.sqrt %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_rsqrt
func.func @test_rsqrt(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
  %ret = math.rsqrt %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_tanh
func.func @test_tanh(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>}
  %ret = math.tanh %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_sin
func.func @test_sin(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
  %ret = math.sin %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_cos
func.func @test_cos(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
  %ret = math.cos %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_erf
func.func @test_erf(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>}
  %ret = math.erf %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_absi
func.func @test_absi(%arg0 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>}
  %ret = math.absi %arg0 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_powf
func.func @test_powf(%arg0 : tensor<6x6xf32>, %arg1 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun =  #hfusion.binary_fn<powf>}
  %ret = math.powf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_log2
func.func @test_log2(%arg0 : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun =  #hfusion.unary_fn<log2>}
  %ret = math.log2 %arg0 : tensor<1024xf32>
  return %ret : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_log10
func.func @test_log10(%arg0 : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun =  #hfusion.unary_fn<log10>}
  %ret = math.log10 %arg0 : tensor<1024xf32>
  return %ret : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_log1p
func.func @test_log1p(%arg0 : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun =  #hfusion.unary_fn<log1p>}
  %ret = math.log1p %arg0 : tensor<1024xf32>
  return %ret : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_exp2
func.func @test_exp2(%arg0 : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun =  #hfusion.unary_fn<exp2>}
  %ret = math.exp2 %arg0 : tensor<1024xf32>
  return %ret : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_expm1
func.func @test_expm1(%arg0 : tensor<1024xf32>) -> tensor<1024xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun =  #hfusion.unary_fn<expm1>}
  %ret = math.expm1 %arg0 : tensor<1024xf32>
  return %ret : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_fma
func.func @test_fma(%arg0 : tensor<64xf32>, %arg1 : tensor<64xf32> , %arg2 : tensor<64xf32>) -> tensor<64xf32> {
  // CHECK:   %[[EMPTY0:.*]] = tensor.empty() : tensor<64xf32>
  // CHECK:   %[[MUL_RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins([[arg0:.*]], %[[arg1:.*]]: tensor<64xf32>, tensor<64xf32>) outs(%[[EMPTY0:.*]] : tensor<64xf32>) -> tensor<64xf32>
  // CHECK:       %[[ADD_RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[MUL_RET:.*]], %[[arg2:.*]] : tensor<64xf32>, tensor<64xf32>) outs(%[[EMPTY0:.*]] : tensor<64xf32>) -> tensor<64xf32>
  %ret = math.fma %arg0, %arg1, %arg2: tensor<64xf32>
  return %ret : tensor<64xf32>
}


