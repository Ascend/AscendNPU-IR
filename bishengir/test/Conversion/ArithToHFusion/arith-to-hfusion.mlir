// RUN: bishengir-opt -convert-arith-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_addf
func.func @test_addf(%arg0 : tensor<6x6xf32>, %arg1 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
  %ret = arith.addf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_addi
func.func @test_addi(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
  %ret = arith.addi %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_subf
func.func @test_subf(%arg0 : tensor<6x6xf32>, %arg1 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
  %ret = arith.subf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_subi
func.func @test_subi(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
  %ret = arith.subi %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_mulf
func.func @test_mulf(%arg0 : tensor<6x6xf32>, %arg1 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
  %ret = arith.mulf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_muli
func.func @test_muli(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
  %ret = arith.muli %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_mulhi
func.func @test_mulhi(%arg0 : tensor<6xi32>, %arg1 : tensor<6xi32>) -> tensor<6xi32> {
  // CHECK:       %[[VAL_2:.*]], %[[VAL_3:.*]] = hfusion.mulext
  %low, %high = arith.mulsi_extended %arg0, %arg1 : tensor<6xi32>
  return %high : tensor<6xi32>
}

// -----

// CHECK-LABEL: func.func @test_divf
func.func @test_divf(%arg0 : tensor<6x6xf32>, %arg1 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
  %ret = arith.divf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_divsi
func.func @test_divsi(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
  %ret = arith.divsi %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_divui
func.func @test_divui(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
  %ret = arith.divui %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_maxsi
func.func @test_maxsi(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
  %ret = arith.maxsi %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_maxui
func.func @test_maxui(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_unsigned>}
  %ret = arith.maxui %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_minsi
func.func @test_minsi(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
  %ret = arith.minsi %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_minui
func.func @test_minui(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_unsigned>}
  %ret = arith.minui %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_shli
func.func @test_shli(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>}
  %ret = arith.shli %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_shrsi
func.func @test_shrsi(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>}
  %ret = arith.shrsi %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_shrui
func.func @test_shrui(%arg0 : tensor<6xi32>, %arg1 : tensor<6xi32>) -> tensor<6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>}
  %ret = arith.shrui %arg0, %arg1 : tensor<6xi32>
  return %ret : tensor<6xi32>
}

// -----

// CHECK-LABEL: func.func @test_andi
func.func @test_andi(%arg0 : tensor<6x6xi1>, %arg1 : tensor<6x6xi1>) -> tensor<6x6xi1> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
  %ret = arith.andi %arg0, %arg1 : tensor<6x6xi1>
  return %ret : tensor<6x6xi1>
}

// -----

// CHECK-LABEL: func.func @test_ori
func.func @test_ori(%arg0 : tensor<6x6xi1>, %arg1 : tensor<6x6xi1>) -> tensor<6x6xi1> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
  %ret = arith.ori %arg0, %arg1 : tensor<6x6xi1>
  return %ret : tensor<6x6xi1>
}

// -----

// CHECK-LABEL: func.func @test_maxf
func.func @test_maxf(%arg0 : tensor<512xf16>, %arg1 : tensor<512xf16>) -> tensor<512xf16> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun =  #hfusion.binary_fn<maxf>}
  %ret = arith.maxnumf %arg0, %arg1 : tensor<512xf16>
  return %ret : tensor<512xf16>
}

// -----

// CHECK-LABEL: func.func @test_minf
func.func @test_minf(%arg0 : tensor<512xf16>, %arg1 : tensor<512xf16>) -> tensor<512xf16> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
  %ret =  arith.minnumf %arg0, %arg1 : tensor<512xf16>
  return %ret : tensor<512xf16>
}

// -----

// CHECK-LABEL: func.func @test_truncf_f32_f16
func.func @test_truncf_f32_f16(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf16> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
  %ret = arith.truncf %arg0 : tensor<6x6xf32> to tensor<6x6xf16>
  return %ret : tensor<6x6xf16>
}

// -----

// CHECK-LABEL: func.func @test_truncf_f32_bf16
func.func @test_truncf_f32_bf16(%arg0 : tensor<6x6xf32>) -> tensor<6x6xbf16> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
  %ret = arith.truncf %arg0 : tensor<6x6xf32> to tensor<6x6xbf16>
  return %ret : tensor<6x6xbf16>
}

// -----

// CHECK-LABEL: func.func @test_extf_f16_f32
func.func @test_extf_f16_f32(%arg0 : tensor<6x6xf16>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
  %ret = arith.extf %arg0 : tensor<6x6xf16> to tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_extf_bf16_f32
func.func @test_extf_bf16_f32(%arg0 : tensor<6x6xbf16>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
  %ret = arith.extf %arg0 : tensor<6x6xbf16> to tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_extui_i1_i8
func.func @test_extui_i1_i8(%arg0 : tensor<6x6xi1>) -> tensor<6x6xi8> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
  %ret = arith.extui %arg0 : tensor<6x6xi1> to tensor<6x6xi8>
  return %ret : tensor<6x6xi8>
}

// -----

// CHECK-LABEL: func.func @test_fptosi_f32_i32
func.func @test_fptosi_f32_i32(%arg0 : tensor<6x6xf32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<trunc>}
  %ret = arith.fptosi %arg0 : tensor<6x6xf32> to tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// -----

// CHECK-LABEL: func.func @test_sitofp_i32_f32
func.func @test_sitofp_i32_f32(%arg0 : tensor<6x6xi32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
  %ret = arith.sitofp %arg0 : tensor<6x6xi32> to tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_mod_i32
func.func @test_mod_i32(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun =  #hfusion.binary_fn<mod>}
  %ret = arith.remsi %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}


// -----

// CHECK-LABEL: func.func @test_maximumf
func.func @test_maximumf(%arg0 : tensor<512xf16>, %arg1 : tensor<512xf16>) -> tensor<512xf16> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
  %ret =  arith.maximumf %arg0, %arg1 : tensor<512xf16>
  return %ret : tensor<512xf16>
}

// -----

// CHECK-LABEL: func.func @test_minimumf
func.func @test_minimumf(%arg0 : tensor<512xf16>, %arg1 : tensor<512xf16>) -> tensor<512xf16> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
  %ret =  arith.minimumf %arg0, %arg1 : tensor<512xf16>
  return %ret : tensor<512xf16>
}

// -----

// CHECK-LABEL: func.func @test_trunci_i64_i32
func.func @test_trunci_i64_i32(%arg0 : tensor<6x6xi64>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
  %ret = arith.trunci %arg0 : tensor<6x6xi64> to tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}




// -----

// CHECK-LABEL: func.func @test_xori
func.func @test_xori(%arg0 : tensor<512xi16>, %arg1 : tensor<512xi16>) -> tensor<512xi16> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>}
  %ret =  arith.xori %arg0, %arg1 : tensor<512xi16>
  return %ret : tensor<512xi16>
}

// -----

// CHECK-LABEL: func.func @test_constant_to_fill
func.func @test_constant_to_fill() -> (tensor<1024xf32>) {
  // CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<1024xf32>
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[ONE:.*]] =  linalg.fill ins(%[[CST:.*]]  : f32) outs(%[[ZERO:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
  %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32>

  // CHECK: %[[CST0:.*]] =  arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %cst0 = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>

  // CHECK: %[[TWO:.*]] = tensor.empty() : tensor<2xf32>
  // CHECK: %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[THREE:.*]] =  linalg.fill ins(%[[CST1:.*]]  : f32) outs(%[[TWO:.*]] : tensor<2xf32>) -> tensor<2xf32>
  %cst1 = arith.constant dense<[1.000000e+00, 1.000000e+00]> : tensor<2xf32>

  return %cst : tensor<1024xf32>
}


// -----

// CHECK-LABEL: func.func @test_ceildivsi
func.func @test_ceildivsi(%arg0 : tensor<6x6xi8>, %arg1 : tensor<6x6xi8>) -> tensor<6x6xi8> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>}
  %ret = arith.ceildivsi %arg0, %arg1 : tensor<6x6xi8>
  return %ret : tensor<6x6xi8>
}


// -----

// CHECK-LABEL: func.func @test_floordivsi
func.func @test_floordivsi(%arg0 : tensor<6x6xi8>, %arg1 : tensor<6x6xi8>) -> tensor<6x6xi8> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>}
  %ret = arith.floordivsi %arg0, %arg1 : tensor<6x6xi8>
  return %ret : tensor<6x6xi8>
}

// -----

// CHECK-LABEL: func.func @test_ceildivsi
func.func @test_ceildivsi(%arg0 : tensor<6x6xi32>, %arg1 : tensor<6x6xi32>) -> tensor<6x6xi32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>}
  %ret = arith.ceildivsi %arg0, %arg1 : tensor<6x6xi32>
  return %ret : tensor<6x6xi32>
}

// ----

// CHECK-LABEL: func.func @test_negf
func.func @test_negf(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
  %ret = arith.negf %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// ----

// CHECK-LABEL: func.func @test_arith_cmpf
func.func @test_arith_cmpf(%arg1 : tensor<32xf32>, %arg2 : tensor<32xf32>) -> tensor<32xi1> {
  // CHECK:       %[[EMPTY:.*]] = tensor.empty()
  // CHECK:       %[[RET:.*]] =  hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins({{.*}}, {{.*}} : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xi1>) -> tensor<32xi1>
  %1 = arith.cmpf une, %arg1, %arg2 : tensor<32xf32>
  return %1 : tensor<32xi1>
}

// ----

// CHECK-LABEL: func.func @test_arith_bitcast
func.func @test_arith_bitcast(%arg : tensor<32xf32>) -> tensor<32xi32> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty()
  // CHECK: %[[RET:.*]] = hfusion.bitcast ins(%[[arg0:.*]] : tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xi32>) -> tensor<32xi32>
  %1 = arith.bitcast %arg : tensor<32xf32> to tensor<32xi32>
  return %1 : tensor<32xi32>
}