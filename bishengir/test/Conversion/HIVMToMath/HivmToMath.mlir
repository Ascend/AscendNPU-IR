// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s
 
// ========== case1-1 vexp F16 ==========
// CHECK-LABEL: func.func @test_func_vexp_f16
func.func @test_func_vexp_f16(%arg0:tensor<16x16xf16>) -> tensor<16x16xf16> {
    %0 = tensor.empty():tensor<16x16xf16>
    //     CHECK: math.exp
    hivm.hir.vexp ins(%arg0 : tensor<16x16xf16>) outs(%0:tensor<16x16xf16>)  -> tensor<16x16xf16>
    return %0 : tensor<16x16xf16>
}

// ========== case1-2 vexp F32 ==========
// CHECK-LABEL: func.func @test_func_vexp_f32
func.func @test_func_vexp_f32(%arg0:tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = tensor.empty():tensor<16x16xf32>
    //     CHECK: math.exp
    hivm.hir.vexp ins(%arg0 : tensor<16x16xf32>) outs(%0:tensor<16x16xf32>)  -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
}

// ========== case2-1 vabs F32 ==========
// CHECK-LABEL: func.func @test_vabs_f32
func.func @test_vabs_f32(%arg0:tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = tensor.empty():tensor<16x16xf32>
  // CHECK: math.absf %arg0 : tensor<16x16xf32>
  hivm.hir.vabs ins(%arg0 : tensor<16x16xf32>) outs(%0:tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// ========== case2-2 vabs I32 ==========
// CHECK-LABEL: func.func @test_vabs_i32
func.func @test_vabs_i32(%arg0:tensor<8x8xi32>) -> tensor<8x8xi32> {
  %0 = tensor.empty():tensor<8x8xi32>
  // CHECK: math.absi %arg0 : tensor<8x8xi32>
  hivm.hir.vabs ins(%arg0 : tensor<8x8xi32>) outs(%0:tensor<8x8xi32>) -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// ========== case3-1 vln F32==========
// CHECK-LABEL: func.func @test_vln_f32_2d
func.func @test_vln_f32_2d(%arg0:tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = tensor.empty():tensor<16x32xf32>
  // CHECK: math.log %arg0 : tensor<16x32xf32>
  %1 = hivm.hir.vln ins(%arg0 : tensor<16x32xf32>) outs(%0:tensor<16x32xf32>) -> tensor<16x32xf32>
  return %1 : tensor<16x32xf32>
}

// ========== case3-2 vln F16==========
// CHECK-LABEL: func.func @test_vln_f16_1d
func.func @test_vln_f16_1d(%arg0:tensor<64xf16>) -> tensor<64xf16> {
  %0 = tensor.empty():tensor<64xf16>
  // CHECK: math.log %arg0 : tensor<64xf16>
  %1 = hivm.hir.vln ins(%arg0 : tensor<64xf16>) outs(%0:tensor<64xf16>) -> tensor<64xf16>
  return %1 : tensor<64xf16>
}

// ========== case3-3 vln F32 Scalar==========
// CHECK-LABEL: func.func @test_vln_f32_scalar
func.func @test_vln_f32_scalar(%arg0:tensor<f32>) -> tensor<f32> {
  %0 = tensor.empty():tensor<f32>
  // CHECK: math.log %arg0 : tensor<f32>
  %1 = hivm.hir.vln ins(%arg0 : tensor<f32>) outs(%0:tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}


// ========== case4-1 vsqrt f32==========
// CHECK-LABEL: func.func @test_vsqrt_f32_2d
func.func @test_vsqrt_f32_2d(%arg0:tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = tensor.empty():tensor<16x32xf32>
  // CHECK: math.sqrt %arg0 : tensor<16x32xf32>
  %1 = hivm.hir.vsqrt ins(%arg0 : tensor<16x32xf32>) outs(%0:tensor<16x32xf32>) -> tensor<16x32xf32>
  return %1 : tensor<16x32xf32>
}

// ========== case4-2 vsqrt f16==========
// CHECK-LABEL: func.func @test_vsqrt_f16_1d
func.func @test_vsqrt_f16_1d(%arg0:tensor<64xf16>) -> tensor<64xf16> {
  %0 = tensor.empty():tensor<64xf16>
  // CHECK: math.sqrt %arg0 : tensor<64xf16>
  %1 = hivm.hir.vsqrt ins(%arg0 : tensor<64xf16>) outs(%0:tensor<64xf16>) -> tensor<64xf16>
  return %1 : tensor<64xf16>
}

// ========== case4-3 vsqrt f32 scalar==========
// CHECK-LABEL: func.func @test_vsqrt_f32_scalar
func.func @test_vsqrt_f32_scalar(%arg0:tensor<f32>) -> tensor<f32> {
  %0 = tensor.empty():tensor<f32>
  // CHECK: math.sqrt %arg0 : tensor<f32>
  %1 = hivm.hir.vsqrt ins(%arg0 : tensor<f32>) outs(%0:tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// ========== case5-1 vrsqrt f32 ==========
// CHECK-LABEL: func.func @test_vrsqrt_f32_2d
func.func @test_vrsqrt_f32_2d(%arg0:tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = tensor.empty():tensor<16x32xf32>
  // CHECK: math.rsqrt %arg0 : tensor<16x32xf32>
  %1 = hivm.hir.vrsqrt ins(%arg0 : tensor<16x32xf32>) outs(%0 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %1 : tensor<16x32xf32>
}

// ========== case6-1 vtanh f32 ==========
// CHECK-LABEL: func.func @test_vtanh_f32_2d
func.func @test_vtanh_f32_2d(%arg0:tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = tensor.empty():tensor<16x32xf32>
  // CHECK: math.tanh %arg0 : tensor<16x32xf32>
  %1 = hivm.hir.vtanh ins(%arg0 : tensor<16x32xf32>) outs(%0:tensor<16x32xf32>) -> tensor<16x32xf32>
  return %1 : tensor<16x32xf32>
}

// ========== case6-2 vtanh f16 ==========
// CHECK-LABEL: func.func @test_vtanh_f16_1d
func.func @test_vtanh_f16_1d(%arg0:tensor<64xf16>) -> tensor<64xf16> {
  %0 = tensor.empty():tensor<64xf16>
  // CHECK: math.tanh %arg0 : tensor<64xf16>
  %1 = hivm.hir.vtanh ins(%arg0 : tensor<64xf16>) outs(%0:tensor<64xf16>) -> tensor<64xf16>
  return %1 : tensor<64xf16>
}

// ========== case7-1 vsin f32 ==========
// CHECK-LABEL: func.func @test_vsin_f32_2d
func.func @test_vsin_f32_2d(%arg0:tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = tensor.empty():tensor<16x32xf32>
  // CHECK: math.sin %arg0 : tensor<16x32xf32>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<16x32xf32>) outs(%0 : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %1 : tensor<16x32xf32>
}

// ========== case7-2 vsin f16 ==========
// CHECK-LABEL: func.func @test_vsin_f16_1d
func.func @test_vsin_f16_1d(%arg0:tensor<64xf16>) -> tensor<64xf16> {
  %0 = tensor.empty():tensor<64xf16>
  // CHECK: math.sin %arg0 : tensor<64xf16>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<64xf16>) outs(%0 : tensor<64xf16>) -> tensor<64xf16>
  return %1 : tensor<64xf16>
}

// ========== case8-1 vcos f32 ==========
// CHECK-LABEL: func.func @test_vcos_f32_2d
func.func @test_vcos_f32_2d(%arg0:tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty():tensor<32x32xf32>
  // CHECK: math.cos %arg0 : tensor<32x32xf32>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// ========== case8-2 vcos f16 ==========
// CHECK-LABEL: func.func @test_vcos_f16_1d
func.func @test_vcos_f16_1d(%arg0:tensor<128xf16>) -> tensor<128xf16> {
  %0 = tensor.empty():tensor<128xf16>
  // CHECK: math.cos %arg0 : tensor<128xf16>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<128xf16>) outs(%0 : tensor<128xf16>) -> tensor<128xf16>
  return %1 : tensor<128xf16>
}

// ========== case9-1 verf f32 ==========
// CHECK-LABEL: func.func @test_verf_f32_2d
func.func @test_verf_f32_2d(%arg0:tensor<24x24xf32>) -> tensor<24x24xf32> {
  %0 = tensor.empty():tensor<24x24xf32>
  // CHECK: math.erf %arg0 : tensor<24x24xf32>
  %1 = hivm.hir.verf ins(%arg0 : tensor<24x24xf32>) outs(%0 : tensor<24x24xf32>) -> tensor<24x24xf32>
  return %1 : tensor<24x24xf32>
}

// ========== case9-2 verf f32 ==========
// CHECK-LABEL: func.func @test_verf_f16_1d
func.func @test_verf_f16_1d(%arg0:tensor<256xf16>) -> tensor<256xf16> {
  %0 = tensor.empty():tensor<256xf16>
  // CHECK: math.erf %arg0 : tensor<256xf16>
  %1 = hivm.hir.verf ins(%arg0 : tensor<256xf16>) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  return %1 : tensor<256xf16>
}


// ========== case10-1 vpow f32 ==========
// CHECK-LABEL: func.func @test_vpow_i32_1d_binary
func.func @test_vpow_i32_1d_binary(%arg0:tensor<64xi32>, %arg1:tensor<64xi32>) -> tensor<64xi32> {
  %0 = tensor.empty():tensor<64xi32>
  // CHECK: math.ipowi %arg0, %arg1 : tensor<64xi32>
  %1 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<64xi32>, tensor<64xi32>) outs(%0 : tensor<64xi32>) -> tensor<64xi32>
  return %1 : tensor<64xi32>
}

// ========== case10-2 vpow i32 ==========
// CHECK-LABEL: func.func @test_vpow_i32_1d_temp_buffer
func.func @test_vpow_i32_1d_temp_buffer(%arg0:tensor<32xi32>, %arg1:tensor<32xi32>, %buf:memref<128xi32>) -> tensor<32xi32> {
  %0 = tensor.empty():tensor<32xi32>
  // CHECK: math.ipowi %arg0, %arg1 : tensor<32xi32>
  %1 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<32xi32>, tensor<32xi32>) outs(%0 : tensor<32xi32>) temp_buffer(%buf : memref<128xi32>) -> tensor<32xi32>
  return %1 : tensor<32xi32>
}

