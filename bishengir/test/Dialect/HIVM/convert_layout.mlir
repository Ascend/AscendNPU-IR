// RUN: bishengir-opt -canonicalize %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// Fractal → ND convert_layout: [K1,M1,16,16] → [M,K]
// CHECK-LABEL: func.func @test_convert_fractal_A_to_ND
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
module {
  func.func @test_convert_fractal_A_to_ND(%arg0: tensor<20x10x16x16xf16>) -> tensor<160x320xf16> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<20x10x16x16xf16>) -> tensor<160x320xf16>
    return %0 : tensor<160x320xf16>
  }
}

// -----
// ND → Fractal C convert_layout: [M,N] → [N1,M1,16,16]
// CHECK-LABEL: func.func @test_convert_ND_to_fractal_C
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [5, 10, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
module {
  func.func @test_convert_ND_to_fractal_C(%arg0: tensor<160x80xf32>) -> tensor<5x10x16x16xf32> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [5, 10, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<160x80xf32>) -> tensor<5x10x16x16xf32>
    return %0 : tensor<5x10x16x16xf32>
  }
}

// -----
// ND → Fractal B convert_layout: [K,N] → [N1,K1,16,16]
// CHECK-LABEL: func.func @test_convert_ND_to_fractal_B
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [5, 20, 16, 16]
module {
  func.func @test_convert_ND_to_fractal_B(%arg0: tensor<320x80xf16>) -> tensor<5x20x16x16xf16> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [5, 20, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<320x80xf16>) -> tensor<5x20x16x16xf16>
    return %0 : tensor<5x20x16x16xf16>
  }
}

// -----
// Fractal int8 A → ND convert_layout: [K1,M1,16,32] → [M,K]
// CHECK-LABEL: func.func @test_convert_fractal_int8_A_to_ND
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [160, 320]
module {
  func.func @test_convert_fractal_int8_A_to_ND(%arg0: tensor<10x10x16x32xi8>) -> tensor<160x320xi8> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>} : (tensor<10x10x16x32xi8>) -> tensor<160x320xi8>
    return %0 : tensor<160x320xi8>
  }
}

// -----
// s_CV_f32: Fractal f32 A → ND, fractalSizes = [16, 8]
// CHECK-LABEL: func.func @test_convert_fractal_f32_A_to_ND
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>}
module {
  func.func @test_convert_fractal_f32_A_to_ND(%arg0: tensor<40x10x16x8xf32>) -> tensor<160x320xf32> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>} : (tensor<40x10x16x8xf32>) -> tensor<160x320xf32>
    return %0 : tensor<160x320xf32>
  }
}

// -----
// s_C_int8: Fractal int8 B → ND, fractalSizes = [32, 32] for B side
// CHECK-LABEL: func.func @test_convert_fractal_int8_B_to_ND
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [320, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>}
module {
  func.func @test_convert_fractal_int8_B_to_ND(%arg0: tensor<10x2x32x32xi8>) -> tensor<320x64xi8> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [320, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>} : (tensor<10x2x32x32xi8>) -> tensor<320x64xi8>
    return %0 : tensor<320x64xi8>
  }
}

// -----
// s_CV_bf16: Fractal bf16 A → ND, with non-square fractalSizes for bf16
// CHECK-LABEL: func.func @test_convert_fractal_bf16_A_to_ND
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [256, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
module {
  func.func @test_convert_fractal_bf16_A_to_ND(%arg0: tensor<8x16x16x16xbf16>) -> tensor<256x128xbf16> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [256, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<8x16x16x16xbf16>) -> tensor<256x128xbf16>
    return %0 : tensor<256x128xbf16>
  }
}

// -----
// s_C_int8: ND→Fractal int8 C, fractalSizes = [16, 32]
// CHECK-LABEL: func.func @test_convert_ND_to_fractal_int8_C
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [2, 10, 16, 32] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, srcLayout = #hivm.data_layout<ND>}
module {
  func.func @test_convert_ND_to_fractal_int8_C(%arg0: tensor<160x64xi8>) -> tensor<2x10x16x32xi8> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [2, 10, 16, 32] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>, srcLayout = #hivm.data_layout<ND>} : (tensor<160x64xi8>) -> tensor<2x10x16x32xi8>
    return %0 : tensor<2x10x16x32xi8>
  }
}

// -----
// s_CV_f32: ND→Fractal f32 C, fractalSizes = [16, 8]
// CHECK-LABEL: func.func @test_convert_ND_to_fractal_f32_C
// CHECK: hivm.hir.convert_layout %{{.*}} output_shape [5, 10, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>}
module {
  func.func @test_convert_ND_to_fractal_f32_C(%arg0: tensor<80x160xf32>) -> tensor<5x10x16x8xf32> {
    %0 = hivm.hir.convert_layout %arg0 output_shape [5, 10, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>} : (tensor<80x160xf32>) -> tensor<5x10x16x8xf32>
    return %0 : tensor<5x10x16x8xf32>
  }
}
