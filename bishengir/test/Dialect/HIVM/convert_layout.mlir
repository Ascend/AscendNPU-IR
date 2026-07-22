// RUN: bishengir-opt -canonicalize %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_propagate_convert_layout_down_scf_if(
// CHECK: %[[select:.*]] = arith.select
// CHECK: hivm.hir.convert_layout %[[select]] output_shape [128, 128]
module {
  func.func @test_propagate_convert_layout_down_scf_if(%arg0: i32){
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %alloc_0 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %alloc_1 = memref.alloc() : memref<128x128xbf16, #hivm.address_space<cbuf>>
    %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %1 = scf.if %0 -> (memref<128x128xbf16, #hivm.address_space<cbuf>>) {
      %2 = hivm.hir.convert_layout %alloc output_shape [128, 128]  {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<128x128xbf16, #hivm.address_space<cbuf>>
      scf.yield %2 : memref<128x128xbf16, #hivm.address_space<cbuf>>
    } else {
      %2 = hivm.hir.convert_layout %alloc_0 output_shape [128, 128]  {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<128x128xbf16, #hivm.address_space<cbuf>>
      scf.yield %2 : memref<128x128xbf16, #hivm.address_space<cbuf>>
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 {already_set_real_mkn} ins(%1, %alloc_1, %true, %c128, %c128, %c128 : memref<128x128xbf16, #hivm.address_space<cbuf>>, memref<128x128xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_2 : memref<128x128xf32, #hivm.address_space<cc>>)
    return
  }
}

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
