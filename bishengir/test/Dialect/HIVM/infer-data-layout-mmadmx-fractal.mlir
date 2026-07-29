// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-data-layout -split-input-file -verify-diagnostics | FileCheck %s

// -----
// 4D fractal A/B with transpose flags + 4D bias for getOperandBiasLayout(kDimFour).
// CHECK-LABEL: func.func @test_infer_layout_mmadmx_fractal_transpose_bias4d
module {
  func.func @test_infer_layout_mmadmx_fractal_transpose_bias4d() {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    // 4D fractal A (nZ / a_transpose): [M1,K1,16,16]
    %a = memref.alloc() : memref<4x2x16x16xf8E5M2, #hivm.address_space<cbuf>>
    // 4D fractal B (nZ / b_transpose): [K1,N1,16,16]
    %b = memref.alloc() : memref<2x4x16x16xf8E5M2, #hivm.address_space<cbuf>>
    %sa = memref.alloc() : memref<64x1xui8, #hivm.address_space<cbuf>>
    %sb = memref.alloc() : memref<64x1xui8, #hivm.address_space<cbuf>>
    // 4D bias -> zN layout branch in getOperandBiasLayout
    %bias = memref.alloc() : memref<1x1x1x64xf32, #hivm.address_space<cbuf>>
    %c = memref.alloc() : memref<64x64xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.mmadmxL1
    hivm.hir.mmadmxL1 {a_transpose, b_transpose}
      ins(%a, %b, %sa, %sb, %true, %c64, %c32, %c64, %bias :
          memref<4x2x16x16xf8E5M2, #hivm.address_space<cbuf>>,
          memref<2x4x16x16xf8E5M2, #hivm.address_space<cbuf>>,
          memref<64x1xui8, #hivm.address_space<cbuf>>,
          memref<64x1xui8, #hivm.address_space<cbuf>>,
          i1, index, index, index,
          memref<1x1x1x64xf32, #hivm.address_space<cbuf>>)
      outs(%c : memref<64x64xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// 2D A/B with fractal_layout hint so transpose is treated as layout-only.
// CHECK-LABEL: func.func @test_infer_layout_mmadmx_fractal_hint_suppress_transpose
module {
  func.func @test_infer_layout_mmadmx_fractal_hint_suppress_transpose() {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %a = memref.alloc() {hivm.fractal_layout = "nZ"} : memref<64x32xf8E5M2, #hivm.address_space<cbuf>>
    %b = memref.alloc() {hivm.fractal_layout = "nZ"} : memref<32x64xf8E5M2, #hivm.address_space<cbuf>>
    %sa = memref.alloc() : memref<64x1xui8, #hivm.address_space<cbuf>>
    %sb = memref.alloc() : memref<64x1xui8, #hivm.address_space<cbuf>>
    %c = memref.alloc() : memref<64x64xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.mmadmxL1
    hivm.hir.mmadmxL1 {a_transpose, b_transpose}
      ins(%a, %b, %sa, %sb, %true, %c64, %c32, %c64 :
          memref<64x32xf8E5M2, #hivm.address_space<cbuf>>,
          memref<32x64xf8E5M2, #hivm.address_space<cbuf>>,
          memref<64x1xui8, #hivm.address_space<cbuf>>,
          memref<64x1xui8, #hivm.address_space<cbuf>>,
          i1, index, index, index)
      outs(%c : memref<64x64xf32, #hivm.address_space<cc>>)
    return
  }
}
