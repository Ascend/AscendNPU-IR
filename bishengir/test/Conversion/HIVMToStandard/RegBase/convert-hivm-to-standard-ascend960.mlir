// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960 -convert-hivm-to-std -split-input-file | FileCheck %s

// Ascend960-only MMAD_MX coverage: fp4/hif4 library-call name mapping, MMAD
// config control bits combined with transposes, and the heterogeneous
// native-A x packed-B form whose format suffix is only appended on Ascend960.

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  // CHECK-LABEL: test_mmadmxL1_fp4_hif4_format
  func.func @test_mmadmxL1_fp4_hif4_format() {
    %ma = memref.alloc() : memref<256x128xi8>
    %mb = memref.alloc() : memref<128x256xi8>
    %scaleA = memref.alloc() : memref<256x4xui8>
    %scaleB = memref.alloc() : memref<256x4xui8>
    %mc = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    // CHECK: call @mmadmxL1_int8_t_to_float_lhs_format_fp4x2_e2m1_t_rhs_format_fp4x2_e2m1_t
    hivm.hir.mmadmxL1 {lhsFormat = 3 : i32, rhsFormat = 3 : i32}
      ins(%ma, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xi8>, memref<128x256xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_int8_t_to_float_lhs_format_fp4_hif4_t_rhs_format_fp4_hif4_t
    hivm.hir.mmadmxL1 {lhsFormat = 4 : i32, rhsFormat = 4 : i32}
      ins(%ma, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xi8>, memref<128x256xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  // CHECK-LABEL: test_mmadmxL1_a16w4
  // Native f16/bf16 activation x packed 4-bit weight (A16W4): the f16/bf16
  // type token and the rhs format token both appear in the library call name.
  func.func @test_mmadmxL1_a16w4() {
    %ma = memref.alloc() : memref<256x128xf16>
    %mb = memref.alloc() : memref<128x256xi8>
    %ma_bf16 = memref.alloc() : memref<256x128xbf16>
    %scaleA = memref.alloc() : memref<256x4xui8>
    %scaleB = memref.alloc() : memref<256x4xui8>
    %mc = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    // CHECK: call @mmadmxL1_half_int8_t_to_float_lhs_format_fp16_rhs_format_fp4x2_e2m1_t
    hivm.hir.mmadmxL1 {lhsFormat = 5 : i32, rhsFormat = 3 : i32}
      ins(%ma, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xf16>, memref<128x256xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_half_int8_t_to_float_lhs_format_fp16_rhs_format_fp4_hif4_t
    hivm.hir.mmadmxL1 {lhsFormat = 5 : i32, rhsFormat = 4 : i32}
      ins(%ma, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xf16>, memref<128x256xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_bfloat16_t_int8_t_to_float_lhs_format_bf16_rhs_format_fp4_hif4_t
    hivm.hir.mmadmxL1 {lhsFormat = 6 : i32, rhsFormat = 4 : i32}
      ins(%ma_bf16, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xbf16>, memref<128x256xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  // CHECK-LABEL: test_mmadmxL1_transpose_with_config
  // Transposes keep their ta/tb suffixes regardless of config control bits.
  func.func @test_mmadmxL1_transpose_with_config() {
    %ma = memref.alloc() : memref<256x128xi8>
    %mb = memref.alloc() : memref<128x256xi8>
    %ma_t = memref.alloc() : memref<128x256xi8>
    %mb_t = memref.alloc() : memref<256x128xi8>
    %scaleA = memref.alloc() : memref<256x4xui8>
    %scaleB = memref.alloc() : memref<256x4xui8>
    %mc = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    // CHECK: call @mmadmxL1_int8_t_to_float_ta_lhs_format_fp8_e5m2_t_rhs_format_fp8_e5m2_t
    hivm.hir.mmadmxL1 {a_transpose, lhsFormat = 1 : i32, rhsFormat = 1 : i32}
      ins(%ma_t, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<128x256xi8>, memref<128x256xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_int8_t_to_float_tb_lhs_format_fp8_e4m3_t_rhs_format_fp8_e4m3_t
    hivm.hir.mmadmxL1 {b_transpose, lhsFormat = 2 : i32, rhsFormat = 2 : i32}
      ins(%ma, %mb_t, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xi8>, memref<256x128xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_int8_t_to_float_ta_tb_lhs_format_fp8_e4m3_t_rhs_format_fp8_e4m3_t
    hivm.hir.mmadmxL1 {a_transpose, b_transpose, lhsFormat = 2 : i32, rhsFormat = 2 : i32}
      ins(%ma_t, %mb_t, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<128x256xi8>, memref<256x128xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  // CHECK-LABEL: test_mmadmxL1_hetero_e4m3_hif4
  // The heterogeneous native-A x packed-B form only appends the format suffix
  // on Ascend960, which the RUN line provides for this file.
  func.func @test_mmadmxL1_hetero_e4m3_hif4() {
    %ma = memref.alloc() : memref<256x128xf8E4M3FN>
    %mb = memref.alloc() : memref<128x256xi8>
    %scaleA = memref.alloc() : memref<256x4xui8>
    %scaleB = memref.alloc() : memref<256x4xui8>
    %mc = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    // CHECK: call @mmadmxL1_float8_e4m3_t_int8_t_to_float_lhs_format_fp8_e4m3_t_rhs_format_fp4_hif4_t
    hivm.hir.mmadmxL1 {lhsFormat = 2 : i32, rhsFormat = 4 : i32}
      ins(%ma, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xf8E4M3FN>, memref<128x256xi8>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)
    return
  }
}
