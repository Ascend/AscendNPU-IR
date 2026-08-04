// RUN: bishengir-opt %s -convert-hivm-to-std -split-input-file -verify-diagnostics | FileCheck %s

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: test_mmadmxL1_with_per_channel_bias
  func.func @test_mmadmxL1_with_per_channel_bias() {
    %ma = memref.alloc() : memref<256x128xf8E5M2>
    %mb = memref.alloc() : memref<128x256xf8E5M2>
    %ma_t = memref.alloc() : memref<128x256xf8E5M2>
    %mb_t = memref.alloc() : memref<256x128xf8E5M2>
    %scaleA = memref.alloc() : memref<256x4xui8>
    %scaleB = memref.alloc() : memref<256x4xui8>
    %bias = memref.alloc() : memref<1x256xf32>
    %mc = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    // CHECK: call @mmadmxL1_with_float_bias_float8_e5m2_t_to_float
    hivm.hir.mmadmxL1
      ins(%ma, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256, %bias :
          memref<256x128xf8E5M2>, memref<128x256xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index,
          memref<1x256xf32>)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_with_float_bias_float8_e5m2_t_to_float_ta
    hivm.hir.mmadmxL1 {a_transpose}
      ins(%ma_t, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256, %bias :
          memref<128x256xf8E5M2>, memref<128x256xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index,
          memref<1x256xf32>)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_with_float_bias_float8_e5m2_t_to_float_tb
    hivm.hir.mmadmxL1 {b_transpose}
      ins(%ma, %mb_t, %scaleA, %scaleB, %init, %c256, %c128, %c256, %bias :
          memref<256x128xf8E5M2>, memref<256x128xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index,
          memref<1x256xf32>)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_with_float_bias_float8_e5m2_t_to_float_ta_tb
    hivm.hir.mmadmxL1 {a_transpose, b_transpose}
      ins(%ma_t, %mb_t, %scaleA, %scaleB, %init, %c256, %c128, %c256, %bias :
          memref<128x256xf8E5M2>, memref<256x128xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index,
          memref<1x256xf32>)
      outs(%mc : memref<256x256xf32>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: test_nd2nz_mmadmx_forbias
  // CHECK-NOT: hivm
  func.func @test_nd2nz_mmadmx_forbias() {
    %gmBias = memref.alloc() : memref<1x256xf32>
    %bias = memref.alloc() : memref<1x1x1x256xf32>
    %ma = memref.alloc() : memref<256x128xf8E5M2>
    %mb = memref.alloc() : memref<128x256xf8E5M2>
    %scaleA = memref.alloc() : memref<256x4xui8>
    %scaleB = memref.alloc() : memref<256x4xui8>
    %mc = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    // CHECK: call @nd2nz_forbias_float
    hivm.hir.nd2nz {dst_continuous}
      ins(%gmBias : memref<1x256xf32>)
      outs(%bias : memref<1x1x1x256xf32>)

    // CHECK: call @mmadmxL1_with_float_bias_float8_e5m2_t_to_float
    hivm.hir.mmadmxL1
      ins(%ma, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256, %bias :
          memref<256x128xf8E5M2>, memref<128x256xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index,
          memref<1x1x1x256xf32>)
      outs(%mc : memref<256x256xf32>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: test_nd2nz_mmadmx_dst_is_c_not_bias
  // ND2NZ dst feeds mmadmx C (not per-channel bias) → plain nd2nz, not forbias.
  func.func @test_nd2nz_mmadmx_dst_is_c_not_bias() {
    %gmC = memref.alloc() : memref<256x256xf32>
    %mc = memref.alloc() : memref<256x256xf32>
    %ma = memref.alloc() : memref<256x128xf8E5M2>
    %mb = memref.alloc() : memref<128x256xf8E5M2>
    %scaleA = memref.alloc() : memref<256x4xui8>
    %scaleB = memref.alloc() : memref<256x4xui8>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    // CHECK: call @nd2nz_float
    // CHECK-NOT: nd2nz_forbias
    hivm.hir.nd2nz {dst_continuous}
      ins(%gmC : memref<256x256xf32>)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_
    hivm.hir.mmadmxL1
      ins(%ma, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xf8E5M2>, memref<128x256xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)
    return
  }
}
