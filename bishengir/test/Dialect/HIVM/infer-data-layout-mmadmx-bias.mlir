// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-data-layout -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_infer_data_layout_mmadmx_with_bias
module {
  func.func @test_infer_data_layout_mmadmx_with_bias(%arg0 : i32,
                                          %arg1 : i32,
                                          %arg2 : i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256xf32>
    %ret = scf.for %iv = %arg0 to %arg1 step %arg2 iter_args(%l0c = %alloc) -> (memref<256x256xf32>) : i32 {
      %l1A = memref.alloc() : memref<256x128xf8E5M2>
      %l1B = memref.alloc() : memref<128x256xf8E5M2>
      %scaleA = memref.alloc() : memref<256x4xui8>
      %scaleB = memref.alloc() : memref<256x4xui8>
      %bias = memref.alloc() : memref<1x256xf32>
      %init_cond = arith.cmpi eq, %iv, %arg1 : i32
      // CHECK: hivm.hir.mmadmxL1
      // expected-warning@+1 {{Unsupported user for propagating data layout}}
      %idk = hivm.hir.mmadmxL1
        ins(%l1A, %l1B, %scaleA, %scaleB, %init_cond, %c256, %c128, %c256, %bias :
            memref<256x128xf8E5M2>, memref<128x256xf8E5M2>,
            memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index,
            memref<1x256xf32>)
        outs(%l0c : memref<256x256xf32>) -> memref<256x256xf32>
      scf.yield %l0c : memref<256x256xf32>
    }
    return
  }
}
