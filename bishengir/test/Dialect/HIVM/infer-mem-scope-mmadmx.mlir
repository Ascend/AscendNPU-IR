// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-mem-scope -split-input-file -verify-diagnostics | FileCheck %s

// -----
module {
  // CHECK-LABEL: test_infer_mem_scope_mmadmx_with_bias
  func.func @test_infer_mem_scope_mmadmx_with_bias(%arg0: i32, %arg1: i32, %arg2: i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    // CHECK: #hivm.address_space<cc>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256xf32>
    %0 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %alloc) -> (memref<256x256xf32>)  : i32 {
      // CHECK: #hivm.address_space<cbuf>
      %alloc_a = memref.alloc() : memref<256x128xf8E5M2>
      // CHECK: #hivm.address_space<cbuf>
      %alloc_b = memref.alloc() : memref<128x256xf8E5M2>
      // CHECK: #hivm.address_space<cbuf>
      %alloc_sa = memref.alloc() : memref<256x4xui8>
      // CHECK: #hivm.address_space<cbuf>
      %alloc_sb = memref.alloc() : memref<256x4xui8>
      // CHECK: #hivm.address_space<cbuf>
      %alloc_bias = memref.alloc() : memref<1x256xf32>
      %1 = arith.cmpi eq, %arg3, %arg1 : i32
      hivm.hir.mmadmxL1
        ins(%alloc_a, %alloc_b, %alloc_sa, %alloc_sb, %1, %c256, %c128, %c256, %alloc_bias :
            memref<256x128xf8E5M2>, memref<128x256xf8E5M2>,
            memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index,
            memref<1x256xf32>)
        outs(%arg4 : memref<256x256xf32>)
      scf.yield %arg4 : memref<256x256xf32>
    }
    return
  }
}
