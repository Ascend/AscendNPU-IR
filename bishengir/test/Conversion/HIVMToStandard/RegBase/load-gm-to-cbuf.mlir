// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std -split-input-file | FileCheck %s

// GM->L1 (cbuf) copies on RegBase targets must lower to 1D library calls:
// the RegBase template library only registers load_gm_to_cbuf_1d_* symbols.
// A strided copy that cannot be flattened to 1D is decomposed into a loop
// of 1D copies instead of emitting an unsupported 2D call.

module {
  // CHECK-LABEL: func.func @load_gm_to_cbuf_2d_strided_to_1d
  func.func @load_gm_to_cbuf_2d_strided_to_1d(%arg0: memref<?xbf16, #hivm.address_space<gm>>, %arg1: memref<4x1024xbf16, #hivm.address_space<cbuf>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 1024], strides: [8192, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<4x1024xbf16, strided<[8192, 1]>, #hivm.address_space<gm>>
    // CHECK: scf.for
    // CHECK: call @load_gm_to_cbuf_1d_bfloat16_t
    // CHECK-NOT: call @load_gm_to_cbuf_2d_bfloat16_t
    hivm.hir.load ins(%reinterpret_cast : memref<4x1024xbf16, strided<[8192, 1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<4x1024xbf16, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @load_gm_to_cbuf_1d_direct
  func.func @load_gm_to_cbuf_1d_direct(%arg0: memref<4096xbf16, #hivm.address_space<gm>>, %arg1: memref<4096xbf16, #hivm.address_space<cbuf>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK-NOT: scf.for
    // CHECK: call @load_gm_to_cbuf_1d_bfloat16_t
    hivm.hir.load ins(%arg0 : memref<4096xbf16, #hivm.address_space<gm>>) outs(%arg1 : memref<4096xbf16, #hivm.address_space<cbuf>>)
    return
  }
}
