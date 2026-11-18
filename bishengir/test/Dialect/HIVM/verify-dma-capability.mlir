// RUN: bishengir-opt --hivm-verify-capability --split-input-file --verify-diagnostics %s

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @native_nd2nz(%src: memref<16x16xf32, #hivm.address_space<gm>>,
                          %dst: memref<2x2x16x16xf32, #hivm.address_space<cbuf>>) {
    hivm.hir.nd2nz ins(%src : memref<16x16xf32, #hivm.address_space<gm>>)
      outs(%dst : memref<2x2x16x16xf32, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @unknown_l12ub(%src: memref<16x16xf32, #hivm.address_space<cbuf>>,
                           %dst: memref<16x16xf32, #hivm.address_space<ub>>) {
    // expected-error @below {{result is unknown; target is Ascend910B1 (dav-c220); coverage is none}}
    hivm.hir.l12ub ins(%src : memref<16x16xf32, #hivm.address_space<cbuf>>)
      outs(%dst : memref<16x16xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @missing_scope(%src: tensor<16x16xf32>, %dst: tensor<2x2x16x16xf32>) {
    // expected-error @below {{result is unknown; src or dst memory scope is not on this operation and was not set for this query}}
    %0 = hivm.hir.nd2nz ins(%src : tensor<16x16xf32>) outs(%dst : tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32>
    return
  }
}
