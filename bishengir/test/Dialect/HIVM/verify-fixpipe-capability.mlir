// RUN: bishengir-opt --hivm-verify-capability --split-input-file --verify-diagnostics %s

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @missing_scope(%src: tensor<16x16xf32>, %dst: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // expected-error @below {{result is unknown; src or dst memory scope is not on this operation and was not set for this query}}
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%src : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @native_c220_gm(%src: memref<16x16xf32, #hivm.address_space<cc>>,
                            %dst: memref<16x16xf32, #hivm.address_space<gm>>) {
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%src : memref<16x16xf32, #hivm.address_space<cc>>)
      outs(%dst : memref<16x16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @unknown_c220_l1(%src: memref<16x16xf32, #hivm.address_space<cc>>,
                             %dst: memref<16x16xf32, #hivm.address_space<cbuf>>) {
    // expected-error @below {{result is unknown; target is Ascend910B1 (dav-c220); coverage is partial}}
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%src : memref<16x16xf32, #hivm.address_space<cc>>)
      outs(%dst : memref<16x16xf32, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @native_c310_ub(%src: memref<16x16xf32, #hivm.address_space<cc>>,
                            %dst: memref<16x16xf16, #hivm.address_space<ub>>) {
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>,
                      pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
      ins(%src : memref<16x16xf32, #hivm.address_space<cc>>)
      outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend310B1">} {
  func.func @no_profile(%src: memref<16x16xf32, #hivm.address_space<cc>>,
                        %dst: memref<16x16xf32, #hivm.address_space<gm>>) {
    // expected-error @below {{result is unknown; target is Ascend310B1 (dav-m300); coverage is none}}
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%src : memref<16x16xf32, #hivm.address_space<cc>>)
      outs(%dst : memref<16x16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @missing_target(%src: tensor<16x16xf32>, %dst: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // expected-error @below {{no #hacc.target on this operation or an enclosing module; this check does not assume Ascend910B}}
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%src : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}

// -----

module attributes {hacc.target = #hacc.target<"not-a-device">} {
  func.func @unrecognized(%src: memref<16x16xf32, #hivm.address_space<cc>>,
                          %dst: memref<16x16xf32, #hivm.address_space<gm>>) {
    // expected-error @below {{result is unknown; target is Unknown (unknown); coverage is none}}
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%src : memref<16x16xf32, #hivm.address_space<cc>>)
      outs(%dst : memref<16x16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @outer_unknown(%src: memref<16x16xf32, #hivm.address_space<cc>>,
                           %dst: memref<16x16xf32, #hivm.address_space<cbuf>>) {
    // expected-error @below {{result is unknown; target is Ascend910B1 (dav-c220); coverage is partial}}
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%src : memref<16x16xf32, #hivm.address_space<cc>>)
      outs(%dst : memref<16x16xf32, #hivm.address_space<cbuf>>)
    return
  }
  module attributes {hacc.target = #hacc.target<"Ascend950">} {
    func.func @inner_native(%src: memref<16x16xf32, #hivm.address_space<cc>>,
                            %dst: memref<16x16xf16, #hivm.address_space<ub>>) {
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>,
                        pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
        ins(%src : memref<16x16xf32, #hivm.address_space<cc>>)
        outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
      return
    }
  }
}
