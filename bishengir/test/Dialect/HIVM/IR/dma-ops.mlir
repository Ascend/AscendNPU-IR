// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect -mlir-print-op-generic %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: test_nd2nz
func.func @test_nd2nz() {
  %gmA = memref.alloc() : memref<1024x2048xf16>
  %gmASubview = memref.subview %gmA[0, 0][256, 128][1, 1]
                       : memref<1024x2048xf16> to
                         memref<256x128xf16, strided<[2048, 1], offset: 0>>
  %l1A = memref.alloc() : memref<256x128xf16>
  hivm.hir.nd2nz ins(%gmASubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
                 outs(%l1A: memref<256x128xf16>)
  hivm.hir.nd2nz {dst_continuous}
                ins(%gmASubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
                outs(%l1A: memref<256x128xf16>)
  return
}

// -----
// CHECK-LABEL: test_nd2nz_tensor
func.func @test_nd2nz_tensor() {
  %gmA = tensor.empty() : tensor<1024x2048xf16>
  %gmASubview = tensor.extract_slice %gmA[0, 0][256, 128][1, 1]
                       : tensor<1024x2048xf16> to
                         tensor<256x128xf16>
  %l1A = tensor.empty() : tensor<256x128xf16>
  %ret = hivm.hir.nd2nz ins(%gmASubview : tensor<256x128xf16>)
                        outs(%l1A: tensor<256x128xf16>) -> tensor<256x128xf16>
  %ret1 = hivm.hir.nd2nz ins(%gmASubview : tensor<256x128xf16>)
                         outs(%l1A: tensor<256x128xf16>) -> tensor<256x128xf16>
  return
}


// -----
// CHECK-LABEL: func @test_nd2nz_tensor_init_out_buffer
func.func @test_nd2nz_tensor_init_out_buffer(%arg0: memref<?x?x?x?xf32, #hivm.address_space<gm>>) {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous}  ins(%arg0 : memref<?x?x?x?xf32, #hivm.address_space<gm>>) outs(%alloc : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = %cst : f32
  return
}

// -----
// CHECK-LABEL: test_fixpipe
func.func @test_fixpipe() {
  %gmC = memref.alloc() : memref<1024x2048xf16>
  %gmCSubview = memref.subview %gmC[0, 0][256, 128][1, 1]
                       : memref<1024x2048xf16> to
                         memref<256x128xf16, strided<[2048, 1], offset: 0>>
  %l0c = memref.alloc() : memref<256x128xf16>
  // Normal data movement
  hivm.hir.fixpipe ins(%l0c : memref<256x128xf16>)
                   outs(%gmCSubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
  // NZ2ND data movement
  hivm.hir.fixpipe {enable_nz2nd} ins(%l0c : memref<256x128xf16>)
                                  outs(%gmCSubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
  return
}

// -----
// CHECK-LABEL: test_fixpipe_tensor
func.func @test_fixpipe_tensor() {
  %gmC = tensor.empty() : tensor<1024x2048xf16>
  %gmCSubview = tensor.extract_slice %gmC[0, 0][256, 128][1, 1]
                       : tensor<1024x2048xf16> to
                         tensor<256x128xf16>
  %l0c = tensor.empty() : tensor<256x128xf16>
  // Normal data movement
  %ret = hivm.hir.fixpipe ins(%l0c : tensor<256x128xf16>)
                          outs(%gmCSubview : tensor<256x128xf16>) -> tensor<256x128xf16>
  // NZ2ND data movement
  %ret1 = hivm.hir.fixpipe {enable_nz2nd} ins(%l0c : tensor<256x128xf16>)
                                          outs(%gmCSubview : tensor<256x128xf16>) -> tensor<256x128xf16>
  // f322f16 pre quant on the fly
  %l0c1 = tensor.empty() : tensor<256x128xf32>
  %ret2 = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
                           ins(%l0c1 : tensor<256x128xf32>)
                           outs(%gmCSubview : tensor<256x128xf16>) -> tensor<256x128xf16>
  // leaky relu on the fly
  %ret3 = hivm.hir.fixpipe {pre_relu = #hivm.fixpipe_pre_relu_mode<LEAKY_RELU>}
                           ins(%l0c : tensor<256x128xf16>)
                           outs(%gmCSubview : tensor<256x128xf16>) -> tensor<256x128xf16>
  return
}

// -----
// CHECK-LABEL: hivm_memref_copy_gm_to_ub_pad_value_only
func.func @hivm_memref_copy_gm_to_ub_pad_value_only() {
  %val = arith.constant 10.0 : f16
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                pad_value = %val : f16
  return
}

// -----
// CHECK-LABEL: hivm_memref_copy_gm_to_ub_pad_value
func.func @hivm_memref_copy_gm_to_ub_pad_value() {
  %val = arith.constant 10.0 : f16
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                pad_mode = #hivm.padmode<PadValue>
                pad_value = %val : f16
  return
}

// -----
// CHECK-LABEL: hivm_memref_copy_gm_to_ub_pad_first
func.func @hivm_memref_copy_gm_to_ub_pad_first() {
  %src = memref.alloc() : memref<16x15xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x15xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                pad_mode = #hivm.padmode<PadFirstElem>
  return
}

// -----
// CHECK-LABEL: hivm_memref_load_gm_to_ub
func.func @hivm_memref_load_gm_to_ub() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}

// -----
// CHECK-LABEL: hivm_memref_copy_gm_to_ub_pad_value
func.func @hivm_memref_copy_gm_to_ub_pad_value() {
  %val = arith.constant 16 : index
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                left_padding_num = %val : index
  return
}

// -----
// CHECK-LABEL: hivm_tensor_copy_gm_to_ub
func.func @hivm_tensor_copy_gm_to_ub() -> tensor<16x16xf32> {
  %src = tensor.empty() : tensor<16x16xf32>
  %dst = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.copy ins(%src : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>)
                       -> tensor<16x16xf32>
  return %res : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: hivm_memref_copy_ub_to_ub
func.func @hivm_memref_copy_ub_to_ub() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}

// -----
// CHECK-LABEL: hivm_memref_store_ub_to_gm
func.func @hivm_memref_store_ub_to_gm() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  hivm.hir.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
  return
}

// -----
// CHECK-LABEL: noncontiguous_copy
func.func @noncontiguous_copy(%src : memref<64x32x4xbf16, strided<[512, 16, 1]>, #hivm.address_space<ub>>,
                              %dst : memref<64x32x4xbf16, #hivm.address_space<ub>>) {
  // CHECK: hivm.hir.copy
  hivm.hir.copy ins(%src : memref<64x32x4xbf16, strided<[512, 16, 1]>, #hivm.address_space<ub>>)
                outs(%dst : memref<64x32x4xbf16, #hivm.address_space<ub>>)
                collapse_reassociation = [[0, 1, 2]]
  return
}