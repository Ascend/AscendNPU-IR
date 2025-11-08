// RUN: bishengir-opt --one-shot-bufferize="dialect-filter=hivm,bufferization copy-before-write unknown-type-conversion=identity-layout-map" -split-input-file %s | FileCheck %s
// RUN: bishengir-opt %s -one-shot-bufferize -split-input-file | FileCheck %s --check-prefix=ONE-SHOT

// CHECK-LABEL: test_fixpipe_mixed_type_bufferize
func.func @test_fixpipe_mixed_type_bufferize() {
  %gmC = memref.alloc() : memref<1024x2048xf16>
  // CHECK: %[[SUBVIEW:.*]] = memref.subview
  %gmCSubview = memref.subview %gmC[0, 0][256, 128][1, 1]
                       : memref<1024x2048xf16> to
                         memref<256x128xf16, strided<[2048, 1], offset: 0>>
  // CHECK: hivm.hir.fixpipe {enable_nz2nd}
  // CHECK-SAME: ins(%[[ALLOC:.*]] : memref<256x128xf16>)
  // CHECK-SAME: outs(%[[SUBVIEW]] : memref<256x128xf16, strided<[2048, 1]>>)
  %l0c = tensor.empty() : tensor<256x128xf16>
  hivm.hir.fixpipe {enable_nz2nd}
      ins(%l0c : tensor<256x128xf16>)
      outs(%gmCSubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)

  return
}

// -----

// CHECK-LABEL: test_mmadL1_bufferize
func.func @test_mmadL1_bufferize() {
  %mc = tensor.empty() : tensor<256x256xf32>
  %start = arith.constant 0 : index
  %end = arith.constant 1024 : index
  %step = arith.constant 128 : index
  %realM = arith.constant 256 : index
  %realK = arith.constant 128 : index
  %realN = arith.constant 256 : index
  %scf_ret = scf.for %arg0 = %start to %end step %step iter_args(%mC_iter = %mc) -> (tensor<256x256xf32>) {
    // Data loaded to L1
    %ma = tensor.empty() : tensor<256x128xf16>
    %mb = tensor.empty() : tensor<128x256xf16>
    // L0C data is cleared for the first iteration
    %init_condition = arith.cmpi eq, %arg0, %start : index
    // CHECK: hivm.hir.mmadL1
    // CHECK-SAME: ins({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}} : memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
    // CHECK-SAME: outs({{.*}} : memref<256x256xf32>)
    %res = hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %realM, %realK, %realN :
                                 tensor<256x128xf16>, tensor<128x256xf16>, i1, index, index, index)
                           outs(%mC_iter : tensor<256x256xf32>) -> tensor<256x256xf32>
    scf.yield %res : tensor<256x256xf32>
  }

  return
}

// -----

// CHECK-LABEL: test_binary_vector_op_vv_vs_bufferize
func.func @test_binary_vector_op_vv_vs_bufferize() {
  %src1 = tensor.empty() : tensor<256x256xf16>
  %src2 = tensor.empty() : tensor<256x256xf16>
  %dst = tensor.empty() : tensor<256x256xf16>
  %cst = arith.constant 256.0 : f16

  // CHECK: hivm.hir.vadd
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, memref<256x256xf16>)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vadd ins(%src1, %src2 : tensor<256x256xf16>, tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vmul
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, memref<256x256xf16>)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vmul ins(%src1, %src2 : tensor<256x256xf16>, tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vsub
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, memref<256x256xf16>)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vsub ins(%src1, %src2 : tensor<256x256xf16>, tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vdiv
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, memref<256x256xf16>)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vdiv ins(%src1, %src2 : tensor<256x256xf16>, tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vmax
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, memref<256x256xf16>)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vmax ins(%src1, %src2 : tensor<256x256xf16>, tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vmin
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, memref<256x256xf16>)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vmin ins(%src1, %src2 : tensor<256x256xf16>, tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vadd
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, f16)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vadd ins(%src1, %cst : tensor<256x256xf16>, f16)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vmul
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, f16)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vmul ins(%src1, %cst : tensor<256x256xf16>, f16)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vmax
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, f16)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vmax ins(%src1, %cst : tensor<256x256xf16>, f16)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vmin
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xf16>, f16)
  // CHECK-SAME: outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vmin ins(%src1, %cst : tensor<256x256xf16>, f16)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  return
}

// -----

// CHECK-LABEL: test_binary_vector_op_vv_vs_int16_bufferize
func.func @test_binary_vector_op_vv_vs_int16_bufferize() {
  %src1 = tensor.empty() : tensor<256x256xi16>
  %src2 = tensor.empty() : tensor<256x256xi16>
  %dst = tensor.empty() : tensor<256x256xi16>

  // CHECK: hivm.hir.vor
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xi16>, memref<256x256xi16>)
  // CHECK-SAME: outs({{.*}} : memref<256x256xi16>)
  hivm.hir.vor ins(%src1, %src2 : tensor<256x256xi16>, tensor<256x256xi16>)
               outs(%dst : tensor<256x256xi16>) -> tensor<256x256xi16>

  // CHECK: hivm.hir.vand
  // CHECK-SAME: ins({{.*}}, {{.*}} : memref<256x256xi16>, memref<256x256xi16>)
  // CHECK-SAME: outs({{.*}} : memref<256x256xi16>)
  hivm.hir.vand ins(%src1, %src2 : tensor<256x256xi16>, tensor<256x256xi16>)
                outs(%dst : tensor<256x256xi16>) -> tensor<256x256xi16>

  return
}

// -----

// CHECK-LABEL: test_unary_vector_op_bufferize
func.func @test_unary_vector_op_bufferize() {
  %src = tensor.empty() : tensor<256x256xf16>
  %dst = tensor.empty() : tensor<256x256xf16>

  // CHECK: hivm.hir.vexp
  // CHECK-SAME: ins({{.*}} : memref<256x256xf16>) outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vexp ins(%src : tensor<256x256xf16>) 
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  // CHECK: hivm.hir.vabs
  // CHECK-SAME: ins({{.*}} : memref<256x256xf16>) outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vabs ins(%src : tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>
  
  // CHECK: hivm.hir.vln
  // CHECK-SAME: ins({{.*}} : memref<256x256xf16>) outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vln ins(%src : tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>
  
  // CHECK: hivm.hir.vrelu
  // CHECK-SAME: ins({{.*}} : memref<256x256xf16>) outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vrelu ins(%src : tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>
  
  // CHECK: hivm.hir.vrsqrt
  // CHECK-SAME: ins({{.*}} : memref<256x256xf16>) outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vrsqrt ins(%src : tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>
  
  // CHECK: hivm.hir.vsqrt
  // CHECK-SAME: ins({{.*}} : memref<256x256xf16>) outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vsqrt ins(%src : tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>
  
  // CHECK: hivm.hir.vrec
  // CHECK-SAME: ins({{.*}} : memref<256x256xf16>) outs({{.*}} : memref<256x256xf16>)
  hivm.hir.vrec ins(%src : tensor<256x256xf16>)
                outs(%dst : tensor<256x256xf16>) -> tensor<256x256xf16>

  return
}

// -----

// CHECK-LABEL: test_unary_vector_op_int16_t_bufferize
func.func @test_unary_vector_op_int16_t_bufferize() {
  %src = tensor.empty() : tensor<256x256xi16>
  %dst = tensor.empty() : tensor<256x256xi16>

  // CHECK: hivm.hir.vnot
  // CHECK-SAME: ins({{.*}} : memref<256x256xi16>) outs({{.*}} : memref<256x256xi16>)
  hivm.hir.vnot ins(%src : tensor<256x256xi16>) outs(%dst : tensor<256x256xi16>) -> tensor<256x256xi16>

  return
}

// -----

// CHECK-LABEL: test_vbrc_bufferize
func.func @test_vbrc_bufferize() {
  %src = tensor.empty() : tensor<1xf16>
  %dst = tensor.empty() : tensor<128xf16>
  // CHECK: hivm.hir.vbrc
  // CHECK-SAME: ins({{.*}} : memref<1xf16>)
  // CHECK-SAME: outs({{.*}} : memref<128xf16>)
  hivm.hir.vbrc ins(%src : tensor<1xf16>)
                outs(%dst : tensor<128xf16>)
                broadcast_dims = [0] -> tensor<128xf16>

  %src1 = tensor.empty() : tensor<1x1xf16>
  %dst1 = tensor.empty() : tensor<128x128xf16>
  // CHECK: hivm.hir.vbrc
  // CHECK-SAME: ins({{.*}} : memref<1x1xf16>)
  // CHECK-SAME: outs({{.*}} : memref<128x128xf16>)
  hivm.hir.vbrc ins(%src1 : tensor<1x1xf16>)
                outs(%dst1 : tensor<128x128xf16>)
                broadcast_dims = [0, 1] -> tensor<128x128xf16>
  
  %cst = arith.constant 256.0 : f16
  // CHECK: hivm.hir.vbrc
  // CHECK-SAME: ins({{.*}} : f16)
  // CHECK-SAME: outs({{.*}} : memref<128xf16>)
  hivm.hir.vbrc ins(%cst : f16)
                outs(%dst : tensor<128xf16>)
                broadcast_dims = [] -> tensor<128xf16>

  return
}

// -----

// CHECK-LABEL: test_vreduce_op_tensor_bufferize
func.func @test_vreduce_op_tensor_bufferize() {
  %src = tensor.empty() : tensor<128xf16>
  %dst = tensor.empty() : tensor<1xf16>

  // CHECK: hivm.hir.vreduce
  // CHECK-SAME: ins({{.*}} : memref<128xf16>)
  // CHECK-SAME: outs({{.*}} : memref<1xf16>)
  hivm.hir.vreduce <sum> ins(%src : tensor<128xf16>)
                         outs(%dst : tensor<1xf16>)
                         reduce_dims = [0] -> tensor<1xf16>

  // CHECK: hivm.hir.vreduce
  // CHECK-SAME: ins({{.*}} : memref<128xf16>)
  // CHECK-SAME: outs({{.*}} : memref<1xf16>)
  hivm.hir.vreduce <max> ins(%src : tensor<128xf16>)
                         outs(%dst : tensor<1xf16>)
                         reduce_dims = [0] -> tensor<1xf16>

  // CHECK: hivm.hir.vreduce
  // CHECK-SAME: ins({{.*}} : memref<128xf16>)
  // CHECK-SAME: outs({{.*}} : memref<1xf16>)
  hivm.hir.vreduce <min> ins(%src : tensor<128xf16>)
                         outs(%dst : tensor<1xf16>)
                         reduce_dims = [0] -> tensor<1xf16>

  return
}

// -----

// CHECK-LABEL: test_vcast_op_bufferize
func.func @test_vcast_op_bufferize() {
  %f16 = tensor.empty() : tensor<128x128xf16>
  %f32 = tensor.empty() : tensor<128x128xf32>

  // CHECK: hivm.hir.vcast
  // CHECK-SAME: ins({{.*}} : memref<128x128xf16>)
  // CHECK-SAME: outs({{.*}} : memref<128x128xf32>)
  hivm.hir.vcast ins(%f16 : tensor<128x128xf16>) outs(%f32 : tensor<128x128xf32>)
                 round_mode = #hivm.round_mode<rint> -> tensor<128x128xf32>

  return
}

// -----
// CHECK-LABEL: hivm_tensor_copy_gm_to_ub
func.func @hivm_tensor_copy_gm_to_ub() -> tensor<16x16xf32> {
  // CHECK: %[[T0:.*]] = tensor.empty() : tensor<16x16xf32>
  // CHECK: %[[T1:.*]] = bufferization.to_memref %[[T0]] : memref<16x16xf32>
  // CHECK: %[[T2:.*]] = tensor.empty() : tensor<16x16xf32>
  // CHECK: %[[T3:.*]] = bufferization.to_memref %[[T2]] : memref<16x16xf32>
  // CHECK: %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  // CHECK: memref.copy %[[T3]], %[[alloc]] : memref<16x16xf32> to memref<16x16xf32>
  // CHECK: hivm.hir.copy ins(%[[T1]] : memref<16x16xf32>) outs(%[[alloc]] : memref<16x16xf32>)
  // CHECK: %[[T4:.*]] = bufferization.to_tensor %[[alloc]] : memref<16x16xf32>

  // ONE-SHOT: %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  // ONE-SHOT: %[[alloc_0:.*]] = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  // ONE-SHOT: hivm.hir.copy ins(%[[alloc]] : memref<16x16xf32>) outs(%[[alloc_0]] : memref<16x16xf32>)
  // ONE-SHOT: %[[T0:.*]] = bufferization.to_tensor %[[alloc_0]] : memref<16x16xf32>

  %src = tensor.empty() : tensor<16x16xf32>
  %dst = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.copy ins(%src : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>)
                       -> tensor<16x16xf32>
  return %res : tensor<16x16xf32>
}

// -----
// CHECK: func.func @hivm_matmul_basic_args(%[[arg0:.*]]: tensor<16x16xf16>, %[[arg1:.*]]: tensor<16x16xf16>, %[[arg2:.*]]: tensor<16x16xf16>)
func.func @hivm_matmul_basic_args(%A : tensor<16x16xf16>,
                                  %B : tensor<16x16xf16>,
                                  %C : tensor<16x16xf16>) -> tensor<16x16xf16> {
  // CHECK: %[[T0:.*]] = bufferization.to_memref %[[arg1]] : memref<16x16xf16>
  // CHECK: %[[T1:.*]] = bufferization.to_memref %[[arg0]] : memref<16x16xf16>
  // CHECK: %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16>
  // CHECK: hivm.hir.matmul ins(%[[T1]], %[[T0]] : memref<16x16xf16>, memref<16x16xf16>) outs(%[[alloc]] : memref<16x16xf16>) descale_mode = <DescaleNull>
  // CHECK: %[[T2:.*]] = bufferization.to_tensor %[[alloc]] : memref<16x16xf16>

  // ONE-SHOT: hivm.hir.matmul ins({{.*}}, {{.*}} : memref<16x16xf16, strided<[?, ?], offset: ?>>, memref<16x16xf16, strided<[?, ?], offset: ?>>) outs({{.*}} : memref<16x16xf16>) descale_mode = <DescaleNull>

  %0 = hivm.hir.matmul
     ins(%A, %B: tensor<16x16xf16>, tensor<16x16xf16>)
     outs(%C : tensor<16x16xf16>)
     descale_mode = #hivm.descale_mode<DescaleNull> -> tensor<16x16xf16>
  return %0 : tensor<16x16xf16>
}

// -----
func.func @hivm_matmul_basic() -> tensor<16x16xf16> {
  %A = tensor.empty() : tensor<16x16xf16>
  %B = tensor.empty() : tensor<16x16xf16>
  %C = tensor.empty() : tensor<16x16xf16>

  // CHECK: hivm.hir.matmul ins({{.*}}, {{.*}} : memref<16x16xf16>, memref<16x16xf16>) outs({{.*}} : memref<16x16xf16>) descale_mode = <DescaleNull>
  // ONE-SHOT: hivm.hir.matmul ins({{.*}}, {{.*}} : memref<16x16xf16>, memref<16x16xf16>) outs({{.*}} : memref<16x16xf16>) descale_mode = <DescaleNull>

  %0 = hivm.hir.matmul
     ins(%A, %B: tensor<16x16xf16>, tensor<16x16xf16>)
     outs(%C : tensor<16x16xf16>)
     descale_mode = #hivm.descale_mode<DescaleNull> -> tensor<16x16xf16>
  return %0 : tensor<16x16xf16>
}

// -----

  func.func @matmul_add_0_300_mix_aiv(%arg0: i64 {hfusion.ffts_base_address},
                                      %arg1: tensor<1024x1024xf16>,
                                      %arg2: tensor<1024x1024xf16>,
                                      %arg3: tensor<1024x1024xf16>,
                                      %arg4: tensor<1024x1024xf16>,
                                      %arg5: tensor<1024x1024xf16>) -> tensor<1024x1024xf16> {
    %c256_i64 = arith.constant 256 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    hivm.hir.set_ffts_base_addr %arg0
    // CHECK: hivm.hir.mix_matmul {post_vector_func = "bishengir_gen_vector_epilogue_func"} ins({{.*}}, {{.*}}  : memref<1024x1024xf16>, memref<1024x1024xf16>) post_vector_func_ins({{.*}} : memref<1024x1024xf16>) workspace_ins({{.*}}  : memref<1024x1024xf16>) outs({{.*}} : memref<1024x1024xf16>) block_sizes({{.*}}, {{.*}}, {{.*}} : i64, i64, i64) process_sizes({{.*}}, {{.*}}, {{.*}} : i64, i64, i64) swizzle_offset ={{.*}} : i64 swizzle_direction = {{.*}} : i64 epilogue_p_tiles = {{.*}} : i64
    %0 = hivm.hir.mix_matmul
       {post_vector_func = "bishengir_gen_vector_epilogue_func"}
       ins(%arg1, %arg2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
       post_vector_func_ins(%arg4 : tensor<1024x1024xf16>)
       workspace_ins(%arg5: tensor<1024x1024xf16>)
       outs(%arg3 : tensor<1024x1024xf16>)
       block_sizes(%c128_i64, %c256_i64, %c256_i64 : i64, i64, i64)
       process_sizes(%c128_i64, %c256_i64, %c64_i64 : i64, i64, i64)
       swizzle_offset = %c1_i64 : i64
       swizzle_direction = %c0_i64 : i64
       epilogue_p_tiles = %c4_i64 : i64 -> tensor<1024x1024xf16>
    return %0 : tensor<1024x1024xf16>
  }

// -----

module {
  func.func @test_vadd_bufferize() {
    %alloc = memref.alloc() : memref<256xf32>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %alloc_2 = memref.alloc() : memref<256xf32>
    %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<256xf32>

    // ONE-SHOT: %[[alloc:.*]] = memref.alloc() : memref<256xf32>
    // ONE-SHOT: %[[alloc_0:.*]] = memref.alloc() : memref<256xf32>
    // ONE-SHOT: hivm.hir.vadd ins(%[[alloc]], %[[alloc_0]] : memref<256xf32>, memref<256xf32>) outs(%[[alloc]] : memref<256xf32>)
    %11 = hivm.hir.vadd ins(%9, %10 : tensor<256xf32>, tensor<256xf32>) outs(%9 : tensor<256xf32>) -> tensor<256xf32>
    return
  }
}

// -----
module {
  func.func @test_vexp_bufferize() {
    %alloc = memref.alloc() : memref<256xf32>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>

    // ONE-SHOT: %[[alloc:.*]] = memref.alloc() : memref<256xf32>
    // ONE-SHOT: hivm.hir.vexp ins(%[[alloc]] : memref<256xf32>) outs(%[[alloc]] : memref<256xf32>)
    %11 = hivm.hir.vexp ins(%9 : tensor<256xf32>) outs(%9 : tensor<256xf32>) -> tensor<256xf32>
    return
  }
}

// -----
module {
  // CHECK: mix_matmul_with_tiling_param
  // CHECK: {{.*}}: tensor<?x4096xf16>
  // CHECK: {{.*}}: tensor<6144x4096xf16>
  // CHECK: {{.*}}: tensor<?x6144xf16>
  // CHECK: %[[TILING:.*]]: memref<20xi64>
  func.func @mix_matmul_with_tiling_param(%arg0: tensor<?x4096xf16>,
                                      %arg1: tensor<6144x4096xf16>,
                                      %arg2: tensor<?x6144xf16>,
                                      %arg3: memref<20xi64>) -> tensor<?x6144xf16> {
    // CHECK: hivm.hir.mix_matmul
    // CHECK: tiling_params = %[[TILING]] : memref<20xi64>
    %0 = hivm.hir.mix_matmul ins(%arg0, %arg1 : tensor<?x4096xf16>, tensor<6144x4096xf16>)
                         outs(%arg2 : tensor<?x6144xf16>)
                         tiling_params = %arg3 : memref<20xi64>
                         b_transpose
                         -> tensor<?x6144xf16>
    return %0 : tensor<?x6144xf16>
  }
}

// -----
module {
  func.func @test_load_pad_bufferize(%arg0: tensor<2047xf32>,%arg1: tensor<2047xf32>) {
    // ONE-SHOT-DAG: %[[arg1:.*]] = bufferization.to_memref %arg0 : memref<2047xf32, strided<[?], offset: ?>>
    // ONE-SHOT-DAG: %[[arg2:.*]] = bufferization.to_memref %arg1 : memref<2047xf32, strided<[?], offset: ?>>
    %c0 = arith.constant 0 : index
    %c2046 = arith.constant 2046 : index
    %cst = arith.constant 0.000000e+00 : f32
    // ONE-SHOT: %[[alloc1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2047xf32>
    // ONE-SHOT: hivm.hir.load ins(%[[arg1]] : memref<2047xf32, strided<[?], offset: ?>>) outs(%[[alloc1]] : memref<2047xf32>)
    %0 = tensor.empty() : tensor<2047xf32>
    %1 = hivm.hir.load ins(%arg0 : tensor<2047xf32>) outs(%0 : tensor<2047xf32>) -> tensor<2047xf32>
    // ONE-SHOT: %[[alloc2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4093xf32>
    // ONE-SHOT: hivm.hir.load ins(%[[arg2]] : memref<2047xf32, strided<[?], offset: ?>>) outs(%[[alloc2]] : memref<4093xf32>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c2046 : index right_padding_num = %c0 : index
    %2 = tensor.empty() : tensor<4093xf32>
    %3 = hivm.hir.load ins(%arg1 : tensor<2047xf32>) outs(%2 : tensor<4093xf32>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c2046 : index right_padding_num = %c0 : index -> tensor<4093xf32>
    return
  }
}

// -----
module {
// CHECK-LABEL: test_bitcast_bufferize
  func.func @test_bitcast_bufferize() {
    %alloc = memref.alloc() : memref<256xf32>
    // CHECK: {{.*}} = hivm.hir.bitcast {{.*}} : memref<256xf32> -> memref<256xi32>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %11 = hivm.hir.bitcast %9 : tensor<256xf32> -> tensor<256xi32>
    return
  }
}

// -----
module {
// CHECK-LABEL: test_bitcast_unranked_bufferize
  func.func @test_bitcast_unranked_bufferize(%t : tensor<*xf32>) {
    hivm.hir.bitcast %t : tensor<*xf32> -> tensor<*xi32>
    return
  }
}

// -----
// CHECK-LABEL: test_nd2nz_bufferize
func.func @test_nd2nz_bufferize() {
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
// CHECK-LABEL: test_mix_group_matmul_bufferize
func.func @test_mix_group_matmul_bufferize(%A_gm : tensor<16x16x16xf16>,
                                  %B_gm :  tensor<16x16xf16>,
                                  %tokens_per_expert_gm :  tensor<16xi64>,
                                  %bias_gm : tensor<16xf16>,
                                  %tiling_params_gm : tensor<16xf16>,
                                  %comm_params_gm : tensor<16xi64>,
                                  %res_gm :tensor<16x16xf16>,
                                  %post_vector_func_ins: tensor<1024x1024xf16>,
                                  %post_vector_func_outs: tensor<1024x1024xf16>) {
    hivm.hir.mix_group_matmul
        ins(%A_gm, %B_gm, %tokens_per_expert_gm:
            tensor<16x16x16xf16>, tensor<16x16xf16>,
            tensor<16xi64>)
        post_vector_func_ins(%post_vector_func_ins : tensor<1024x1024xf16>)
        post_vector_func_outs(%post_vector_func_outs : tensor<1024x1024xf16>)
        outs(%res_gm : tensor<16x16xf16>)
        tiling_params = %tiling_params_gm : tensor<16xf16>
        comm_params = %comm_params_gm : tensor<16xi64>
        -> tensor<16x16xf16>
    return
}

// -----
// CHECK-LABEL: test_vconcat_bufferize
func.func @test_vconcat_bufferize(%a_f16 : tensor<2x16xf16>, %b_f16 : tensor<2x16xf16>,
                               %c_f16 : tensor<2x32xf16>) {
  %res0 = hivm.hir.vconcat dim(1) ins(%a_f16, %b_f16: tensor<2x16xf16>, tensor<2x16xf16>)
                                  outs(%c_f16 : tensor<2x32xf16>) -> tensor<2x32xf16>
  return
}