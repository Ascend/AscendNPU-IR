// RUN: bishengir-opt --hivm-insert-cv-tight-coupled-buffer -split-input-file %s | FileCheck %s


// -----
module {
  func.func @vf_mm_func(%x : tensor<16x16xf16>)
                      -> tensor<16x16xf16>
                      attributes {hivm.vector_function} {
    %dst = tensor.empty() : tensor<16x16xf16>
    %one = arith.constant 1.000000e+00 : f16
    %r = hivm.hir.vmul ins(%x, %one : tensor<16x16xf16>, f16)
                       outs(%dst : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %r : tensor<16x16xf16>
  }
  
  // CHECK-LABEL: func.func @test_fixpipe_vf(
  func.func @test_fixpipe_vf(%src : tensor<16x16xf32>)
             -> tensor<16x16xf16> {
    %dst = tensor.empty() : tensor<16x16xf16>


    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
    // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
    // CHECK: hivm.hir.fixpipe {{.*}} ins({{.*}} : tensor<16x16xf32>) outs(%[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>>
    // CHECK: %[[TOT:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16x16xf16>
    // CHECK: %[[RES:.*]] = call @vf_mm_func(%[[TOT]])


    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
             ins(%src : tensor<16x16xf32>)
             outs(%dst : tensor<16x16xf16>) -> tensor<16x16xf16>

    %res = call @vf_mm_func(%fix) {hivm.vector_function} 
           : (tensor<16x16xf16>) -> tensor<16x16xf16>

    return %res : tensor<16x16xf16>
  }
}

// -----
module {
  func.func @vf_mm_func(%x : tensor<16x16xf16>)
                      -> tensor<16x16xf16>
                      attributes {hivm.vector_function} {
    %dst = tensor.empty() : tensor<16x16xf16>
    %one = arith.constant 1.000000e+00 : f16
    %r = hivm.hir.vmul ins(%x, %one : tensor<16x16xf16>, f16)
                       outs(%dst : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %r : tensor<16x16xf16>
  }

  // CHECK-LABEL: func.func @test_vf_mm(
  func.func @test_vf_mm(%a : tensor<16x16xf16>,
                        %b : tensor<16x16xf16>)
             -> tensor<16x16xf32> {
    %true = arith.constant 1 : i1
    %c16 = arith.constant 16 : index
    %mm_dst = tensor.empty() : tensor<16x16xf32>

    %vf_res = func.call @vf_mm_func(%a) {hivm.vector_function}
              : (tensor<16x16xf16>) -> tensor<16x16xf16>
    // CHECK: %[[EXPAND:.*]] = tensor.expand_shape {{.*}} {{\[\[0, 1\], \[2, 3\]\]}} output_shape {{\[1, 16, 1, 16\]}} : tensor<16x16xf16> into tensor<1x16x1x16xf16>
    // CHECK: %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<1x1x16x16xf16>
    // CHECK: %[[TRANSPOSE:.*]] =  hivm.hir.vtranspose ins(%[[EXPAND]] : {{.*}}) outs(%[[EMPTY_TENSOR]]

    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : 
    // CHECK: %[[EMPTY_T:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : 
    // CHECK: hivm.hir.copy ins(%[[TRANSPOSE]] : tensor<1x1x16x16xf16>) outs(%[[CAST:.*]] : memref<1x1x16x16xf16>)
    %mm = hivm.hir.mmadL1
            ins(%vf_res, %b, %true, %c16, %c16, %c16
                : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
            outs(%mm_dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mm : tensor<16x16xf32>
  }
}

// -----
module {
  // CHECK-LABEL: func.func @test_insert_slice_mm(
  // CHECK-SAME: %[[ARG0:.*]]: tensor<16x16xf16>, %[[ARG1:.*]]: tensor<16x16xf16>) -> tensor<16x16xf16> {
  func.func @test_insert_slice_mm(%a : tensor<16x16xf16>,
                          %b : tensor<16x16xf16>)
               -> tensor<16x16xf16>
  {
    %true = arith.constant 1 : i1
    %mm_dst = tensor.empty() : tensor<16x16xf16>
    %c16 = arith.constant 16 : index
    %cst_4 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = hivm.hir.vbrc ins(%cst_4 : f16) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %extracted_slice_30 = tensor.extract_slice %a[0, 0] [%c16, %c16] [1, 1] : tensor<16x16xf16> to tensor<?x?xf16>
    %inserted_slice = tensor.insert_slice %extracted_slice_30 into %1[0, 0] [%c16, %c16] [1, 1] : tensor<?x?xf16> into tensor<16x16xf16>
    %mm = hivm.hir.mmadL1
        ins(%a, %inserted_slice, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%mm_dst : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %mm : tensor<16x16xf16>
  }

  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK:   %[[C16:.*]] = arith.constant 16 : index
  // CHECK:   %[[TRUE:.*]] = arith.constant true
  // CHECK:   %[[EMPTY1:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK:   %[[EMPTY2:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK:   %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f16) outs(%[[EMPTY2]] : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK:   %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [%[[C16]], %[[C16]]] [1, 1] : tensor<16x16xf16> to tensor<?x?xf16>
  // CHECK:   %[[INSERT:.*]] = tensor.insert_slice %[[EXTRACT]] into %[[VBRC]][0, 0] [%[[C16]], %[[C16]]] [1, 1] : tensor<?x?xf16> into tensor<16x16xf16>
  // CHECK: %[[EXPAND:.*]] = tensor.expand_shape %[[INSERT]] {{\[\[0, 1\], \[2, 3\]\]}} output_shape {{\[1, 16, 1, 16\]}} : tensor<16x16xf16> into tensor<1x16x1x16xf16>
  // CHECK:   %[[EMPTY_T:.*]] = tensor.empty() : tensor<1x1x16x16xf16>
  // CHECK:   %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[EXPAND]] : tensor<1x16x1x16xf16>) outs(%[[EMPTY_T]] : tensor<1x1x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<1x1x16x16xf16>
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK:   %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
  // CHECK:   %[[BUF_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<1x1x16x16xf16>
  // CHECK:   hivm.hir.copy ins(%[[TRANSPOSE]] : tensor<1x1x16x16xf16>) outs(%[[CAST:.*]] : memref<1x1x16x16xf16>)
  // CHECK:   %[[MM:.*]] = hivm.hir.mmadL1 ins(%[[ARG0]], %[[BUF_TENSOR:.*]], %[[TRUE]], %[[C16]], %[[C16]], %[[C16]] : tensor<16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%[[EMPTY1]] : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK:   return %[[MM]] : tensor<16x16xf16>
}

// -----
module {
  // CHECK-LABEL: func.func @test_insert_slice_scf_for_mm(
  // CHECK-SAME: %[[ARG0:.*]]: tensor<16x16xf16>, %[[ARG1:.*]]: tensor<16x16xf16>) -> tensor<16x16xf16> {
  func.func @test_insert_slice_scf_for_mm(%a : tensor<16x16xf16>,
                          %b : tensor<16x16xf16>)
               -> tensor<16x16xf16>
  {
    %true = arith.constant 1 : i1
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mm_dst = tensor.empty() : tensor<16x16xf16>
    %c16 = arith.constant 16 : index
    %cst_4 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = scf.for %arg46 = %c0 to %c16 step %c1 iter_args(%arg47 = %mm_dst) -> (tensor<16x16xf16>) {
       %7 = tensor.empty() : tensor<16x16xf16>
       scf.yield %7 : tensor<16x16xf16>
    } {ExtractedLoadOrStore}
    %extracted_slice_30 = tensor.extract_slice %a[0, 0] [%c16, %c16] [1, 1] : tensor<16x16xf16> to tensor<?x?xf16>
    %inserted_slice = tensor.insert_slice %extracted_slice_30 into %1[0, 0] [%c16, %c16] [1, 1] : tensor<?x?xf16> into tensor<16x16xf16>
    %mm = hivm.hir.mmadL1
        ins(%a, %inserted_slice, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%mm_dst : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %mm : tensor<16x16xf16>
  }

  // CHECK:   %[[C16:.*]] = arith.constant 16 : index
  // CHECK:   %[[TRUE:.*]] = arith.constant true
  // CHECK:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK:   %[[C1:.*]] = arith.constant 1 : index
  // CHECK:   %[[EMPTY0:.*]] = tensor.empty() : tensor<16x16xf16>
  // CHECK:   %[[FOR_RES:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%{{.*}} = %[[EMPTY0]]) -> (tensor<16x16xf16>) {
  // CHECK:     %{{.*}} = tensor.empty() : tensor<16x16xf16>
  // CHECK:     scf.yield %{{.*}} : tensor<16x16xf16>
  // CHECK:   } {ExtractedLoadOrStore}
  // CHECK:   %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [%[[C16]], %[[C16]]] [1, 1] : tensor<16x16xf16> to tensor<?x?xf16>
  // CHECK:   %[[INSERT:.*]] = tensor.insert_slice %[[EXTRACT]] into %[[FOR_RES]][0, 0] [%[[C16]], %[[C16]]] [1, 1] : tensor<?x?xf16> into tensor<16x16xf16>
  // CHECK:   %[[EXPAND:.*]] = tensor.expand_shape %[[INSERT]] {{\[\[0, 1\], \[2, 3\]\]}} output_shape {{\[1, 16, 1, 16\]}} : tensor<16x16xf16> into tensor<1x16x1x16xf16>
  // CHECK:   %[[EMPTY_T:.*]] = tensor.empty() : tensor<1x1x16x16xf16>
  // CHECK:   %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[EXPAND]] : tensor<1x16x1x16xf16>) outs(%[[EMPTY_T]] : tensor<1x1x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<1x1x16x16xf16>
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK:   %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
  // CHECK:   %[[BUF_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<1x1x16x16xf16>
  // CHECK:   hivm.hir.copy ins(%[[TRANSPOSE]] : tensor<1x1x16x16xf16>) outs(%[[CAST:.*]] : memref<1x1x16x16xf16>)
  // CHECK:   %[[MM:.*]] = hivm.hir.mmadL1 ins(%[[ARG0]], %[[BUF_TENSOR:.*]], %[[TRUE]], %[[C16]], %[[C16]], %[[C16]] : tensor<16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%[[EMPTY0]] : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK:   return %[[MM]] : tensor<16x16xf16>
}

// -----
module {
  // CHECK-LABEL: func.func @test_vector_mm(
  // CHECK: %[[VMUL:.*]] = hivm.hir.vmul ins({{.*}} : tensor<16x16xf32>, f32) outs({{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
  // CHECK: %[[EXPAND:.*]] = tensor.expand_shape {{.*}} {{\[\[0, 1\], \[2, 3\]\]}} output_shape {{\[1, 16, 2, 8\]}} : tensor<16x16xf32> into tensor<1x16x2x8xf32>
  // CHECK: %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<2x1x16x8xf32>
  // CHECK: %[[TRANSPOSE:.*]] =  hivm.hir.vtranspose ins(%[[EXPAND]] : {{.*}}) outs(%[[EMPTY_TENSOR]]
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
  // CHECK: %[[CAST0:.*]] = memref.memory_space_cast %[[ALLOC0]] :
  // CHECK: %[[T0:.*]] = bufferization.to_tensor %[[CAST0]] restrict writable : memref<2x1x16x8xf32>
  // CHECK: hivm.hir.copy ins(%[[TRANSPOSE]] : tensor<2x1x16x8xf32>) outs(%[[CAST0:.*]] : memref<2x1x16x8xf32>)
  func.func @test_vector_mm(%arg0 : memref<?xf32>) {
    %cst_1 = arith.constant 2.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %init_condition = arith.constant 0 : i1
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
    %2 = bufferization.to_tensor %1  restrict writable : memref<16x16xf32, strided<[16, 1], offset: 0>>
    %3 = hivm.hir.vmul ins(%2, %cst_1 : tensor<16x16xf32>, f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = tensor.empty() : tensor<16x16xf32>
    %5 = hivm.hir.mmadL1 ins(%3, %3, %init_condition, %c16, %c16, %c16 :
                                  tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
                            outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
    hivm.hir.fixpipe ins(%5 : tensor<16x16xf32>) outs(%reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1], offset: 0>>)
    %6 = tensor.empty() : tensor<16x16xf32>
    %7 = hivm.hir.vmul ins(%3, %cst_1 : tensor<16x16xf32>, f32) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [1024], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 1024>>
    hivm.hir.store ins(%7 : tensor<16x16xf32>) outs(%reinterpret_cast_1 : memref<16x16xf32, strided<[16, 1], offset: 1024>>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: func.func @test_fixpipe_vector(
  // CHECK: %[[ALLOC_UB:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC_UB]] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
  // CHECK: hivm.hir.fixpipe {{.*}} ins({{.*}} : tensor<16x16xf32>) outs(%[[ALLOC_UB]] : memref<16x16xf16, #hivm.address_space<ub>>)
  // CHECK: %[[TT:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16x16xf16>
  // CHECK: %[[VMUL:.*]] = hivm.hir.vmul ins(%[[TT]], {{.*}} : tensor<16x16xf16>, f16) outs({{.*}} : tensor<16x16xf16>) -> tensor<16x16xf16>

  func.func @test_fixpipe_vector(%arg0 : memref<?xf16>, %arg1 : memref<?xi8>) {
    %cst_1 = arith.constant 2.000000e+00 : f16
    %1 = tensor.empty() : tensor<16x16xf32>
    %2 = tensor.empty() : tensor<16x16xf16>
    %3 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%1 : tensor<16x16xf32>) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %4 = hivm.hir.vmul ins(%3, %cst_1 : tensor<16x16xf16>, f16) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512], strides: [ 1] : memref<?xi8> to memref<512xi8, strided<[1], offset: 0>>
    %cst0 = arith.constant 0 : index
    %view = memref.view %reinterpret_cast_0[%cst0][] : memref<512xi8, strided<[1], offset: 0>> to memref<16x16xf16>
    hivm.hir.store ins(%4 : tensor<16x16xf16>) outs(%view : memref<16x16xf16>)
    return
  }
}

// -----
module {
  
  func.func @_attn_fwd_outlined_vf_3(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> attributes {hivm.vector_function} {
    %cst = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %arg1) -> (tensor<16x16xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg2, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
      %1 = vector.constant_mask [1, 16] : vector<1x64xi1>
      %2 = vector.transfer_read %extracted_slice[%c0, %c0], %cst_0, %1 {in_bounds = [true, true]} : tensor<1x16xf32>, vector<1x64xf32>
      %3 = arith.mulf %2, %cst : vector<1x64xf32>
      %4 = vector.transfer_write %3, %extracted_slice_1[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x16xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg3[%arg2, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
      scf.yield %inserted_slice : tensor<16x16xf32>
    }
    return %0 : tensor<16x16xf32>
  }
  // CHECK-LABEL: func.func @test_not_affect_affinity(
  func.func @test_not_affect_affinity() -> tensor<16x16xf32> {
    %c16 = arith.constant 16 : index
    
    %true = arith.constant true
    // CHECK: %[[ALLOC:.*]] = memref.alloc() {{.*}}
    // CHECK: %[[ALLOC0:.*]] = memref.alloc() {{.*}}
    // CHECK: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %[[ALLOC]]
    // CHECK: %[[MEMSPACECAST1:.*]] = memref.memory_space_cast %[[ALLOC0]]
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cbuf>>
    %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %memspacecast_1 = memref.memory_space_cast %alloc_0 : memref<16x16xf32, #hivm.address_space<cbuf>> to memref<16x16xf32>
    %alloc_2 = memref.alloc() : memref<16x16xf16>
    %0 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x16xf16>
    %alloc_3 = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xf16>
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = hivm.hir.mmadL1 {b_transpose} ins(%1, %0, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    // CHECK: hivm.hir.fixpipe {{.*}} ins({{.*}}) outs(%[[ALLOC]]
    // CHECK: %[[TENSOR_UB:.*]] = bufferization.to_tensor %[[MEMSPACECAST]] restrict writable :
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%3 : tensor<16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
    %4 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
    %5 = tensor.empty() : tensor<16x16xf32>
    // CHECK: {{.*}} = call @_attn_fwd_outlined_vf_3(%[[TENSOR_UB]], {{.*}}) 
    %6 = call @_attn_fwd_outlined_vf_3(%4, %5) {hivm.vector_function} : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    // CHECK: %[[TENSOR_L1:.*]] = bufferization.to_tensor %[[MEMSPACECAST1]] restrict writable :
    %7 = bufferization.to_tensor %memspacecast_1 restrict writable : memref<16x16xf32>
    // CHECK: %[[COPYED_TENSOR_L1:.*]] = hivm.hir.copy ins({{.*}}) outs(%[[TENSOR_L1]] : {{.*}})
    %8 = hivm.hir.copy ins(%6 : tensor<16x16xf32>) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc_4 = memref.alloc() : memref<16x16xf16>
    %9 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x16xf16>
    // CHECK: {{.*}} = hivm.hir.mmadL1 {a_transpose} ins({{.*}}, %[[COPYED_TENSOR_L1]]
    %10 = hivm.hir.mmadL1 {a_transpose} ins(%9, %8, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf32>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %10 : tensor<16x16xf32>
  }
}