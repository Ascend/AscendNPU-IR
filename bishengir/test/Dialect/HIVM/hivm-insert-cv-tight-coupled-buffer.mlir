// RUN: bishengir-opt --hivm-insert-cv-tight-coupled-buffer -split-input-file %s | FileCheck %s


// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
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
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
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
    // CHECK: %[[EXPAND:.*]] = tensor.expand_shape {{.*}} {{\[\[0\], \[1, 2\]\]}} output_shape {{\[16, 1, 16\]}} : tensor<16x16xf16> into tensor<16x1x16xf16>
    // CHECK: %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<1x16x16xf16>
    // CHECK: %[[TRANSPOSE:.*]] =  hivm.hir.vtranspose ins(%[[EXPAND]] : {{.*}}) outs(%[[EMPTY_TENSOR]]
    // CHECK: %[[EXPAND0:.*]] = tensor.expand_shape {{.*}} {{\[\[0\], \[1, 2\], \[3]\]}} output_shape {{\[1, 1, 16, 16\]}} : tensor<1x16x16xf16> into tensor<1x1x16x16xf16>

    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : 
    // CHECK: %[[EMPTY_T:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : 
    // CHECK: hivm.hir.copy ins(%[[EXPAND0]] : tensor<1x1x16x16xf16>) outs(%[[CAST:.*]] : memref<1x1x16x16xf16>)
    %mm = hivm.hir.mmadL1
            ins(%vf_res, %b, %true, %c16, %c16, %c16
                : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
            outs(%mm_dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mm : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
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
  // CHECK:   %[[EXPAND:.*]] = tensor.expand_shape %[[INSERT]] {{\[\[0\], \[1, 2\]\]}} output_shape {{\[16, 1, 16\]}} : tensor<16x16xf16> into tensor<16x1x16xf16>
  // CHECK:   %[[EMPTY_T:.*]] = tensor.empty() : tensor<1x16x16xf16>
  // CHECK:   %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[EXPAND]] : tensor<16x1x16xf16>) outs(%[[EMPTY_T]] : tensor<1x16x16xf16>) permutation = [1, 0, 2] -> tensor<1x16x16xf16>
  // CHECK:   %[[EXPAND0:.*]] = tensor.expand_shape {{.*}} {{\[\[0\], \[1, 2\], \[3]\]}} output_shape {{\[1, 1, 16, 16\]}} : tensor<1x16x16xf16> into tensor<1x1x16x16xf16>
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK:   %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
  // CHECK:   %[[BUF_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<1x1x16x16xf16>
  // CHECK:   hivm.hir.copy ins(%[[EXPAND0]] : tensor<1x1x16x16xf16>) outs(%[[CAST:.*]] : memref<1x1x16x16xf16>)
  // CHECK:   %[[MM:.*]] = hivm.hir.mmadL1 ins(%[[ARG0]], %[[BUF_TENSOR:.*]], %[[TRUE]], %[[C16]], %[[C16]], %[[C16]] : tensor<16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%[[EMPTY1]] : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK:   return %[[MM]] : tensor<16x16xf16>
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
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
  // CHECK:   %[[EXPAND:.*]] = tensor.expand_shape %[[INSERT]] {{\[\[0\], \[1, 2\]\]}} output_shape {{\[16, 1, 16\]}} : tensor<16x16xf16> into tensor<16x1x16xf16>
  // CHECK:   %[[EMPTY_T:.*]] = tensor.empty() : tensor<1x16x16xf16>
  // CHECK:   %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[EXPAND]] : tensor<16x1x16xf16>) outs(%[[EMPTY_T]] : tensor<1x16x16xf16>) permutation = [1, 0, 2] -> tensor<1x16x16xf16>
  // CHECK:   %[[EXPAND0:.*]] = tensor.expand_shape {{.*}} {{\[\[0\], \[1, 2\], \[3]\]}} output_shape {{\[1, 1, 16, 16\]}} : tensor<1x16x16xf16> into tensor<1x1x16x16xf16>
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK:   %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
  // CHECK:   %[[BUF_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<1x1x16x16xf16>
  // CHECK:   hivm.hir.copy ins(%[[EXPAND0]] : tensor<1x1x16x16xf16>) outs(%[[CAST:.*]] : memref<1x1x16x16xf16>)
  // CHECK:   %[[MM:.*]] = hivm.hir.mmadL1 ins(%[[ARG0]], %[[BUF_TENSOR:.*]], %[[TRUE]], %[[C16]], %[[C16]], %[[C16]] : tensor<16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%[[EMPTY0]] : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK:   return %[[MM]] : tensor<16x16xf16>
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_vector_mm(
  // CHECK: %[[VMUL:.*]] = hivm.hir.vmul ins({{.*}} : tensor<16x16xf32>, f32) outs({{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
  // CHECK: %[[EXPAND:.*]] = tensor.expand_shape {{.*}} {{\[\[0\], \[1, 2\]\]}} output_shape {{\[16, 2, 8\]}} : tensor<16x16xf32> into tensor<16x2x8xf32>
  // CHECK: %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<2x16x8xf32>
  // CHECK: %[[TRANSPOSE:.*]] =  hivm.hir.vtranspose ins(%[[EXPAND]] : {{.*}}) outs(%[[EMPTY_TENSOR]]
  // CHECK: %[[EXPAND0:.*]] = tensor.expand_shape {{.*}} {{\[\[0\], \[1, 2\], \[3]\]}} output_shape {{\[2, 1, 16, 8\]}} : tensor<2x16x8xf32> into tensor<2x1x16x8xf32>
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
  // CHECK: %[[CAST0:.*]] = memref.memory_space_cast %[[ALLOC0]] :
  // CHECK: %[[T0:.*]] = bufferization.to_tensor %[[CAST0]] restrict writable : memref<2x1x16x8xf32>
  // CHECK: hivm.hir.copy ins(%[[EXPAND0]] : tensor<2x1x16x8xf32>) outs(%[[CAST0:.*]] : memref<2x1x16x8xf32>)
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
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
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
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_fixpipe_store(
  // CHECK: %[[ALLOC_UB:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC_UB]] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
  // CHECK: hivm.hir.fixpipe {{.*}} ins({{.*}} : tensor<16x16xf32>) outs(%[[ALLOC_UB]] : memref<16x16xf16, #hivm.address_space<ub>>)
  // CHECK: %[[TT:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16x16xf16>
  // CHECK: hivm.hir.store ins(%[[TT]]

  func.func @test_fixpipe_store(%arg0 : memref<?xf16>, %arg1 : memref<?xi8>) {
    %cst_1 = arith.constant 2.000000e+00 : f16
    %1 = tensor.empty() : tensor<16x16xf32>
    %2 = tensor.empty() : tensor<16x16xf16>
    %3 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%1 : tensor<16x16xf32>) outs(%2 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512], strides: [ 1] : memref<?xi8> to memref<512xi8, strided<[1], offset: 0>>
    %cst0 = arith.constant 0 : index
    %view = memref.view %reinterpret_cast_0[%cst0][] : memref<512xi8, strided<[1], offset: 0>> to memref<16x16xf16>
    hivm.hir.store ins(%3 : tensor<16x16xf16>) outs(%view : memref<16x16xf16>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_mmad_fixpipe_vadd_dynamic(
  // CHECK-SAME: %{{.*}}: tensor<16x32xf32>, %{{.*}}: tensor<32x16xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index
  // CHECK: %[[MINSI:.*]] = arith.minsi %[[ARG2]], %{{.*}} : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[MINSI]], %[[ARG3]]) : memref<?x?xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 1024 : i64} : memref<?x?xf32, #hivm.address_space<ub>>

  func.func @test_mmad_fixpipe_vadd_dynamic(%arg0: tensor<16x32xf32>, %arg1: tensor<32x16xf32>, %arg2: index, %arg3: index, %arg4: tensor<16x16xf32>, %arg5: tensor<16x16xf32>, %arg6: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%arg0, %arg1, %true, %c16, %c32, %c16 : tensor<16x32xf32>, tensor<32x16xf32>, i1, index, index, index) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = arith.minsi %arg2, %c16 : index
    %extracted_slice = tensor.extract_slice %1[0, 0] [%2, %arg3] [1, 1] : tensor<16x16xf32> to tensor<?x?xf32>
    %3 = tensor.empty(%2, %arg3) : tensor<?x?xf32>
    %4 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%extracted_slice : tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %4 into %arg4[0, 0] [%2, %arg3] [1, 1] : tensor<?x?xf32> into tensor<16x16xf32>
    %5 = hivm.hir.vadd ins(%inserted_slice, %arg5 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%arg6 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %5 : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  
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

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_vector_for_next_loop_cube
  func.func @test_vector_for_next_loop_cube() -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %init = arith.constant false
    %dst = tensor.empty() : tensor<16x16xf32>
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
    // CHECK: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %[[ALLOC:.*]] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
    %alloc_ub = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc_ub restrict writable : memref<16x16xf16>
    %2, %3 = scf.for %arg1 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg2 = %1, %arg3 = %dst) -> (tensor<16x16xf16>, tensor<16x16xf32>) : i32 {
      // CHECK: hivm.hir.copy
      // CHECK-NEXT: hivm.hir.mmadL1
      %4 = hivm.hir.mmadL1 ins(%arg2, %arg2, %init, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
      %empty = tensor.empty() : tensor<16x16xf16>
      %5 = hivm.hir.vadd ins(%arg2, %arg2: tensor<16x16xf16>, tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
      scf.yield %5, %4 : tensor<16x16xf16>, tensor<16x16xf32>
    }
    return %3 : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_cube_for_next_loop_vector
  func.func @test_cube_for_next_loop_vector() -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %dst = tensor.empty() : tensor<16x16xf32>
    %0 = tensor.empty() : tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %2:2 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %1, %arg2 = %0) -> (tensor<16x16xf16>, tensor<16x16xf32>) : i32 {
      %3 = tensor.empty() : tensor<16x16xf16>
      %4 = hivm.hir.vadd ins(%arg1, %arg1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %5 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%4, %4, %false, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %6 = tensor.empty() : tensor<16x16xf32>
      // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC:.*]] : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      // CHECK: hivm.hir.fixpipe
      %7 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %8 = tensor.empty() : tensor<16x16xf16>
      %9 = hivm.hir.vcast ins(%7 : tensor<16x16xf32>) outs(%8 : tensor<16x16xf16>) -> tensor<16x16xf16>
      scf.yield %9, %7 : tensor<16x16xf16>, tensor<16x16xf32>
    }
    return %2#1 : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_fixpipe_to_mmadl1_tight_coupled
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf32>, %[[ARG1:.*]]: tensor<16x16xf32>, %[[ARG2:.*]]: tensor<16x16xf32>, %[[ARG3:.*]]: tensor<16x16xf32>)
  func.func @test_fixpipe_to_mmadl1_tight_coupled(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>, %arg3: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    // CHECK: %[[MMAD_OUT:.*]] = hivm.hir.mmadL1 {{.*}} ins(%[[ARG1]], %[[ARG2]]
    %0 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%arg1, %arg2, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%arg3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    // CHECK-NEXT: %[[UB_ALLOC:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK-NEXT: %[[UB_CAST:.*]] = memref.memory_space_cast %[[UB_ALLOC]] : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    // CHECK-NEXT: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[MMAD_OUT]] : tensor<16x16xf32>) outs(%[[UB_ALLOC]] : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK-NEXT: %[[UB_TENSOR:.*]] = bufferization.to_tensor %[[UB_CAST]] restrict writable : memref<16x16xf32>
    %empty_fixpipe = tensor.empty() : tensor<16x16xf32>
    %1 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : tensor<16x16xf32>) outs(%empty_fixpipe : tensor<16x16xf32>) -> tensor<16x16xf32>
    
    // CHECK: %[[EXPAND1:.*]] = tensor.expand_shape %[[UB_TENSOR]] {{\[\[}}0], [1, 2]] output_shape [16, 2, 8]
    // CHECK: %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[EXPAND1]] : tensor<16x2x8xf32>) {{.*}} permutation = [1, 0, 2]
    // CHECK: %[[EXPAND2:.*]] = tensor.expand_shape %[[TRANSPOSE]] {{\[\[}}0], [1, 2], [3]] output_shape [2, 1, 16, 8]
    
    // CHECK: %[[CBUF_ALLOC:.*]] = memref.alloc() : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    // CHECK-NEXT: %[[CBUF_CAST:.*]] = memref.memory_space_cast %[[CBUF_ALLOC]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x1x16x8xf32>
    // CHECK-NEXT: %[[CBUF_TENSOR:.*]] = bufferization.to_tensor %[[CBUF_CAST]] restrict writable : memref<2x1x16x8xf32>
    // CHECK-NEXT: hivm.hir.copy ins(%[[EXPAND2]] : tensor<2x1x16x8xf32>) outs(%[[CBUF_CAST]] : memref<2x1x16x8xf32>)
    
    // CHECK: %[[FINAL_MMAD:.*]] = hivm.hir.mmadL1 {{.*}} ins(%[[CBUF_TENSOR]], %[[ARG0]]
    %empty_mmad = tensor.empty() : tensor<16x16xf32>
    %2 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%1, %arg0, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%empty_mmad : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_fixpipe_extract(
  func.func @test_fixpipe_extract(%src : tensor<16x16xf32>) -> f16 {
    %dst_init = tensor.empty() : tensor<16x16xf16>
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
    // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
    // CHECK: hivm.hir.fixpipe {{.*}} ins({{.*}} : tensor<16x16xf32>) outs(%[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>>
    // CHECK: %[[TOT:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16x16xf16>
    // CHECK: %[[RES:.*]] = tensor.extract %[[TOT]]
    %fix_res = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
               ins(%src : tensor<16x16xf32>)
               outs(%dst_init : tensor<16x16xf16>) -> tensor<16x16xf16>
    %c0 = arith.constant 0 : index
    %val = tensor.extract %fix_res[%c0, %c0] : tensor<16x16xf16>
    return %val : f16
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_connect_point_with_if(
  func.func @test_connect_point_with_if() -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %dst = tensor.empty() : tensor<16x16xf32>
    %0 = tensor.empty() : tensor<16x16xf32>
    %alloc = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %2:2 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %1, %arg2 = %0) -> (tensor<16x16xf16>, tensor<16x16xf32>) : i32 {
      %3 = tensor.empty() : tensor<16x16xf16>
      %and = arith.andi %arg0, %c1_i32 : i32
      %cond = arith.cmpi ne, %and, %c0_i32 : i32
      %4 = scf.if %cond -> (tensor<16x16xf16>) {
        scf.yield %arg1 : tensor<16x16xf16>
      } else {
        %5 = hivm.hir.vadd ins(%arg1, %arg1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
        scf.yield %5 : tensor<16x16xf16>
      }
      // CHECK: tensor.expand_shape
      // CHECK: tensor.empty
      // CHECK: hivm.hir.vtranspose
      // CHECK: hivm.hir.copy
      // CHECK-NEXT: hivm.hir.mmadL1
      %5 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%4, %4, %false, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %6 = tensor.empty() : tensor<16x16xf32>
      // CHECK: memref.alloc
      // CHECK: memref.memory_space_cast
      // CHECK: hivm.hir.fixpipe
      %7 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %8 = tensor.empty() : tensor<16x16xf16>
      %9 = scf.if %cond -> (tensor<16x16xf16>) {
        %10 = hivm.hir.vcast ins(%7 : tensor<16x16xf32>) outs(%8 : tensor<16x16xf16>) -> tensor<16x16xf16>
        scf.yield %10 : tensor<16x16xf16>
      } else {
        scf.yield %4 : tensor<16x16xf16>
      }
      scf.yield %9, %7 : tensor<16x16xf16>, tensor<16x16xf32>
    }
    return %2#1 : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_marked_to_tensor_b_operand(
  // CHECK: %[[B_TENSOR:.*]] = bufferization.to_tensor %[[B_MEM:.*]] restrict writable : memref<16x64xf16>
  // CHECK: annotation.mark %[[B_TENSOR]] {MayImplicitTransposeWithLastAxis} : tensor<16x64xf16>
  // CHECK: %[[EXPAND:.*]] = tensor.expand_shape %[[B_TENSOR]] {{\[\[0\], \[1, 2\]\]}} output_shape {{\[16, 4, 16\]}} : tensor<16x64xf16> into tensor<16x4x16xf16>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x16x16xf16>
  // CHECK: %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[EXPAND]] : tensor<16x4x16xf16>) outs(%[[EMPTY]] : tensor<4x16x16xf16>) permutation = [1, 0, 2] -> tensor<4x16x16xf16>
  // CHECK: %[[EXPAND0:.*]] = tensor.expand_shape %[[TRANSPOSE]] {{\[\[0\], \[1, 2\], \[3]\]}} output_shape {{\[4, 1, 16, 16\]}} : tensor<4x16x16xf16> into tensor<4x1x16x16xf16>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<4x1x16x16xf16, #hivm.address_space<cbuf>> to memref<4x1x16x16xf16>
  // CHECK: %[[BUF_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<4x1x16x16xf16>
  // CHECK: hivm.hir.copy ins(%[[EXPAND0]] : tensor<4x1x16x16xf16>) outs(%[[CAST]] : memref<4x1x16x16xf16>)
  // CHECK: hivm.hir.mmadL1 ins(%arg0, %[[BUF_TENSOR]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x16xf16>, tensor<4x1x16x16xf16>, i1, index, index, index)
  func.func @test_marked_to_tensor_b_operand(%arg0 : tensor<16x16xf16>,
                                            %arg1 : memref<16x64xf16>)
      -> tensor<16x64xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %b = bufferization.to_tensor %arg1 restrict writable : memref<16x64xf16>
    annotation.mark %b {MayImplicitTransposeWithLastAxis} : tensor<16x64xf16>
    %out = tensor.empty() : tensor<16x64xf32>
    %mm = hivm.hir.mmadL1 ins(%arg0, %b, %true, %c16, %c16, %c64 :
                                  tensor<16x16xf16>, tensor<16x64xf16>, i1, index, index, index)
                          outs(%out : tensor<16x64xf32>) -> tensor<16x64xf32>
    return %mm : tensor<16x64xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_collapse_shape_with_annotation(
  func.func @test_collapse_shape_with_annotation(%arg1 : memref<4x4x16xf16>, 
                                                %arg2 : tensor<16x64xf16>) -> tensor<16x64xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %b = bufferization.to_tensor %arg1 restrict writable : memref<4x4x16xf16>
    // CHECK: %[[EXPANDED:.*]] = tensor.expand_shape {{.*}} {{\[\[0\], \[1, 2\]\]}} output_shape {{\[16, 1, 16\]}} : tensor<16x16xf16> into tensor<16x1x16xf16>
    // CHECK: %[[EMPTY_T:.*]] = tensor.empty() : tensor<1x16x16xf16>
    // CHECK: %[[TRANSPOSED:.*]] = hivm.hir.vtranspose ins(%[[EXPANDED]] : tensor<16x1x16xf16>) outs(%[[EMPTY_T]] : tensor<1x16x16xf16>) permutation = [1, 0, 2] -> tensor<1x16x16xf16>
    // CHECK: %[[EXPANDED_0:.*]] = tensor.expand_shape %[[TRANSPOSED]] {{\[\[0\], \[1, 2\], \[3\]\]}} output_shape {{\[1, 1, 16, 16\]}} : tensor<1x16x16xf16> into tensor<1x1x16x16xf16>
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
    // CHECK: %[[BUF_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<1x1x16x16xf16>
    // CHECK: hivm.hir.copy ins(%[[EXPANDED_0]] : tensor<1x1x16x16xf16>) outs(%[[CAST]] : memref<1x1x16x16xf16>)
    %collapsed = tensor.collapse_shape %b [[0, 1], [2]] : tensor<4x4x16xf16> into tensor<16x16xf16>
    annotation.mark %collapsed {maybeUnCollapsibleReshape} : tensor<16x16xf16>
    %out = tensor.empty() : tensor<16x64xf32>
    %mm = hivm.hir.mmadL1 ins(%collapsed, %arg2, %true, %c16, %c16, %c64 : 
                                tensor<16x16xf16>, tensor<16x64xf16>, i1, index, index, index)
                          outs(%out : tensor<16x64xf32>) -> tensor<16x64xf32>
                          
    return %mm : tensor<16x64xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @triton_dot_2(
  func.func @triton_dot_2(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi8> to memref<16x16xi8, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<16x16xi8>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xi8, strided<[16, 1]>>) outs(%alloc : memref<16x16xi8>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x16xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi8> to memref<16x16xi8, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x16xi8>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xi8, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xi8>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xi8>
    %4 = tensor.empty() : tensor<16x16xi8>
    %5 = hivm.hir.vadd ins(%2, %3 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%4 : tensor<16x16xi8>) -> tensor<16x16xi8>
    %6 = tensor.empty() : tensor<16x16xi32>
    
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x32xi8>
    // CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : i8) outs(%[[EMPTY]] : tensor<16x32xi8>) -> tensor<16x32xi8>
    // CHECK: %[[INSERTED:.*]] = tensor.insert_slice %{{.*}} into %[[VBRC]][0, 0] [16, 16] [1, 1] : tensor<16x16xi8> into tensor<16x32xi8>
    // CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[INSERTED]] {{\[\[}}0], [1, 2]] output_shape [16, 1, 32] : tensor<16x32xi8> into tensor<16x1x32xi8>
    // CHECK: %[[EMPTY_1:.*]] = tensor.empty() : tensor<1x16x32xi8>
    // CHECK: %[[VTRANS:.*]] = hivm.hir.vtranspose ins(%[[EXPANDED]] : tensor<16x1x32xi8>) outs(%[[EMPTY_1]] : tensor<1x16x32xi8>) permutation = [1, 0, 2] -> tensor<1x16x32xi8>
    // CHECK: %[[EXPANDED_2:.*]] = tensor.expand_shape %[[VTRANS]] {{\[\[}}0], [1, 2], [3]] output_shape [1, 1, 16, 32] : tensor<1x16x32xi8> into tensor<1x1x16x32xi8>
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x32xi8, #hivm.address_space<cbuf>>
    // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x32xi8, #hivm.address_space<cbuf>> to memref<1x1x16x32xi8>
    // CHECK: %[[TO_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<1x1x16x32xi8>
    // CHECK: hivm.hir.copy ins(%[[EXPANDED_2]] : tensor<1x1x16x32xi8>) outs(%[[CAST]] : memref<1x1x16x32xi8>)


    %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%5, %3, %true, %c16, %c16, %c16 : tensor<16x16xi8>, tensor<16x16xi8>, i1, index, index, index) outs(%6 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xi32> to memref<16x16xi32, strided<[16, 1]>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : tensor<16x16xi32>) outs(%reinterpret_cast_2 : memref<16x16xi32, strided<[16, 1]>>)
    return
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

  // CHECK-LABEL: func.func @test_vf_mm_with_bias(
  func.func @test_vf_mm_with_bias(%a : tensor<16x16xf16>,
                                 %b : tensor<16x16xf16>,
                                 %bias : tensor<16x16xf16>)
               -> tensor<16x16xf32> {
    %true = arith.constant 1 : i1
    %c16 = arith.constant 16 : index
    %mm_dst = tensor.empty() : tensor<16x16xf32>

    %vf_res = func.call @vf_mm_func(%bias) {hivm.vector_function}
              : (tensor<16x16xf16>) -> tensor<16x16xf16>
    
    // Bias should skip nd2nz conversion, just do simple copy
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
    // CHECK: %[[BUF_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16x16xf16>
    // CHECK: hivm.hir.copy ins(%{{.*}} : tensor<16x16xf16>) outs(%[[CAST]] : memref<16x16xf16>)
    
    %mm = hivm.hir.mmadL1
            ins(%a, %b, %true, %c16, %c16, %c16, %vf_res
                : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index, tensor<16x16xf16>)
            outs(%mm_dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mm : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_indirect_load_mmad(
  // CHECK: %[[INDIRECT:.*]] = hivm.hir.indirect_load ins(%[[BASE:.*]] : memref<?xf16>, %[[IDX:.*]] : tensor<16x16xi64>) outs(%{{.*}} : tensor<16x16xf16>) -> tensor<16x16xf16>
  // CHECK: %[[EXPAND:.*]] = tensor.expand_shape %[[INDIRECT]] {{\[\[0\], \[1, 2\]\]}} output_shape {{\[16, 1, 16\]}} : tensor<16x16xf16> into tensor<16x1x16xf16>
  // CHECK: %[[EMPTY_TRANSPOSE:.*]] = tensor.empty() : tensor<1x16x16xf16>
  // CHECK: %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[EXPAND]] : tensor<16x1x16xf16>) outs(%[[EMPTY_TRANSPOSE]] : tensor<1x16x16xf16>) permutation = [1, 0, 2] -> tensor<1x16x16xf16>
  // CHECK: %[[EXPAND0:.*]] = tensor.expand_shape %[[TRANSPOSE]] {{\[\[0\], \[1, 2\], \[3]\]}} output_shape {{\[1, 1, 16, 16\]}} : tensor<1x16x16xf16> into tensor<1x1x16x16xf16>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
  // CHECK: %[[BUF_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<1x1x16x16xf16>
  // CHECK: hivm.hir.copy ins(%[[EXPAND0]] : tensor<1x1x16x16xf16>) outs(%[[CAST]] : memref<1x1x16x16xf16>)
  // CHECK: hivm.hir.mmadL1 ins(%[[BUF_TENSOR]], %[[B:.*]], %[[TRUE:.*]], %[[C16:.*]], %[[C16]], %[[C16]] : tensor<1x1x16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
  func.func @test_indirect_load_mmad(%base : memref<?xf16>,
                                     %idx : tensor<16x16xi64>,
                                     %b : tensor<16x16xf16>)
      -> tensor<16x16xf32> {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %indirect_dst = tensor.empty() : tensor<16x16xf16>
    %indirect = hivm.hir.indirect_load
        ins(%base : memref<?xf16>, %idx : tensor<16x16xi64>)
        outs(%indirect_dst : tensor<16x16xf16>) -> tensor<16x16xf16>
    %mm_dst = tensor.empty() : tensor<16x16xf32>
    %mm = hivm.hir.mmadL1
        ins(%indirect, %b, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%mm_dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mm : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @test_fixpipe_indirect_store(
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
  // CHECK: hivm.hir.fixpipe {{.*}} ins(%[[SRC:.*]] : tensor<16x16xf32>) outs(%[[ALLOC]] : memref<16x16xf16, #hivm.address_space<ub>>)
  // CHECK: %[[TO_TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable : memref<16x16xf16>
  // CHECK: hivm.hir.indirect_store ins(%[[TO_TENSOR]] : tensor<16x16xf16>, %[[IDX:.*]] : tensor<16x16xi64>) outs(%[[BASE:.*]] : memref<?xf16>)
  func.func @test_fixpipe_indirect_store(%src : tensor<16x16xf32>,
                                         %base : memref<?xf16>,
                                         %idx : tensor<16x16xi64>) {
    %dst_init = tensor.empty() : tensor<16x16xf16>
    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%src : tensor<16x16xf32>)
        outs(%dst_init : tensor<16x16xf16>) -> tensor<16x16xf16>
    hivm.hir.indirect_store
        ins(%fix : tensor<16x16xf16>, %idx : tensor<16x16xi64>)
        outs(%base : memref<?xf16>)
    return
  }
}
