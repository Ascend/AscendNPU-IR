// RUN: bishengir-opt %s -hivm-mark-multi-buffer="enable-auto=false enable-preload=true" | FileCheck %s
// RUN: bishengir-opt %s -hivm-mark-multi-buffer="enable-auto=false enable-preload=false" | FileCheck %s --check-prefix=DISABLED

// A fusion boundary nested in VECTOR stage 1 must not hide the stage of the
// QK read or P copy. Both flat and nested forms need the same four TCB buffers:
// QK/dP from CUBE(2) to VECTOR(1), and P/dS from VECTOR(1) to CUBE(0).
// DISABLED-NOT: hivm.multi_buffer
// DISABLED-NOT: hivm.preload_local_buffer

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @flat_preload_stages
  // CHECK: %[[DS:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK-NEXT: annotation.mark %[[DS]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
  // CHECK: %[[P:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK-NEXT: annotation.mark %[[P]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
  // CHECK: %[[DP:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK-NEXT: annotation.mark %[[DP]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
  // CHECK: %[[QK:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK-NEXT: annotation.mark %[[QK]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
  func.func @flat_preload_stages(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x1x16x16xf16>, %arg2: tensor<1x1x16x16xf16>, %arg3: memref<16x16xf32>, %arg4: memref<16x16xf32>, %arg5: i32, %arg6: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %arg7 = %c0_i32 to %arg5 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_0 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scope.scope : () -> () {
        %4 = tensor.empty() : tensor<1x1x16x16xf32>
        %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%4 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<1x1x16x16xf32>) outs(%alloc_2 : memref<16x16xf32, #hivm.address_space<ub>>)
        %6 = tensor.empty() : tensor<1x1x16x16xf32>
        %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%6 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : tensor<1x1x16x16xf32>) outs(%alloc_1 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
      %0 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      scope.scope : () -> () {
        %4 = tensor.empty() : tensor<16x16xf32>
        %5 = tensor.empty() : tensor<16x16xf16>
        %6 = hivm.hir.vmul ins(%0, %arg6 : tensor<16x16xf32>, f32) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %7 = hivm.hir.vexp ins(%6 : tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %8 = hivm.hir.vcast ins(%7 : tensor<16x16xf32>) outs(%5 : tensor<16x16xf16>) -> tensor<16x16xf16>
        %expanded = tensor.expand_shape %8 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded : tensor<1x1x16x16xf16>) outs(%alloc_0 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        %9 = hivm.hir.vmul ins(%7, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %10 = hivm.hir.vcast ins(%9 : tensor<16x16xf32>) outs(%5 : tensor<16x16xf16>) -> tensor<16x16xf16>
        %expanded_3 = tensor.expand_shape %10 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_3 : tensor<1x1x16x16xf16>) outs(%alloc : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
      %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %3 = bufferization.to_tensor %alloc restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      scope.scope : () -> () {
        %4 = tensor.empty() : tensor<1x1x16x16xf32>
        %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%2, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%4 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<1x1x16x16xf32>) outs(%arg3 : memref<16x16xf32>)
        %6 = tensor.empty() : tensor<1x1x16x16xf32>
        %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%3, %arg2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%6 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : tensor<1x1x16x16xf32>) outs(%arg4 : memref<16x16xf32>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
    } {hivm.cv_pipelined_loop}
    return
  }

  // CHECK-LABEL: func.func @nested_preload_stages
  // CHECK: %[[DS:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK-NEXT: annotation.mark %[[DS]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
  // CHECK: %[[P:.*]] = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
  // CHECK-NEXT: annotation.mark %[[P]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
  // CHECK: %[[DP:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK-NEXT: annotation.mark %[[DP]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
  // CHECK: %[[QK:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK-NEXT: annotation.mark %[[QK]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
  func.func @nested_preload_stages(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x1x16x16xf16>, %arg2: tensor<1x1x16x16xf16>, %arg3: memref<16x16xf32>, %arg4: memref<16x16xf32>, %arg5: i32, %arg6: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %arg7 = %c0_i32 to %arg5 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_0 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      scope.scope : () -> () {
        %4 = tensor.empty() : tensor<1x1x16x16xf32>
        %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%4 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<1x1x16x16xf32>) outs(%alloc_2 : memref<16x16xf32, #hivm.address_space<ub>>)
        %6 = tensor.empty() : tensor<1x1x16x16xf32>
        %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%6 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : tensor<1x1x16x16xf32>) outs(%alloc_1 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
      %0 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      scope.scope : () -> () {
        %4 = tensor.empty() : tensor<16x16xf32>
        %5 = tensor.empty() : tensor<16x16xf16>
        annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %early_p = scope.scope : () -> tensor<16x16xf32> {
          %6 = hivm.hir.vmul ins(%0, %arg6 : tensor<16x16xf32>, f32) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
          %7 = hivm.hir.vexp ins(%6 : tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
          %8 = hivm.hir.vcast ins(%7 : tensor<16x16xf32>) outs(%5 : tensor<16x16xf16>) -> tensor<16x16xf16>
          %expanded = tensor.expand_shape %8 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
          hivm.hir.copy ins(%expanded : tensor<1x1x16x16xf16>) outs(%alloc_0 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
          scope.return %7 : tensor<16x16xf32>
        } {no_inline}
        %9 = hivm.hir.vmul ins(%early_p, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %10 = hivm.hir.vcast ins(%9 : tensor<16x16xf32>) outs(%5 : tensor<16x16xf16>) -> tensor<16x16xf16>
        %expanded_3 = tensor.expand_shape %10 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_3 : tensor<1x1x16x16xf16>) outs(%alloc : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
      %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %3 = bufferization.to_tensor %alloc restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      scope.scope : () -> () {
        %4 = tensor.empty() : tensor<1x1x16x16xf32>
        %5 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%2, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%4 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%5 : tensor<1x1x16x16xf32>) outs(%arg3 : memref<16x16xf32>)
        %6 = tensor.empty() : tensor<1x1x16x16xf32>
        %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%3, %arg2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%6 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : tensor<1x1x16x16xf32>) outs(%arg4 : memref<16x16xf32>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
    } {hivm.cv_pipelined_loop}
    return
  }

  // The other preload marker derives its distance from a yielded scope
  // result. A nested consumer still belongs to its outer stage 0, so a result
  // produced at stage 2 requires three slots.
  // CHECK-LABEL: func.func @nested_scope_result_consumer
  // CHECK: %[[YIELD_ALLOC:.*]] = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // CHECK-NEXT: annotation.mark %[[YIELD_ALLOC]] {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32}
  func.func @nested_scope_result_consumer(%input: memref<16x16xf16, #hivm.address_space<gm>>, %output: memref<16x16xf16, #hivm.address_space<gm>>) {
    %value = scope.scope : () -> tensor<16x16xf16> {
      %buffer = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%input : memref<16x16xf16, #hivm.address_space<gm>>) outs(%buffer : memref<16x16xf16, #hivm.address_space<ub>>)
      %tensor = bufferization.to_tensor %buffer restrict writable : memref<16x16xf16, #hivm.address_space<ub>>
      scope.return %tensor : tensor<16x16xf16>
    } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.scope : () -> () {
        hivm.hir.store ins(%value : tensor<16x16xf16>) outs(%output : memref<16x16xf16, #hivm.address_space<gm>>)
        scope.return
      } {no_inline}
      scope.return
    } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
    return
  }

  // A nested scope without an enclosing preload stage does not acquire a
  // preload buffer merely because this pass is enabled.
  // CHECK-LABEL: func.func @no_preload_stage
  // CHECK-NOT: hivm.multi_buffer
  // CHECK-NOT: hivm.preload_local_buffer
  func.func @no_preload_stage(%input: tensor<1x1x16x16xf32>, %output: memref<16x16xf32, #hivm.address_space<gm>>) {
    %buffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %buffer {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%input : tensor<1x1x16x16xf32>) outs(%buffer : memref<16x16xf32, #hivm.address_space<ub>>)
      scope.return
    } {no_inline}
    scope.scope : () -> () {
      scope.scope : () -> () {
        hivm.hir.store ins(%buffer : memref<16x16xf32, #hivm.address_space<ub>>) outs(%output : memref<16x16xf32, #hivm.address_space<gm>>)
        scope.return
      } {no_inline}
      scope.return
    } {no_inline}
    return
  }
}
