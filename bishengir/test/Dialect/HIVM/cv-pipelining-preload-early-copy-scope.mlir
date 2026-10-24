// RUN: bishengir-opt %s -cv-pipelining="pipeline-mode=skew enable-preload=true bypass-shape-registry=true" -inline-scope -hivm-mark-multi-buffer="enable-auto=false enable-preload=true" | FileCheck %s
// RUN: bishengir-opt %s -cv-pipelining="pipeline-mode=skew enable-preload=true bypass-shape-registry=true" -inline-scope -hivm-inline-vf-scope | FileCheck %s --check-prefix=INLINE-VF
// RUN: bishengir-opt %s -cv-pipelining="pipeline-mode=skew enable-preload=true" | FileCheck %s --check-prefix=SAME --implicit-check-not='= scope.scope'
// RUN: sed 's/, %dp :/, %qk :/' %s | bishengir-opt -cv-pipelining="pipeline-mode=skew enable-preload=true bypass-shape-registry=true" | FileCheck %s --check-prefix=SAME --implicit-check-not='= scope.scope'
// RUN: bishengir-opt %s -cv-pipelining="pipeline-mode=unroll enable-preload=false bypass-shape-registry=true" | FileCheck %s --check-prefix=UNROLL --implicit-check-not=scope.scope

// Publish P before consuming the independent dP input. The nested boundary
// survives inlining without adding a stage or hiding any TCB's two slots.
// CHECK-LABEL: func.func @early_copy
// CHECK-COUNT-3: hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32
// CHECK: hivm.preload_num = 2 : i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x16xf16>
// CHECK: %[[P:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK: %[[EXP:.*]] = hivm.hir.vexp
// CHECK: hivm.hir.vcast {{.*}} outs(%[[EMPTY]]
// CHECK: hivm.hir.copy
// CHECK-NEXT: scope.return %[[EXP]] : tensor<16x16xf32>
// CHECK-NEXT: } {no_inline, "vf-scope"}
// CHECK: hivm.hir.vmul ins(%[[P]],
// CHECK: hivm.preload_num = 1 : i32
// CHECK: hivm.preload_num = 0 : i32
// INLINE-VF-LABEL: func.func @early_copy
// INLINE-VF: hivm.preload_num = 2 : i32
// INLINE-VF: scope.scope : () -> () {
// INLINE-VF-NOT: scope.scope
// INLINE-VF: %[[EXP_INLINED:.*]] = hivm.hir.vexp
// INLINE-VF: hivm.hir.vcast
// INLINE-VF: hivm.hir.copy
// INLINE-VF: hivm.hir.vmul ins(%[[EXP_INLINED]],
// INLINE-VF: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
// SAME-LABEL: func.func @early_copy
// SAME: hivm.preload_num = 2 : i32
// SAME: hivm.preload_num = 1 : i32
// SAME: hivm.preload_num = 0 : i32
// UNROLL-LABEL: func.func @early_copy

!t32 = tensor<16x16xf32>
!t16 = tensor<16x16xf16>
!nz32 = tensor<1x1x16x16xf32>
!nz16 = tensor<1x1x16x16xf16>
!ub = memref<16x16xf32, #hivm.address_space<ub>>
!l1 = memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @early_copy(%qk_in: !nz32, %dp_in: !nz32, %rhs: !nz16,
      %out: memref<16x16xf32>, %ds_out: memref<16x16xf32>, %bound: i32)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : !ub
      %dpbuf = memref.alloc() : !ub
      %pbuf = memref.alloc() : !l1
      annotation.mark %qkbuf {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : !ub
      annotation.mark %dpbuf {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : !ub
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%qk_in : !nz32) outs(%qkbuf : !ub)
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dp_in : !nz32) outs(%dpbuf : !ub)
      %qk = bufferization.to_tensor %qkbuf restrict writable : !ub
      %dp = bufferization.to_tensor %dpbuf restrict writable : !ub
      %empty32 = tensor.empty() : !t32
      %p = hivm.hir.vexp ins(%qk : !t32) outs(%empty32 : !t32) -> !t32
      %empty16 = tensor.empty() : !t16
      %p16 = hivm.hir.vcast ins(%p : !t32) outs(%empty16 : !t16) -> !t16
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : !t16 into !nz16
      annotation.mark %pbuf {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : !l1
      hivm.hir.copy ins(%pnz : !nz16) outs(%pbuf : !l1) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %dp : !t32, !t32) outs(%empty32 : !t32) -> !t32
      hivm.hir.store ins(%ds : !t32) outs(%ds_out : memref<16x16xf32>)
      %pcube = bufferization.to_tensor %pbuf restrict writable : !l1
      %init = tensor.empty() : !nz32
      %dot = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
        ins(%pcube, %rhs, %true, %c16, %c16, %c16 : !nz16, !nz16, i1, index, index, index)
        outs(%init : !nz32) -> !nz32
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : !nz32) outs(%out : memref<16x16xf32>)
    }
    return
  }
}
