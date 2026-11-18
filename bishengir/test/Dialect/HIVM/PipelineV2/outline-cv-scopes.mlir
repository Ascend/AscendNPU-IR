// RUN: bishengir-opt -outline-cv-scopes -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @outline_basic_and_fanout
// CHECK: %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<16x16xf32>)
// CHECK:   %[[S0:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.nd2nz
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S1:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[VEXP:.*]] = hivm.hir.vexp ins(%[[S0]] : tensor<16x16xf32>)
// CHECK:     scope.return %[[VEXP]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S2:.*]] = scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK:     hivm.hir.vcast ins(%[[S1]] : tensor<16x16xf32>)
// CHECK:     hivm.hir.copy
// CHECK:     scope.return {{.*}} : tensor<1x1x16x16xf16>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S3:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.nd2nz
// CHECK:     hivm.hir.mmadL1 ins(%[[S2]], {{.*}})
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S4:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[VADD0:.*]] = hivm.hir.vadd ins(%[[ACC]], %[[S1]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     %[[VADD1:.*]] = hivm.hir.vadd ins(%[[VADD0]], %[[S3]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     scope.return %[[VADD1]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   scf.yield %[[S4]] : tensor<16x16xf32>
// CHECK: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_basic_and_fanout(
      %a: tensor<1x1x16x16xf16>,
      %k_gm: memref<16x16xf16>,
      %v_gm: memref<16x16xf16>,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %k_alloc = memref.alloc() : memref<1x1x16x16xf16>
      hivm.hir.nd2nz {dst_continuous} ins(%k_gm : memref<16x16xf16>) outs(%k_alloc : memref<1x1x16x16xf16>)
      %k = bufferization.to_tensor %k_alloc restrict writable : memref<1x1x16x16xf16>
      %dot0_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dot0 = hivm.hir.mmadL1 ins(%a, %k, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %s0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)

      // %p is consumed by both the CUBE path (via vcast + copy) and the final VECTOR vadd.
      %p_init = tensor.empty() : tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%s0 : tensor<16x16xf32>) outs(%p_init : tensor<16x16xf32>) -> tensor<16x16xf32>

      %p_f16_init = tensor.empty() : tensor<16x16xf16>
      %p_f16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%p_f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %p_4d = tensor.expand_shape %p_f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %p_cbuf = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%p_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)

      %v_alloc = memref.alloc() : memref<1x1x16x16xf16>
      hivm.hir.nd2nz {dst_continuous} ins(%v_gm : memref<16x16xf16>) outs(%v_alloc : memref<1x1x16x16xf16>)
      %v = bufferization.to_tensor %v_alloc restrict writable : memref<1x1x16x16xf16>
      %dot1_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dot1 = hivm.hir.mmadL1 ins(%p_cbuf, %v, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %ub1_cast = memref.memory_space_cast %ub1 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %s1 = bufferization.to_tensor %ub1_cast restrict writable : memref<16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot1 : tensor<1x1x16x16xf32>) outs(%ub1 : memref<16x16xf32, #hivm.address_space<ub>>)

      %sum0 = hivm.hir.vadd ins(%acc, %p : tensor<16x16xf32>, tensor<16x16xf32>) outs(%p_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %sum1 = hivm.hir.vadd ins(%sum0, %s1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%p_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %sum1 : tensor<16x16xf32>
    }
    return %res : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @outline_scf_if_hoisted_alloc_and_value_yield
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<16x16xf32>)
// CHECK:   %[[S0:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[UB_ALLOC:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
// CHECK:     %[[UB_CAST:.*]] = memref.memory_space_cast %[[UB_ALLOC]]
// CHECK:     %[[UB_TENSOR:.*]] = bufferization.to_tensor %[[UB_CAST]]
// CHECK:     scf.if {{.*}} {
// CHECK:       hivm.hir.mmadL1
// CHECK:       hivm.hir.fixpipe {{.*}} outs(%[[UB_ALLOC]] : memref<16x16xf32, #hivm.address_space<ub>>)
// CHECK:     } {pipeline.if.cond = 0 : i64}
// CHECK:     scope.return %[[UB_TENSOR]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S1:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[IF_RES:.*]] = scf.if {{.*}} -> (tensor<16x16xf32>) {
// CHECK:       %[[VADD:.*]] = hivm.hir.vadd ins(%[[ACC]], %[[S0]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:       scf.yield %[[VADD]] : tensor<16x16xf32>
// CHECK:     } else {
// CHECK:       scf.yield %[[ACC]] : tensor<16x16xf32>
// CHECK:     } {pipeline.if.cond = 0 : i64}
// CHECK:     scope.return %[[IF_RES]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   scf.yield %[[S1]] : tensor<16x16xf32>
// CHECK: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_scf_if_hoisted_alloc_and_value_yield(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %cond = arith.cmpi slt, %i, %c2 : i32
      %if_res = scf.if %cond -> (tensor<16x16xf32>) {
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub_alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub_cast = memref.memory_space_cast %ub_alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %s = bufferization.to_tensor %ub_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub_alloc : memref<16x16xf32, #hivm.address_space<ub>>)
        %v_init = tensor.empty() : tensor<16x16xf32>
        %sum = hivm.hir.vadd ins(%acc, %s : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %sum : tensor<16x16xf32>
      } else {
        scf.yield %acc : tensor<16x16xf32>
      }
      scf.yield %if_res : tensor<16x16xf32>
    }
    return %res : tensor<16x16xf32>
  }
}
