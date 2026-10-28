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

// -----

// CHECK-LABEL: func.func @outline_store_and_fixpipe_to_gm_without_to_tensor
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<16x16xf32>)
// CHECK:   %[[S0:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe {{.*}} outs({{.*}} : memref<16x16xf32, #hivm.address_space<ub>>)
// CHECK:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   scope.scope : () -> () {
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   scope.scope : () -> () {
// CHECK:     hivm.hir.vmul ins(%[[S0]], {{.*}})
// CHECK:     hivm.hir.store
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S3:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[SUM:.*]] = hivm.hir.vadd ins(%[[ACC]], %[[S0]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     scope.return %[[SUM]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   scf.yield %[[S3]] : tensor<16x16xf32>
// CHECK: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_store_and_fixpipe_to_gm_without_to_tensor(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %gm_vec_out: memref<16x16xf32>,
      %gm_cube_out: memref<16x16xf32>,
      %scale: f32,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %dot0_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dot0 = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %s0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)

      %vmul_init = tensor.empty() : tensor<16x16xf32>
      %scaled = hivm.hir.vmul ins(%s0, %scale : tensor<16x16xf32>, f32) outs(%vmul_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %gm_vec_sub = memref.subview %gm_vec_out[0, 0] [16, 16] [1, 1] : memref<16x16xf32> to memref<16x16xf32, strided<[16, 1]>>
      hivm.hir.store ins(%scaled : tensor<16x16xf32>) outs(%gm_vec_sub : memref<16x16xf32, strided<[16, 1]>>)

      %dot1_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dot1 = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %gm_cube_sub = memref.subview %gm_cube_out[0, 0] [16, 16] [1, 1] : memref<16x16xf32> to memref<16x16xf32, strided<[16, 1]>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot1 : tensor<1x1x16x16xf32>) outs(%gm_cube_sub : memref<16x16xf32, strided<[16, 1]>>)

      %v_init = tensor.empty() : tensor<16x16xf32>
      %sum = hivm.hir.vadd ins(%acc, %s0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %sum : tensor<16x16xf32>
    }
    return %res : tensor<16x16xf32>
  }
}
// Nested mixed loops remain inside the original conditional. Outlining the
// inner loop must not make its internal scope results escape into an
// enclosing loop that the outliner does not support.
// CHECK-LABEL: func.func @nested_then
// CHECK: scf.for %{{.*}} =
// CHECK-NEXT: scf.if %{{.*}} {
// CHECK-NEXT: %[[INNER:.*]] = scf.for
// CHECK: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.cv_pipelined_loop}
// CHECK-NEXT: hivm.hir.store ins(%[[INNER]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @nested_then(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x1x16x16xf16>, %arg2: index, %arg3: i1, %arg4: memref<16x16xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<16x16xf32>
    scf.for %arg5 = %c0 to %arg2 step %c1 {
      scf.if %arg3 {
        %1 = scf.for %arg6 = %c0 to %arg2 step %c1 iter_args(%arg7 = %0) -> (tensor<16x16xf32>) {
          %2 = tensor.empty() : tensor<1x1x16x16xf32>
          %3 = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%2 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
          %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
          %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
          %4 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%3 : tensor<1x1x16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
          %5 = tensor.empty() : tensor<16x16xf32>
          %6 = hivm.hir.vadd ins(%arg7, %4 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
          scf.yield %6 : tensor<16x16xf32>
        }
        hivm.hir.store ins(%1 : tensor<16x16xf32>) outs(%arg4 : memref<16x16xf32>)
      }
    }
    return
  }
}

// -----

// Nested mixed loops remain inside the original conditional. Outlining the
// inner loop must not make its internal scope results escape into an
// enclosing loop that the outliner does not support.
// CHECK-LABEL: func.func @nested_else
// CHECK: scf.for %{{.*}} =
// CHECK-NEXT: scf.if %{{.*}} {
// CHECK-NEXT: } else {
// CHECK-NEXT: %[[INNER:.*]] = scf.for
// CHECK: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: } {hivm.cv_pipelined_loop}
// CHECK-NEXT: hivm.hir.store ins(%[[INNER]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @nested_else(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x1x16x16xf16>, %arg2: index, %arg3: i1, %arg4: memref<16x16xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<16x16xf32>
    scf.for %arg5 = %c0 to %arg2 step %c1 {
      scf.if %arg3 {
      } else {
        %1 = scf.for %arg6 = %c0 to %arg2 step %c1 iter_args(%arg7 = %0) -> (tensor<16x16xf32>) {
          %2 = tensor.empty() : tensor<1x1x16x16xf32>
          %3 = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%2 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
          %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
          %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
          %4 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%3 : tensor<1x1x16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
          %5 = tensor.empty() : tensor<16x16xf32>
          %6 = hivm.hir.vadd ins(%arg7, %4 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
          scf.yield %6 : tensor<16x16xf32>
        }
        hivm.hir.store ins(%1 : tensor<16x16xf32>) outs(%arg4 : memref<16x16xf32>)
      }
    }
    return
  }
}

// -----

// Resultless loops must end with one yield after their outlined scopes.
// CHECK-LABEL: func.func @outline_no_iter_args
// CHECK: scf.for
// CHECK-NEXT: %{{.*}} = scope.scope
// CHECK: hivm.hir.mmadL1
// CHECK: scope.return
// CHECK: scope.scope
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.store
// CHECK: scope.return
// CHECK-NEXT: } {hivm.loop_core_type
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_no_iter_args(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x1x16x16xf16>, %arg2: index, %arg3: i1, %arg4: memref<16x16xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<16x16xf32>
    scf.for %arg5 = %c0 to %arg2 step %c1 {
      %1 = tensor.empty() : tensor<1x1x16x16xf32>
      %2 = hivm.hir.mmadL1 ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%1 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %3 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%2 : tensor<1x1x16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
      %4 = tensor.empty() : tensor<16x16xf32>
      %5 = hivm.hir.vadd ins(%0, %3 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.store ins(%5 : tensor<16x16xf32>) outs(%arg4 : memref<16x16xf32>)
    }
    return
  }
}

// -----

// ELSE-local descriptors must dominate scope returns, and raw tensor
// live-outs must leave their branch through scf.if results.
// CHECK-LABEL: func.func @outline_else_live_out
// CHECK: scf.for
// CHECK-NEXT: %{{.*}} = scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK: %[[ELSE_CB:.*]] = memref.alloc()
// CHECK: %[[ELSE_TENSOR:.*]] = bufferization.to_tensor %[[ELSE_CB]]
// CHECK: scf.if
// CHECK: } else {
// CHECK: hivm.hir.copy
// CHECK: scope.return %[[ELSE_TENSOR]]
// CHECK: %{{.*}} = scope.scope : () -> tensor<1x1x16x16xf32> {
// CHECK: %[[ELSE_RESULT:.*]] = scf.if {{.*}} -> (tensor<1x1x16x16xf32>) {
// CHECK: tensor.empty
// CHECK: scf.yield
// CHECK: } else {
// CHECK: %[[ELSE_DOT:.*]] = hivm.hir.mmadL1
// CHECK: scf.yield %[[ELSE_DOT]]
// CHECK: scope.return %[[ELSE_RESULT]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_else_live_out(%arg0: tensor<1x1x16x16xf16>, %arg1: tensor<1x1x16x16xf16>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>, %arg4: index, %arg5: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 2.000000e+00 : f32
    %true = arith.constant true
    scf.for %arg6 = %c0 to %arg4 step %c1 {
      scf.if %arg5 {
      } else {
        %alloc = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%arg0 : tensor<1x1x16x16xf16>) outs(%alloc : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        %0 = bufferization.to_tensor %alloc restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %1 = tensor.empty() : tensor<1x1x16x16xf32>
        %2 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%0, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%1 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %alloc_0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%2 : tensor<1x1x16x16xf32>) outs(%alloc_0 : memref<16x16xf32, #hivm.address_space<ub>>)
        %3 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
        %4 = tensor.empty() : tensor<16x16xf32>
        %5 = hivm.hir.vmul ins(%3, %cst : tensor<16x16xf32>, f32) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
        hivm.hir.store ins(%5 : tensor<16x16xf32>) outs(%arg2 : memref<16x16xf32>)
        %6 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%arg0, %arg1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%2 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%6 : tensor<1x1x16x16xf32>) outs(%arg3 : memref<16x16xf32>)
      }
    }
    return
  }
}

// -----

// Export the UB descriptor from CUBE and keep the conditional merge in a
// separate shared scope. The loop must still be outlined in either branch.
// CHECK-LABEL: func.func @outline_conditional_merge_then
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}})
// CHECK: %[[CUBE:.*]]:2 = scope.scope : () -> (tensor<16x16xf32>, i1) {
// CHECK: %[[BUFFER:.*]] = bufferization.to_tensor
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.fixpipe
// CHECK-NOT: scf.yield %[[ACC]]
// CHECK: scope.return %[[BUFFER]], {{.*}} : tensor<16x16xf32>, i1
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: %[[MERGE:.*]] = scope.scope
// CHECK: %[[SELECT:.*]] = arith.select %[[CUBE]]#1, %[[CUBE]]#0, %[[ACC]]
// CHECK-NEXT: scope.return %[[SELECT]]
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK: %[[VECTOR:.*]] = scope.scope
// CHECK: hivm.hir.vadd ins(%[[ACC]], %[[MERGE]]
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scf.yield %[[VECTOR]]
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_conditional_merge_then(%a: tensor<1x1x16x16xf16>, %b: tensor<1x1x16x16xf16>, %ub: i32, %cond: i1) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %true = arith.constant true
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = hivm.hir.vbrc ins(%zero : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %result = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %selected = scf.if %cond -> (tensor<16x16xf32>) {
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %buf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%buf : memref<16x16xf32, #hivm.address_space<ub>>)
        %value = bufferization.to_tensor %buf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
        scf.yield %value : tensor<16x16xf32>
      } else {
        scf.yield %acc : tensor<16x16xf32>
      } {may_not_exec, normalized_in_L0C = [0 : i32]}
      %next = hivm.hir.vadd ins(%acc, %selected : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %next : tensor<16x16xf32>
    }
    return %result : tensor<16x16xf32>
  }
}



// -----

// Export the UB descriptor from CUBE and keep the conditional merge in a
// separate shared scope. The loop must still be outlined in either branch.
// CHECK-LABEL: func.func @outline_conditional_merge_else
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}})
// CHECK: %[[CUBE:.*]]:2 = scope.scope : () -> (tensor<16x16xf32>, i1) {
// CHECK: %[[BUFFER:.*]] = bufferization.to_tensor
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.fixpipe
// CHECK-NOT: scf.yield %[[ACC]]
// CHECK: scope.return %[[BUFFER]], {{.*}} : tensor<16x16xf32>, i1
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: %[[MERGE:.*]] = scope.scope
// CHECK: %[[SELECT:.*]] = arith.select %[[CUBE]]#1, %[[ACC]], %[[CUBE]]#0
// CHECK-NEXT: scope.return %[[SELECT]]
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK: %[[VECTOR:.*]] = scope.scope
// CHECK: hivm.hir.vadd ins(%[[ACC]], %[[MERGE]]
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scf.yield %[[VECTOR]]
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_conditional_merge_else(%a: tensor<1x1x16x16xf16>, %b: tensor<1x1x16x16xf16>, %ub: i32, %cond: i1) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %true = arith.constant true
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = hivm.hir.vbrc ins(%zero : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %result = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %selected = scf.if %cond -> (tensor<16x16xf32>) {
        scf.yield %acc : tensor<16x16xf32>
      } else {
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %buf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%buf : memref<16x16xf32, #hivm.address_space<ub>>)
        %value = bufferization.to_tensor %buf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
        scf.yield %value : tensor<16x16xf32>
      } {may_not_exec, normalized_in_L0C = [0 : i32]}
      %next = hivm.hir.vadd ins(%acc, %selected : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %next : tensor<16x16xf32>
    }
    return %result : tensor<16x16xf32>
  }
}

// -----

// Counter storage must not be cloned without its initialization. Both core
// consumers read scalar results from one shared state scope.
// CHECK-LABEL: func.func @outline_conditional_counter
// CHECK: scf.for
// CHECK-NEXT: %[[STATE:.*]]:2 = scope.scope : () -> (i32, i32) {
// CHECK: %[[COUNTER:.*]] = memref.alloca()
// CHECK-NEXT: memref.store {{.*}}, %[[COUNTER]]
// CHECK: %[[BEFORE:.*]] = scf.if
// CHECK: %[[READ:.*]] = memref.load %[[COUNTER]]
// CHECK: %[[INC:.*]] = arith.addi %[[READ]]
// CHECK-NEXT: memref.store %[[INC]], %[[COUNTER]]
// CHECK-NEXT: scf.yield %[[READ]]
// CHECK: %[[AFTER:.*]] = memref.load %[[COUNTER]]
// CHECK-NEXT: scope.return %[[BEFORE]], %[[AFTER]]
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK-NOT: memref.alloca
// CHECK: arith.cmpi eq, %[[STATE]]#0
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: memref.alloca
// CHECK: arith.cmpi eq, %[[STATE]]#1
// CHECK: hivm.hir.vadd
// CHECK-NOT: memref.alloca
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_conditional_counter(%a: tensor<1x1x16x16xf16>, %b: tensor<1x1x16x16xf16>, %ub: i32, %cond: i1) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %true = arith.constant true
    %empty = tensor.empty() : tensor<16x16xf32>
    %init = hivm.hir.vbrc ins(%zero : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %result = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %counter = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
      memref.store %c0, %counter[] : memref<i32>
      %selected = scf.if %cond -> (tensor<16x16xf32>) {
        %before = memref.load %counter[] : memref<i32>
        %clear = arith.cmpi eq, %before, %c0 : i32
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %clear, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %buf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%buf : memref<16x16xf32, #hivm.address_space<ub>>)
        %value = bufferization.to_tensor %buf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
        %inc = arith.addi %before, %c1 : i32
        memref.store %inc, %counter[] : memref<i32>
        scf.yield %value : tensor<16x16xf32>
      } else {
        scf.yield %acc : tensor<16x16xf32>
      } {may_not_exec, normalized_in_L0C = [0 : i32]}
      %after = memref.load %counter[] : memref<i32>
      %keep = arith.cmpi eq, %after, %c0 : i32
      %next = scf.if %keep -> (tensor<16x16xf32>) {
        scf.yield %acc : tensor<16x16xf32>
      } else {
      %sum = hivm.hir.vadd ins(%acc, %selected : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %sum : tensor<16x16xf32>
      }
      scf.yield %next : tensor<16x16xf32>
    }
    return %result : tensor<16x16xf32>
  }
}
