// RUN: bishengir-opt -outline-cv-scopes -canonicalize -cse -split-input-file %s | FileCheck %s
// RUN: bishengir-opt -outline-cv-scopes -verify-each -split-input-file %s | FileCheck %s --check-prefix=RESHAPE
// RUN: bishengir-opt -outline-cv-scopes -split-input-file %s | FileCheck %s --check-prefix=TCB

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
// CHECK:     %[[VADD0:.*]] = hivm.hir.vadd ins(%[[ACC]], %[[S1]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     scope.return %[[VADD0]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S4:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.nd2nz
// CHECK:     hivm.hir.mmadL1 ins(%[[S2]], {{.*}})
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S5:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[VADD1:.*]] = hivm.hir.vadd ins(%[[S3]], %[[S4]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     scope.return %[[VADD1]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   scf.yield %[[S5]] : tensor<16x16xf32>
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

// -----

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
// CHECK: %[[ELSE_INIT:.*]] = tensor.empty() : tensor<1x1x16x16xf32>
// CHECK: %[[ELSE_RESULT:.*]] = scf.if {{.*}} -> (tensor<1x1x16x16xf32>) {
// CHECK: scf.yield %[[ELSE_INIT]]
// CHECK: } else {
// CHECK: %[[ELSE_DOT:.*]] = hivm.hir.mmadL1 {{.*}} outs(%[[ELSE_INIT]] : tensor<1x1x16x16xf32>)
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
// CHECK: %[[CUBE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK: %[[BUFFER:.*]] = bufferization.to_tensor
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.fixpipe
// CHECK-NOT: scf.yield %[[ACC]]
// CHECK: scope.return %[[BUFFER]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: %[[MERGE:.*]] = scope.scope
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %[[CUBE]], %[[ACC]]
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
// CHECK: %[[CUBE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK: %[[BUFFER:.*]] = bufferization.to_tensor
// CHECK: scf.if
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.fixpipe
// CHECK-NOT: scf.yield %[[ACC]]
// CHECK: scope.return %[[BUFFER]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: %[[MERGE:.*]] = scope.scope
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %[[ACC]], %[[CUBE]]
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

// Counter storage must not be cloned without its initialization. Each counter
// state scope outputs at most one result.
// CHECK-LABEL: func.func @outline_conditional_counter
// CHECK: scf.for
// CHECK-NEXT: %[[COUNTER_SCOPE:.*]] = scope.scope : () -> memref<i32> {
// CHECK-NEXT: %[[COUNTER:.*]] = memref.alloca()
// CHECK-NEXT: scope.return %[[COUNTER]] : memref<i32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: memref.store {{.*}}, %[[COUNTER_SCOPE]][]
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK-NEXT: %[[BEFORE_SCOPE:.*]] = scope.scope : () -> i32 {
// CHECK: %[[BEFORE:.*]] = scf.if
// CHECK: %[[READ:.*]] = memref.load %[[COUNTER_SCOPE]][]
// CHECK: %[[INC:.*]] = arith.addi %[[READ]]
// CHECK-NEXT: memref.store %[[INC]], %[[COUNTER_SCOPE]][]
// CHECK-NEXT: scf.yield %[[READ]]
// CHECK: scope.return %[[BEFORE]] : i32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK-NEXT: %[[AFTER_SCOPE:.*]] = scope.scope : () -> i32 {
// CHECK-NEXT: %[[AFTER:.*]] = memref.load %[[COUNTER_SCOPE]][]
// CHECK-NEXT: scope.return %[[AFTER]] : i32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK-NOT: memref.alloca
// CHECK: arith.cmpi eq, %[[BEFORE_SCOPE]]
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: memref.alloca
// CHECK: arith.cmpi eq, %[[AFTER_SCOPE]]
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

// -----

// Both branches produce tensors in different core scopes.
// CHECK-LABEL: func.func @outline_mixed_branch_vector_then
// CHECK: %[[VECTOR:.*]] = scope.scope
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: %[[CUBE:.*]] = scope.scope
// CHECK: bufferization.to_tensor
// CHECK: scf.if
// CHECK: hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: %[[MERGE:.*]] = scope.scope
// CHECK: %[[SELECT:.*]] = arith.select {{.*}}, %[[VECTOR]], %[[CUBE]]
// CHECK-NEXT: scope.return %[[SELECT]]
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK: hivm.hir.store ins(%[[MERGE]]
// CHECK: hivm.cv_pipelined_loop
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_mixed_branch_vector_then(%dot: tensor<2x2x16x16xf32>, %dst: memref<32x32xf32>, %n: index, %cond: i1) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      %selected = scf.if %cond -> tensor<32x32xf32> {
        %empty = tensor.empty() : tensor<32x32xf32>
        %v = hivm.hir.vbrc ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %v : tensor<32x32xf32>
      } else {
        %buf = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
        %cast = memref.memory_space_cast %buf : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
        %value = bufferization.to_tensor %cast restrict writable : memref<32x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<2x2x16x16xf32>) outs(%buf : memref<32x32xf32, #hivm.address_space<ub>>)
        scf.yield %value : tensor<32x32xf32>
      }
      hivm.hir.store ins(%selected : tensor<32x32xf32>) outs(%dst : memref<32x32xf32>)
    }
    return
  }
}

// -----

// Both branches produce tensors in different core scopes.
// CHECK-LABEL: func.func @outline_mixed_branch_vector_else
// CHECK: %[[CUBE:.*]] = scope.scope
// CHECK: bufferization.to_tensor
// CHECK: scf.if
// CHECK: hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: %[[VECTOR:.*]] = scope.scope
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: %[[MERGE:.*]] = scope.scope
// CHECK: %[[SELECT:.*]] = arith.select {{.*}}, %[[CUBE]], %[[VECTOR]]
// CHECK-NEXT: scope.return %[[SELECT]]
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK: hivm.hir.store ins(%[[MERGE]]
// CHECK: hivm.cv_pipelined_loop
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_mixed_branch_vector_else(%dot: tensor<2x2x16x16xf32>, %dst: memref<32x32xf32>, %n: index, %cond: i1) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      %selected = scf.if %cond -> tensor<32x32xf32> {
        %buf = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
        %cast = memref.memory_space_cast %buf : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
        %value = bufferization.to_tensor %cast restrict writable : memref<32x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<2x2x16x16xf32>) outs(%buf : memref<32x32xf32, #hivm.address_space<ub>>)
        scf.yield %value : tensor<32x32xf32>
      } else {
        %empty = tensor.empty() : tensor<32x32xf32>
        %v = hivm.hir.vbrc ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %v : tensor<32x32xf32>
      }
      hivm.hir.store ins(%selected : tensor<32x32xf32>) outs(%dst : memref<32x32xf32>)
    }
    return
  }
}

// -----

// A classified nested loop implicitly captures an outer-loop-local value.
// CHECK-LABEL: func.func @outline_mixed_branch_nested_capture
// CHECK: scf.for
// CHECK: %[[OFFSET:.*]] = arith.addi
// CHECK: scf.for
// CHECK: arith.addi %[[OFFSET]],
// CHECK: captured_offset
// CHECK: hivm.cv_pipelined_loop
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_mixed_branch_nested_capture(%dot: tensor<2x2x16x16xf32>, %dst: memref<32x32xf32>, %n: index, %cond: i1) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      %offset = arith.addi %i, %n : index
      scf.for %j = %c0 to %n step %c1 {
        %sum = arith.addi %offset, %j : index
        annotation.mark %sum {captured_offset} : index
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
      %selected = scf.if %cond -> tensor<32x32xf32> {
        %empty = tensor.empty() : tensor<32x32xf32>
        %v = hivm.hir.vbrc ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %v : tensor<32x32xf32>
      } else {
        %buf = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
        %cast = memref.memory_space_cast %buf : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
        %value = bufferization.to_tensor %cast restrict writable : memref<32x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<2x2x16x16xf32>) outs(%buf : memref<32x32xf32, #hivm.address_space<ub>>)
        scf.yield %value : tensor<32x32xf32>
      }
      hivm.hir.store ins(%selected : tensor<32x32xf32>) outs(%dst : memref<32x32xf32>)
    }
    return
  }
}

// -----

// The final counter read must remain after updates in the nested loop.
// CHECK-LABEL: func.func @outline_mixed_branch_nested_counter
// CHECK: scf.for
// CHECK: %[[COUNTER:.*]] = scope.scope
// CHECK: memref.alloca
// CHECK: memref.store
// CHECK: scf.for
// CHECK: memref.load %[[COUNTER]]
// CHECK: memref.store
// CHECK: %[[READ:.*]] = scope.scope
// CHECK: memref.load %[[COUNTER]]
// CHECK: arith.cmpi eq, %[[READ]],
// CHECK: arith.select
// CHECK: hivm.cv_pipelined_loop
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_mixed_branch_nested_counter(%dot: tensor<2x2x16x16xf32>, %dst: memref<32x32xf32>, %n: index, %cond: i1) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      %counter = memref.alloca() {normalize_matmul_counter = 2 : i32} : memref<i32>
      memref.store %c0_i32, %counter[] : memref<i32>
      scf.for %j = %c0 to %n step %c1 {
        %old = memref.load %counter[] : memref<i32>
        %next = arith.addi %old, %c1_i32 : i32
        memref.store %next, %counter[] : memref<i32>
      }
      %count = memref.load %counter[] : memref<i32>
      %no_matmul = arith.cmpi eq, %count, %c0_i32 : i32
      %selected = scf.if %no_matmul -> tensor<32x32xf32> {
        %empty = tensor.empty() : tensor<32x32xf32>
        %v = hivm.hir.vbrc ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %v : tensor<32x32xf32>
      } else {
        %buf = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
        %cast = memref.memory_space_cast %buf : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
        %value = bufferization.to_tensor %cast restrict writable : memref<32x32xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<2x2x16x16xf32>) outs(%buf : memref<32x32xf32, #hivm.address_space<ub>>)
        scf.yield %value : tensor<32x32xf32>
      }
      hivm.hir.store ins(%selected : tensor<32x32xf32>) outs(%dst : memref<32x32xf32>)
    }
    return
  }
}

// -----

// Scalar values loaded via memref.load from a loop-invariant memref are cloned
// into each consuming scope instead of being returned as scope results.
// CHECK-LABEL: func.func @outline_distribute_memref_load
// CHECK: scf.for
// CHECK-NEXT: %[[CUBE:.*]] = scope.scope : () -> tensor<32x32xf32> {
// CHECK: memref.reinterpret_cast
// CHECK: memref.load
// CHECK: hivm.hir.fixpipe
// CHECK: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK: memref.reinterpret_cast
// CHECK: memref.load
// CHECK: hivm.hir.store ins(%[[CUBE]]
// CHECK: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: hivm.cv_pipelined_loop
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_distribute_memref_load(
      %idx_buf: memref<16xi64>, %dot: tensor<2x2x16x16xf32>, %dst: memref<32x32xf32>, %n: index)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %view = memref.reinterpret_cast %idx_buf to offset: [%i], sizes: [1], strides: [1] : memref<16xi64> to memref<1xi64, strided<[1], offset: ?>>
      %idx_i64 = memref.load %view[%c0] : memref<1xi64, strided<[1], offset: ?>>
      %idx = arith.index_cast %idx_i64 : i64 to index
      %buf = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
      %sub = memref.subview %buf[%idx, 0] [32, 32] [1, 1] : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<2x2x16x16xf32>) outs(%sub : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>)
      %t = bufferization.to_tensor %buf restrict writable : memref<32x32xf32, #hivm.address_space<ub>>
      %dst_sub = memref.subview %dst[%idx, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      hivm.hir.store ins(%t : tensor<32x32xf32>) outs(%dst_sub : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}

// -----

// Intermediate tensor values exported from an scf.if branch reuse their outs()
// tensor.empty buffer in the opposite branch, hoisting branch-local tensor.empty
// above the scf.if so both branches share the same buffer.
// CHECK-LABEL: func.func @outline_if_intermediate_share_outs_buffer
// CHECK: %[[OUTER_INIT:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: scf.for
// CHECK:   %[[S1:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK-NEXT: %[[IF1:.*]] = scf.if %{{.*}} -> (tensor<16x16xf32>) {
// CHECK-NEXT:   %[[VRELU:.*]] = hivm.hir.vrelu ins(%{{.*}} : tensor<16x16xf32>) outs(%[[OUTER_INIT]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:        scf.yield %[[VRELU]] : tensor<16x16xf32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[OUTER_INIT]] : tensor<16x16xf32>
// CHECK-NEXT: }
// CHECK-NEXT: scope.return %[[IF1]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S0:.*]] = scope.scope : () -> tensor<16x32xf32> {
// CHECK-NEXT: %[[HOISTED_INIT:.*]] = tensor.empty() : tensor<16x32xf32>
// CHECK-NEXT: %[[IF0:.*]] = scf.if %{{.*}} -> (tensor<16x32xf32>) {
// CHECK-NEXT:   %[[VEXP:.*]] = hivm.hir.vexp ins(%{{.*}} : tensor<16x32xf32>) outs(%[[HOISTED_INIT]] : tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK-NEXT:   hivm.hir.store ins(%[[VEXP]] : tensor<16x32xf32>) outs(%{{.*}} : memref<16x32xf32>)
// CHECK-NEXT:   scf.yield %[[VEXP]] : tensor<16x32xf32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[HOISTED_INIT]] : tensor<16x32xf32>
// CHECK-NEXT: }
// CHECK-NEXT: scope.return %[[IF0]] : tensor<16x32xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_if_intermediate_share_outs_buffer(
      %x: tensor<16x32xf32>, %y: tensor<16x16xf32>, %b: tensor<1x1x16x16xf16>, %dst0: memref<16x16xf32>, %dst1: memref<16x32xf32>, %dst2: memref<16x16xf32>, %n: index, %cond: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %outer_init = tensor.empty() : tensor<16x16xf32>
    scf.for %i = %c0 to %n step %c1 {
      scf.if %cond {
        %inner_init = tensor.empty() : tensor<16x32xf32>
        %exp = hivm.hir.vexp ins(%x : tensor<16x32xf32>) outs(%inner_init : tensor<16x32xf32>) -> tensor<16x32xf32>
        %relu = hivm.hir.vrelu ins(%y : tensor<16x16xf32>) outs(%outer_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        %f16_init = tensor.empty() : tensor<16x16xf16>
        %f16 = hivm.hir.vcast ins(%relu : tensor<16x16xf32>) outs(%f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %f16_4d = tensor.expand_shape %f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%f16_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        %a = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%dst0 : memref<16x16xf32>)
        hivm.hir.store ins(%exp : tensor<16x32xf32>) outs(%dst1 : memref<16x32xf32>)
        %sum_exp = hivm.hir.vadd ins(%exp, %x : tensor<16x32xf32>, tensor<16x32xf32>) outs(%inner_init : tensor<16x32xf32>) -> tensor<16x32xf32>
        hivm.hir.store ins(%sum_exp : tensor<16x32xf32>) outs(%dst1 : memref<16x32xf32>)
        %sum = hivm.hir.vadd ins(%relu, %y : tensor<16x16xf32>, tensor<16x16xf32>) outs(%outer_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        hivm.hir.store ins(%sum : tensor<16x16xf32>) outs(%dst2 : memref<16x16xf32>)
      }
    }
    return
  }
}

// -----

// Both branch-local and enclosing broadcasts keep their padded destinations
// on the active path, while the inactive path reuses the backing tensor.empty.
// CHECK-LABEL: func.func @outline_if_intermediate_broadcast_padding
// CHECK: %[[OUTER_EMPTY:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: %[[OUTER_PAD:.*]] = hivm.hir.vbrc {{.*}} outs(%[[OUTER_EMPTY]] : tensor<16x16xf32>)
// CHECK: scf.for
// CHECK: scf.if {{.*}} -> (tensor<16x16xf32>) {
// CHECK: %[[RELU:.*]] = hivm.hir.vrelu {{.*}} outs(%[[OUTER_PAD]] : tensor<16x16xf32>)
// CHECK: scf.yield %[[RELU]] : tensor<16x16xf32>
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[OUTER_EMPTY]] : tensor<16x16xf32>
// CHECK: %[[INNER_EMPTY:.*]] = tensor.empty() : tensor<16x32xf32>
// CHECK-NEXT: %{{.*}} = scf.if {{.*}} -> (tensor<16x32xf32>) {
// CHECK-NEXT: %[[INNER_PAD:.*]] = hivm.hir.vbrc {{.*}} outs(%[[INNER_EMPTY]] : tensor<16x32xf32>)
// CHECK-NEXT: %[[EXP:.*]] = hivm.hir.vexp {{.*}} outs(%[[INNER_PAD]] : tensor<16x32xf32>)
// CHECK: scf.yield %[[EXP]] : tensor<16x32xf32>
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[INNER_EMPTY]] : tensor<16x32xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_if_intermediate_broadcast_padding(
      %x: tensor<16x32xf32>, %y: tensor<16x16xf32>, %b: tensor<1x1x16x16xf16>, %dst0: memref<16x16xf32>, %dst1: memref<16x32xf32>, %dst2: memref<16x16xf32>, %n: index, %cond: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %pad = arith.constant 0.0 : f32
    %outer_init = tensor.empty() : tensor<16x16xf32>
    %outer_padded = hivm.hir.vbrc ins(%pad : f32) outs(%outer_init : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.for %i = %c0 to %n step %c1 {
      scf.if %cond {
        %inner_init = tensor.empty() : tensor<16x32xf32>
        %inner_padded = hivm.hir.vbrc ins(%pad : f32) outs(%inner_init : tensor<16x32xf32>) -> tensor<16x32xf32>
        %exp = hivm.hir.vexp ins(%x : tensor<16x32xf32>) outs(%inner_padded : tensor<16x32xf32>) -> tensor<16x32xf32>
        %relu = hivm.hir.vrelu ins(%y : tensor<16x16xf32>) outs(%outer_padded : tensor<16x16xf32>) -> tensor<16x16xf32>
        %f16_init = tensor.empty() : tensor<16x16xf16>
        %f16 = hivm.hir.vcast ins(%relu : tensor<16x16xf32>) outs(%f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %f16_4d = tensor.expand_shape %f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%f16_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        %a = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%dst0 : memref<16x16xf32>)
        hivm.hir.store ins(%exp : tensor<16x32xf32>) outs(%dst1 : memref<16x32xf32>)
        %sum_exp = hivm.hir.vadd ins(%exp, %x : tensor<16x32xf32>, tensor<16x32xf32>) outs(%inner_init : tensor<16x32xf32>) -> tensor<16x32xf32>
        hivm.hir.store ins(%sum_exp : tensor<16x32xf32>) outs(%dst1 : memref<16x32xf32>)
        %sum = hivm.hir.vadd ins(%relu, %y : tensor<16x16xf32>, tensor<16x16xf32>) outs(%outer_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        hivm.hir.store ins(%sum : tensor<16x16xf32>) outs(%dst2 : memref<16x16xf32>)
      }
    }
    return
  }
}

// -----

// When a normalize_matmul_counter alloca is defined outside the pipelined loop,
// its increment and store inside the loop are absorbed into the consuming CUBE
// scope so the scope returns only the tensor result.
// CHECK-LABEL: func.func @outline_external_normalize_matmul_counter
// CHECK: %[[COUNTER:.*]] = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<1x1x16x16xf32>)
// CHECK-NEXT: %[[VEC:.*]] = scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK: hivm.hir.copy
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[CUBE:.*]] = scope.scope : () -> tensor<1x1x16x16xf32> {
// CHECK-NEXT: %[[LOAD:.*]] = memref.load %[[COUNTER]][]
// CHECK-NEXT: %[[COND:.*]] = arith.cmpi eq, %[[LOAD]],
// CHECK-NEXT: %[[MMAD:.*]] = hivm.hir.mmadL1 ins(%{{.*}}, %[[VEC]], %[[COND]], {{.*}}) outs(%[[ACC]] : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
// CHECK-NEXT: %[[INC:.*]] = arith.addi %[[LOAD]],
// CHECK-NEXT: memref.store %[[INC]], %[[COUNTER]][]
// CHECK-NEXT: scope.return %[[MMAD]] : tensor<1x1x16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scf.yield %[[CUBE]] : tensor<1x1x16x16xf32>
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_external_normalize_matmul_counter(
      %a: tensor<1x1x16x16xf16>, %x: tensor<16x16xf16>, %ub: i32) -> tensor<1x1x16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %counter = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    memref.store %c0, %counter[] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>
    %init = tensor.empty() : tensor<1x1x16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<1x1x16x16xf32>) : i32 {
      %exp_init = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%x : tensor<16x16xf16>) outs(%exp_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %exp_4d = tensor.expand_shape %exp [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %b = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%exp_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %before = memref.load %counter[] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>
      %clear = arith.cmpi eq, %before, %c0 : i32
      %dot = hivm.hir.mmadL1 ins(%a, %b, %clear, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %inc = arith.addi %before, %c1 : i32
      memref.store %inc, %counter[] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : memref<i32>
      scf.yield %dot : tensor<1x1x16x16xf32>
    }
    return %res : tensor<1x1x16x16xf32>
  }
}

// -----

// duplicateExtractScalarForCube clones the scalar chain feeding CUBE ops onto
// the workspace re-extract while preserving the original scalar chain for
// VECTOR ops and erasing the replacementLabel mark.
// CHECK-LABEL: func.func @outline_duplicate_extract_scalar_for_cube
// CHECK-NOT: DuplicateTensorExtractForCube::replacementLabel
// CHECK: scf.for
// CHECK: %[[VRED_SCOPE:.*]] = scope.scope : () -> tensor<1xf32> {
// CHECK:   %[[RED:.*]] = hivm.hir.vreduce
// CHECK:   scope.return %[[RED]] : tensor<1xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: %[[WS_SCOPE:.*]] = scope.scope : () -> tensor<1xf32> {
// CHECK:   %[[WS:.*]] = memref_ext.alloc_workspace()
// CHECK:   hivm.hir.store ins(%[[VRED_SCOPE]] : tensor<1xf32>) outs(%[[WS]] : memref<1xf32>)
// CHECK:   %[[WST:.*]] = bufferization.to_tensor %[[WS]]
// CHECK:   scope.return %[[WST]] : tensor<1xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: %[[CUBE_SCOPE:.*]] = scope.scope : () -> tensor<16x16xf16> {
// CHECK:   %[[CEXT:.*]] = tensor.extract %[[WS_SCOPE]][%{{.*}}] {{.*}}DuplicateTensorExtractForCube::newExtractLabel{{.*}} : tensor<1xf32>
// CHECK:   %[[KF:.*]] = arith.fptosi %[[CEXT]] : f32 to i32
// CHECK:   %[[KIDX:.*]] = arith.index_cast %[[KF]] : i32 to index
// CHECK:   hivm.hir.mmadL1 {{.*}} %[[KIDX]]
// CHECK:   hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK: %[[VEC2_SCOPE:.*]] = scope.scope : () -> tensor<16x16xf16> {
// CHECK:   %[[VEXT:.*]] = tensor.extract %[[VRED_SCOPE]][%{{.*}}] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
// CHECK:   %[[VF16:.*]] = arith.truncf %[[VEXT]] : f32 to f16
// CHECK:   hivm.hir.vadd ins(%[[CUBE_SCOPE]], %[[VF16]] : tensor<16x16xf16>, f16)
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT: DuplicateTensorExtractForCube::replacementLabel
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_duplicate_extract_scalar_for_cube(
      %a: tensor<1x1x16x16xf16>, %b: tensor<1x1x16x16xf16>,
      %svin: tensor<8xf32>, %vinit: tensor<16x16xf16>, %bound: i32) -> tensor<16x16xf16>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0idx = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %res = scf.for %i = %c0 to %bound step %c1 iter_args(%vacc = %vinit) -> (tensor<16x16xf16>) : i32 {
      %sexp = tensor.empty() : tensor<1xf32>
      %red = hivm.hir.vreduce <sum> ins(%svin : tensor<8xf32>) outs(%sexp : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
      %ext = tensor.extract %red[%c0idx] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
      %ws = memref_ext.alloc_workspace() : memref<1xf32>
      hivm.hir.store ins(%red : tensor<1xf32>) outs(%ws : memref<1xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, "inserted-store"}
      %stored = bufferization.to_tensor %ws restrict writable : memref<1xf32>
      %newext = tensor.extract %stored[%c0idx] {"DuplicateTensorExtractForCube::newExtractLabel" = 1 : i32, "DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
      annotation.mark %ext {"DuplicateTensorExtractForCube::replacementLabel" = 1 : i32} keys = [] values = [%newext : f32] : f32

      %kf = arith.fptosi %ext : f32 to i32
      %kidx = arith.index_cast %kf : i32 to index
      %cdest = tensor.empty() : tensor<1x1x16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %kidx : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%cdest : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf16>

      %extf16 = arith.truncf %ext : f32 to f16
      %vdest = tensor.empty() : tensor<16x16xf16>
      %vadd = hivm.hir.vadd ins(%wst, %extf16 : tensor<16x16xf16>, f16) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      scf.yield %vadd : tensor<16x16xf16>
    }
    return %res : tensor<16x16xf16>
  }
}

// -----

// Mixed scf.if conditioned on a scalar marked with replacementLabel routes the
// CUBE scope's scf.if condition through the workspace extract while keeping the
// VECTOR scope's scf.if condition on the UB extract, preserving pipeline.if.cond.
// CHECK-LABEL: func.func @outline_duplicate_extract_scalar_mixed_if_cond
// CHECK-NOT: DuplicateTensorExtractForCube::replacementLabel
// CHECK: scf.for
// CHECK: %[[RED_SCOPE:.*]] = scope.scope : () -> tensor<1xi32> {
// CHECK:   hivm.hir.vreduce
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: %[[WS_SCOPE:.*]] = scope.scope : () -> tensor<i32> {
// CHECK:   %[[WS:.*]] = memref_ext.alloc_workspace() : memref<i32>
// CHECK:   hivm.hir.store {{.*}} outs(%[[WS]] : memref<i32>)
// CHECK:   %[[WST:.*]] = bufferization.to_tensor %[[WS]]
// CHECK:   scope.return %[[WST]] : tensor<i32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[RED_SCOPE]]
// CHECK:   %[[VEXT:.*]] = tensor.extract %[[COLLAPSED]][] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<i32>
// CHECK:   %[[VCOND:.*]] = arith.cmpi ugt, %[[VEXT]], %{{.*}} : i32
// CHECK:   scf.if %[[VCOND]] {
// CHECK:     hivm.hir.copy
// CHECK:   } {pipeline.if.cond = 0 : i64}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK: %[[CUBE_SCOPE:.*]] = scope.scope : () -> tensor<1x1x16x16xf32> {
// CHECK:   %[[CEXT:.*]] = tensor.extract %[[WS_SCOPE]][] {{.*}}DuplicateTensorExtractForCube::newExtractLabel{{.*}} : tensor<i32>
// CHECK:   %[[CCOND:.*]] = arith.cmpi ugt, %[[CEXT]], %{{.*}} : i32
// CHECK:   %[[CIF:.*]] = scf.if %[[CCOND]] -> (tensor<1x1x16x16xf32>) {
// CHECK:     hivm.hir.mmadL1
// CHECK:   } else {
// CHECK:     scf.yield
// CHECK:   } {pipeline.if.cond = 0 : i64}
// CHECK:   scope.return %[[CIF]] : tensor<1x1x16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NOT: DuplicateTensorExtractForCube::replacementLabel
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_duplicate_extract_scalar_mixed_if_cond(
      %x: tensor<16x16xf16>, %b: tensor<1x1x16x16xf16>, %mask: tensor<256xi32>, %bound: i32) -> tensor<1x1x16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<1x1x16x16xf32>
    %res = scf.for %i = %c0 to %bound step %c1 iter_args(%acc = %init) -> (tensor<1x1x16x16xf32>) : i32 {
      %red_init = tensor.empty() : tensor<1xi32>
      %red = hivm.hir.vreduce <sum> ins(%mask : tensor<256xi32>) outs(%red_init : tensor<1xi32>) reduce_dims = [0] -> tensor<1xi32>
      %collapsed = tensor.collapse_shape %red [] : tensor<1xi32> into tensor<i32>
      %ext = tensor.extract %collapsed[] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<i32>
      %ws = memref_ext.alloc_workspace() : memref<i32>
      hivm.hir.store ins(%collapsed : tensor<i32>) outs(%ws : memref<i32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, "hivm.inserted-store"}
      %wst = bufferization.to_tensor %ws restrict writable : memref<i32>
      %newext = tensor.extract %wst[] {"DuplicateTensorExtractForCube::newExtractLabel" = 1 : i32, "DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<i32>
      annotation.mark %ext {"DuplicateTensorExtractForCube::replacementLabel" = 1 : i32} keys = [] values = [%newext : i32] : i32

      %cond = arith.cmpi ugt, %ext, %c0 : i32
      %next = scf.if %cond -> (tensor<1x1x16x16xf32>) {
        %exp_init = tensor.empty() : tensor<16x16xf16>
        %exp = hivm.hir.vexp ins(%x : tensor<16x16xf16>) outs(%exp_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %exp_4d = tensor.expand_shape %exp [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %a = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%exp_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        scf.yield %dot : tensor<1x1x16x16xf32>
      } else {
        scf.yield %acc : tensor<1x1x16x16xf32>
      }
      scf.yield %next : tensor<1x1x16x16xf32>
    }
    return %res : tensor<1x1x16x16xf32>
  }
}

// -----

// Map unmapped annotation.mark and hivm.debug ops to the work item of their
// operand's definition inside the loop instead of a trailing catch-all scope.
// CHECK-LABEL: func.func @outline_mark_and_debug_with_operand_def
// CHECK: scf.for
// CHECK: %[[VEXP_SCOPE:.*]] = scope.scope : () -> tensor<16x16xf16> {
// CHECK:   %[[EXP:.*]] = hivm.hir.vexp
// CHECK:   annotation.mark %[[EXP]] {marked_exp} : tensor<16x16xf16>
// CHECK:   scope.return %[[EXP]] : tensor<16x16xf16>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK:   tensor.expand_shape %[[VEXP_SCOPE]]
// CHECK:   hivm.hir.copy
// CHECK:   scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[CUBE:.*]] = scope.scope : () -> tensor<1x1x16x16xf32> {
// CHECK:   %[[DOT:.*]] = hivm.hir.mmadL1
// CHECK:   hivm.hir.debug {debugtype = "print", hex = false, prefix = "dot: "} %[[DOT]] : tensor<1x1x16x16xf32>
// CHECK:   scope.return %[[DOT]] : tensor<1x1x16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scf.yield %[[CUBE]] : tensor<1x1x16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_mark_and_debug_with_operand_def(
      %x: tensor<16x16xf16>, %b: tensor<1x1x16x16xf16>, %bound: i32) -> tensor<1x1x16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<1x1x16x16xf32>
    %res = scf.for %i = %c0 to %bound step %c1 iter_args(%acc = %init) -> (tensor<1x1x16x16xf32>) : i32 {
      %exp_init = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%x : tensor<16x16xf16>) outs(%exp_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      annotation.mark %exp {marked_exp} : tensor<16x16xf16>
      %exp_4d = tensor.expand_shape %exp [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %a = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%exp_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.debug {debugtype = "print", hex = false, prefix = "dot: "} %dot : tensor<1x1x16x16xf32>
      scf.yield %dot : tensor<1x1x16x16xf32>
    }
    return %res : tensor<1x1x16x16xf32>
  }
}

// -----

// Propagate Cube outputs along user chains inside Vector work items and split
// untouched Vector operations into their own Vector work item, while keeping
// Cube work items (e.g. nd2nz + mmadL1 + fixpipe) intact.
// CHECK-LABEL: func.func @outline_split_untouched_ops_in_vector_work_item
// CHECK-SAME: (%{{.*}}: tensor<16x16xf16>, %[[Y:.*]]: tensor<16x16xf32>,
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<16x16xf32>)
// CHECK:   %[[S_COPY:.*]] = scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK:     hivm.hir.vexp
// CHECK:     hivm.hir.copy
// CHECK:     scope.return {{.*}} : tensor<1x1x16x16xf16>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S_CUBE:.*]] = scope.scope : () -> tensor<1x16x16xf32> {
// CHECK:     hivm.hir.nd2nz
// CHECK:     hivm.hir.mmadL1 ins(%[[S_COPY]], {{.*}})
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return {{.*}} : tensor<1x16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S_UNTOUCHED:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.load
// CHECK:     %[[U0:.*]] = hivm.hir.vmul ins(%[[Y]], {{.*}})
// CHECK:     %[[U1:.*]] = hivm.hir.vmul ins(%[[U0]], {{.*}})
// CHECK:     scope.return %[[U1]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S_TOUCHED:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S_CUBE]]
// CHECK:     %[[T0:.*]] = hivm.hir.vmul ins(%[[Y]], %[[COLLAPSED]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     %[[T1:.*]] = hivm.hir.vadd ins(%[[T0]], %[[S_UNTOUCHED]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     scope.return %[[T1]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   scf.yield %[[S_TOUCHED]] : tensor<16x16xf32>
// CHECK: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_split_untouched_ops_in_vector_work_item(
      %x: tensor<16x16xf16>,
      %y: tensor<16x16xf32>,
      %b_gm: memref<16x16xf16>,
      %d_gm: memref<16x16xf32>,
      %scale: f32,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %exp_init = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%x : tensor<16x16xf16>) outs(%exp_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %exp_4d = tensor.expand_shape %exp [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %a_cbuf = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%exp_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}

      %b_alloc = memref.alloc() : memref<1x1x16x16xf16>
      hivm.hir.nd2nz {dst_continuous} ins(%b_gm : memref<16x16xf16>) outs(%b_alloc : memref<1x1x16x16xf16>)
      %b_l1 = bufferization.to_tensor %b_alloc restrict writable : memref<1x1x16x16xf16>
      %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a_cbuf, %b_l1, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub_alloc = memref.alloc() : memref<1x16x16xf32, #hivm.address_space<ub>>
      %ub_cast = memref.memory_space_cast %ub_alloc : memref<1x16x16xf32, #hivm.address_space<ub>> to memref<1x16x16xf32>
      %cube_3d = bufferization.to_tensor %ub_cast restrict writable : memref<1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub_alloc : memref<1x16x16xf32, #hivm.address_space<ub>>)

      %d_alloc = memref.alloc() : memref<16x16xf32>
      hivm.hir.load ins(%d_gm : memref<16x16xf32>) outs(%d_alloc : memref<16x16xf32>) core_type = <VECTOR>
      %d = bufferization.to_tensor %d_alloc restrict writable : memref<16x16xf32>

      %shared_empty = tensor.empty() : tensor<16x16xf32>
      %u0 = hivm.hir.vmul ins(%y, %scale : tensor<16x16xf32>, f32) outs(%shared_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %u1 = hivm.hir.vmul ins(%u0, %d : tensor<16x16xf32>, tensor<16x16xf32>) outs(%shared_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cube_2d = tensor.collapse_shape %cube_3d [[0, 1], [2]] : tensor<1x16x16xf32> into tensor<16x16xf32>
      %t0 = hivm.hir.vmul ins(%y, %cube_2d : tensor<16x16xf32>, tensor<16x16xf32>) outs(%shared_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %t1 = hivm.hir.vadd ins(%t0, %u1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%shared_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %t1 : tensor<16x16xf32>
    }
    return %res : tensor<16x16xf32>
  }
}

// -----

// Define a CoreOp's result as an Output when it uses a loop iter_arg directly
// or via view-like operations (e.g. collapse_shape, extract_slice).
// CHECK-LABEL: func.func @outline_iter_arg_derived_core_op_as_output
// CHECK: scf.for {{.*}} iter_args(%[[ACC0:.*]] = {{.*}}, %[[ACC1:.*]] = {{.*}}) -> (tensor<16x16xf32>, tensor<1x32x16xf32>)
// CHECK-NEXT: %[[S_CUBE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:        hivm.hir.mmadL1
// CHECK:        hivm.hir.fixpipe
// CHECK-NEXT:   scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: %[[S_DIRECT:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK-NEXT:   %[[VSUB:.*]] = hivm.hir.vsub ins(%[[ACC0]], {{.*}})
// CHECK-NEXT:   scope.return %[[VSUB]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[S_DERIVED:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK-NEXT:   %[[VEXP:.*]] = hivm.hir.vexp ins(%[[S_DIRECT]] : tensor<16x16xf32>)
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ACC1]]
// CHECK-NEXT:   %[[SLICED:.*]] = tensor.extract_slice %[[COLLAPSED]]
// CHECK-NEXT:   %[[VMUL:.*]] = hivm.hir.vmul ins(%[[SLICED]], %[[VEXP]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK-NEXT:   scope.return %[[VMUL]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[S_YIELD0:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK-NEXT:   %[[VADD:.*]] = hivm.hir.vadd ins(%[[S_CUBE]], %[[S_DERIVED]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK-NEXT:   scope.return %[[VADD]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[S_YIELD1:.*]] = scope.scope : () -> tensor<1x32x16xf32> {
// CHECK-NEXT:   %[[EXP1:.*]] = tensor.expand_shape %[[S_YIELD0]]
// CHECK-NEXT:   %[[VADD1:.*]] = hivm.hir.vadd ins(%[[ACC1]], %[[EXP1]] : tensor<1x32x16xf32>, tensor<1x16x16xf32>)
// CHECK-NEXT:   scope.return %[[VADD1]] : tensor<1x32x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scf.yield %[[S_YIELD0]], %[[S_YIELD1]] : tensor<16x16xf32>, tensor<1x32x16xf32>
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_iter_arg_derived_core_op_as_output(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %x: tensor<16x16xf32>,
      %ub: i32) -> (tensor<16x16xf32>, tensor<1x32x16xf32>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init0 = tensor.empty() : tensor<16x16xf32>
    %init1 = tensor.empty() : tensor<1x32x16xf32>
    %res:2 = scf.for %i = %c0 to %ub step %c1 iter_args(%acc0 = %init0, %acc1 = %init1) -> (tensor<16x16xf32>, tensor<1x32x16xf32>) : i32 {
      %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub_alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %ub_cast = memref.memory_space_cast %ub_alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube_s = bufferization.to_tensor %ub_cast restrict writable : memref<16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub_alloc : memref<16x16xf32, #hivm.address_space<ub>>)

      %d_sub = hivm.hir.vsub ins(%acc0, %x : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init0 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %d_exp = hivm.hir.vexp ins(%d_sub : tensor<16x16xf32>) outs(%init0 : tensor<16x16xf32>) -> tensor<16x16xf32>

      %collapsed = tensor.collapse_shape %acc1 [[0, 1], [2]] : tensor<1x32x16xf32> into tensor<32x16xf32>
      %sliced = tensor.extract_slice %collapsed[0, 0] [16, 16] [1, 1] : tensor<32x16xf32> to tensor<16x16xf32>
      %v_mul = hivm.hir.vmul ins(%sliced, %d_exp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init0 : tensor<16x16xf32>) -> tensor<16x16xf32>

      %next0 = hivm.hir.vadd ins(%cube_s, %v_mul : tensor<16x16xf32>, tensor<16x16xf32>) outs(%init0 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %next0_3d = tensor.expand_shape %next0 [[0, 1], [2]] output_shape [1, 16, 16] : tensor<16x16xf32> into tensor<1x16x16xf32>
      %next1 = hivm.hir.vadd ins(%acc1, %next0_3d : tensor<1x32x16xf32>, tensor<1x16x16xf32>) outs(%init1 : tensor<1x32x16xf32>) broadcast = [1] -> tensor<1x32x16xf32>
      scf.yield %next0, %next1 : tensor<16x16xf32>, tensor<1x32x16xf32>
    }
    return %res#0, %res#1 : tensor<16x16xf32>, tensor<1x32x16xf32>
  }
}

// -----

// Split multiple independent untouched Vector subgraphs inside an scf.if into
// separate single-result Vector scopes while keeping Cube-touched ops together.
// CHECK-LABEL: func.func @outline_split_multiple_untouched_ops_in_scf_if
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<16x16xf32>)
// CHECK:   %[[S_CUBE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     scf.if {{.*}} {
// CHECK:       hivm.hir.mmadL1
// CHECK:       hivm.hir.fixpipe
// CHECK:     } {pipeline.if.cond = 0 : i64}
// CHECK:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S_U0:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[IF_U0:.*]] = scf.if {{.*}} -> (tensor<16x16xf32>) {
// CHECK:       %[[EXP:.*]] = hivm.hir.vexp
// CHECK:       scf.yield %[[EXP]] : tensor<16x16xf32>
// CHECK:     } else {
// CHECK:       scf.yield {{.*}} : tensor<16x16xf32>
// CHECK:     } {pipeline.if.cond = 0 : i64}
// CHECK:     scope.return %[[IF_U0]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S_U1:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[IF_U1:.*]] = scf.if {{.*}} -> (tensor<16x16xf32>) {
// CHECK:       %[[LN:.*]] = hivm.hir.vln
// CHECK:       scf.yield %[[LN]] : tensor<16x16xf32>
// CHECK:     } else {
// CHECK:       scf.yield {{.*}} : tensor<16x16xf32>
// CHECK:     } {pipeline.if.cond = 0 : i64}
// CHECK:     scope.return %[[IF_U1]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S_TOUCHED:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[IF_T:.*]] = scf.if {{.*}} -> (tensor<16x16xf32>) {
// CHECK:       %[[T0:.*]] = hivm.hir.vadd ins(%[[S_CUBE]], %[[S_U0]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:       %[[T1:.*]] = hivm.hir.vsub ins(%[[T0]], %[[S_U1]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:       scf.yield %[[T1]] : tensor<16x16xf32>
// CHECK:     } else {
// CHECK:       scf.yield %[[ACC]] : tensor<16x16xf32>
// CHECK:     } {pipeline.if.cond = 0 : i64}
// CHECK:     scope.return %[[IF_T]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   scf.yield %[[S_TOUCHED]] : tensor<16x16xf32>
// CHECK: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_split_multiple_untouched_ops_in_scf_if(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %x: tensor<16x16xf32>,
      %y: tensor<16x16xf32>,
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

        %v_empty = tensor.empty() : tensor<16x16xf32>
        %u0 = hivm.hir.vexp ins(%x : tensor<16x16xf32>) outs(%v_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
        %u1 = hivm.hir.vln ins(%y : tensor<16x16xf32>) outs(%v_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
        %t0 = hivm.hir.vadd ins(%s, %u0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
        %t1 = hivm.hir.vsub ins(%t0, %u1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %t1 : tensor<16x16xf32>
      } else {
        scf.yield %acc : tensor<16x16xf32>
      }
      scf.yield %if_res : tensor<16x16xf32>
    }
    return %res : tensor<16x16xf32>
  }
}

// -----

// Only split off untouched slices that contain at least one CoreOp; keep
// non-CoreOp tensor helpers like `tensor.insert_slice` inside the consuming
// Vector scope instead of forming a CUBE_AND_VECTOR scope.
// CHECK-LABEL: func.func @outline_do_not_split_non_core_untouched_slice
// CHECK: scf.for {{.*}} iter_args(%[[CNT:.*]] = {{.*}}, %[[ACC:.*]] = {{.*}}) -> (i64, tensor<16x16xf32>)
// CHECK-NEXT: %[[S_CNT:.*]] = scope.scope : () -> i64 {
// CHECK-NEXT:   %[[NEXT:.*]] = arith.addi %[[CNT]],
// CHECK-NEXT:   scope.return %[[NEXT]] : i64
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK-NEXT: %[[S_CUBE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:        hivm.hir.mmadL1
// CHECK:        hivm.hir.fixpipe
// CHECK-NEXT:   scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: %[[S_UNTOUCHED:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:        %[[EXP:.*]] = hivm.hir.vexp
// CHECK-NEXT:   scope.return %[[EXP]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[S_TOUCHED:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:        arith.index_cast %[[CNT]] : i64 to index
// CHECK:        tensor.extract_slice
// CHECK:        %[[INS:.*]] = tensor.insert_slice
// CHECK:        %[[T0:.*]] = hivm.hir.vadd ins(%[[S_CUBE]], %[[S_UNTOUCHED]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:        %[[T1:.*]] = hivm.hir.vadd ins(%[[T0]], %[[INS]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK-NEXT:   scope.return %[[T1]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scf.yield %[[S_CNT]], %[[S_TOUCHED]] : i64, tensor<16x16xf32>
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_do_not_split_non_core_untouched_slice(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %x: tensor<16x16xf32>,
      %pad0: tensor<16x16xf32>,
      %pad1: tensor<16x16xf32>,
      %lens: memref<?xi32>,
      %ub: i32) -> (i64, tensor<16x16xf32>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_idx = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res:2 = scf.for %i = %c0 to %ub step %c1 iter_args(%cnt = %c0_i64, %acc = %init) -> (i64, tensor<16x16xf32>) : i32 {
      %next = arith.addi %cnt, %c1_i64 : i64

      %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub_alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %ub_cast = memref.memory_space_cast %ub_alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube_s = bufferization.to_tensor %ub_cast restrict writable : memref<16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub_alloc : memref<16x16xf32, #hivm.address_space<ub>>)

      %v_empty = tensor.empty() : tensor<16x16xf32>
      %u_exp = hivm.hir.vexp ins(%x : tensor<16x16xf32>) outs(%v_empty : tensor<16x16xf32>) -> tensor<16x16xf32>

      %cnt_idx = arith.index_cast %cnt : i64 to index
      %rc = memref.reinterpret_cast %lens to offset: [%cnt_idx], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %len_i32 = memref.load %rc[%c0_idx] : memref<1xi32, strided<[1], offset: ?>>
      %len = arith.index_cast %len_i32 : i32 to index
      %ext = tensor.extract_slice %pad0[0, 0] [16, %len] [1, 1] : tensor<16x16xf32> to tensor<16x?xf32>
      %ins = tensor.insert_slice %ext into %pad1[0, 0] [16, %len] [1, 1] : tensor<16x?xf32> into tensor<16x16xf32>

      %t0 = hivm.hir.vadd ins(%cube_s, %u_exp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %t1 = hivm.hir.vadd ins(%t0, %ins : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %next, %t1 : i64, tensor<16x16xf32>
    }
    return %res#0, %res#1 : i64, tensor<16x16xf32>
  }
}

// -----

// Check raw outlining before canonicalization folds reshapes of tensor.empty.
// Only the empty is shared; the fallback expand_shape belongs inside else.
// CHECK-LABEL: func.func @outline_expand_init_then
// CHECK: } {hivm.cv_pipelined_loop}
// RESHAPE-LABEL: func.func @outline_expand_init_then
// RESHAPE: %[[SCOPE:.*]] = scope.scope : () -> tensor<64x1xf32> {
// RESHAPE-NEXT: %[[EMPTY:.*]] = tensor.empty() : tensor<64xf32>
// RESHAPE-NEXT: %[[IF:.*]] = scf.if %[[COND:.*]] -> (tensor<64x1xf32>) {
// RESHAPE-NEXT: %[[ZEROS:.*]] = hivm.hir.vbrc {{.*}} outs(%[[EMPTY]] : tensor<64xf32>)
// RESHAPE-NEXT: %[[INIT:.*]] = tensor.expand_shape %[[ZEROS]] {{.*}} : tensor<64xf32> into tensor<64x1xf32>
// RESHAPE-NEXT: %[[SUM:.*]] = hivm.hir.vreduce <sum> {{.*}} outs(%[[INIT]] : tensor<64x1xf32>)
// RESHAPE-NEXT: hivm.hir.store ins(%[[SUM]] : tensor<64x1xf32>)
// RESHAPE-NEXT: scf.yield %[[SUM]] : tensor<64x1xf32>
// RESHAPE-NEXT: } else {
// RESHAPE-NEXT: %[[INACTIVE:.*]] = tensor.expand_shape %[[EMPTY]] {{.*}} : tensor<64xf32> into tensor<64x1xf32>
// RESHAPE-NEXT: scf.yield %[[INACTIVE]] : tensor<64x1xf32>
// RESHAPE-NEXT: }
// RESHAPE-NEXT: scope.return %[[IF]] : tensor<64x1xf32>
// RESHAPE: scf.if %[[COND]] {
// RESHAPE: hivm.hir.vadd ins(%[[SCOPE]], %[[SCOPE]] : tensor<64x1xf32>, tensor<64x1xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @outline_expand_init_then(
      %x: tensor<64x64xf32>, %dst0: memref<64x1xf32>,
      %dst1: memref<64x1xf32>, %cube_dst: memref<64x64xf32>,
      %n: index, %cond: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      scf.if %cond {
        %empty = tensor.empty() : tensor<64xf32>
        %zeros = hivm.hir.vbrc ins(%zero : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
        %init = tensor.expand_shape %zeros [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %dot_init = tensor.empty() : tensor<64x64xf32>
        %dot = hivm.hir.mmadL1 ins(%x, %x, %true, %c64, %c64, %c64 : tensor<64x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%dot_init : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<64x64xf32>) outs(%cube_dst : memref<64x64xf32>)
        %sum = hivm.hir.vreduce <sum> ins(%x : tensor<64x64xf32>) outs(%init : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
        hivm.hir.store ins(%sum : tensor<64x1xf32>) outs(%dst0 : memref<64x1xf32>)
        // Fanout makes the reduction result cross scope boundaries.
        %next_init = tensor.empty() : tensor<64x1xf32>
        %next = hivm.hir.vadd ins(%sum, %sum : tensor<64x1xf32>, tensor<64x1xf32>) outs(%next_init : tensor<64x1xf32>) -> tensor<64x1xf32>
        hivm.hir.store ins(%next : tensor<64x1xf32>) outs(%dst1 : memref<64x1xf32>)
      }
    }
    return
  }
}

// -----

// Legacy loops bypass outlining, but their TCB marks must still follow the
// allocations hoisted out of the preload scopes. Preserve all attributes and
// the order of multiple marks on the same allocation, including nested scopes.
// TCB-LABEL: func.func @legacy_tcb_marks_to_allocations
// TCB: scf.for
// TCB-NEXT: %[[A:.*]] = memref.alloc()
// TCB-NEXT: annotation.mark %[[A]] {effects = ["write", "read"], first, hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
// TCB-NEXT: annotation.mark %[[A]] {effects = ["read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, second} keys = ["tag"] values = [%{{.*}} : index]
// TCB-NEXT: %[[B:.*]] = memref.alloc()
// TCB-NEXT: annotation.mark %[[B]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>}
// TCB-NEXT: scope.scope
// TCB-NOT: annotation.mark
// TCB: } {"hivm.legacy-cv-pipeline"}
module {
  func.func @legacy_tcb_marks_to_allocations(%n: index, %tag: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %a = memref.alloc() : memref<8xf32, #hivm.address_space<cbuf>>
      %b = memref.alloc() : memref<8xf32, #hivm.address_space<cbuf>>
      scope.scope : () -> () {
        annotation.mark %b {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<8xf32, #hivm.address_space<cbuf>>
        annotation.mark %a {effects = ["write", "read"], first, hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8xf32, #hivm.address_space<cbuf>>
        scope.scope : () -> () {
          annotation.mark %a {effects = ["read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, second} keys = ["tag"] values = [%tag : index] : memref<8xf32, #hivm.address_space<cbuf>>
          scope.return
        } {no_inline}
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
      scf.yield
    } {"hivm.legacy-cv-pipeline"}
    return
  }

  // Crossing an if or a loop would change execution of the mark.
  // TCB-LABEL: func.func @preserve_guarded_tcb_marks
  // TCB: %[[GUARDED:.*]] = memref.alloc()
  // TCB-NEXT: scope.scope
  // TCB-NEXT: scf.if
  // TCB-NEXT: annotation.mark %[[GUARDED]]
  // TCB: scf.for
  // TCB-NEXT: annotation.mark %[[GUARDED]]
  func.func @preserve_guarded_tcb_marks(%cond: i1, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = memref.alloc() : memref<8xf32>
    scope.scope : () -> () {
      scf.if %cond {
        annotation.mark %a {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8xf32>
      }
      scf.for %i = %c0 to %n step %c1 {
        annotation.mark %a {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8xf32>
      } {"hivm.legacy-cv-pipeline"}
      scope.return
    } {no_inline}
    return
  }

  // Do not move marks whose dynamic values would no longer dominate, marks on
  // block arguments, unrelated annotations, or marks already beside their alloc.
  // TCB-LABEL: func.func @preserve_other_marks
  // TCB: %[[LOCAL:.*]] = memref.alloc()
  // TCB-NEXT: annotation.mark %[[LOCAL]] {hivm.tightly_coupled_buffer
  // TCB-NEXT: scope.scope
  // TCB-NEXT: %[[TAG:.*]] = arith.addi
  // TCB-NEXT: annotation.mark %[[LOCAL]] {{.*}} keys = ["tag"] values = [%[[TAG]] : index]
  // TCB-NEXT: annotation.mark %{{.*}} {hivm.tightly_coupled_buffer
  // TCB-NEXT: annotation.mark %[[LOCAL]] {unrelated}
  func.func @preserve_other_marks(%arg: memref<8xf32>, %x: index, %y: index) {
    %a = memref.alloc() : memref<8xf32>
    annotation.mark %a {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8xf32>
    scope.scope : () -> () {
      %tag = arith.addi %x, %y : index
      annotation.mark %a {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} keys = ["tag"] values = [%tag : index] : memref<8xf32>
      annotation.mark %arg {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<8xf32>
      annotation.mark %a {unrelated} : memref<8xf32>
      scope.return
    } {no_inline}
    return
  }
}
