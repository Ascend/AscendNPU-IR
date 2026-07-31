// RUN: bishengir-opt %s -hivm-partition-and-bind-sub-block=enable-load-balanced=false -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hivm-partition-and-bind-sub-block=enable-load-balanced=true  -split-input-file | FileCheck %s --check-prefix=BAL

// CHECK-LABEL: func.func @shared_value_both
// CHECK-NOT:     scf.if
// CHECK:         %[[P:.*]] = hivm.hir.vmul ins(%{{[0-9a-z_]+}}, %{{[0-9a-z_]+}} : tensor<32x64xf16>, f16)
// CHECK:         %[[I0:.*]] = hivm.hir.get_sub_block_idx
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[CND0:.*]] = arith.cmpi eq, %{{.*}}, %[[C0]]
// CHECK:         scf.if %[[CND0]]
// CHECK:           hivm.hir.vmul ins(%[[P]], %{{[0-9a-z_]+}} : tensor<32x64xf16>, f16)
// CHECK:         %[[I1:.*]] = hivm.hir.get_sub_block_idx
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[CND1:.*]] = arith.cmpi eq, %{{.*}}, %[[C1]]
// CHECK:         scf.if %[[CND1]]
// CHECK:           hivm.hir.vmul ins(%[[P]], %{{[0-9a-z_]+}} : tensor<32x64xf16>, f16)
func.func @shared_value_both(%arg0: memref<?xf16>) attributes {hacc.entry} {
  %cst0 = arith.constant 2.000000e+00 : f16
  %cst1 = arith.constant 3.000000e+00 : f16
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index

  %pe = tensor.empty() : tensor<32x64xf16>
  %pf = linalg.fill ins(%cst0 : f16) outs(%pe : tensor<32x64xf16>) -> tensor<32x64xf16>
  %p = hivm.hir.vmul ins(%pf, %cst1 : tensor<32x64xf16>, f16) outs(%pe : tensor<32x64xf16>) -> tensor<32x64xf16>

  %ea0 = tensor.empty() : tensor<32x64xf16>
  %s0 = scope.scope : () -> tensor<32x64xf16> {
    %a = hivm.hir.vmul ins(%p, %cst0 : tensor<32x64xf16>, f16) outs(%ea0 : tensor<32x64xf16>) -> tensor<32x64xf16>
    scope.return %a : tensor<32x64xf16>
  } {sub_block = 0 : i64}

  %ea1 = tensor.empty() : tensor<32x64xf16>
  %s1 = scope.scope : () -> tensor<32x64xf16> {
    %b = hivm.hir.vmul ins(%p, %cst1 : tensor<32x64xf16>, f16) outs(%ea1 : tensor<32x64xf16>) -> tensor<32x64xf16>
    scope.return %b : tensor<32x64xf16>
  } {sub_block = 1 : i64}

  %da = tensor.empty() : tensor<32x64xf16>
  %eb = tensor.empty() : tensor<64x16xf16>
  %ec = tensor.empty() : tensor<32x16xf32>
  %mm = hivm.hir.mmadL1 ins(%da, %eb, %true, %c32, %c64, %c16 : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%ec : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}

// -----

// CHECK-LABEL: func.func @nested_carry
// CHECK: scf.for %{{[a-zA-Z0-9_]+}} = %{{.*}} iter_args(%[[ACC:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}})
// CHECK:   %[[COND:[a-zA-Z0-9_]+]] = arith.cmpi eq
// CHECK:   %{{.*}} = scf.if %[[COND]]
// CHECK:     hivm.hir.vmul
// CHECK:   } else {
// CHECK-NOT:   tensor.empty
// CHECK:     scf.yield %[[ACC]]
// CHECK:   }
// CHECK:   scf.yield
func.func @nested_carry(%arg0: memref<?xf16>) attributes {hacc.entry} {
  %cst = arith.constant 1.000000e+00 : f16
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %init = tensor.empty() : tensor<32x64xf16>

  %r = scf.for %i = %c0 to %c8 step %c1
      iter_args(%acc = %init) -> tensor<32x64xf16> {
    %ea = tensor.empty() : tensor<32x64xf16>
    %s = scope.scope : () -> tensor<32x64xf16> {
      %v = hivm.hir.vmul ins(%acc, %cst : tensor<32x64xf16>, f16)
                         outs(%ea : tensor<32x64xf16>) -> tensor<32x64xf16>
      scope.return %v : tensor<32x64xf16>
    } {sub_block = 0 : i64}
    scf.yield %s : tensor<32x64xf16>
  }

  %da = tensor.empty() : tensor<32x64xf16>
  %eb = tensor.empty() : tensor<64x16xf16>
  %ec = tensor.empty() : tensor<32x16xf32>
  %mm = hivm.hir.mmadL1 ins(%da, %eb, %true, %c32, %c64, %c16 : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%ec : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}

// -----

// CHECK-LABEL: func.func @load_balance
// CHECK:         hivm.hir.mmadL1
// CHECK:         arith.constant 0 : index
// CHECK:         scf.if
// CHECK:           hivm.hir.vmul ins(%arg4, %arg2 : tensor<32x64xf16>, f16) outs(%arg3
// CHECK-NOT:     arith.constant 1 : index

// BAL-LABEL: func.func @load_balance
// BAL:         hivm.hir.mmadL1
// BAL:         %[[C0:.*]] = arith.constant 0 : index
// BAL:         %[[CND0:.*]] = arith.cmpi eq, %{{[0-9a-z_]+}}, %[[C0]]
// BAL:         %[[FC0:.*]] = scf.if %[[CND0]]
// BAL:           hivm.hir.vmul ins(%arg4, %arg2 : tensor<32x64xf16>, f16) outs(%arg3
// BAL:         %[[C0B:.*]] = arith.constant 0 : index
// BAL:         %[[CND1:.*]] = arith.cmpi eq, %{{[0-9a-z_]+}}, %[[C0B]]
// BAL:         scf.if %[[CND1]]
// BAL:           hivm.hir.vmul ins(%[[FC0]], %arg2 : tensor<32x64xf16>, f16) outs(%arg3
// BAL:         %[[C1:.*]] = arith.constant 1 : index
// BAL:         %[[CND2:.*]] = arith.cmpi eq, %{{[0-9a-z_]+}}, %[[C1]]
// BAL:         scf.if %[[CND2]]
// BAL:           hivm.hir.vmul ins(%arg4, %arg2 : tensor<32x64xf16>, f16) outs(%arg3
// BAL:         %[[C1B:.*]] = arith.constant 1 : index
// BAL:         scf.if
// BAL:           hivm.hir.vmul ins(%arg4, %arg2 : tensor<32x64xf16>, f16) outs(%arg3
func.func @load_balance(%arg0: memref<?xf16>,
                        %ts: tensor<32x64xf16>, %sc: f16, %tout: tensor<32x64xf16>,
                        %tin: tensor<32x64xf16>,
                        %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                        %flag: i1, %m: index, %k: index, %n: index)
    attributes {hacc.entry} {
  %s0 = scope.scope : () -> tensor<32x64xf16> {
    %a = hivm.hir.vmul ins(%ts, %sc : tensor<32x64xf16>, f16) outs(%tout : tensor<32x64xf16>) -> tensor<32x64xf16>
    scope.return %a : tensor<32x64xf16>
  } {sub_block = 0 : i64}
  %s1 = scope.scope : () -> tensor<32x64xf16> {
    %b = hivm.hir.vmul ins(%ts, %sc : tensor<32x64xf16>, f16) outs(%tout : tensor<32x64xf16>) -> tensor<32x64xf16>
    scope.return %b : tensor<32x64xf16>
  } {sub_block = 1 : i64}

  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>

  %fc0 = hivm.hir.vmul ins(%tin, %sc : tensor<32x64xf16>, f16) outs(%tout : tensor<32x64xf16>) -> tensor<32x64xf16>
  %fc1 = hivm.hir.vmul ins(%fc0, %sc : tensor<32x64xf16>, f16) outs(%tout : tensor<32x64xf16>) -> tensor<32x64xf16>

  %fi0 = hivm.hir.vmul ins(%tin, %sc : tensor<32x64xf16>, f16) outs(%tout : tensor<32x64xf16>) -> tensor<32x64xf16>
  %fi1 = hivm.hir.vmul ins(%tin, %sc : tensor<32x64xf16>, f16) outs(%tout : tensor<32x64xf16>) -> tensor<32x64xf16>
  return
}

// -----

// CHECK-LABEL: func.func @double_write_fixpipe
// CHECK:         hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins
// CHECK:         hivm.hir.fixpipe {{.*}}sub_block_idx = #hivm.fixpipe_sub_block<sub_block_1>
// CHECK:         scf.if
// CHECK:         scf.if
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @double_write_fixpipe(%ci: tensor<16x16xf32>, %sc: f32) attributes {hacc.entry} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ci : tensor<16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  %cast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
  %ubt = bufferization.to_tensor %cast restrict writable : memref<16x16xf32>
  %e0 = tensor.empty() : tensor<16x16xf32>
  %s0 = scope.scope : () -> tensor<16x16xf32> {
    %a = hivm.hir.vmul ins(%ubt, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %a : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  %e1 = tensor.empty() : tensor<16x16xf32>
  %s1 = scope.scope : () -> tensor<16x16xf32> {
    %b = hivm.hir.vmul ins(%ubt, %sc : tensor<16x16xf32>, f32) outs(%e1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %b : tensor<16x16xf32>
  } {sub_block = 1 : i64}
  return
}
}

// -----

// CHECK-LABEL: func.func @single_dest_fixpipe_v1
// CHECK:         hivm.hir.fixpipe {{.*}}sub_block_idx = #hivm.fixpipe_sub_block<sub_block_1>
// CHECK-NOT:     hivm.hir.fixpipe
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @single_dest_fixpipe_v0(%ci: tensor<16x16xf32>, %sc: f32) attributes {hacc.entry} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ci : tensor<16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  %cast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
  %ubt = bufferization.to_tensor %cast restrict writable : memref<16x16xf32>
  %e0 = tensor.empty() : tensor<16x16xf32>
  %s0 = scope.scope : () -> tensor<16x16xf32> {
    %a = hivm.hir.vmul ins(%ubt, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %a : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  return
}
func.func @single_dest_fixpipe_v1(%ci: tensor<16x16xf32>, %sc: f32) attributes {hacc.entry} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ci : tensor<16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  %cast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
  %ubt = bufferization.to_tensor %cast restrict writable : memref<16x16xf32>
  %e1 = tensor.empty() : tensor<16x16xf32>
  %s1 = scope.scope : () -> tensor<16x16xf32> {
    %b = hivm.hir.vmul ins(%ubt, %sc : tensor<16x16xf32>, f32) outs(%e1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %b : tensor<16x16xf32>
  } {sub_block = 1 : i64}
  return
}
}

// -----

// CHECK-LABEL: func.func @fixpipe_subview_base
// CHECK:         hivm.hir.fixpipe {{.*}}sub_block_idx = #hivm.fixpipe_sub_block<sub_block_1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @fixpipe_subview_base(%ci: tensor<16x16xf32>, %sc: f32) attributes {hacc.entry} {
  %alloc = memref.alloc() : memref<2x16x16xf32, #hivm.address_space<ub>>
  %sv = memref.subview %alloc[0, 0, 0] [1, 16, 16] [1, 1, 1]
      : memref<2x16x16xf32, #hivm.address_space<ub>>
     to memref<16x16xf32, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%ci : tensor<16x16xf32>) outs(%sv : memref<16x16xf32, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>)
  %wholecast = memref.memory_space_cast %alloc : memref<2x16x16xf32, #hivm.address_space<ub>> to memref<2x16x16xf32>
  %whole = bufferization.to_tensor %wholecast restrict writable : memref<2x16x16xf32>
  %e = tensor.empty() : tensor<16x16xf32>
  %s1 = scope.scope : () -> tensor<16x16xf32> {
    %slice = tensor.extract_slice %whole[0, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<2x16x16xf32> to tensor<16x16xf32>
    %v = hivm.hir.vmul ins(%slice, %sc : tensor<16x16xf32>, f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %v : tensor<16x16xf32>
  } {sub_block = 1 : i64}
  return
}
}

// -----

// CHECK-LABEL: func.func @memref_bridge
// CHECK:         scf.if
// CHECK:           hivm.hir.copy {{.*}}outs(%{{.*}} : memref<16x16xf32>) {already_sub_block_bound}
// CHECK:         %[[T:.*]] = bufferization.to_tensor
// CHECK:         scf.if
// CHECK:           hivm.hir.vmul ins(%[[T]]
func.func @memref_bridge(%ts: tensor<16x16xf32>, %sc: f32,
                         %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                         %flag: i1, %m: index, %k: index, %n: index) attributes {hacc.entry} {
  %e = tensor.empty() : tensor<16x16xf32>
  %s0 = scope.scope : () -> tensor<16x16xf32> {
    %v = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %v : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  %cast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
  hivm.hir.copy ins(%s0 : tensor<16x16xf32>) outs(%cast : memref<16x16xf32>)
  %t = bufferization.to_tensor %cast restrict writable : memref<16x16xf32>
  %e2 = tensor.empty() : tensor<16x16xf32>
  %r = hivm.hir.vmul ins(%t, %sc : tensor<16x16xf32>, f32) outs(%e2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}

// -----

// CHECK-LABEL: func.func @legality_conflict
// CHECK-NOT:     scf.if
// CHECK:         hivm.hir.vadd ins(%{{.*}}, %{{.*}} : tensor<16x16xf32>, tensor<16x16xf32>)
func.func @legality_conflict(%ts: tensor<16x16xf32>, %sc: f32,
                             %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                             %flag: i1, %m: index, %k: index, %n: index) attributes {hacc.entry} {
  %e0 = tensor.empty() : tensor<16x16xf32>
  %s0 = scope.scope : () -> tensor<16x16xf32> {
    %a = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %a : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  %e1 = tensor.empty() : tensor<16x16xf32>
  %s1 = scope.scope : () -> tensor<16x16xf32> {
    %b = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%e1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %b : tensor<16x16xf32>
  } {sub_block = 1 : i64}
  %e2 = tensor.empty() : tensor<16x16xf32>
  %conflict = hivm.hir.vadd ins(%s0, %s1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%e2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}

// -----

// CHECK-LABEL: func.func @no_cube
// CHECK-NOT:     scf.if
// CHECK:         hivm.hir.vmul
// CHECK:         return
func.func @no_cube(%ts: tensor<16x16xf32>, %sc: f32) attributes {hacc.entry} {
  %e0 = tensor.empty() : tensor<16x16xf32>
  %s0 = scope.scope : () -> tensor<16x16xf32> {
    %a = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %a : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @if_plumb
// CHECK:         %[[G:.*]] = scf.if %{{.*}} -> (tensor<16x16xf32>) {
// CHECK:           hivm.hir.vmul
// CHECK:         %{{.*}} = scf.if %arg2 -> (tensor<16x16xf32>) {
// CHECK:           scf.yield %[[G]]
func.func @if_plumb(%ts: tensor<16x16xf32>, %sc: f32, %cond: i1,
                    %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                    %flag: i1, %m: index, %k: index, %n: index) attributes {hacc.entry} {
  %e0 = tensor.empty() : tensor<16x16xf32>
  %s0 = scope.scope : () -> tensor<16x16xf32> {
    %a = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %a : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  %e1 = tensor.empty() : tensor<16x16xf32>
  %r = scf.if %cond -> (tensor<16x16xf32>) {
    scf.yield %s0 : tensor<16x16xf32>
  } else {
    scf.yield %e1 : tensor<16x16xf32>
  }
  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}

// -----

// CHECK-LABEL: func.func @hoist_dps_init
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:         scf.if %{{.*}} -> (tensor<16x16xf32>) {
// CHECK:           hivm.hir.vmul {{.*}}outs(%[[E]]
// CHECK:         } else {
// CHECK:           scf.yield %[[E]]
func.func @hoist_dps_init(%ts: tensor<16x16xf32>, %sc: f32,
                          %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                          %flag: i1, %m: index, %k: index, %n: index) attributes {hacc.entry} {
  %s = scope.scope : () -> tensor<16x16xf32> {
    %ei = tensor.empty() : tensor<16x16xf32>
    %v = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%ei : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %v : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}

// -----

// CHECK-LABEL: func.func @annotation_mark_unguarded
// CHECK:         annotation.mark %{{.*}} {hivm.cv_pipelined_multi_buffer}
// CHECK-NEXT:    return
func.func @annotation_mark_unguarded(%ts: tensor<16x16xf32>, %sc: f32,
                                     %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                                     %flag: i1, %m: index, %k: index, %n: index) attributes {hacc.entry} {
  %e = tensor.empty() : tensor<16x16xf32>
  %s0 = scope.scope : () -> tensor<16x16xf32> {
    %v = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %v : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
  %cast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<cbuf>> to memref<16x16xf32>
  hivm.hir.copy ins(%s0 : tensor<16x16xf32>) outs(%cast : memref<16x16xf32>)
  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>
  annotation.mark %cast {hivm.cv_pipelined_multi_buffer} : memref<16x16xf32>
  return
}

// -----

// BAL-LABEL: func.func @diamond_free_nodes
// BAL:         hivm.hir.mmadL1
// BAL:         arith.constant 0 : index
// BAL:         %[[P1:.*]] = scf.if
// BAL:           hivm.hir.vmul ins(%arg2, %arg1
// BAL:         arith.constant 0 : index
// BAL:         %[[P2:.*]] = scf.if
// BAL:           hivm.hir.vmul ins(%arg2, %arg1
// BAL:         arith.constant 0 : index
// BAL:         scf.if
// BAL:           hivm.hir.vadd ins(%[[P1]], %[[P2]]
func.func @diamond_free_nodes(%ts: tensor<16x16xf32>, %sc: f32, %tin: tensor<16x16xf32>,
                        %tout: tensor<16x16xf32>,
                        %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                        %flag: i1, %m: index, %k: index, %n: index) attributes {hacc.entry} {
  %s0 = scope.scope : () -> tensor<16x16xf32> {
    %e0 = tensor.empty() : tensor<16x16xf32>
    %a = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %a : tensor<16x16xf32>
  } {sub_block = 0 : i64}
  %s1 = scope.scope : () -> tensor<16x16xf32> {
    %e1 = tensor.empty() : tensor<16x16xf32>
    %b = hivm.hir.vmul ins(%ts, %sc : tensor<16x16xf32>, f32) outs(%e1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %b : tensor<16x16xf32>
  } {sub_block = 1 : i64}
  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>

  %p1 = hivm.hir.vmul ins(%tin, %sc : tensor<16x16xf32>, f32) outs(%tout : tensor<16x16xf32>) -> tensor<16x16xf32>
  %p2 = hivm.hir.vmul ins(%tin, %sc : tensor<16x16xf32>, f32) outs(%tout : tensor<16x16xf32>) -> tensor<16x16xf32>
  %f = hivm.hir.vadd ins(%p1, %p2 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%tout : tensor<16x16xf32>) -> tensor<16x16xf32>
  return
}

// -----

// CHECK-LABEL: func.func @free_simt_scope_pinned
// CHECK:         scf.if
// CHECK-NEXT:      scope.scope
// CHECK:             hivm.hir.vmul
// CHECK:           vector_type = "simt"
// CHECK:         scf.if
// CHECK-NEXT:      scope.scope
// CHECK:             hivm.hir.vmul
// CHECK:           vector_type = "simt"
// CHECK:         scf.if
// CHECK-NEXT:      scope.scope
// CHECK:             hivm.hir.vmul
// CHECK:           vector_type = "simt"
// CHECK:         hivm.hir.mmadL1
func.func @free_simt_scope_pinned(%vin: tensor<16x16xf32>, %sc: f32,
                                  %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                                  %flag: i1, %m: index, %k: index, %n: index) attributes {hacc.entry} {
  %e0 = tensor.empty() : tensor<16x16xf32>
  %a0 = scope.scope : () -> tensor<16x16xf32> {
    %s = scope.scope : () -> tensor<16x16xf32> {
      %v = hivm.hir.vmul ins(%vin, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return %v : tensor<16x16xf32>
    } {noinline, vector_type = "simt"}
    scope.return %s : tensor<16x16xf32>
  } {noinline, sub_block = 0 : i32}
  %e1 = tensor.empty() : tensor<16x16xf32>
  %a1 = scope.scope : () -> tensor<16x16xf32> {
    %s = scope.scope : () -> tensor<16x16xf32> {
      %v = hivm.hir.vmul ins(%vin, %sc : tensor<16x16xf32>, f32) outs(%e1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return %v : tensor<16x16xf32>
    } {noinline, vector_type = "simt"}
    scope.return %s : tensor<16x16xf32>
  } {noinline, sub_block = 1 : i32}
  %e2 = tensor.empty() : tensor<16x16xf32>
  %free = scope.scope : () -> tensor<16x16xf32> {
    %v = hivm.hir.vmul ins(%vin, %sc : tensor<16x16xf32>, f32) outs(%e2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %v : tensor<16x16xf32>
  } {noinline, vector_type = "simt"}
  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}

// -----

// CHECK-LABEL: func.func @free_simt_scope_both
// CHECK-NOT:     scf.if
// CHECK:         scope.scope
// CHECK-NEXT:      hivm.hir.vmul
// CHECK-NEXT:      scope.return
// CHECK-NEXT:    } {noinline, vector_type = "simt"}
// CHECK:         scf.if
func.func @free_simt_scope_both(%vin: tensor<16x16xf32>, %sc: f32,
                                %ca: tensor<32x64xf16>, %cb: tensor<64x16xf16>, %cc: tensor<32x16xf32>,
                                %flag: i1, %m: index, %k: index, %n: index) attributes {hacc.entry} {
  %e2 = tensor.empty() : tensor<16x16xf32>
  %shared = scope.scope : () -> tensor<16x16xf32> {
    %v = hivm.hir.vmul ins(%vin, %sc : tensor<16x16xf32>, f32) outs(%e2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scope.return %v : tensor<16x16xf32>
  } {noinline, vector_type = "simt"}
  %e0 = tensor.empty() : tensor<16x16xf32>
  %a0 = scope.scope : () -> tensor<16x16xf32> {
    %s = scope.scope : () -> tensor<16x16xf32> {
      %v = hivm.hir.vmul ins(%shared, %sc : tensor<16x16xf32>, f32) outs(%e0 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return %v : tensor<16x16xf32>
    } {noinline, vector_type = "simt"}
    scope.return %s : tensor<16x16xf32>
  } {noinline, sub_block = 0 : i32}
  %e1 = tensor.empty() : tensor<16x16xf32>
  %a1 = scope.scope : () -> tensor<16x16xf32> {
    %s = scope.scope : () -> tensor<16x16xf32> {
      %v = hivm.hir.vmul ins(%shared, %sc : tensor<16x16xf32>, f32) outs(%e1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return %v : tensor<16x16xf32>
    } {noinline, vector_type = "simt"}
    scope.return %s : tensor<16x16xf32>
  } {noinline, sub_block = 1 : i32}
  %mm = hivm.hir.mmadL1 ins(%ca, %cb, %flag, %m, %k, %n : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%cc : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}
