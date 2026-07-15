// RUN: bishengir-opt -cv-pipelining="pipeline-depth=2" -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s --implicit-check-not="DuplicateTensorExtractForCube::replacementLabel"

// Test: a single-element scalar produced on the VECTOR side is consumed by both
// a CUBE stage and a later VECTOR stage -- the V[producer], C[user], V[user]
// shape.  The scalar is materialized through the `InsertLoadStoreForScalar`
// pattern: it is stored to a one-element `memref_ext.alloc_workspace` and
// re-extracted, with the cube consumer paired to the workspace re-extract by a
// `DuplicateTensorExtractForCube::replacementLabel` `annotation.mark`.
//
// cv-pipelining must:
//   1. run `duplicateExtractScalarForCube` first and clone *only* the CUBE
//      consumer's scalar chain (`fptosi` + `index_cast` feeding the mmadL1 K
//      dimension) onto the workspace re-extract (`newExtractLabel`); the VECTOR
//      consumer keeps reading the original scalar.  The replacement-label mark
//      is then erased (asserted by --implicit-check-not).
//   2. expand the one-element workspace with a leading multibuffer dim and
//      rewire the inserted store into a per-slot subview so the CUBE stage can
//      read its slot back via `extract_slice` + `tensor.extract`.

// CHECK-LABEL: func.func @test_scalar_vcv
// Expanded one-element workspace carries the multibuffer dim + multi-buffer mark.
// CHECK: %[[WS:.*]] = memref_ext.alloc_workspace() : memref<2x1xf32>
// CHECK: annotation.mark %[[WS]] {hivm.cv_pipelined_multi_buffer} : memref<2x1xf32>
// V1 (VECTOR scalar producer): reduce, then store into a per-slot subview of %[[WS]].
// CHECK: scf.for
// CHECK: %[[SV:.*]] = memref.subview %[[WS]][%{{.*}}, 0] [1, 1] [1, 1]
// CHECK: hivm.hir.vreduce
// CHECK: hivm.hir.store ins(%{{.*}} : tensor<1xf32>) outs(%[[SV]]
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// C (CUBE scalar user): the cube reads the scalar back through the cloned
// (newExtractLabel) chain rooted at the workspace re-extract, feeding mmadL1 K.
// CHECK: %[[WST:.*]] = bufferization.to_tensor %[[WS]] restrict writable : memref<2x1xf32>
// CHECK: scf.for
// CHECK: %[[CESL:.*]] = tensor.extract_slice %[[WST]][%{{.*}}, 0] [1, 1] [1, 1] : tensor<2x1xf32> to tensor<1xf32>
// CHECK: %[[CEXT:.*]] = tensor.extract %[[CESL]][%{{.*}}] {{.*}}DuplicateTensorExtractForCube::newExtractLabel{{.*}} : tensor<1xf32>
// CHECK: %[[KF:.*]] = arith.fptosi %[[CEXT]] : f32 to i32
// CHECK: %[[K:.*]] = arith.index_cast %[[KF]] : i32 to index
// CHECK: hivm.hir.mmadL1 ins(%{{.*}}, %{{.*}}, %true, %c16, %c16, %[[K]]
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// V2 (VECTOR scalar user): the vector keeps the *original* scalar -- a plain
// visitedLabel extract (not rewired to the workspace) -> truncf -> vadd.
// CHECK: scf.for
// CHECK: %[[VEXT:.*]] = tensor.extract %{{.*}}[%{{.*}}] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
// CHECK: %[[VF:.*]] = arith.truncf %[[VEXT]] : f32 to f16
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.vadd ins(%{{.*}}, %[[VF]] : tensor<16x16xf16>, f16)
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_scalar_vcv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %scalarin: memref<8xf32>) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %input1 = "some_op"() : () -> memref<16x16xf16>
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    %input2 = "some_op"() : () -> memref<?xf16>
    %initin = memref.reinterpret_cast %input2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>
    %offset = "some_op"() : () -> index
    %c0 = arith.constant 0 : i32
    %c0idx = arith.constant 0 : index
    %true = arith.constant true
    %c0i = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    %vinit = "some_op"() : () -> tensor<16x16xf16>
    scf.for %i = %c0 to %bound step %step iter_args(%sliding_input = %initin, %inc = %c0i, %vacc = %vinit) -> (memref<16x16xf16>, index, tensor<16x16xf16>) : i32 {
      // ===== V1: scalar producer (vector) =====
      // Single-element tensor stored to a one-element workspace and re-extracted.
      %svin = bufferization.to_tensor %scalarin restrict writable : memref<8xf32>
      %salloc = bufferization.alloc_tensor() : tensor<f32>
      %sexp = tensor.expand_shape %salloc [] output_shape [1] : tensor<f32> into tensor<1xf32>
      %red = hivm.hir.vreduce <sum> ins(%svin : tensor<8xf32>) outs(%sexp : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
      %ext = tensor.extract %red[%c0idx] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
      %ws = memref_ext.alloc_workspace() : memref<1xf32>
      %wst3 = bufferization.to_tensor %ws restrict writable : memref<1xf32>
      %stored = hivm.hir.store ins(%red : tensor<1xf32>) outs(%wst3 : tensor<1xf32>) {"inserted-store"} -> tensor<1xf32>
      annotation.mark %stored {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<1xf32>
      %newext = tensor.extract %stored[%c0idx] {"DuplicateTensorExtractForCube::newExtractLabel" = 1 : i32, "DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
      annotation.mark %ext {"DuplicateTensorExtractForCube::replacementLabel" = 1 : i32} keys = [] values = [%newext : f32] : f32

      // ===== C: cube user of the scalar (K dimension via fptosi + index_cast) =====
      %kf = arith.fptosi %ext : f32 to i32
      %kidx = arith.index_cast %kf : i32 to index
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      %cdest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %kidx : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>
      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // ===== V2: vector user of the scalar (truncf) and the cube result =====
      %extf16 = arith.truncf %ext : f32 to f16
      %vdest1 = tensor.empty() : tensor<16x16xf16>
      %vexp = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest1 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %vdest2 = tensor.empty() : tensor<16x16xf16>
      %vadd = hivm.hir.vadd ins(%vexp, %extf16 : tensor<16x16xf16>, f16) outs(%vdest2 : tensor<16x16xf16>) -> tensor<16x16xf16>
      scf.yield %next, %newinc, %vadd : memref<16x16xf16>, index, tensor<16x16xf16>
    }
    return
  }
}
