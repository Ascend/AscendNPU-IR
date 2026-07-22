// REQUIRES: regbase
// RUN: bishengir-opt -cv-pipelining="pipeline-depth=2" -allow-unregistered-dialect -split-input-file -verify-diagnostics %s

// Test: a single tensor carries two `annotation.mark` ops both tagged with
// `cv_pipeline_lazy_load` -- the pass keeps the first-wins semantics but
// emits a warning on the first mark plus a note on each duplicate so the
// user can clean up redundant hints.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_load_duplicate_marks(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %input1 = "some_op"() : () -> memref<16x16xf16>
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    %input2 = "some_op"() : () -> memref<?xf16>
    %initin = memref.reinterpret_cast %input2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>
    %offset = "some_op"() : () -> index
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c0i = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    scf.for %i = %c0 to %bound step %step iter_args(%sliding_input = %initin, %inc = %c0i) -> (memref<16x16xf16>, index) : i32 {
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      // Probe is the first mark in `v.getUsers()` order, which is LIFO of
      // creation order -- i.e. the source-order LAST mark.  The warning lands
      // there; the note points back to the source-order earlier mark.
      // expected-note@+1 {{duplicate `cv_pipeline_lazy_load` mark here}}
      annotation.mark %tensor2 {cv_pipeline_lazy_load = true} : tensor<16x16xf16>
      // expected-warning@+1 {{tensor carries 2 `cv_pipeline_lazy_load` annotation.mark ops}}
      annotation.mark %tensor2 {cv_pipeline_lazy_load = true} : tensor<16x16xf16>

      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      %vdest = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%tensor2 : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%exp : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)

      scf.yield %next, %newinc : memref<16x16xf16>, index
    }
    return
  }
}

// -----

// Test: a `cv_pipeline_lazy_load` hint is placed on a value that is not
// produced by `bufferization.to_tensor` (here, a `tensor.empty`).  The hint
// cannot be honored -- the pass emits a warning explaining that the hint is
// ignored, and continues normally.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_load_mark_on_non_to_tensor(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %input1 = "some_op"() : () -> memref<16x16xf16>
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    %input2 = "some_op"() : () -> memref<?xf16>
    %initin = memref.reinterpret_cast %input2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>
    %offset = "some_op"() : () -> index
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c0i = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    scf.for %i = %c0 to %bound step %step iter_args(%sliding_input = %initin, %inc = %c0i) -> (memref<16x16xf16>, index) : i32 {
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>

      %dest = tensor.empty() : tensor<16x16xf16>
      // expected-warning@+1 {{hint is ignored: marked value is not produced by `bufferization.to_tensor`}}
      annotation.mark %dest {cv_pipeline_lazy_load = true} : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      %vdest = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%tensor2 : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%exp : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)

      scf.yield %next, %newinc : memref<16x16xf16>, index
    }
    return
  }
}
