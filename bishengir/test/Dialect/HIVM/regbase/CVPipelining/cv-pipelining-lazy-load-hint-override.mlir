// REQUIRES: regbase
// RUN: bishengir-opt -cv-pipelining="pipeline-depth=2 enable-lazy-loading=true" -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// Test: with kernel-level enable-lazy-loading=true, an explicit per-tensor
// `cv_pipeline_lazy_load = false` hint OVERRIDES the kernel switch -- the
// load-backed to_tensor is NOT cloned across stages and its backing alloc
// is expanded into a multi-buffered memref<2x...>.  A warning must be
// emitted on the offending annotation.mark to flag the override.  The
// load is non-cross-core (mmadL1 is the only consumer of %tensor2; vexp
// reads %wst instead) so the cross-core legality path does not kick in.

// CHECK-LABEL: func.func @test_lazy_loading_hint_false_override
// CHECK: memref<2x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_loading_hint_false_override(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
      // expected-warning@+1 {{cv_pipeline_lazy_load=false overrides kernel-level enable-lazy-loading=true}}
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      // expected-note@+1 {{see `cv_pipeline_lazy_load = false` hint here}}
      annotation.mark %tensor2 {cv_pipeline_lazy_load = false} : tensor<16x16xf16>

      // CUBE-only consumer of the load result.
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // VECTOR op consumes the fixpipe output, NOT the load result.
      %vdest = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%exp : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)

      scf.yield %next, %newinc : memref<16x16xf16>, index
    }
    return
  }
}

// -----

// Test (regression guard): with kernel-level enable-lazy-loading=true and a
// per-tensor `cv_pipeline_lazy_load = true` hint (concurring with the kernel
// switch), the load-backed to_tensor still goes through the lazy-load path
// (cloned into each consuming work item) -- no `memref<2x...>` multi-buffer
// expansion -- and no warning is emitted.

// CHECK-LABEL: func.func @test_lazy_loading_hint_true_concur
// CHECK-NOT: memref<2x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_loading_hint_true_concur(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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

// Test (legality): with kernel-level enable-lazy-loading=true, an explicit
// `cv_pipeline_lazy_load = false` hint on a cross-core load (consumed by
// both CUBE and VECTOR) is IGNORED -- lazy loading is required for
// correctness.  A "hint is ignored" warning is emitted to flag this to
// the user; the produced IR has no `memref<2x...>` multi-buffer expansion
// for the load result.

// CHECK-LABEL: func.func @test_lazy_loading_hint_false_cross_core_ignored
// CHECK-NOT: memref<2x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_loading_hint_false_cross_core_ignored(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
      // expected-warning@+1 {{cv_pipeline_lazy_load=false is ignored: load result is consumed by both CUBE and VECTOR cores}}
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      // expected-note@+1 {{see `cv_pipeline_lazy_load = false` hint here}}
      annotation.mark %tensor2 {cv_pipeline_lazy_load = false} : tensor<16x16xf16>

      // CUBE consumer
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // VECTOR consumer of the SAME load result -- this is what makes it
      // cross-core and therefore not opt-out-able via the hint.
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
