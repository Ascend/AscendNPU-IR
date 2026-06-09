// RUN: bishengir-opt -cv-pipelining="pipeline-depth=2" -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,CHECK-HINT,CHECK-NEG-HINT
// RUN: bishengir-opt -cv-pipelining="pipeline-depth=2 enable-lazy-loading=true" -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,CHECK-LAZY,CHECK-HINT

// CHECK-LABEL: func.func @test_pipeline
// CHECK: scf.for
// CHECK: memref.alloc() : memref<2x16x16xf16, #hivm.address_space<ub>>
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK-DAG: memref.memory_space_cast
// CHECK-DAG: bufferization.to_tensor
// CHECK-DAG: memref.alloc() : memref<2x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// CHECK: memref.memory_space_cast
// CHECK: bufferization.to_tensor
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipeline(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
    %cinit = "some_op"() : () -> tensor<16x16xf16>
    %cond = "some_op"() : () -> i1
    %gm = "some_op"() : () -> tensor<16x16xf32>
    %gm2 = "some_op"() : () -> tensor<16x16xf16>
    %vdest = tensor.empty() : tensor<16x16xf16>
    scf.for %i = %c0 to %bound step %step iter_args(%sliding_input = %initin, %inc = %c0i, %itercube = %cinit) -> (memref<16x16xf16>, index, tensor<16x16xf16>) : i32 {
      // Cube ops
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>
      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // Vector ops
      %vdest1 = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest1 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %if = scf.if %cond -> tensor<16x16xf16> {
        %newexp = hivm.hir.vexp ins(%exp : tensor<16x16xf16>) outs(%vdest1:tensor<16x16xf16>) -> tensor<16x16xf16>
        scf.yield %newexp : tensor<16x16xf16>
      } else {
        scf.yield %exp : tensor<16x16xf16>
      }
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%if:tensor<16x16xf16>) outs(%ws1_cast:memref<16x16xf16>)
      %wso = bufferization.to_tensor %ws1_cast : memref<16x16xf16>

      // Another cube with iter arg/yield
      %t1 = tensor.empty() : tensor<16x16xf16>
      %dot1 = hivm.hir.mmadL1 ins(%itercube, %wso, %true, %c16, %c16, %c16: tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%t1:tensor<16x16xf16>) -> tensor<16x16xf16>
      scf.yield %next, %newinc, %dot1 : memref<16x16xf16>, index, tensor<16x16xf16>
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_pipeline_atomic
// Outer unrolled loop
// CHECK: scf.for
// Cube loop: set_atomic brackets around fixpipe
// CHECK:   scf.for
// CHECK:     hivm.hir.set_atomic kind = <add>[type = f32]
// CHECK:     hivm.hir.fixpipe
// CHECK:     hivm.hir.set_atomic kind = <none>[type = f32]
// CHECK:     hivm.loop_core_type = #hivm.tcore_type<CUBE>
// Vector loop
// CHECK:   scf.for
// CHECK:     hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipeline_atomic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
    %cond = "some_op"() : () -> i1
    scf.for %i = %c0 to %bound step %step iter_args(%sliding_input = %initin, %inc = %c0i) -> (memref<16x16xf16>, index) : i32 {
      // Cube ops with atomic fixpipe
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.set_atomic kind = <add>[type = f32]
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.set_atomic kind = <none>[type = f32]
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>
      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // Vector ops
      %vdest1 = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest1 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%exp:tensor<16x16xf16>) outs(%ws1_cast:memref<16x16xf16>)

      scf.yield %next, %newinc : memref<16x16xf16>, index
    }
    return
  }
}

// -----

// Test: lazy loading -- a single load whose result feeds both CUBE (mmadL1)
// and VECTOR (vexp) work items.
//
// Without lazy loading (default): the load's to_tensor result becomes a
// localOutput and is expanded into a multi-buffered alloc (memref<2x...>).
//
// With lazy loading (enable-lazy-loading=true): the load is cloned into each
// consuming work item independently; no expanded alloc is created for it.

// CHECK-LAZY-LABEL: func.func @test_lazy_loading
// Outer unrolled loop
// CHECK-LAZY: scf.for
// CUBE stage loop: contains its own hivm.hir.load AND hivm.hir.mmadL1
// CHECK-LAZY: scf.for
// CHECK-LAZY: hivm.hir.load
// CHECK-LAZY: hivm.hir.mmadL1
// CHECK-LAZY: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// VECTOR stage loop: has its own independent hivm.hir.load (cloned) AND hivm.hir.vexp
// CHECK-LAZY: scf.for
// CHECK-LAZY: hivm.hir.load
// CHECK-LAZY: hivm.hir.vexp
// CHECK-LAZY: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// No expanded 2x buffer for the load result:
// CHECK-LAZY-NOT: memref<2x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_loading(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
      // Load from GM -- result (%tensor2) is consumed by both CUBE and VECTOR.
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>

      // CUBE op consuming the load result
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // VECTOR op also consuming the same load result (%tensor2)
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

// Test: per-tensor compile hint -- annotation.mark on the load-backed
// to_tensor result with cv_pipeline_lazy_load = true should opt the tensor
// into the lazy-load path even when the kernel-level enable-lazy-loading
// switch is OFF.  This function therefore must produce lazy IR under both
// RUN lines (default-off and explicit-on), so we assert via CHECK-HINT
// (active in both runs).

// CHECK-HINT-LABEL: func.func @test_lazy_loading_via_hint
// Outer unrolled loop
// CHECK-HINT: scf.for
// CUBE stage loop: independent hivm.hir.load + hivm.hir.mmadL1
// CHECK-HINT: scf.for
// CHECK-HINT: hivm.hir.load
// CHECK-HINT: hivm.hir.mmadL1
// CHECK-HINT: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// VECTOR stage loop: own cloned hivm.hir.load + hivm.hir.vexp
// CHECK-HINT: scf.for
// CHECK-HINT: hivm.hir.load
// CHECK-HINT: hivm.hir.vexp
// CHECK-HINT: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// No expanded 2x buffer for the load result:
// CHECK-HINT-NOT: memref<2x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_loading_via_hint(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
      // Load whose result is consumed by both CUBE and VECTOR; the per-tensor
      // hint requests lazy-load on this specific tensor.
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

// Nested separator: a hivm.hir.fixpipe sitting inside an scf.if under the
// pipelineLoop. createWorkItems should lift the scf.if to the separators
// list via getContainedParent so partitioning still produces a CUBE and
// VECTOR work item.

// CHECK-LABEL: func.func @test_pipeline_nested_fixpipe
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipeline_nested_fixpipe(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
    %cond = "some_op"() : () -> i1
    scf.for %i = %c0 to %bound step %step iter_args(%sliding_input = %initin, %inc = %c0i) -> (memref<16x16xf16>, index) : i32 {
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      // Fixpipe nested inside scf.if — the scf.if must be lifted to a
      // separator so the workitem boundary sits across it.
      scf.if %cond {
        hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      }
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>
      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      %vdest1 = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest1 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%exp:tensor<16x16xf16>) outs(%ws1_cast:memref<16x16xf16>)

      scf.yield %next, %newinc : memref<16x16xf16>, index
    }
    return
  }
}

// -----

// Test: the lazy-load hint must only apply to LoadOp-backed to_tensors.
// Marking a fixpipe-backed to_tensor with cv_pipeline_lazy_load = true is
// silently ignored; the fixpipe output still flows through the default
// multi-buffered cross-stage path (memref<2x...>).  This is asserted under
// the kernel-switch-OFF run via CHECK-NEG-HINT.

// CHECK-HINT-LABEL: func.func @test_lazy_loading_hint_on_fixpipe
// CHECK-NEG-HINT: memref<2x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_loading_hint_on_fixpipe(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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

      // CUBE consumer
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      // Fixpipe-backed to_tensor; hint here must be ignored.
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>
      // expected-warning@+1 {{hint is ignored: tensor is not backed by `hivm.hir.load`}}
      annotation.mark %wst {cv_pipeline_lazy_load = true} : tensor<16x16xf16>

      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // VECTOR consumer of the fixpipe-backed tensor
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

// GM-alias rejection: a hivm.hir.fixpipe (cube-only) writes directly to a
// function argument; a hivm.hir.load in the vector work item reads the same
// function argument. markOutputs must reject this cross-workitem pattern.

// CHECK-LABEL: func.func @test_gm_alias_rejected
// CHECK-NOT: hivm.loop_core_type
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_gm_alias_rejected(%gmArg: memref<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %input1 = "some_op"() : () -> memref<16x16xf16>
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    %input2 = "some_op"() : () -> memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    scf.for %i = %c0 to %bound step %step : i32 {
      // CUBE work item: mmad then fixpipe writes directly to the GM func arg.
      %allocC = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%input2 : memref<16x16xf16>) outs(%allocC : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %allocC : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%gmArg : memref<16x16xf16>)

      // VECTOR work item: loads from gmArg — the same function argument the
      // cube work item just wrote to. Pipelining would reorder the fixpipe
      // past the load.
      %allocV = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %allocV_cast = memref.memory_space_cast %allocV : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      // expected-warning@+1 {{using GM as intermediate buffer is unsupported}}
      hivm.hir.load ins(%gmArg : memref<16x16xf16>) outs(%allocV_cast : memref<16x16xf16>)
      %tv = bufferization.to_tensor %allocV_cast : memref<16x16xf16>
      %vdest = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%tv : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%exp : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)
    }
    return
  }
}

// -----

// Same pattern as above, but the hivm.hir.fixpipe is nested inside an scf.if.
// The nested-region walk in markOutputs must still catch it.

// CHECK-LABEL: func.func @test_gm_alias_nested_rejected
// CHECK-NOT: hivm.loop_core_type
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_gm_alias_nested_rejected(%gmArg: memref<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %input1 = "some_op"() : () -> memref<16x16xf16>
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    %input2 = "some_op"() : () -> memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    %cond = "some_op"() : () -> i1
    scf.for %i = %c0 to %bound step %step : i32 {
      %allocC = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%input2 : memref<16x16xf16>) outs(%allocC : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %allocC : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      // Fixpipe (cube-only) writing to the GM func arg is nested inside
      // scf.if — the nested-region walk in markOutputs must still catch it.
      scf.if %cond {
        hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%gmArg : memref<16x16xf16>)
      }

      %allocV = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %allocV_cast = memref.memory_space_cast %allocV : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      // expected-warning@+1 {{using GM as intermediate buffer is unsupported}}
      hivm.hir.load ins(%gmArg : memref<16x16xf16>) outs(%allocV_cast : memref<16x16xf16>)
      %tv = bufferization.to_tensor %allocV_cast : memref<16x16xf16>
      %vdest = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%tv : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%exp : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)
    }
    return
  }
}

// -----

// Two mmadL1 ops chained via accumulator (%dot1 = mmadL1 outs(%dot0)) where
// %dot0's init is neither tensor.empty nor a to_tensor. Without the chain
// coalescing in extractAvailableOps, mmad0 would land in an earlier CUBE
// WorkItem and %dot0 would become a cross-WorkItem localOutput that
// expandOutputInits cannot expand (`expected to_tensor for non-tensor-empty
// output`). The fix defers mmad0 so both mmads share one CUBE WorkItem.
// CHECK-LABEL: func.func @test_pipeline_mmad_acc_chain
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipeline_mmad_acc_chain(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %inA = "some_op"() : () -> memref<16x16xf16>
    %A = bufferization.to_tensor %inA : memref<16x16xf16>
    %inB1 = "some_op"() : () -> memref<16x16xf16>
    %B1 = bufferization.to_tensor %inB1 : memref<16x16xf16>
    %unrelated_in = "some_op"() : () -> tensor<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    scf.for %i = %c0 to %bound step %step : i32 {
      // Accumulator init from an opaque op so expandOutputInits cannot
      // expand it via the tensor.empty / to_tensor branches.
      %init_complex = "some_op"() : () -> tensor<16x16xf16>
      %dot0 = hivm.hir.mmadL1 ins(%A, %B1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%init_complex : tensor<16x16xf16>) -> tensor<16x16xf16>

      // Unrelated separator chain producing a value mmad1 reads via %B2.
      // This forces mmad1 to be blocked behind the cross-core copy and
      // would otherwise drop into a separate CUBE WorkItem from mmad0.
      %ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%unrelated_in : tensor<16x16xf16>) outs(%ub : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub_cast = memref.memory_space_cast %ub : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %ub_t = bufferization.to_tensor %ub_cast : memref<16x16xf16>
      %vd = tensor.empty() : tensor<16x16xf16>
      %vexp = hivm.hir.vexp ins(%ub_t : tensor<16x16xf16>) outs(%vd : tensor<16x16xf16>) -> tensor<16x16xf16>
      %cb = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %cb_cast = memref.memory_space_cast %cb : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%vexp : tensor<16x16xf16>) outs(%cb_cast : memref<16x16xf16>)
      %B2 = bufferization.to_tensor %cb_cast : memref<16x16xf16>

      // Chained mmad: outs(%dot0) and ins downstream of the cross-core copy.
      %dot1 = hivm.hir.mmadL1 ins(%A, %B2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot0 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %out_buf = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot1 : tensor<16x16xf16>) outs(%out_buf : memref<16x16xf16, #hivm.address_space<ub>>)
    }
    return
  }
}

// -----

// Regression for two cv-pipelining bugs surfaced by the HSTU FP8 attention
// kernel under --enable-layout-optimization=true:
//
//   1. `hivm.hir.nd2nz` (the fused GM->L1 load with NZ layout conversion that
//      replaces a plain LoadOp under layout optimization) was not treated as
//      a load-like op. When its `to_tensor` result was consumed in a different
//      cv-pipelining workitem, the writer was missing from `outputMemrefMap`
//      and migrateOps asserted in IRMapping::lookup. Fix routes nd2nz through
//      `isLoadLikeOp` (pulled into consumer workitems) and
//      `isMemrefSubnetWriter` (registered in outputMemrefMap).
//
//   2. The cross-core `hivm.hir.copy`'s DPS init is a `to_tensor` of an L1
//      alloc — a TensorType statically, but backed by a memref. migrateOps
//      dispatched on the init's static type, took the tensor path, and crashed
//      in createExtractSlice on a memref iter_arg. Fix dispatches on
//      `expanded`'s actual type (the buffer expandOutputInits built).
//
// The test mirrors the failing pattern: two mmadL1 ops sandwiching vector
// ops and a cross-core copy, with the second mmad's B operand loaded via
// `hivm.hir.nd2nz` so the load is consumed in a downstream workitem.

// CHECK-LABEL: func.func @test_pipeline_nd2nz_cross_workitem
// Outer unrolled loop, then per-stage scf.fors for cube/vector/cube.
// CHECK:   scf.for
// First stage: CUBE — load + mmad + fixpipe to UB.
// CHECK:     scf.for
// CHECK:       hivm.hir.mmadL1
// CHECK:     {{.*}}hivm.loop_core_type = #hivm.tcore_type<CUBE>
// Second stage: VECTOR — vexp + cross-core copy.
// CHECK:     scf.for
// CHECK:       hivm.hir.copy
// CHECK:     {{.*}}hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// Third stage: CUBE — the nd2nz writer must land here (with its consumer
// mmadL1), not in the first cube stage.
// CHECK:     scf.for
// CHECK:       hivm.hir.nd2nz
// CHECK:       hivm.hir.mmadL1
// CHECK:     {{.*}}hivm.loop_core_type = #hivm.tcore_type<CUBE>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipeline_nd2nz_cross_workitem(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %gm_dst: memref<16x16xf16>) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true, true]> : vector<2xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %A = "some_op"() : () -> tensor<16x16xf16>
    %v_src = "some_op"() : () -> memref<16x16xf16>
    %k_src = "some_op"() : () -> memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    scf.for %i = %c0 to %bound step %step : i32 {
      // First mmad's B comes from a plain load.
      %allocK = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%k_src : memref<16x16xf16>) outs(%allocK : memref<16x16xf16>)
      %tensorK = bufferization.to_tensor %allocK : memref<16x16xf16>
      %dest1 = tensor.empty() : tensor<16x16xf16>
      %dot1 = hivm.hir.mmadL1 ins(%A, %tensorK, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest1 : tensor<16x16xf16>) -> tensor<16x16xf16>

      // Cube -> UB fixpipe (separator #1).
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot1 : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      // Vector op.
      %vdest = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>

      // Cross-core copy with TENSOR-typed DPS init (backed by to_tensor of
      // a cbuf alloc). Bug #2: migrateOps dispatched on the init's static
      // type and went down the wrong branch.
      %ws_alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws_cast = memref.memory_space_cast %ws_alloc : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      %ws_t = bufferization.to_tensor %ws_cast : memref<16x16xf16>
      %copy_out = hivm.hir.copy ins(%exp : tensor<16x16xf16>) outs(%ws_t : tensor<16x16xf16>) -> tensor<16x16xf16>

      // Second mmad's B operand comes from a hivm.hir.nd2nz — a load-like
      // op not previously recognized by cv-pipelining. Bug #1: nd2nz was
      // forced into workitem #0 alongside the first mmad's loads, then its
      // to_tensor became a cross-workitem localOutput with no writer
      // registered in outputMemrefMap.
      %allocV = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%v_src : memref<16x16xf16>) outs(%allocV : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      %allocV_cast = memref.memory_space_cast %allocV : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
      %tensorV = bufferization.to_tensor %allocV_cast : memref<1x1x16x16xf16>

      // Second mmad consumes the cross-core copy result and the nd2nz V tile.
      %dest2 = tensor.empty() : tensor<16x16xf16>
      %dot2 = hivm.hir.mmadL1 ins(%copy_out, %tensorV, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dest2 : tensor<16x16xf16>) -> tensor<16x16xf16>

      // Final fixpipe to GM (separator #3) — write to a func arg so
      // populateDependencies accepts the destination.
      hivm.hir.fixpipe ins(%dot2 : tensor<16x16xf16>) outs(%gm_dst : memref<16x16xf16>)
    }
    return
  }
}

 
// -----

// Test: auto cross-core detection -- a load whose tensor result is consumed
// by ops on both the CUBE (mmadL1) and the VECTOR (vexp) cores is treated
// as lazy-loaded automatically, without any per-tensor
// `cv_pipeline_lazy_load` annotation and even when the kernel-level
// enable-lazy-loading switch is off.  The load is cloned into each
// consuming work item; no expanded multi-buffer alloc is produced for its
// result.  CHECK-HINT covers both RUN lines (kernel-off and kernel-on), so
// this asserts the behavior under either configuration.

// CHECK-HINT-LABEL: func.func @test_auto_cross_core_lazy_load
// Outer unrolled loop
// CHECK-HINT: scf.for
// CUBE stage loop: own hivm.hir.load + hivm.hir.mmadL1
// CHECK-HINT: scf.for
// CHECK-HINT: hivm.hir.load
// CHECK-HINT: hivm.hir.mmadL1
// CHECK-HINT: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// VECTOR stage loop: own cloned hivm.hir.load + hivm.hir.vexp
// CHECK-HINT: scf.for
// CHECK-HINT: hivm.hir.load
// CHECK-HINT: hivm.hir.vexp
// CHECK-HINT: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// No expanded 2x buffer for the load result:
// CHECK-HINT-NOT: memref<2x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_auto_cross_core_lazy_load(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
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
      // Load whose result is consumed by both CUBE and VECTOR -- no hint,
      // no kernel switch; auto cross-core detection alone must enable
      // lazy loading.
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>

      // CUBE consumer
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // VECTOR consumer of the same load result
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

// Regression: a scalar indirect-gather lowered to
//   scf.for {ExtractedLoadOrStore} {
//     %v = memref.load <GM memref>
//     memref.store %v, <UB memref>
//   }
// has no HIVM op in its body. illegalRegionedOp's HIVM-walk finds nothing,
// so the loop ends up untagged; extractAvailableOps then refuses it as a
// workitem seed (`!isCoreOp` skip). The loop only enters a workitem
// reactively via consumer tracing -- which pulls it into the *second*
// Vector workitem (after the cube), even though its data dependencies
// allow it to run before the cube alongside the other gathers. That
// mis-placement breaks the intended schedule and produces wrong results
// downstream.
//
// Fix: illegalRegionedOp recognizes the {ExtractedLoadOrStore} +
// memref-load + tensor-write shape and tags the loop with
// `pipeline.veconly`, so it becomes a Vector seed in extractAvailableOps
// round #1 and lands in the first Vector workitem (before the cube).

// CHECK-LABEL: func.func @test_extracted_load_or_store_vec_seed
// Outer unrolled loop.
// CHECK: scf.for
// First stage: VECTOR -- contains the ExtractedLoadOrStore scalar gather.
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.load{{.*}}#hivm.address_space<gm>
// CHECK:       memref.store{{.*}}#hivm.address_space<ub>
// CHECK:     {{.*}}ExtractedLoadOrStore, pipeline.veconly
// CHECK:   {{.*}}hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// Second stage: CUBE -- mmadL1 + fixpipe.
// CHECK:   scf.for
// CHECK:     hivm.hir.mmadL1
// CHECK:   {{.*}}hivm.loop_core_type = #hivm.tcore_type<CUBE>
// Third stage: VECTOR -- the consumer that uses the cube output.
// CHECK:   scf.for
// CHECK:     hivm.hir.vadd
// CHECK:   {{.*}}hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_extracted_load_or_store_vec_seed(
      %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
      %gm_scalar: memref<?xf32, #hivm.address_space<gm>>,
      %gm_k: memref<16x16xf16>)
      attributes {WorkspaceArgIdx = 0 : i16,
                  func_dyn_memref_args = dense<[true, true, true]> : vector<3xi1>,
                  global_kernel = "local", hacc.entry,
                  hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  mix_mode = "mix"} {
    %A = "some_op"() : () -> tensor<16x16xf16>
    %c0 = arith.constant 0 : i32
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c16_idx = arith.constant 16 : index
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %bound = "some_op"() : () -> i32
    %ub_scalar_buf = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    scf.for %i = %c0 to %bound step %c2_i32 : i32 {
      // (A) Vector-side scalar gather: scf.for {ExtractedLoadOrStore}
      // reading from GM and writing to UB.
      scf.for %j = %c0_idx to %c16_idx step %c1_idx {
        %off = "some_op"() : () -> index
        %addr = memref.reinterpret_cast %gm_scalar to offset: [%off], sizes: [1], strides: [1]
            : memref<?xf32, #hivm.address_space<gm>>
              to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %v = memref.load %addr[%c0_idx]
            : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %v, %ub_scalar_buf[%j] : memref<16xf32, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}

      // (B) Cube-feeder: a regular hivm.hir.load on the K side.
      %allocK = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%gm_k : memref<16x16xf16>) outs(%allocK : memref<16x16xf16>)
      %K = bufferization.to_tensor %allocK : memref<16x16xf16>

      // (C) Cube: mmadL1 + fixpipe (separator) to UB.
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%A, %K, %true, %c16_idx, %c16_idx, %c16_idx
          : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
          outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>)
          outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0
          : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %dot_t = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      // (D) Second Vector stage: combines the cube output with the scalar
      // gather result.
      %vdest = tensor.empty() : tensor<16x16xf16>
      %sum = hivm.hir.vadd ins(%dot_t, %dot_t : tensor<16x16xf16>, tensor<16x16xf16>)
          outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%sum : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)
    }
    return
  }
}

// -----

// Test: non-core "merger" ops (arith.cmpi + arith.select) sitting between a
// VECTOR work-item op (scf.if) and the loop's scf.yield must be absorbed into
// the producing VECTOR work item, so they are cloned into the vector stage
// loop and the outer loop's yield references the vector forOp's result
// instead of a dangling op inside the soon-to-be-erased original loop.
//
// Regression test for: "operation destroyed but still has uses" crash in
// cv-pipelining when a yielded value is produced by arith.select(cond,
// init, scf.if_result) rather than directly by a core op.

// CHECK-LABEL: func.func @test_merger_absorption
// Outer unrolled loop preserves the original iter_arg.
// CHECK: scf.for {{.*}} iter_args
// Cube stage loop.
// CHECK:   scf.for
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe
// CHECK:     hivm.loop_core_type = #hivm.tcore_type<CUBE>
// Vector stage loop must (a) carry an iter_arg for the absorbed yield,
// (b) contain the cloned scf.if AND the absorbed cmpi/select, (c) yield
// the select result.
// CHECK:   scf.for {{.*}} iter_args
// CHECK:     scf.if
// CHECK:       hivm.hir.vexp
// CHECK:     arith.cmpi
// CHECK:     arith.select
// CHECK:     scf.yield {{.*}} : tensor<16x16xf16>
// CHECK:     hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_merger_absorption(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %input1 = "some_op"() : () -> memref<16x16xf16>
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    %input2 = "some_op"() : () -> memref<?xf16>
    %initin = memref.reinterpret_cast %input2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    %vinit = "some_op"() : () -> tensor<16x16xf16>
    %cond = "some_op"() : () -> i1
    %gm_dst = "some_op"() : () -> memref<16x16xf16>
    scf.for %i = %c0 to %bound step %step iter_args(%acc = %vinit) -> (tensor<16x16xf16>) : i32 {
      // CUBE: load -> mmad -> fixpipe -> to_tensor
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%initin : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      // VECTOR scf.if
      %vdest = tensor.empty() : tensor<16x16xf16>
      %if = scf.if %cond -> tensor<16x16xf16> {
        %new = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
        scf.yield %new : tensor<16x16xf16>
      } else {
        scf.yield %wst : tensor<16x16xf16>
      }

      // Cross-core separator -- write vector result somewhere so the
      // pipeline has a complete cycle.
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%if : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)

      // MERGER: cmpi + select between init and vec result, yielded out.
      // Both ops are non-core and unclassified by the worklist builder,
      // and must be absorbed into the VECTOR work item.
      %is_first = arith.cmpi eq, %i, %c0 : i32
      %sel = arith.select %is_first, %vinit, %if : tensor<16x16xf16>
      scf.yield %sel : tensor<16x16xf16>
    }
    return
  }
}

