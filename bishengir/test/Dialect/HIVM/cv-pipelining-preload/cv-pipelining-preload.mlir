// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew bypass-shape-registry=true" -allow-unregistered-dialect %s | FileCheck %s
// RUN: bishengir-opt -cv-pipelining="bypass-shape-registry=true" -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @a5_preload_workspace
// CHECK-DAG: %[[WS0:.*]] = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<4x16x16xf32>
// CHECK-DAG: %[[WS1:.*]] = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<4x16x16xf16>
// CHECK-DAG: %[[WS2:.*]] = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<4x16x16xf32>
// CHECK: scope.scope
// CHECK:   memref.subview {{.*}} {hivm.preload_workspace} : memref<4x16x16xf32>
// CHECK:   hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
// CHECK: bufferization.to_tensor {{.*}} : memref<4x16x16xf32>
// CHECK: scope.scope
// CHECK:   tensor.extract_slice {{.*}} {hivm.preload_workspace} : tensor<4x16x16xf32> to tensor<16x16xf32>
// CHECK:   hivm.hir.vexp
// CHECK:   memref.subview {{.*}} {hivm.preload_workspace} : memref<4x16x16xf16>
// CHECK:   hivm.hir.store
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
// CHECK: scope.scope
// CHECK:   tensor.extract_slice {{.*}} {hivm.preload_workspace} : tensor<4x16x16xf16> to tensor<16x16xf16>
// CHECK:   hivm.hir.mmadL1
// CHECK:   memref.subview {{.*}} {hivm.preload_workspace} : memref<4x16x16xf32>
// CHECK:   hivm.hir.fixpipe
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK: scope.scope
// CHECK:   tensor.extract_slice {{.*}} {hivm.preload_workspace} : tensor<4x16x16xf32> to tensor<16x16xf32>
// CHECK:   hivm.hir.vexp
// CHECK:   scope.return
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @a5_preload_workspace(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %a_mem = "some_op"() : () -> memref<16x16xf16>
    %a = bufferization.to_tensor %a_mem : memref<16x16xf16>
    %k_mem = "some_op"() : () -> memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %init = tensor.empty() : tensor<16x16xf32>
    %result = scf.for %i = %c0 to %bound step %step iter_args(%acc = %init) -> tensor<16x16xf32> : i32 {
      %k0_alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%k_mem : memref<16x16xf16>) outs(%k0_alloc : memref<16x16xf16>)
      %k0 = bufferization.to_tensor %k0_alloc : memref<16x16xf16>
      %dot0_init = tensor.empty() : tensor<16x16xf32>
      %dot0 = hivm.hir.mmadL1 ins(%a, %k0, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws0 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws0 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws0_tensor = bufferization.to_tensor %ws0 restrict writable : memref<16x16xf32>
      %fix0 = hivm.hir.fixpipe ins(%dot0 : tensor<16x16xf32>) outs(%ws0_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      %load0_init = tensor.empty() : tensor<16x16xf32>
      %load0 = hivm.hir.load ins(%fix0 : tensor<16x16xf32>) outs(%load0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v0_init = tensor.empty() : tensor<16x16xf32>
      %v0 = hivm.hir.vexp ins(%load0 : tensor<16x16xf32>) outs(%v0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v0_f16_init = tensor.empty() : tensor<16x16xf16>
      %v0_f16 = hivm.hir.vcast ins(%v0 : tensor<16x16xf32>) outs(%v0_f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf16>
      annotation.mark %ws1 {hivm.multi_buffer = 4 : i32} : memref<16x16xf16>
      %ws1_tensor = bufferization.to_tensor %ws1 restrict writable : memref<16x16xf16>
      %store1 = hivm.hir.store ins(%v0_f16 : tensor<16x16xf16>) outs(%ws1_tensor : tensor<16x16xf16>) -> tensor<16x16xf16>

      %load1_init = tensor.empty() : tensor<16x16xf16>
      %load1 = hivm.hir.load ins(%store1 : tensor<16x16xf16>) outs(%load1_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %k1_alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%k_mem : memref<16x16xf16>) outs(%k1_alloc : memref<16x16xf16>)
      %k1 = bufferization.to_tensor %k1_alloc : memref<16x16xf16>
      %dot1_init = tensor.empty() : tensor<16x16xf32>
      %dot1 = hivm.hir.mmadL1 ins(%load1, %k1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws2 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws2 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws2_tensor = bufferization.to_tensor %ws2 restrict writable : memref<16x16xf32>
      %fix1 = hivm.hir.fixpipe ins(%dot1 : tensor<16x16xf32>) outs(%ws2_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      %load2_init = tensor.empty() : tensor<16x16xf32>
      %load2 = hivm.hir.load ins(%fix1 : tensor<16x16xf32>) outs(%load2_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v1_init = tensor.empty() : tensor<16x16xf32>
      %v1 = hivm.hir.vexp ins(%load2 : tensor<16x16xf32>) outs(%v1_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %v1 : tensor<16x16xf32>
    }
    "some_consume"(%result) : (tensor<16x16xf32>) -> ()
    return
  }

  // Minimal preload case for the TCB mixed-CV inserted-load path:
  //   CUBE mmadL1 -> fixpipe(nz2nd) -> VECTOR inserted-load.
  // The TCB tensor view must be rebuilt after the CUBE scope.
  // CHECK-LABEL: func.func @preload_tcb_fixpipe_inserted_load
  // CHECK: scf.for
  // CHECK: %[[TCB_ALLOC:.*]] = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
  // CHECK: scope.scope : () -> () {
  // CHECK: hivm.hir.mmadL1
  // CHECK: annotation.mark %[[TCB_ALLOC]] {{.*}}hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>{{.*}} : memref<128x128xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.fixpipe
  // CHECK: scope.return
  // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
  // CHECK-NEXT: %[[TCB_TENSOR:.*]] = bufferization.to_tensor %[[TCB_ALLOC]] restrict writable : memref{{.*}}
  // CHECK: scope.scope
  // CHECK: hivm.hir.load ins(%[[TCB_TENSOR]] : tensor<128x128xf32>)
  func.func @preload_tcb_fixpipe_inserted_load(%arg0: tensor<8x8x16x16xf16>, %arg1: tensor<8x8x16x16xf16>) attributes {func_dyn_memref_args = dense<[false, false]> : vector<2xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %init = tensor.empty() : tensor<128x128xf32>
    %result = scf.for %i = %c0 to %bound step %step iter_args(%acc = %init) -> tensor<128x128xf32> : i32 {
      %dot0_init = tensor.empty() : tensor<8x8x16x16xf32>
      %dot0 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg1, %true, %c128, %c128, %c128 : tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<8x8x16x16xf32>) -> tensor<8x8x16x16xf32>
      %tcb_alloc = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %tcb_alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x128xf32, #hivm.address_space<ub>>
      %tcb_tensor = bufferization.to_tensor %tcb_alloc restrict writable : memref<128x128xf32, #hivm.address_space<ub>>
      %fix0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<8x8x16x16xf32>) outs(%tcb_tensor : tensor<128x128xf32>) -> tensor<128x128xf32>

      %load0_init = tensor.empty() : tensor<128x128xf32>
      %load0 = hivm.hir.load ins(%fix0 : tensor<128x128xf32>) outs(%load0_init : tensor<128x128xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<128x128xf32>
      %v0_init = tensor.empty() : tensor<128x128xf32>
      %v0 = hivm.hir.vexp ins(%load0 : tensor<128x128xf32>) outs(%v0_init : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %v0 : tensor<128x128xf32>
    } {fixpipe_for_mmad_result_already_inserted = true}
    "some_consume"(%result) : (tensor<128x128xf32>) -> ()
    return
  }

  // A constantized dynamic TCB has byte storage plus a logical memref.view.
  // The physical storage must bypass the producer scope result while the
  // logical view is rebuilt inside the VECTOR consumer scope.
  // CHECK-LABEL: func.func @preload_tcb_dynamic_view
  // CHECK: scf.for
  // CHECK: %[[TCB_STORAGE:.*]] = memref.alloc() : memref<65536xi8, #hivm.address_space<ub>>
  // CHECK: scope.scope : () -> () {
  // CHECK: annotation.mark %[[TCB_STORAGE]] {{.*}}hivm.tightly_coupled_buffer
  // CHECK: hivm.hir.fixpipe
  // CHECK: scope.return
  // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
  // CHECK-NOT: bufferization.to_tensor %[[TCB_STORAGE]]
  // CHECK: scope.scope
  // CHECK: %[[TCB_VIEW:.*]] = memref.view %[[TCB_STORAGE]]
  // CHECK: %[[TCB_CAST:.*]] = memref.memory_space_cast %[[TCB_VIEW]]
  // CHECK: %[[TCB_TENSOR:.*]] = bufferization.to_tensor %[[TCB_CAST]] restrict writable
  // CHECK: scf.if
  // CHECK: hivm.hir.load ins(%[[TCB_TENSOR]] : tensor<?x128xf32>)
  // CHECK: hivm.hir.load ins(%[[TCB_TENSOR]] : tensor<?x128xf32>)
  func.func @preload_tcb_dynamic_view(
      %rows: index,
      %arg0: tensor<8x8x16x16xf16>,
      %arg1: tensor<8x8x16x16xf16>,
      %cond: i1) attributes {
        func_dyn_memref_args = dense<[false, false, false, false]> : vector<4xi1>,
        global_kernel = "local",
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>,
        mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %c0_index = arith.constant 0 : index
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %loop_init = tensor.empty(%rows) : tensor<?x128xf32>
    %result = scf.for %i = %c0 to %bound step %step
        iter_args(%acc = %loop_init) -> tensor<?x128xf32> : i32 {
      %dot_init = tensor.empty() : tensor<8x8x16x16xf32>
      %dot = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg1, %true, %c128, %c128, %c128 : tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<8x8x16x16xf32>) -> tensor<8x8x16x16xf32>
      %storage = memref.alloc() : memref<65536xi8, #hivm.address_space<ub>>
      annotation.mark %storage {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<65536xi8, #hivm.address_space<ub>>
      %view = memref.view %storage[%c0_index][%rows] : memref<65536xi8, #hivm.address_space<ub>> to memref<?x128xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %view : memref<?x128xf32, #hivm.address_space<ub>> to memref<?x128xf32>
      %tensor = bufferization.to_tensor %cast restrict writable : memref<?x128xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<8x8x16x16xf32>) outs(%tensor : tensor<?x128xf32>) -> tensor<?x128xf32>

      // The first consumer is nested; the same scope also has a direct use.
      %nested = scf.if %cond -> tensor<?x128xf32> {
        %nested_init = tensor.empty(%rows) : tensor<?x128xf32>
        %nested_load = hivm.hir.load ins(%fix : tensor<?x128xf32>) outs(%nested_init : tensor<?x128xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<?x128xf32>
        scf.yield %nested_load : tensor<?x128xf32>
      } else {
        scf.yield %acc : tensor<?x128xf32>
      }
      %load_init = tensor.empty(%rows) : tensor<?x128xf32>
      %load = hivm.hir.load ins(%fix : tensor<?x128xf32>) outs(%load_init : tensor<?x128xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<?x128xf32>
      %vector_init = tensor.empty(%rows) : tensor<?x128xf32>
      %vector = hivm.hir.vadd ins(%load, %nested : tensor<?x128xf32>, tensor<?x128xf32>) outs(%vector_init : tensor<?x128xf32>) -> tensor<?x128xf32>
      scf.yield %vector : tensor<?x128xf32>
    }
    "some_consume"(%result) : (tensor<?x128xf32>) -> ()
    return
  }

  // View offsets and sizes used only by the producer must be recomputed in
  // each consumer scope, not captured from inside the producer scope.
  // CHECK-LABEL: func.func @preload_tcb_local_view_operands
  // CHECK-SAME: %[[ROWS:[^:]+]]: index
  // CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[BYTES:.*]] = arith.constant 512 : index
  // CHECK-DAG: %[[LIMIT:.*]] = arith.constant 128 : index
  // CHECK: scf.for %[[IV:[^ ]+]] =
  // CHECK: %[[STORAGE:.*]] = memref.alloc() : memref<66048xi8, #hivm.address_space<ub>>
  // CHECK: scope.scope : () -> () {
  // CHECK: hivm.hir.fixpipe
  // CHECK: scope.return
  // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
  // CHECK: scope.scope
  // CHECK: %[[INDEX:.*]] = arith.index_cast %[[IV]] : i32 to index
  // CHECK: %[[SLOT:.*]] = arith.remui %[[INDEX]], %[[TWO]] : index
  // CHECK: %[[OFFSET:.*]] = arith.muli %[[SLOT]], %[[BYTES]] : index
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[ROWS]], %[[LIMIT]] : index
  // CHECK: %[[VIEW:.*]] = memref.view %[[STORAGE]][%[[OFFSET]]][%[[SIZE]]]
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[VIEW]]
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]] restrict writable
  // CHECK: hivm.hir.load ins(%[[TENSOR]] : tensor<?x128xf32>)
  // CHECK: hivm.hir.vexp
  // CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
  func.func @preload_tcb_local_view_operands(
      %rows: index,
      %arg0: tensor<8x8x16x16xf16>,
      %arg1: tensor<8x8x16x16xf16>) attributes {
        func_dyn_memref_args = dense<[false, false, false]> : vector<3xi1>,
        global_kernel = "local",
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>,
        mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %c2_index = arith.constant 2 : index
    %c512_index = arith.constant 512 : index
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %bounded_rows = arith.minsi %rows, %c128 : index
    %loop_init = tensor.empty(%bounded_rows) : tensor<?x128xf32>
    %result = scf.for %i = %c0 to %bound step %step
        iter_args(%acc = %loop_init) -> tensor<?x128xf32> : i32 {
      %dot_init = tensor.empty() : tensor<8x8x16x16xf32>
      %dot = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%arg0, %arg1, %true, %c128, %c128, %c128 : tensor<8x8x16x16xf16>, tensor<8x8x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<8x8x16x16xf32>) -> tensor<8x8x16x16xf32>
      %storage = memref.alloc() : memref<66048xi8, #hivm.address_space<ub>>
      annotation.mark %storage {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<66048xi8, #hivm.address_space<ub>>
      %iv_index = arith.index_cast %i : i32 to index
      %slot = arith.remui %iv_index, %c2_index : index
      %byte_offset = arith.muli %slot, %c512_index : index
      %view_rows = arith.minsi %rows, %c128 : index
      %view = memref.view %storage[%byte_offset][%view_rows] : memref<66048xi8, #hivm.address_space<ub>> to memref<?x128xf32, #hivm.address_space<ub>>
      %cast = memref.memory_space_cast %view : memref<?x128xf32, #hivm.address_space<ub>> to memref<?x128xf32>
      %tensor = bufferization.to_tensor %cast restrict writable : memref<?x128xf32>
      %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<8x8x16x16xf32>) outs(%tensor : tensor<?x128xf32>) -> tensor<?x128xf32>
      %load_init = tensor.empty(%bounded_rows) : tensor<?x128xf32>
      %load = hivm.hir.load ins(%fix : tensor<?x128xf32>) outs(%load_init : tensor<?x128xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<?x128xf32>
      %vector_init = tensor.empty(%bounded_rows) : tensor<?x128xf32>
      %vector = hivm.hir.vexp ins(%load : tensor<?x128xf32>) outs(%vector_init : tensor<?x128xf32>) -> tensor<?x128xf32>
      scf.yield %vector : tensor<?x128xf32>
    }
    "some_consume"(%result) : (tensor<?x128xf32>) -> ()
    return
  }

  // CHECK-LABEL: func.func @preload_outputless_atomic
  // CHECK: scf.for
  // CHECK-NOT: hivm.hir.set_atomic
  // CHECK: scope.scope : () -> () {
  // CHECK:   hivm.hir.mmadL1
  // CHECK:   hivm.hir.set_atomic kind = <add>[type = f32]
  // CHECK:   hivm.hir.fixpipe
  // CHECK-NEXT:   hivm.hir.set_atomic kind = <none>[type = f32]
  // CHECK:   scope.return
  // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
  // CHECK: scope.scope : () -> () {
  // CHECK:   hivm.hir.vexp
  // CHECK:   hivm.hir.store
  // CHECK:   scope.return
  // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
  func.func @preload_outputless_atomic(
      %a: tensor<16x16xf16>,
      %b: tensor<16x16xf16>,
      %vector_in: tensor<16x16xf32>,
      %cube_out: memref<16x16xf32>,
      %vector_out: memref<16x16xf32>) attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>,
        mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %bound step %step : i32 {
      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.set_atomic kind = <add>[type = f32]
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cube_out : memref<16x16xf32>)
      hivm.hir.set_atomic kind = <none>[type = f32]

      %vector_init = tensor.empty() : tensor<16x16xf32>
      %vector = hivm.hir.vexp ins(%vector_in : tensor<16x16xf32>) outs(%vector_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.store ins(%vector : tensor<16x16xf32>) outs(%vector_out : memref<16x16xf32>)
      scf.yield
    }
    return
  }
}
