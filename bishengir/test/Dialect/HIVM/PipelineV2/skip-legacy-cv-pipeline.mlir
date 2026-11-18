// RUN: bishengir-opt %s -normalize-workspace-store | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -save-cv-pipeline-snapshot | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -outline-cv-scopes | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -merge-cv-scopes | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -intra-stage-scheduling | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -pipeline-staging="max-pipeline-multibuffer=3" | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -cv-lazy-load | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -guard-cv-pipeline-ub="simt-vf-dynamic-size=1" | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -wrap-cv-preload-scopes | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope
// RUN: bishengir-opt %s -normalize-workspace-store -save-cv-pipeline-snapshot -outline-cv-scopes -merge-cv-scopes -intra-stage-scheduling -pipeline-staging="max-pipeline-multibuffer=3" -cv-lazy-load -guard-cv-pipeline-ub="simt-vf-dynamic-size=1" -wrap-cv-preload-scopes | FileCheck %s --implicit-check-not=cv_pipeline_snapshot --implicit-check-not=vf-scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

// CHECK-LABEL: func.func @skip_normalize_and_snapshot
// CHECK:       %[[WS:.*]] = memref_ext.alloc_workspace() : memref<1xf32>
// CHECK-NEXT:  %[[WST:.*]] = bufferization.to_tensor %[[WS]] restrict writable : memref<1xf32>
// CHECK-NEXT:  %[[STORED:.*]] = hivm.hir.store ins(%{{.*}} : tensor<1xf32>) outs(%[[WST]] : tensor<1xf32>) {"hivm.inserted-store"} -> tensor<1xf32>
// CHECK-NEXT:  annotation.mark %[[STORED]] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<1xf32>
// CHECK:       } {"hivm.legacy-cv-pipeline"}
func.func @skip_normalize_and_snapshot(%src: tensor<1xf32>) -> f32
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ws = memref_ext.alloc_workspace() : memref<1xf32>
  %wst = bufferization.to_tensor %ws restrict writable : memref<1xf32>
  %stored = hivm.hir.store ins(%src : tensor<1xf32>) outs(%wst : tensor<1xf32>) {"hivm.inserted-store"} -> tensor<1xf32>
  annotation.mark %stored {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<1xf32>
  scf.for %i = %c0 to %c1 step %c1 {
    scf.yield
  } {"hivm.legacy-cv-pipeline"}
  %ext = tensor.extract %stored[%c0] : tensor<1xf32>
  return %ext : f32
}

// CHECK-LABEL: func.func @skip_outline_cv_scopes
// CHECK:       memref.alloc() : memref<8xf32, #hivm.address_space<cbuf>>
// CHECK-NEXT:  scope.scope : () -> () {
// CHECK-NEXT:    annotation.mark %{{.*}} {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
// CHECK:       } {"hivm.legacy-cv-pipeline"}
// CHECK:       scf.for
// CHECK-NOT:   scope.scope
// CHECK:       hivm.hir.indirect_store
// CHECK-NEXT:  }
func.func @skip_outline_cv_scopes(
    %a: tensor<1x1x16x16xf16>, %b: tensor<1x1x16x16xf16>,
    %offsets: tensor<16x16xi64>, %dst: memref<?xf32>, %n: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %tcb = memref.alloc() : memref<8xf32, #hivm.address_space<cbuf>>
  scope.scope : () -> () {
    annotation.mark %tcb {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<8xf32, #hivm.address_space<cbuf>>
    scope.return
  } {no_inline}
  scf.for %i = %c0 to %c1 step %c1 {
    scf.yield
  } {"hivm.legacy-cv-pipeline"}
  scf.for %i = %c0 to %n step %c1 {
    %init = tensor.empty() : tensor<1x1x16x16xf32>
    %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
    %ub = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub : memref<16x16xf32, #hivm.address_space<ub>>)
    %data = bufferization.to_tensor %ub restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
    %vinit = tensor.empty() : tensor<16x16xf32>
    %value = hivm.hir.vexp ins(%data : tensor<16x16xf32>) outs(%vinit : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.indirect_store ins(%value : tensor<16x16xf32>, %offsets : tensor<16x16xi64>) outs(%dst : memref<?xf32>)
  }
  return
}

// CHECK-LABEL: func.func @skip_merge_cv_scopes
// CHECK:       } {"hivm.legacy-cv-pipeline"}
// CHECK:       scf.for
// CHECK-NEXT:    %[[S0:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT:      arith.addf
// CHECK:         %[[S1:.*]] = scope.scope : () -> f32 {
// CHECK-NEXT:      arith.mulf %[[S0]], %[[S0]]
func.func @skip_merge_cv_scopes(%init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scf.yield
  } {"hivm.legacy-cv-pipeline"}
  %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %init) -> (f32) {
    %0 = scope.scope : () -> f32 {
      %a = arith.addf %acc, %acc : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %1 = scope.scope : () -> f32 {
      %b = arith.mulf %0, %0 : f32
      scope.return %b : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %1 : f32
  } {hivm.cv_pipelined_loop}
  return %r : f32
}

// CHECK-LABEL: func.func @skip_intra_stage_scheduling
// CHECK:       } {"hivm.legacy-cv-pipeline"}
// CHECK:       scope.scope : () -> (tensor<128x128xbf16>, tensor<128x128xf32>) {
// CHECK-NEXT:    scope.scope : () -> (tensor<128x128xf32>, tensor<128x128xbf16>, tensor<8x8x16x16xbf16>) {
func.func @skip_intra_stage_scheduling() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c1 step %c1 {
    scf.yield
  } {"hivm.legacy-cv-pipeline"}
  %alloc = memref.alloc() : memref<128x128xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
  %1 = scf.for %arg0 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg1 = %0) -> (tensor<128x128xf32>) : i32 {
    %2:2 = scope.scope : () -> (tensor<128x128xbf16>, tensor<128x128xf32>) {
      %3:3 = scope.scope : () -> (tensor<128x128xf32>, tensor<128x128xbf16>, tensor<8x8x16x16xbf16>) {
        %alloc_0 = memref.alloc() : memref<128xf32>
        %6 = bufferization.to_tensor %alloc_0 restrict writable : memref<128xf32>
        %7 = tensor.empty() : tensor<128xf32>
        %8 = hivm.hir.vexp ins(%6 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %expanded = tensor.expand_shape %8 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %9 = tensor.empty() : tensor<128x128xf32>
        %10 = hivm.hir.vmul ins(%arg1, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%9 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
        %11 = tensor.empty() : tensor<128x128xbf16>
        %12 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%arg1 : tensor<128x128xf32>) outs(%11 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %expanded_1 = tensor.expand_shape %12 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xbf16> into tensor<128x8x16xbf16>
        %13 = tensor.empty() : tensor<8x128x16xbf16>
        %14 = hivm.hir.vtranspose ins(%expanded_1 : tensor<128x8x16xbf16>) outs(%13 : tensor<8x128x16xbf16>) permutation = [1, 0, 2] -> tensor<8x128x16xbf16>
        %expanded_2 = tensor.expand_shape %14 [[0], [1, 2], [3]] output_shape [8, 8, 16, 16] : tensor<8x128x16xbf16> into tensor<8x8x16x16xbf16>
        %alloc_3 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %15 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_2 : tensor<8x8x16x16xbf16>) outs(%alloc_3 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        scope.return %10, %12, %15 : tensor<128x128xf32>, tensor<128x128xbf16>, tensor<8x8x16x16xbf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      %4 = scope.scope : () -> tensor<64x128xf32> {
        %alloc_0 = memref.alloc() : memref<8x4x16x16xbf16>
        %6 = bufferization.to_tensor %alloc_0 restrict writable : memref<8x4x16x16xbf16>
        %7 = tensor.empty() : tensor<8x4x16x16xf32>
        %8 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%6, %3#2, %true, %c64, %c128, %c128 : tensor<8x4x16x16xbf16>, tensor<8x8x16x16xbf16>, i1, index, index, index) outs(%7 : tensor<8x4x16x16xf32>) -> tensor<8x4x16x16xf32>
        %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_1 : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %9 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%8 : tensor<8x4x16x16xf32>) outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>)
        scope.return %9 : tensor<64x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %5 = scope.scope : () -> tensor<128x128xf32> {
        %6 = tensor.empty() : tensor<64x128xf32>
        %7 = hivm.hir.vexp ins(%4 : tensor<64x128xf32>) outs(%6 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %8 = tensor.empty() : tensor<128x128xf32>
        %9 = hivm.hir.vexp ins(%3#0 : tensor<128x128xf32>) outs(%8 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scope.return %9 : tensor<128x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %3#1, %5 : tensor<128x128xbf16>, tensor<128x128xf32>
    } {cv_merged_scope}
    scf.yield %2#1 : tensor<128x128xf32>
  } {hivm.cv_pipelined_loop}
  return
}

// CHECK-LABEL: func.func @skip_pipeline_staging
// CHECK:       } {"hivm.legacy-cv-pipeline"}
// CHECK:       scf.for
// CHECK-NEXT:    %[[T:.*]]:2 = scope.scope : () -> (tensor<16xf32>, i32) {
// CHECK-NEXT:      memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
// CHECK:         scope.scope : () -> () {
// CHECK-NOT:     hivm.preload_num
// CHECK:       } {hivm.cv_pipelined_loop}
func.func @skip_pipeline_staging(%src: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scf.yield
  } {"hivm.legacy-cv-pipeline"}
  scf.for %i = %c0 to %c1 step %c1 {
    %tensor, %scalar = scope.scope : () -> (tensor<16xf32>, i32) {
      %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%src : memref<16xf32>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
      %t = bufferization.to_tensor %alloc restrict writable : memref<16xf32, #hivm.address_space<cbuf>>
      %v = arith.constant 7 : i32
      scope.return %t, %v : tensor<16xf32>, i32
    }
    scope.scope : () -> () {
      %dim = tensor.dim %tensor, %c0 : tensor<16xf32>
      %sum = arith.addi %scalar, %scalar : i32
      scope.return
    }
  } {hivm.cv_pipelined_loop}
  return
}

// CHECK-LABEL: func.func @skip_cv_lazy_load
// CHECK:       } {"hivm.legacy-cv-pipeline"}
// CHECK:       %[[LAZY_V:.*]] = scope.scope : () -> tensor<4x2xf32> {
// CHECK:         hivm.hir.load
// CHECK:         scope.return %{{.*}} : tensor<4x2xf32>
// CHECK:       } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
// CHECK:       } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
// CHECK-NEXT:  scope.scope : () -> () {
// CHECK-NEXT:    tensor.empty() : tensor<4x2xf32>
// CHECK-NEXT:    hivm.hir.vmul ins(%[[LAZY_V]], %[[LAZY_V]]
// CHECK-NEXT:    hivm.hir.store
// CHECK-NEXT:    scope.return
func.func @skip_cv_lazy_load(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c1 step %c1 {
    scf.yield
  } {"hivm.legacy-cv-pipeline"}
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}

// CHECK-LABEL: func.func @skip_guard_cv_pipeline_ub
// CHECK:       scope.scope : () -> tensor<64x128xf32>
// CHECK:       hivm.preload_num = 2 : i32
// CHECK:       scope.scope : () -> tensor<64x128xf32>
// CHECK:       hivm.preload_num = 0 : i32
// CHECK:       } {hivm.cv_pipelined_loop, "hivm.legacy-cv-pipeline"}
// CHECK:       scope.scope : () -> tensor<64x128xf32>
// CHECK:       hivm.preload_num = 2 : i32
// CHECK:       scope.scope : () -> tensor<64x128xf32>
// CHECK:       hivm.preload_num = 0 : i32
// CHECK:       } {hivm.cv_pipelined_loop}
func.func @skip_guard_cv_pipeline_ub(%in: tensor<64x128xf32>) -> tensor<64x128xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %out0 = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %in) -> (tensor<64x128xf32>) {
    %produced = scope.scope : () -> tensor<64x128xf32> {
      scope.return %acc : tensor<64x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 2 : i32, no_inline}
    %consumed = scope.scope : () -> tensor<64x128xf32> {
      %empty = tensor.empty() : tensor<64x128xf32>
      %added = hivm.hir.vadd ins(%produced, %produced : tensor<64x128xf32>, tensor<64x128xf32>) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
      scope.return %added : tensor<64x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %consumed : tensor<64x128xf32>
  } {hivm.cv_pipelined_loop, "hivm.legacy-cv-pipeline"}
  %out1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %out0) -> (tensor<64x128xf32>) {
    %produced = scope.scope : () -> tensor<64x128xf32> {
      scope.return %acc : tensor<64x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 2 : i32, no_inline}
    %consumed = scope.scope : () -> tensor<64x128xf32> {
      %empty = tensor.empty() : tensor<64x128xf32>
      %added = hivm.hir.vadd ins(%produced, %produced : tensor<64x128xf32>, tensor<64x128xf32>) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
      scope.return %added : tensor<64x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %consumed : tensor<64x128xf32>
  } {hivm.cv_pipelined_loop}
  return %out1 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @skip_wrap_cv_preload_scopes
// CHECK:       } {"hivm.legacy-cv-pipeline"}
// CHECK:       scf.for
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32}
// CHECK-NEXT:    scope.scope : () -> () {
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32}
// CHECK-NOT:   CUBE_AND_VECTOR
func.func @skip_wrap_cv_preload_scopes(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    scf.yield
  } {"hivm.legacy-cv-pipeline"}
  scf.for %i = %c0 to %n step %c1 {
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}

}
