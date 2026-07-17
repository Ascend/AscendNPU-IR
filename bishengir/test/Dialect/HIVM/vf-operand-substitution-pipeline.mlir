// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -lower-hivm-pipeline="enable-vf-operand-substitution=true" -split-input-file -mlir-disable-threading -mlir-print-ir-before=hivm-plan-memory 2>&1 | FileCheck %s

func.func @vf_reuse_direct(
    %arg0: memref<64x128xf32, #hivm.address_space<ub>>,
    %arg1: memref<64x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} :
      memref<64x128xf32, #hivm.address_space<ub>>, vector<1x64xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] {in_bounds = [true, true]} :
      vector<1x64xf32>, memref<64x128xf32, #hivm.address_space<ub>>
  return
}

// CHECK: func.func @positive_select_source(
// CHECK: call @vf_reuse_direct(%[[SELECTED:[a-zA-Z0-9_]+]], %[[SELECTED]])
func.func @positive_select_source(
    %arg0: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg1: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg2: memref<64x128xf32, #hivm.address_space<gm>>,
    %cond: i1)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %left = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %right = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64x128xf32, #hivm.address_space<gm>>)
      outs(%left : memref<64x128xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>)
      outs(%right : memref<64x128xf32, #hivm.address_space<ub>>)
  %selected = arith.select %cond, %left, %right :
      memref<64x128xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_direct(%selected, %dst) {hivm.vector_function, no_inline} :
      (memref<64x128xf32, #hivm.address_space<ub>>,
       memref<64x128xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.store ins(%dst : memref<64x128xf32, #hivm.address_space<ub>>)
      outs(%arg2 : memref<64x128xf32, #hivm.address_space<gm>>)
  return
}

// -----

func.func @vf_reuse_dynamic(
    %arg0: memref<?x?xf32, #hivm.address_space<ub>>,
    %arg1: memref<?x?xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} :
      memref<?x?xf32, #hivm.address_space<ub>>, vector<1x64xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] {in_bounds = [true, true]} :
      vector<1x64xf32>, memref<?x?xf32, #hivm.address_space<ub>>
  return
}

// CHECK: func.func @positive_select_source_through_cast(
// CHECK: call @vf_reuse_dynamic(%[[SELECTED:[a-zA-Z0-9_]+]], %[[SELECTED]])
func.func @positive_select_source_through_cast(
    %arg0: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg1: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg2: memref<64x128xf32, #hivm.address_space<gm>>,
    %cond: i1)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %left_alloc = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %right_alloc = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %left = memref.cast %left_alloc :
      memref<64x128xf32, #hivm.address_space<ub>> to
      memref<?x?xf32, #hivm.address_space<ub>>
  %right = memref.cast %right_alloc :
      memref<64x128xf32, #hivm.address_space<ub>> to
      memref<?x?xf32, #hivm.address_space<ub>>
  %selected = arith.select %cond, %left, %right :
      memref<?x?xf32, #hivm.address_space<ub>>
  %dst_alloc = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %dst = memref.cast %dst_alloc :
      memref<64x128xf32, #hivm.address_space<ub>> to
      memref<?x?xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_dynamic(%selected, %dst) {hivm.vector_function, no_inline} :
      (memref<?x?xf32, #hivm.address_space<ub>>,
       memref<?x?xf32, #hivm.address_space<ub>>) -> ()
  %out = memref.cast %arg2 :
      memref<64x128xf32, #hivm.address_space<gm>> to
      memref<?x?xf32, #hivm.address_space<gm>>
  hivm.hir.store ins(%dst : memref<?x?xf32, #hivm.address_space<ub>>)
      outs(%out : memref<?x?xf32, #hivm.address_space<gm>>)
  return
}

// -----

func.func @vf_reuse_extra_use(
    %arg0: memref<64x128xf32, #hivm.address_space<ub>>,
    %arg1: memref<64x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} :
      memref<64x128xf32, #hivm.address_space<ub>>, vector<1x64xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] {in_bounds = [true, true]} :
      vector<1x64xf32>, memref<64x128xf32, #hivm.address_space<ub>>
  return
}

// CHECK: func.func @negative_select_has_extra_use(
// CHECK: annotation.mark %[[DST:[a-zA-Z0-9_]+]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>}
// CHECK: call @vf_reuse_extra_use(%{{[a-zA-Z0-9_]+}}, %[[DST]])
// CHECK: memref.collapse_shape %[[DST]]
func.func @negative_select_has_extra_use(
    %arg0: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg1: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg2: memref<64x128xf32, #hivm.address_space<gm>>,
    %cond: i1)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %left = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %right = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %selected = arith.select %cond, %left, %right :
      memref<64x128xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_extra_use(%selected, %dst) {hivm.vector_function, no_inline} :
      (memref<64x128xf32, #hivm.address_space<ub>>,
       memref<64x128xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.debug {debugtype = "print", hex = false, prefix = "selected: ",
                  tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %selected :
      memref<64x128xf32, #hivm.address_space<ub>>
  hivm.hir.store ins(%dst : memref<64x128xf32, #hivm.address_space<ub>>)
      outs(%arg2 : memref<64x128xf32, #hivm.address_space<gm>>)
  return
}

// -----

func.func @vf_reuse_cross_sync(
    %arg0: memref<64x128xf32, #hivm.address_space<ub>>,
    %arg1: memref<64x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} :
      memref<64x128xf32, #hivm.address_space<ub>>, vector<1x64xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] {in_bounds = [true, true]} :
      vector<1x64xf32>, memref<64x128xf32, #hivm.address_space<ub>>
  return
}

// CHECK: func.func @negative_dst_use_crosses_sync_set(
// CHECK: annotation.mark %[[DST:[a-zA-Z0-9_]+]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>}
// CHECK: call @vf_reuse_cross_sync(%{{[a-zA-Z0-9_]+}}, %[[DST]])
// CHECK: memref.collapse_shape %[[DST]]
func.func @negative_dst_use_crosses_sync_set(
    %arg0: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg1: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg2: memref<64x128xf32, #hivm.address_space<gm>>,
    %cond: i1)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %left = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %right = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %selected = arith.select %cond, %left, %right :
      memref<64x128xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_cross_sync(%selected, %dst) {hivm.vector_function, no_inline} :
      (memref<64x128xf32, #hivm.address_space<ub>>,
       memref<64x128xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
  hivm.hir.store ins(%dst : memref<64x128xf32, #hivm.address_space<ub>>)
      outs(%arg2 : memref<64x128xf32, #hivm.address_space<gm>>)
  return
}

func.func @vf_reuse_non_select(
    %arg0: memref<64x128xf32, #hivm.address_space<ub>>,
    %arg1: memref<64x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} :
      memref<64x128xf32, #hivm.address_space<ub>>, vector<1x64xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] {in_bounds = [true, true]} :
      vector<1x64xf32>, memref<64x128xf32, #hivm.address_space<ub>>
  return
}

// CHECK: func.func @negative_source_is_not_select(
// CHECK: annotation.mark %[[DST:[a-zA-Z0-9_]+]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
// CHECK: call @vf_reuse_non_select(%{{[a-zA-Z0-9_]+}}, %[[DST]])
// CHECK: memref.collapse_shape %[[DST]]
func.func @negative_source_is_not_select(
    %arg0: memref<64x128xf32, #hivm.address_space<gm>>,
    %arg1: memref<64x128xf32, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %src = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() {alignment = 64 : i64} :
      memref<64x128xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_non_select(%src, %dst) {hivm.vector_function, no_inline} :
      (memref<64x128xf32, #hivm.address_space<ub>>,
       memref<64x128xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.store ins(%dst : memref<64x128xf32, #hivm.address_space<ub>>)
      outs(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func.func @drop_multibuffer_mark_on_store_source(
// CHECK: hivm.hir.pointer_cast
// CHECK-NOT: hivm.multi_buffer
// CHECK: hivm.hir.store
func.func @drop_multibuffer_mark_on_store_source(
    %arg0: memref<64xf32, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.multi_buffer = 2 : i32} :
      memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.store ins(%alloc : memref<64xf32, #hivm.address_space<ub>>)
      outs(%arg0 : memref<64xf32, #hivm.address_space<gm>>)
  return
}
