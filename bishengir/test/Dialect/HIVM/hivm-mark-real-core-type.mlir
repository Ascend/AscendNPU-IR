// RUN: bishengir-opt --hivm-mark-real-core-type %s -o %t
// RUN: FileCheck %s --check-prefix=MARK < %t
// RUN: bishengir-opt --hivm-insert-anchors-and-backup %t | FileCheck %s --check-prefix=ANCHOR
// RUN: bishengir-opt --hivm-mark-real-core-type=remove-core-type-attrs %t | FileCheck %s --check-prefix=CLEAN

// The tensor read takes the core of the split kernel where it survives.
// The both-core barrier retains its mark and the region keeps boundary anchors.
// MARK-LABEL: func.func @vector_branch(
// ANCHOR-LABEL: func.func @vector_branch(
// CLEAN-LABEL: func.func @vector_branch(
func.func @vector_branch(%cond: i1, %buf: memref<16xf32>, %out: memref<16xf32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // ANCHOR-NEXT: hivm.hir.anchor
  // ANCHOR-NEXT: scf.if
  scf.if %cond {
    // MARK: bufferization.to_tensor {{.*}} {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    // ANCHOR-NEXT: bufferization.to_tensor
    // CLEAN: bufferization.to_tensor {{.*}} restrict writable : memref<16xf32>
    %tensor = bufferization.to_tensor %buf restrict writable : memref<16xf32>
    // ANCHOR-NEXT: hivm.hir.store
    hivm.hir.store ins(%tensor : tensor<16xf32>) outs(%out : memref<16xf32>)
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    // MARK: hivm.hir.pipe_barrier[<PIPE_ALL>] {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
    // ANCHOR-NEXT: hivm.hir.pipe_barrier
    // CLEAN: hivm.hir.pipe_barrier[<PIPE_ALL>]{{$}}
    hivm.hir.pipe_barrier[<PIPE_ALL>]
  // ANCHOR-NEXT: } {hivm.keep_control_flow}
  }
  // ANCHOR-NEXT: hivm.hir.anchor
  // ANCHOR-NEXT: return
  return
}

// Destination-style ops are marked and lifetime-only ops remain neutral.
// Cleanup preserves the placement supplied by custom ops.
// MARK-LABEL: func.func @mark_ops(
// CLEAN-LABEL: func.func @mark_ops(
func.func @mark_ops(%value: f32, %src: memref<16xf32>, %out: memref<16xf32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  // MARK: memref.alloc() : memref<16xf32>
  %tmp = memref.alloc() : memref<16xf32>
  %empty = tensor.empty() : tensor<16xf32>
  // MARK: linalg.fill {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  // CLEAN: linalg.fill ins(
  %filled = linalg.fill ins(%value : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  hivm.hir.store ins(%filled : tensor<16xf32>) outs(%tmp : memref<16xf32>)
  // MARK: hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  // CLEAN: hivm.hir.vbrc ins(
  hivm.hir.vbrc ins(%value : f32) outs(%out : memref<16xf32>)
  // MARK: hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>
  // CLEAN: hivm.hir.custom {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>
  hivm.hir.custom {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>,
                   hivm.pipe = #hivm.pipe<PIPE_V>, symbol = "user_custom"}
      "user.custom" ins(%src : memref<16xf32>) outs(%tmp : memref<16xf32>)
  // MARK: hivm.hir.custom_macro {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>
  // CLEAN: hivm.hir.custom_macro {{.*}}hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>
  hivm.hir.custom_macro {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>,
                         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
                         symbol = "user_macro"}
      "user.macro" ins(%tmp : memref<16xf32>) outs(%out : memref<16xf32>)
  // MARK: memref.dealloc %{{[^ ]+}} : memref<16xf32>
  memref.dealloc %tmp : memref<16xf32>
  return
}
