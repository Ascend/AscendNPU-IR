// RUN: bishengir-opt %s -hivm-pre-mark-stride-align | FileCheck %s
// RUN: bishengir-opt %s -hivm-pre-mark-stride-align \
// RUN:   -hivm-mark-stride-align -hivm-enable-stride-align \
// RUN: | FileCheck %s --check-prefix=LAYOUT

// A vector function may flatten a DMA-loaded UB buffer before reading it.
// Stride alignment would insert gaps between logical rows and invalidate the
// collapse_shape. PreMarkStrideAlign must follow the view-like chain and mark
// the root allocation so that MarkStrideAlign preserves its dense layout.

// CHECK-LABEL: func.func @kernel(
// CHECK: %[[INPUT:.*]] = memref.alloc() : memref<5x3x2x68xf16, #hivm.address_space<ub>>
// CHECK-NEXT: annotation.mark %[[INPUT]] {hivm.skip_stride_align_for_vload = #hivm.skip_stride_align_for_vload}

// LAYOUT-LABEL: func.func @outlined_vf(
// LAYOUT-SAME: %arg0: memref<5x3x2x68xf16, #hivm.address_space<ub>>
// LAYOUT-NOT: strided<[480, 160, 80, 1]>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @outlined_vf(%arg0: memref<5x3x2x68xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16
    %flat = memref.collapse_shape %arg0 [[0, 1, 2, 3]] : memref<5x3x2x68xf16, #hivm.address_space<ub>> into memref<2040xf16, #hivm.address_space<ub>>
    %read = vector.transfer_read %flat[%c0], %zero {in_bounds = [true]} : memref<2040xf16, #hivm.address_space<ub>>, vector<64xf16>
    return
  }

  func.func @kernel(%arg0: memref<5x3x2x68xf16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %input = memref.alloc() : memref<5x3x2x68xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<5x3x2x68xf16, #hivm.address_space<gm>>) outs(%input : memref<5x3x2x68xf16, #hivm.address_space<ub>>)
    func.call @outlined_vf(%input) {hivm.vector_function, no_inline} : (memref<5x3x2x68xf16, #hivm.address_space<ub>>) -> ()
    return
  }
}
