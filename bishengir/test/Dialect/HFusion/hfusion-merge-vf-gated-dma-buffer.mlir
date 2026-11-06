// RUN: bishengir-opt -hfusion-merge-vf %s -split-input-file | FileCheck %s
// RUN: bishengir-opt -hfusion-merge-vf=merge-level=2 %s -split-input-file \
// RUN:   | FileCheck %s --check-prefix=L2

// A VF is one atomic region for the sync solver, which releases a DMA-refilled
// buffer right after the last vector function that reads it. Appending work
// that does not read the buffer pushes that release later, so the next trip's
// DMA is issued late. Refuse the merge in that case.

// CHECK-LABEL: func.func @gated_vf_0
func.func @gated_vf_0(%arg0: memref<64xf16, #hivm.address_space<ub>>)
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}

// CHECK-LABEL: func.func @gated_vf_1
func.func @gated_vf_1(%arg0: memref<128xf32, #hivm.address_space<ub>>)
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @gated_main
func.func @gated_main(%gm: memref<64xf16, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %alloc = memref.alloc() : memref<64xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  // CHECK-NOT: gated_merged_vf
  scf.for %t = %c0 to %c8 step %c1 {
    hivm.hir.load ins(%gm : memref<64xf16, #hivm.address_space<gm>>)
                  outs(%alloc : memref<64xf16, #hivm.address_space<ub>>)
    // CHECK: call @gated_vf_0
    func.call @gated_vf_0(%alloc) {hivm.vector_function, no_inline}
        : (memref<64xf16, #hivm.address_space<ub>>) -> ()
    // CHECK: call @gated_vf_1
    func.call @gated_vf_1(%alloc_0) {hivm.vector_function, no_inline}
        : (memref<128xf32, #hivm.address_space<ub>>) -> ()
    scf.yield
  }
  return
}

// -----

// A later vector function still reads the buffer, so the release already sits
// behind the merge candidate and merging cannot delay it: allow the merge even
// though the candidate carries a multi-trip loop. Checked at merge-level=2
// because at level 1 the later read already relates vf_0 to vf_2, which keeps
// vf_0 out of the merge bases before this guard is consulted.

func.func @later_reader_vf_0(%arg0: memref<64xf16, #hivm.address_space<ub>>)
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}

func.func @later_reader_vf_1(%arg0: memref<128xf32, #hivm.address_space<ub>>)
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c32 step %c1 {
    scf.yield
  }
  return
}

func.func @later_reader_vf_2(%arg0: memref<64xf16, #hivm.address_space<ub>>)
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}

// L2-LABEL: func.func @later_reader_main
func.func @later_reader_main(%gm: memref<64xf16, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %alloc = memref.alloc() : memref<64xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  scf.for %t = %c0 to %c8 step %c1 {
    hivm.hir.load ins(%gm : memref<64xf16, #hivm.address_space<gm>>)
                  outs(%alloc : memref<64xf16, #hivm.address_space<ub>>)
    // L2: call @later_reader_merged_vf
    func.call @later_reader_vf_0(%alloc) {hivm.vector_function, no_inline}
        : (memref<64xf16, #hivm.address_space<ub>>) -> ()
    func.call @later_reader_vf_1(%alloc_0) {hivm.vector_function, no_inline}
        : (memref<128xf32, #hivm.address_space<ub>>) -> ()
    func.call @later_reader_vf_2(%alloc) {hivm.vector_function, no_inline}
        : (memref<64xf16, #hivm.address_space<ub>>) -> ()
    scf.yield
  }
  return
}

// -----

// A straight-line candidate has a bounded delay and measures as harmless, so
// only a multi-trip loop blocks the merge.

func.func @no_loop_vf_0(%arg0: memref<64xf16, #hivm.address_space<ub>>)
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}

func.func @no_loop_vf_1(%arg0: memref<128xf32, #hivm.address_space<ub>>)
    attributes {hivm.vector_function, no_inline,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  return
}

// CHECK-LABEL: func.func @no_loop_main
func.func @no_loop_main(%gm: memref<64xf16, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %alloc = memref.alloc() : memref<64xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  scf.for %t = %c0 to %c8 step %c1 {
    hivm.hir.load ins(%gm : memref<64xf16, #hivm.address_space<gm>>)
                  outs(%alloc : memref<64xf16, #hivm.address_space<ub>>)
    // CHECK: call @no_loop_merged_vf
    func.call @no_loop_vf_0(%alloc) {hivm.vector_function, no_inline}
        : (memref<64xf16, #hivm.address_space<ub>>) -> ()
    func.call @no_loop_vf_1(%alloc_0) {hivm.vector_function, no_inline}
        : (memref<128xf32, #hivm.address_space<ub>>) -> ()
    scf.yield
  }
  return
}
