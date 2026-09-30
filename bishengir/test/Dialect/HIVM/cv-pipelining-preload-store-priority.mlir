// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew enable-preload=true" %s | FileCheck %s

// The first function uses a registered LIT name plus `--enable-preload`
// so store-priority is on without `--bypass-shape-registry` (bypass would
// also enable other registry heuristics on the later negative cases).
// The GM output and the transpose read the same tensor. Prioritize the
// independent CBUF write consumed by the next CUBE scope over the GM stores,
// while preserving the order between those stores. Also sink past the trailing
// pure VECTOR computation so later vector-scope merging preserves the
// priority of the CBUF write. The second output is an unranked GM function
// argument reached through a ranked memref.cast.
// CHECK-LABEL: func.func @cross_core_loop_carry_skew
// CHECK: %[[RANKED_OUTPUT:.*]] = memref.cast {{.*}} : memref<*xf16, #hivm.address_space<gm>> to memref<16x16xf16, #hivm.address_space<gm>>
// CHECK: scope.scope
// CHECK: %[[CAST:.*]] = hivm.hir.vcast
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[CAST]]
// CHECK-NOT: hivm.hir.store
// CHECK: hivm.hir.vtranspose
// CHECK-NOT: hivm.hir.store
// CHECK: hivm.hir.copy
// CHECK-NOT: hivm.hir.store
// CHECK: %[[SCALED:.*]] = hivm.hir.vmul
// CHECK-NEXT: hivm.hir.store ins(%[[SLICE]] : tensor<16x16xf16>)
// CHECK-NEXT: hivm.hir.store ins(%[[CAST]] : tensor<16x16xf16>) outs(%[[RANKED_OUTPUT]] : memref<16x16xf16, #hivm.address_space<gm>>)
// CHECK-NEXT: scope.return %[[SCALED]]
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// CHECK: scope.scope
// CHECK: hivm.hir.mmadL1
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cross_core_loop_carry_skew(
      %input: tensor<16x16xf32>, %rhs: tensor<1x1x16x16xf16>,
      %output: memref<?xf16>,
      %second_output: memref<*xf16, #hivm.address_space<gm>>,
      %cube_output: memref<16x16xf32>, %scaled_output: memref<16x16xf32>,
      %bound: i32) attributes {
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %scale = arith.constant 2.0 : f32
    %ranked_output = memref.cast %second_output : memref<*xf16, #hivm.address_space<gm>> to memref<16x16xf16, #hivm.address_space<gm>>
    %result = scf.for %i = %c0 to %bound step %c1 iter_args(%unused = %input) -> tensor<16x16xf32> : i32 {
      %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %empty = tensor.empty() : tensor<16x16xf16>
      %cast = hivm.hir.vcast ins(%input : tensor<16x16xf32>) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
      %slice = tensor.extract_slice %cast[0, 0] [16, 16] [1, 1] : tensor<16x16xf16> to tensor<16x16xf16>
      %view = memref.reinterpret_cast %output to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
      %subview = memref.subview %view[0, 0] [16, 16] [1, 1] : memref<16x16xf16, strided<[16, 1]>> to memref<16x16xf16, strided<[16, 1]>>
      %dynamic = memref.cast %subview : memref<16x16xf16, strided<[16, 1]>> to memref<?x16xf16, strided<[16, 1]>>
      hivm.hir.store ins(%slice : tensor<16x16xf16>) outs(%dynamic : memref<?x16xf16, strided<[16, 1]>>)
      hivm.hir.store ins(%cast : tensor<16x16xf16>) outs(%ranked_output : memref<16x16xf16, #hivm.address_space<gm>>)
      %expanded = tensor.expand_shape %cast [[0], [1, 2]] output_shape [16, 1, 16] : tensor<16x16xf16> into tensor<16x1x16xf16>
      %transpose_init = tensor.empty() : tensor<1x16x16xf16>
      %transposed = hivm.hir.vtranspose ins(%expanded : tensor<16x1x16xf16>) outs(%transpose_init : tensor<1x16x16xf16>) permutation = [1, 0, 2] -> tensor<1x16x16xf16>
      %nz = tensor.expand_shape %transposed [[0], [1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<1x16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %cbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%nz : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %scaled_init = tensor.empty() : tensor<16x16xf32>
      %scaled = hivm.hir.vmul ins(%input, %scale : tensor<16x16xf32>, f32) outs(%scaled_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %lhs = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dot = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%cube_output : memref<16x16xf32>)
      scf.yield %scaled : tensor<16x16xf32>
    }
    hivm.hir.store ins(%result : tensor<16x16xf32>) outs(%scaled_output : memref<16x16xf32>)
    return
  }

  // Atomic outputs retain their position even when the following CBUF copy
  // would otherwise qualify for prioritization.
  // CHECK-LABEL: func.func @keep_atomic_store
  // CHECK: scope.scope
  // CHECK: %[[CAST:.*]] = hivm.hir.vcast
  // CHECK-NEXT: hivm.hir.store ins(%[[CAST]] : tensor<16x16xf16>) {{.*}} atomic = <add>
  // CHECK: hivm.hir.copy
  // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>
  func.func @keep_atomic_store(
      %input: tensor<16x16xf32>, %rhs: tensor<16x16xf16>,
      %output: memref<16x16xf16>, %cube_output: memref<16x16xf32>,
      %bound: i32) attributes {
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %cbuf = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %empty = tensor.empty() : tensor<16x16xf16>
      %cast = hivm.hir.vcast ins(%input : tensor<16x16xf32>) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%cast : tensor<16x16xf16>) outs(%output : memref<16x16xf16>) atomic = <add>
      %transpose_init = tensor.empty() : tensor<16x16xf16>
      %transposed = hivm.hir.vtranspose ins(%cast : tensor<16x16xf16>) outs(%transpose_init : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
      annotation.mark %cbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%transposed : tensor<16x16xf16>) outs(%cbuf : memref<16x16xf16, #hivm.address_space<cbuf>>)
      %lhs = bufferization.to_tensor %cbuf restrict writable : memref<16x16xf16, #hivm.address_space<cbuf>>
      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cube_output : memref<16x16xf32>)
    }
    return
  }

  // A potentially aliasing GM load is a scheduling barrier, even when the
  // following CBUF copy has an independent source. Current worklist
  // construction leaves this loop unpipelined; the store still precedes
  // the load and the CBUF copy.
  // CHECK-LABEL: func.func @keep_store_before_gm_load
  // CHECK: scf.for
  // CHECK: hivm.hir.store
  // CHECK: hivm.hir.load
  // CHECK: hivm.hir.copy
  func.func @keep_store_before_gm_load(
      %input: tensor<16x16xf32>, %rhs: tensor<16x16xf16>,
      %output: memref<16x16xf16>, %cube_output: memref<16x16xf32>,
      %gm_input: memref<16x16xf16>, %bound: i32) attributes {
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %cbuf = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<16x16xf16>
      %cast = hivm.hir.vcast ins(%input : tensor<16x16xf32>) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%cast : tensor<16x16xf16>) outs(%output : memref<16x16xf16>)
      hivm.hir.load ins(%gm_input : memref<16x16xf16>) outs(%ub : memref<16x16xf16, #hivm.address_space<ub>>) {pipeline.veconly} core_type = <VECTOR>
      %transpose_init = tensor.empty() : tensor<16x16xf16>
      %transposed = hivm.hir.vtranspose ins(%cast : tensor<16x16xf16>) outs(%transpose_init : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
      annotation.mark %cbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%transposed : tensor<16x16xf16>) outs(%cbuf : memref<16x16xf16, #hivm.address_space<cbuf>>)
      %lhs = bufferization.to_tensor %cbuf restrict writable : memref<16x16xf16, #hivm.address_space<cbuf>>
      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cube_output : memref<16x16xf32>)
    }
    return
  }

  // UB copies do not feed another core. Keep the GM store before a UB copy,
  // even when a qualifying CBUF copy follows it.
  // CHECK-LABEL: func.func @keep_store_before_ub_copy
  // CHECK: scope.scope
  // CHECK: hivm.hir.store
  // CHECK: hivm.hir.copy {{.*}} outs({{.*}} : memref<16x16xf16, #hivm.address_space<ub>>)
  // CHECK: hivm.hir.copy {{.*}} outs({{.*}} : memref<16x16xf16, #hivm.address_space<cbuf>>)
  // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>
  func.func @keep_store_before_ub_copy(
      %input: tensor<16x16xf32>, %rhs: tensor<16x16xf16>,
      %output: memref<16x16xf16>, %cube_output: memref<16x16xf32>,
      %bound: i32) attributes {
        hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %cbuf = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<16x16xf16>
      %cast = hivm.hir.vcast ins(%input : tensor<16x16xf32>) outs(%empty : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%cast : tensor<16x16xf16>) outs(%output : memref<16x16xf16>)
      hivm.hir.copy ins(%cast : tensor<16x16xf16>) outs(%ub : memref<16x16xf16, #hivm.address_space<ub>>) {pipeline.veconly}
      %transpose_init = tensor.empty() : tensor<16x16xf16>
      %transposed = hivm.hir.vtranspose ins(%cast : tensor<16x16xf16>) outs(%transpose_init : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
      annotation.mark %cbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%transposed : tensor<16x16xf16>) outs(%cbuf : memref<16x16xf16, #hivm.address_space<cbuf>>)
      %lhs = bufferization.to_tensor %cbuf restrict writable : memref<16x16xf16, #hivm.address_space<cbuf>>
      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cube_output : memref<16x16xf32>)
    }
    return
  }
}
