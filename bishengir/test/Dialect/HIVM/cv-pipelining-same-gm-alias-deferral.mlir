// RUN: bishengir-opt -cv-pipelining="set-depth-in-unroll-mode=2 pipeline-mode=unroll enable-lazy-loading=true" -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=UNROLL
// RUN: bishengir-opt -cv-pipelining="set-depth-in-unroll-mode=2 pipeline-mode=skew enable-lazy-loading=true" -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=SKEW
// RUN: bishengir-opt -cv-pipelining="set-depth-in-unroll-mode=2 pipeline-mode=skew" -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=NO-LAZY

// The same-GM alias deferral applies when Load and Store use the exact same SSA
// GM view. It keeps them in the final VECTOR WorkItem solely to avoid the
// cross-WorkItem GM-alias rejection; it is not a general scheduling
// optimization.

// UNROLL-LABEL: func.func @same_gm_alias_deferral_colocates_load_store
// UNROLL: scf.for
// UNROLL: scf.for
// UNROLL-NOT: hivm.hir.load
// UNROLL-NOT: hivm.hir.store
// UNROLL: hivm.hir.vexp
// UNROLL-NOT: hivm.hir.load
// UNROLL-NOT: hivm.hir.store
// UNROLL: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// UNROLL: scf.for
// UNROLL-NOT: hivm.hir.load
// UNROLL-NOT: hivm.hir.store
// UNROLL: hivm.hir.mmadL1
// UNROLL: hivm.hir.fixpipe
// UNROLL-NOT: hivm.hir.load
// UNROLL-NOT: hivm.hir.store
// UNROLL: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// UNROLL: scf.for
// UNROLL: %[[UNROLL_GM_VIEW:.*]] = memref.subview {{.*}} : memref<64x64xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
// UNROLL: hivm.hir.load ins(%[[UNROLL_GM_VIEW]]
// UNROLL-NOT: scf.for
// UNROLL-NOT: hivm.loop_core_type
// UNROLL-NOT: hivm.hir.load
// UNROLL: hivm.hir.store {{.*}} outs(%[[UNROLL_GM_VIEW]]
// UNROLL: hivm.loop_core_type = #hivm.tcore_type<VECTOR>

// SKEW-LABEL: func.func @same_gm_alias_deferral_colocates_load_store
// SKEW: scf.for
// SKEW-NOT: hivm.hir.load
// SKEW-NOT: hivm.hir.store
// SKEW: scope.scope : () -> tensor<64x64xf16>
// SKEW-NOT: hivm.hir.load
// SKEW-NOT: hivm.hir.store
// SKEW: hivm.hir.vexp
// SKEW-NOT: hivm.hir.load
// SKEW-NOT: hivm.hir.store
// SKEW: scope.return
// SKEW: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// SKEW-SAME: hivm.max_preload_num = 3
// SKEW-SAME: hivm.preload_num = 2
// SKEW: scope.scope : () -> tensor<64x64xf32>
// SKEW-NOT: hivm.hir.load
// SKEW-NOT: hivm.hir.store
// SKEW: hivm.hir.mmadL1
// SKEW: hivm.hir.fixpipe
// SKEW-NOT: hivm.hir.load
// SKEW-NOT: hivm.hir.store
// SKEW: scope.return
// SKEW: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// SKEW-SAME: hivm.max_preload_num = 3
// SKEW-SAME: hivm.preload_num = 1
// SKEW: scope.scope : () -> ()
// SKEW: %[[SKEW_GM_VIEW:.*]] = memref.subview {{.*}} : memref<64x64xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
// SKEW: hivm.hir.load ins(%[[SKEW_GM_VIEW]]
// SKEW-NOT: scope.return
// SKEW-NOT: scope.scope
// SKEW-NOT: hivm.loop_core_type
// SKEW-NOT: hivm.hir.load
// SKEW: hivm.hir.store {{.*}} outs(%[[SKEW_GM_VIEW]]
// SKEW: scope.return
// SKEW: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// SKEW-SAME: hivm.max_preload_num = 3
// SKEW-SAME: hivm.preload_num = 0

// NO-LAZY-LABEL: func.func @same_gm_alias_deferral_colocates_load_store
// NO-LAZY-NOT: hivm.loop_core_type
// NO-LAZY: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @same_gm_alias_deferral_colocates_load_store(%gm: memref<?xbf16>, %a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %rows: index, %cols: index) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %f16Empty = tensor.empty() : tensor<64x64xf16>
    %bf16Empty = tensor.empty() : tensor<64x64xbf16>
    %f32Empty = tensor.empty() : tensor<64x64xf32>
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %prep = hivm.hir.vexp ins(%a : tensor<64x64xf16>) outs(%f16Empty : tensor<64x64xf16>) -> tensor<64x64xf16>
      %dot = hivm.hir.mmadL1 ins(%prep, %b, %true, %c64, %c64, %c64 : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index) outs(%f32Empty : tensor<64x64xf32>) -> tensor<64x64xf32>
      %cubeBuffer = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<64x64xf32>) outs(%cubeBuffer : memref<64x64xf32, #hivm.address_space<ub>>)
      %cubeBufferCast = memref.memory_space_cast %cubeBuffer : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %cube = bufferization.to_tensor %cubeBufferCast : memref<64x64xf32>

      %iv = arith.index_cast %i : i32 to index
      %gmTile = memref.reinterpret_cast %gm to offset: [%iv], sizes: [64, 64], strides: [4096, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[4096, 1], offset: ?>>
      %gmView = memref.subview %gmTile[0, 0] [%rows, %cols] [1, 1] : memref<64x64xbf16, strided<[4096, 1], offset: ?>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>>
      %localBuffer = memref.alloc() : memref<64x64xbf16>
      %localView = memref.subview %localBuffer[0, 0] [%rows, %cols] [1, 1] : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
      hivm.hir.load ins(%gmView : memref<?x?xbf16, strided<[4096, 1], offset: ?>>) outs(%localView : memref<?x?xbf16, strided<[64, 1]>>)
      %loadedTensor = bufferization.to_tensor %localBuffer : memref<64x64xbf16>
      %old = hivm.hir.vcast ins(%loadedTensor : tensor<64x64xbf16>) outs(%f32Empty : tensor<64x64xf32>) -> tensor<64x64xf32>
      %sum = hivm.hir.vadd ins(%cube, %old : tensor<64x64xf32>, tensor<64x64xf32>) outs(%f32Empty : tensor<64x64xf32>) -> tensor<64x64xf32>
      %updated = hivm.hir.vcast ins(%sum : tensor<64x64xf32>) outs(%bf16Empty : tensor<64x64xbf16>) -> tensor<64x64xbf16>
      %slice = tensor.extract_slice %updated[0, 0] [%rows, %cols] [1, 1] : tensor<64x64xbf16> to tensor<?x?xbf16>
      hivm.hir.store ins(%slice : tensor<?x?xbf16>) outs(%gmView : memref<?x?xbf16, strided<[4096, 1], offset: ?>>)
    }
    return
  }
}

// -----

// The same-GM alias deferral rejects equivalent-looking views: the Load source
// and Store destination must be the exact same SSA value.

// UNROLL-LABEL: func.func @same_gm_alias_deferral_rejects_distinct_ssa_views
// UNROLL-NOT: hivm.loop_core_type
// SKEW-LABEL: func.func @same_gm_alias_deferral_rejects_distinct_ssa_views
// SKEW-NOT: hivm.loop_core_type

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @same_gm_alias_deferral_rejects_distinct_ssa_views(%gm: memref<16x16xbf16>, %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %f16Empty = tensor.empty() : tensor<16x16xf16>
    %bf16Empty = tensor.empty() : tensor<16x16xbf16>
    %f32Empty = tensor.empty() : tensor<16x16xf32>
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %prep = hivm.hir.vexp ins(%a : tensor<16x16xf16>) outs(%f16Empty : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%prep, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cubeBuffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>>)
      %cubeBufferCast = memref.memory_space_cast %cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube = bufferization.to_tensor %cubeBufferCast : memref<16x16xf32>

      %gmLoadView = memref.subview %gm[0, 0] [16, 16] [1, 1] : memref<16x16xbf16> to memref<16x16xbf16, strided<[16, 1]>>
      %gmStoreView = memref.subview %gm[0, 0] [16, 16] [1, 1] : memref<16x16xbf16> to memref<16x16xbf16, strided<[16, 1]>>
      %localBuffer = memref.alloc() : memref<16x16xbf16>
      // expected-warning@+1 {{using GM as intermediate buffer is unsupported}}
      hivm.hir.load ins(%gmLoadView : memref<16x16xbf16, strided<[16, 1]>>) outs(%localBuffer : memref<16x16xbf16>)
      %loadedTensor = bufferization.to_tensor %localBuffer : memref<16x16xbf16>
      %old = hivm.hir.vcast ins(%loadedTensor : tensor<16x16xbf16>) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %sum = hivm.hir.vadd ins(%cube, %old : tensor<16x16xf32>, tensor<16x16xf32>) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %updated = hivm.hir.vcast ins(%sum : tensor<16x16xf32>) outs(%bf16Empty : tensor<16x16xbf16>) -> tensor<16x16xbf16>
      %slice = tensor.extract_slice %updated[0, 0] [16, 16] [1, 1] : tensor<16x16xbf16> to tensor<16x16xbf16>
      hivm.hir.store ins(%slice : tensor<16x16xbf16>) outs(%gmStoreView : memref<16x16xbf16, strided<[16, 1]>>)
    }
    return
  }
}

// -----

// The same-GM alias deferral rejects an extra, unmodeled access on the same
// function argument and keeps the existing conservative fallback.

// UNROLL-LABEL: func.func @same_gm_alias_deferral_rejects_extra_gm_access
// UNROLL-NOT: hivm.loop_core_type
// SKEW-LABEL: func.func @same_gm_alias_deferral_rejects_extra_gm_access
// SKEW-NOT: hivm.loop_core_type

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @same_gm_alias_deferral_rejects_extra_gm_access(%gm: memref<16x16xbf16>, %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %true = arith.constant true
    %c0Index = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %f16Empty = tensor.empty() : tensor<16x16xf16>
    %bf16Empty = tensor.empty() : tensor<16x16xbf16>
    %f32Empty = tensor.empty() : tensor<16x16xf32>
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %prep = hivm.hir.vexp ins(%a : tensor<16x16xf16>) outs(%f16Empty : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%prep, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cubeBuffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>>)
      %cubeBufferCast = memref.memory_space_cast %cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube = bufferization.to_tensor %cubeBufferCast : memref<16x16xf32>

      %localBuffer = memref.alloc() : memref<16x16xbf16>
      // expected-warning@+1 {{using GM as intermediate buffer is unsupported}}
      hivm.hir.load ins(%gm : memref<16x16xbf16>) outs(%localBuffer : memref<16x16xbf16>)
      %loadedTensor = bufferization.to_tensor %localBuffer : memref<16x16xbf16>
      %old = hivm.hir.vcast ins(%loadedTensor : tensor<16x16xbf16>) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %extra = memref.load %gm[%c0Index, %c0Index] : memref<16x16xbf16>
      %sum = hivm.hir.vadd ins(%cube, %old : tensor<16x16xf32>, tensor<16x16xf32>) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %updated = hivm.hir.vcast ins(%sum : tensor<16x16xf32>) outs(%bf16Empty : tensor<16x16xbf16>) -> tensor<16x16xbf16>
      %slice = tensor.extract_slice %updated[0, 0] [16, 16] [1, 1] : tensor<16x16xbf16> to tensor<16x16xbf16>
      hivm.hir.store ins(%slice : tensor<16x16xbf16>) outs(%gm : memref<16x16xbf16>)
    }
    return
  }
}

// -----

// The same-GM alias deferral requires one closed path to the Store. A second
// VECTOR consumer could otherwise pull the Load into an earlier WorkItem.

// UNROLL-LABEL: func.func @same_gm_alias_deferral_rejects_branched_load_path
// UNROLL-NOT: hivm.loop_core_type
// SKEW-LABEL: func.func @same_gm_alias_deferral_rejects_branched_load_path
// SKEW-NOT: hivm.loop_core_type

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @same_gm_alias_deferral_rejects_branched_load_path(%gm: memref<16x16xbf16>, %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %f16Empty = tensor.empty() : tensor<16x16xf16>
    %bf16Empty = tensor.empty() : tensor<16x16xbf16>
    %f32Empty = tensor.empty() : tensor<16x16xf32>
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %prep = hivm.hir.vexp ins(%a : tensor<16x16xf16>) outs(%f16Empty : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%prep, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cubeBuffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>>)
      %cubeBufferCast = memref.memory_space_cast %cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube = bufferization.to_tensor %cubeBufferCast : memref<16x16xf32>

      %localBuffer = memref.alloc() : memref<16x16xbf16>
      // expected-warning@+1 {{using GM as intermediate buffer is unsupported}}
      hivm.hir.load ins(%gm : memref<16x16xbf16>) outs(%localBuffer : memref<16x16xbf16>)
      %loadedTensor = bufferization.to_tensor %localBuffer : memref<16x16xbf16>
      %old = hivm.hir.vcast ins(%loadedTensor : tensor<16x16xbf16>) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %side = hivm.hir.vcast ins(%old : tensor<16x16xf32>) outs(%bf16Empty : tensor<16x16xbf16>) -> tensor<16x16xbf16>
      %sum = hivm.hir.vadd ins(%cube, %old : tensor<16x16xf32>, tensor<16x16xf32>) outs(%f32Empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      %updated = hivm.hir.vcast ins(%sum : tensor<16x16xf32>) outs(%bf16Empty : tensor<16x16xbf16>) -> tensor<16x16xbf16>
      %slice = tensor.extract_slice %updated[0, 0] [16, 16] [1, 1] : tensor<16x16xbf16> to tensor<16x16xbf16>
      hivm.hir.store ins(%slice : tensor<16x16xbf16>) outs(%gm : memref<16x16xbf16>)
    }
    return
  }
}
