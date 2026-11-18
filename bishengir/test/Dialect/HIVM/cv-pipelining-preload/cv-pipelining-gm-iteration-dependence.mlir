// RUN: bishengir-opt %s -split-input-file -cv-pipelining="pipeline-mode=skew" -verify-diagnostics | FileCheck %s

// Static bounds, sizes, and strides: gap = span = 256 elements.
// Alias analysis proves different iterations disjoint and enables preload.
// CHECK-LABEL: func.func @static_gm_tiles
// CHECK: scope.scope
// CHECK: hivm.hir.load
// CHECK: hivm.preload_num = {{[1-9][0-9]*}}
// CHECK: scope.scope
// CHECK: hivm.hir.store
// CHECK: hivm.preload_num = 0
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @static_gm_tiles(%gm: memref<2048xf16>, %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %step = arith.constant 16 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %f16 = tensor.empty() : tensor<16x16xf16>
    %f32 = tensor.empty() : tensor<16x16xf32>
    scf.for %i = %c0 to %c128 step %step : i32 {
      %product = arith.muli %i, %step : i32
      %offset = arith.index_cast %product : i32 to index
      %readView = memref.reinterpret_cast %gm to offset: [%offset], sizes: [16, 16], strides: [16, 1] : memref<2048xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
      %writeView = memref.reinterpret_cast %gm to offset: [%offset], sizes: [16, 16], strides: [16, 1] : memref<2048xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
      %local = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%readView : memref<16x16xf16, strided<[16, 1], offset: ?>>) outs(%local : memref<16x16xf16>)
      %loaded = bufferization.to_tensor %local : memref<16x16xf16>
      %old = hivm.hir.vcast ins(%loaded : tensor<16x16xf16>) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %prep = hivm.hir.vexp ins(%a : tensor<16x16xf16>) outs(%f16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%prep, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cubeBuffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>>)
      %cast = memref.memory_space_cast %cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube = bufferization.to_tensor %cast : memref<16x16xf32>
      %sum = hivm.hir.vadd ins(%cube, %old : tensor<16x16xf32>, tensor<16x16xf32>) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %updated = hivm.hir.vcast ins(%sum : tensor<16x16xf32>) outs(%f16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%updated : tensor<16x16xf16>) outs(%writeView : memref<16x16xf16, strided<[16, 1], offset: ?>>)
    }
    return
  }
}

// -----

// Dynamic row stride S: gap = 16*S and span = 15*S+16 elements.
// Without a bound on S, alias analysis cannot prove disjointness.
// CHECK-LABEL: func.func @dynamic_gm_tiles
// CHECK-NOT: scope.scope
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_gm_tiles(%gm: memref<?xf16>, %stride: i32, %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %step = arith.constant 16 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %f16 = tensor.empty() : tensor<16x16xf16>
    %f32 = tensor.empty() : tensor<16x16xf32>
    %s = arith.index_cast %stride : i32 to index
    scf.for %i = %c0 to %c128 step %step : i32 {
      %product = arith.muli %i, %stride : i32
      %offset = arith.index_cast %product : i32 to index
      %readView = memref.reinterpret_cast %gm to offset: [%offset], sizes: [16, 16], strides: [%s, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %writeView = memref.reinterpret_cast %gm to offset: [%offset], sizes: [16, 16], strides: [%s, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %local = memref.alloc() : memref<16x16xf16>
      // expected-warning@+1 {{using GM as intermediate buffer is unsupported}}
      hivm.hir.load ins(%readView : memref<16x16xf16, strided<[?, 1], offset: ?>>) outs(%local : memref<16x16xf16>)
      %loaded = bufferization.to_tensor %local : memref<16x16xf16>
      %old = hivm.hir.vcast ins(%loaded : tensor<16x16xf16>) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %prep = hivm.hir.vexp ins(%a : tensor<16x16xf16>) outs(%f16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%prep, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cubeBuffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>>)
      %cast = memref.memory_space_cast %cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube = bufferization.to_tensor %cast : memref<16x16xf32>
      %sum = hivm.hir.vadd ins(%cube, %old : tensor<16x16xf32>, tensor<16x16xf32>) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %updated = hivm.hir.vcast ins(%sum : tensor<16x16xf32>) outs(%f16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%updated : tensor<16x16xf16>) outs(%writeView : memref<16x16xf16, strided<[?, 1], offset: ?>>)
    }
    return
  }
}

// -----

// The same dynamic case with S >= 16 established before the loop.
// This proves gap >= span, so preload is enabled.
// CHECK-LABEL: func.func @dynamic_gm_tiles_with_assume
// CHECK: scope.scope
// CHECK: hivm.hir.load
// CHECK: hivm.preload_num = {{[1-9][0-9]*}}
// CHECK: scope.scope
// CHECK: hivm.hir.store
// CHECK: hivm.preload_num = 0
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_gm_tiles_with_assume(%gm: memref<?xf16>, %stride: i32, %a: tensor<16x16xf16>, %b: tensor<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %step = arith.constant 16 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %f16 = tensor.empty() : tensor<16x16xf16>
    %f32 = tensor.empty() : tensor<16x16xf32>
    %s = arith.index_cast %stride : i32 to index
    %validStride = arith.cmpi sge, %stride, %step : i32
    "llvm.intr.assume"(%validStride) : (i1) -> ()
    scf.for %i = %c0 to %c128 step %step : i32 {
      %product = arith.muli %i, %stride : i32
      %offset = arith.index_cast %product : i32 to index
      %readView = memref.reinterpret_cast %gm to offset: [%offset], sizes: [16, 16], strides: [%s, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %writeView = memref.reinterpret_cast %gm to offset: [%offset], sizes: [16, 16], strides: [%s, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %local = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%readView : memref<16x16xf16, strided<[?, 1], offset: ?>>) outs(%local : memref<16x16xf16>)
      %loaded = bufferization.to_tensor %local : memref<16x16xf16>
      %old = hivm.hir.vcast ins(%loaded : tensor<16x16xf16>) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %prep = hivm.hir.vexp ins(%a : tensor<16x16xf16>) outs(%f16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%prep, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cubeBuffer = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>>)
      %cast = memref.memory_space_cast %cubeBuffer : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube = bufferization.to_tensor %cast : memref<16x16xf32>
      %sum = hivm.hir.vadd ins(%cube, %old : tensor<16x16xf32>, tensor<16x16xf32>) outs(%f32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %updated = hivm.hir.vcast ins(%sum : tensor<16x16xf32>) outs(%f16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%updated : tensor<16x16xf16>) outs(%writeView : memref<16x16xf16, strided<[?, 1], offset: ?>>)
    }
    return
  }
}
