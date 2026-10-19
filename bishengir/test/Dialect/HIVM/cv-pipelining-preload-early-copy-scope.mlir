// RUN: bishengir-opt %s -cv-pipelining="pipeline-mode=skew enable-preload=true bypass-shape-registry=true" | FileCheck %s --check-prefix=EARLY
// RUN: bishengir-opt %s -cv-pipelining="pipeline-mode=skew enable-preload=true bypass-shape-registry=true" -inline-scope | FileCheck %s --check-prefix=EARLY
// RUN: bishengir-opt %s -cv-pipelining="pipeline-mode=unroll enable-preload=false bypass-shape-registry=true" | FileCheck %s --check-prefix=UNROLL
// RUN: bishengir-opt %s -cv-pipelining="pipeline-mode=skew enable-preload=true bypass-shape-registry=true" -hivm-mark-multi-buffer="enable-auto=false enable-preload=true" | FileCheck %s --check-prefix=BUFFER

// The functions retain CUBE(2), VECTOR(1), CUBE(0). Only the first has a
// contiguous early-copy chain with one live-out and a new Cube input later.
// The no_inline inner scope must survive the ordinary scope inliner; it is
// not another preload stage and must not hide the original TCB buffer stages.
// EARLY-LABEL: func.func @separate_cube_input
// EARLY: hivm.preload_num = 2 : i32
// EARLY: %[[CAST_INIT:.*]] = tensor.empty() : tensor<16x16xf16>
// EARLY: %[[TRANSPOSE_INIT:.*]] = tensor.empty() : tensor<16x16xf16>
// EARLY: %[[P:.*]] = scope.scope : () -> tensor<16x16xf32> {
// EARLY-NOT: tensor.empty
// EARLY: %[[EXP:.*]] = hivm.hir.vexp
// EARLY-NOT: tensor.empty
// EARLY: hivm.hir.vcast {{.*}} outs(%[[CAST_INIT]]
// EARLY-NOT: tensor.empty
// EARLY: hivm.hir.vtranspose {{.*}} outs(%[[TRANSPOSE_INIT]]
// EARLY-NOT: tensor.empty
// EARLY: hivm.hir.copy
// EARLY-NEXT: scope.return %[[EXP]] : tensor<16x16xf32>
// EARLY-NEXT: } {no_inline}
// EARLY: hivm.hir.vmul ins(%[[P]],
// EARLY: hivm.hir.copy
// EARLY: hivm.preload_num = 1 : i32
// EARLY: hivm.preload_num = 0 : i32
// EARLY-LABEL: func.func @same_cube_input
// EARLY-NOT: = scope.scope
// EARLY: hivm.preload_num = 2 : i32
// EARLY-NOT: = scope.scope
// EARLY: hivm.hir.vexp
// EARLY: hivm.hir.copy
// EARLY: hivm.hir.vmul
// EARLY: hivm.hir.copy
// EARLY: hivm.preload_num = 1 : i32
// EARLY-NOT: = scope.scope
// EARLY: hivm.preload_num = 0 : i32
// EARLY-NOT: = scope.scope

// BUFFER-LABEL: func.func @separate_cube_input
// BUFFER-COUNT-4: hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32
// BUFFER: scope.scope
// BUFFER-LABEL: func.func @same_cube_input

// UNROLL-LABEL: func.func @separate_cube_input
// UNROLL-NOT: scope.scope
// UNROLL-LABEL: func.func @same_cube_input
// UNROLL-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @separate_cube_input(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %scaled = hivm.hir.vmul ins(%qk, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %transpose_empty = tensor.empty() : tensor<16x16xf16>
      %transposed = hivm.hir.vtranspose ins(%p16 : tensor<16x16xf16>) outs(%transpose_empty : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %transposed [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %dp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
    }
    return
  }

  func.func @same_cube_input(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%qk, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %qk : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
    }
    return
  }

  // Two SSA views of one incoming TCB are not independent Cube inputs. The
  // views live outside the loop so the CV output rewriter still has one
  // to_tensor per loop-local allocation.
  // EARLY-LABEL: func.func @same_cube_buffer_alias
  // EARLY: %[[ALIAS:.*]] = memref.{{(subview|cast)}}
  // EARLY: bufferization.to_tensor %[[ALIAS]]
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 2 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.vexp
  // EARLY: hivm.hir.copy
  // EARLY: hivm.hir.vmul
  // EARLY: hivm.hir.copy
  // EARLY: hivm.preload_num = 1 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 0 : i32
  // EARLY-NOT: = scope.scope
  // UNROLL-LABEL: func.func @same_cube_buffer_alias
  // UNROLL-NOT: scope.scope
  func.func @same_cube_buffer_alias(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32, %qk_initial: tensor<1x1x16x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %qk_external = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %qk_external {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%qk_initial : tensor<1x1x16x16xf32>) outs(%qk_external : memref<16x16xf32, #hivm.address_space<ub>>)
    %external_tensor = bufferization.to_tensor %qk_external : memref<16x16xf32, #hivm.address_space<ub>>
    %external_view = memref.subview %qk_external[0, 0] [16, 16] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %external_alias_tensor = bufferization.to_tensor %external_view : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%external_tensor, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %external_alias_tensor : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
    }
    return
  }

  // Do not move a producer DAG across interleaved independent tensor work.
  // EARLY-LABEL: func.func @noncontiguous_producer_chain
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 2 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.vexp
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.vexp
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.copy
  // EARLY: hivm.hir.vmul
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 1 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 0 : i32
  // EARLY-NOT: = scope.scope
  // UNROLL-LABEL: func.func @noncontiguous_producer_chain
  // UNROLL-NOT: scope.scope
  func.func @noncontiguous_producer_chain(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%qk, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %unrelated = hivm.hir.vexp ins(%dp : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %unrelated : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
    }
    return
  }

  // The publication boundary must not move computation past a GM store.
  // EARLY-LABEL: func.func @effect_between_producer_and_copy
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 2 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.vexp
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.store
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.copy
  // EARLY: hivm.hir.vmul
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 1 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 0 : i32
  // EARLY-NOT: = scope.scope
  // UNROLL-LABEL: func.func @effect_between_producer_and_copy
  // UNROLL-NOT: scope.scope
  func.func @effect_between_producer_and_copy(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32, %debug: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%qk, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      hivm.hir.store ins(%p : tensor<16x16xf32>) outs(%debug : memref<16x16xf32, #hivm.address_space<gm>>)
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %dp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
    }
    return
  }

  // Returning P and another full tensor can increase cross-VF UB use.
  // EARLY-LABEL: func.func @multiple_producer_liveouts
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 2 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: %[[SCALED:.*]] = hivm.hir.vmul
  // EARLY: %[[PROB:.*]] = hivm.hir.vexp
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.copy
  // EARLY: hivm.hir.vmul ins(%[[PROB]],
  // EARLY: hivm.hir.vadd ins({{.*}}, %[[SCALED]]
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 1 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 0 : i32
  // EARLY-NOT: = scope.scope
  // UNROLL-LABEL: func.func @multiple_producer_liveouts
  // UNROLL-NOT: scope.scope
  func.func @multiple_producer_liveouts(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%qk, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds_partial = hivm.hir.vmul ins(%p, %dp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds = hivm.hir.vadd ins(%ds_partial, %scaled : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
    }
    return
  }

  // Splitting cannot remove a Cube dependency already needed by P.
  // EARLY-LABEL: func.func @producer_already_needs_second_cube_input
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 2 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.vmul
  // EARLY: hivm.hir.vexp
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.copy
  // EARLY: hivm.hir.vmul
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 1 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 0 : i32
  // EARLY-NOT: = scope.scope
  // UNROLL-LABEL: func.func @producer_already_needs_second_cube_input
  // UNROLL-NOT: scope.scope
  func.func @producer_already_needs_second_cube_input(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%qk, %dp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %dp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
    }
    return
  }

  // There is no downstream consumer of P to preserve a boundary for.
  // EARLY-LABEL: func.func @no_producer_liveout
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 2 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.vexp
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.hir.copy
  // EARLY: hivm.hir.vmul
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 1 : i32
  // EARLY-NOT: = scope.scope
  // EARLY: hivm.preload_num = 0 : i32
  // EARLY-NOT: = scope.scope
  // UNROLL-LABEL: func.func @no_producer_liveout
  // UNROLL-NOT: scope.scope
  func.func @no_producer_liveout(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %bound step %c1 : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%qk, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%qk, %dp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
    }
    return
  }

  // A stage result must retain its existing preload-buffer lifetime tracking.
  // Do not hide its producer behind an additional yielded scope result.
  // EARLY-LABEL: func.func @yielded_producer_liveout
  // EARLY: hivm.preload_num = 2 : i32
  // EARLY: %[[STAGE_RESULT:.*]] = scope.scope : () -> tensor<16x16xf32> {
  // EARLY-NOT: scope.scope
  // EARLY: %[[YIELDED_P:.*]] = hivm.hir.vexp
  // EARLY-NOT: scope.scope
  // EARLY: hivm.hir.copy
  // EARLY: hivm.hir.vmul ins(%[[YIELDED_P]],
  // EARLY-NOT: scope.scope
  // EARLY: scope.return %[[YIELDED_P]] : tensor<16x16xf32>
  // EARLY-NEXT: } {{.*}}hivm.preload_num = 1 : i32
  // EARLY: hivm.preload_num = 0 : i32
  // EARLY: scf.yield %[[STAGE_RESULT]] : tensor<16x16xf32>
  // EARLY-NOT: scope.scope
  // UNROLL-LABEL: func.func @yielded_producer_liveout
  // UNROLL-NOT: scope.scope
  func.func @yielded_producer_liveout(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32, %initial_p: tensor<16x16xf32>, %outp: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %last_p = scf.for %i = %c0 to %bound step %c1 iter_args(%carry = %initial_p) -> tensor<16x16xf32> : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%qk, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %dp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
      scf.yield %p : tensor<16x16xf32>
    }
    hivm.hir.store ins(%last_p : tensor<16x16xf32>) outs(%outp : memref<16x16xf32, #hivm.address_space<gm>>)
    return
  }

  // A tensor view can hide the path from the producer to the stage result.
  // Restrict the boundary to stages without tensor results altogether.
  // EARLY-LABEL: func.func @yielded_producer_view
  // EARLY: hivm.preload_num = 2 : i32
  // EARLY: %[[VIEW_STAGE_RESULT:.*]] = scope.scope : () -> tensor<1x1x16x16xf32> {
  // EARLY-NOT: scope.scope
  // EARLY: %[[VIEW_P:.*]] = hivm.hir.vexp
  // EARLY-NOT: scope.scope
  // EARLY: hivm.hir.copy
  // EARLY: hivm.hir.vmul ins(%[[VIEW_P]],
  // EARLY-NOT: scope.scope
  // EARLY: %[[YIELDED_VIEW:.*]] = tensor.expand_shape %[[VIEW_P]]
  // EARLY-NEXT: scope.return %[[YIELDED_VIEW]] : tensor<1x1x16x16xf32>
  // EARLY-NEXT: } {{.*}}hivm.preload_num = 1 : i32
  // EARLY: hivm.preload_num = 0 : i32
  // EARLY: scf.yield %[[VIEW_STAGE_RESULT]] : tensor<1x1x16x16xf32>
  // EARLY-NOT: scope.scope
  // UNROLL-LABEL: func.func @yielded_producer_view
  // UNROLL-NOT: scope.scope
  func.func @yielded_producer_view(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x16x16xf16>, %rhs2: tensor<1x1x16x16xf16>, %outv: memref<16x16xf32>, %outk: memref<16x16xf32>, %bound: i32, %scale: f32, %initial_p: tensor<1x1x16x16xf32>, %outp: memref<1x1x16x16xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %last_p = scf.for %i = %c0 to %bound step %c1 iter_args(%carry = %initial_p) -> tensor<1x1x16x16xf32> : i32 {
      %qkbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %dpbuf = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %pbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dsbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dotqk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotqk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotqk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %qkbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotqk : tensor<1x1x16x16xf32>) outs(%qkbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %dotdp_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dotdp = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dotdp_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      annotation.mark %dpbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dotdp : tensor<1x1x16x16xf32>) outs(%dpbuf : memref<16x16xf32, #hivm.address_space<ub>>)
      %qk = bufferization.to_tensor %qkbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %dp = bufferization.to_tensor %dpbuf restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
      %empty32 = tensor.empty() : tensor<16x16xf32>
      %empty16 = tensor.empty() : tensor<16x16xf16>
      %scaled = hivm.hir.vmul ins(%qk, %scale : tensor<16x16xf32>, f32) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p = hivm.hir.vexp ins(%scaled : tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %p16 = hivm.hir.vcast ins(%p : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %pnz = tensor.expand_shape %p16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %pbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%pnz : tensor<1x1x16x16xf16>) outs(%pbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %ds = hivm.hir.vmul ins(%p, %dp : tensor<16x16xf32>, tensor<16x16xf32>) outs(%empty32 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ds16 = hivm.hir.vcast ins(%ds : tensor<16x16xf32>) outs(%empty16 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %dsnz = tensor.expand_shape %ds16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
      annotation.mark %dsbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%dsnz : tensor<1x1x16x16xf16>) outs(%dsbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %pcube = bufferization.to_tensor %pbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dscube = bufferization.to_tensor %dsbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      %dv_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dv = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%pcube, %rhs, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dv_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dv : tensor<1x1x16x16xf32>) outs(%outv : memref<16x16xf32>)
      %dk_init = tensor.empty() : tensor<1x1x16x16xf32>
      %dk = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%dscube, %rhs2, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dk_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dk : tensor<1x1x16x16xf32>) outs(%outk : memref<16x16xf32>)
      %p_view = tensor.expand_shape %p [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf32> into tensor<1x1x16x16xf32>
      scf.yield %p_view : tensor<1x1x16x16xf32>
    }
    hivm.hir.store ins(%last_p : tensor<1x1x16x16xf32>) outs(%outp : memref<1x1x16x16xf32, #hivm.address_space<gm>>)
    return
  }
}
