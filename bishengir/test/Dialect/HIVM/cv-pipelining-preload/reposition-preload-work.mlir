// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s
// RUN: sed -E 's/hivm.preload_num = [0-9]+/hivm.preload_num = 0/g; s/hivm.max_preload_num = [0-9]+/hivm.max_preload_num = 1/g' %s > %t
// RUN: bishengir-opt %t -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=ZERO --implicit-check-not=hivm.independent_preload_prefix

func.func @cube_touch(%arg0: memref<32x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIC>,
                no_inline} {
  return
}

// All-zero loops retain same-iteration scope grouping. The mask chain stays
// beside the cube-dependent multiply; do not add a hoisted prefix.
// ZERO-LABEL: func.func @hoist_mask_chain_before_cube_use
// ZERO: scope.scope
// ZERO: func.call @cube_touch
// ZERO: scope.return
// ZERO: scope.scope
// ZERO: hivm.hir.vmul
// ZERO: hivm.hir.vcast
// ZERO: hivm.hir.vcast
// ZERO: hivm.hir.vadd
// ZERO: hivm.hir.vsub
// ZERO: scope.return
// ZERO: } {{.*}}hivm.preload_num = 0

// The mask vcast chain does not use the cube tensor. With a third independent
// vector op (vadd) the chain meets the three-op hoist threshold and moves into
// a new scope before the scope that multiplies the cube result. The vsub that
// joins them stays.
// CHECK-LABEL: func.func @hoist_mask_chain_before_cube_use
// CHECK:         scope.scope
// CHECK:           func.call @cube_touch
// CHECK:         } {{{.*}}hivm.preload_num = 3
// CHECK:         %[[CUBE:.*]] = bufferization.to_tensor
// CHECK:         scope.scope
// CHECK:           hivm.hir.vcast {{.*}} ins(%arg1
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vadd
// CHECK:           scope.return
// CHECK:         } {{{.*}}hivm.independent_preload_prefix{{.*}}hivm.preload_num = 2
// CHECK:         scope.scope
// CHECK:           hivm.hir.vmul ins(%[[CUBE]]
// CHECK:           hivm.hir.vsub
// CHECK-NOT:       hivm.hir.vcast
// CHECK:         } {{{.*}}hivm.preload_num = 2 : i32, no_inline}
func.func @hoist_mask_chain_before_cube_use(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>,
    %arg1: tensor<32x128xi8>, %scale: f32)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    %cube = bufferization.to_tensor %arg0 restrict writable :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %empty = tensor.empty() : tensor<32x128xf32>
      %empty16 = tensor.empty() : tensor<32x128xf16>
      %scaled = hivm.hir.vmul ins(%cube, %scale : tensor<32x128xf32>, f32)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      %cast16 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                                hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%arg1 : tensor<32x128xi8>) outs(%empty16 : tensor<32x128xf16>)
          -> tensor<32x128xf16>
      %mask = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                              hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%cast16 : tensor<32x128xf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      %bump = hivm.hir.vadd ins(%mask, %mask : tensor<32x128xf32>, tensor<32x128xf32>)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      %sub = hivm.hir.vsub ins(%scaled, %mask : tensor<32x128xf32>, tensor<32x128xf32>)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
  }
  return
}

// A non-cube scope between the VECTOR scope and the CUBE scope does not
// block the split. The cube that matters is the first subsequent CUBE
// scope, not the immediate next sibling. With three independent vector ops
// the mask chain hoists across the intervening scope.
// CHECK-LABEL: func.func @hoist_across_intervening_vector
// CHECK:         scope.scope
// CHECK:           hivm.hir.vcast {{.*}} ins(%arg1
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vadd
// CHECK:         } {{{.*}}hivm.independent_preload_prefix{{.*}}hivm.preload_num = 2
// CHECK:         hivm.hir.vmul ins(%{{.*}} : tensor<32x128xf32>, f32)
// CHECK:         } {{{.*}}hivm.preload_num = 2 : i32, no_inline}
// CHECK:         } {{{.*}}hivm.preload_num = 4
// CHECK:         } {{{.*}}hivm.preload_num = 1
func.func @hoist_across_intervening_vector(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>,
    %arg1: tensor<32x128xi8>, %scale: f32)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    %cube = bufferization.to_tensor %arg0 restrict writable :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %empty = tensor.empty() : tensor<32x128xf32>
      %empty16 = tensor.empty() : tensor<32x128xf16>
      %scaled = hivm.hir.vmul ins(%cube, %scale : tensor<32x128xf32>, f32)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      %cast16 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                                hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%arg1 : tensor<32x128xi8>) outs(%empty16 : tensor<32x128xf16>)
          -> tensor<32x128xf16>
      %mask = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                              hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%cast16 : tensor<32x128xf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      %bump = hivm.hir.vadd ins(%mask, %mask : tensor<32x128xf32>, tensor<32x128xf32>)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      %sub = hivm.hir.vsub ins(%scaled, %mask : tensor<32x128xf32>, tensor<32x128xf32>)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 4 : i32, no_inline}
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
  }
  return
}

// The same mask chain is not split when no CUBE scope follows. The first
// subsequent cube is what gates the hoist.
// CHECK-LABEL: func.func @no_hoist_without_following_cube
// CHECK-NOT:     hivm.independent_preload_prefix
// CHECK:         hivm.hir.vmul
// CHECK:         hivm.hir.vcast
// CHECK:         hivm.hir.vsub
func.func @no_hoist_without_following_cube(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>,
    %arg1: tensor<32x128xi8>, %scale: f32)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    %cube = bufferization.to_tensor %arg0 restrict writable :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %empty = tensor.empty() : tensor<32x128xf32>
      %empty16 = tensor.empty() : tensor<32x128xf16>
      %scaled = hivm.hir.vmul ins(%cube, %scale : tensor<32x128xf32>, f32)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      %cast16 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                                hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%arg1 : tensor<32x128xi8>) outs(%empty16 : tensor<32x128xf16>)
          -> tensor<32x128xf16>
      %mask = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                              hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%cast16 : tensor<32x128xf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      %sub = hivm.hir.vsub ins(%scaled, %mask : tensor<32x128xf32>, tensor<32x128xf32>)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
  }
  return
}

// A bare destination read only by an op that stays behind must stay with it.
// Hoisting it would export it through scope.return, and EnableStrideAlign
// needs the defining tensor.empty / alloc to reshape it. %dst is only written
// by the vtranspose that depends on the cube result, so it stays; %empty is
// also written by the hoisted vcast, so it still moves. The vadd gives the
// mask chain the third vector op needed to meet the hoist threshold.
// CHECK-LABEL: func.func @bare_destination_stays_with_consumer
// CHECK:         scope.scope : () -> (tensor<32x128xf32>, tensor<32x128xf32>) {
// CHECK-NOT:       tensor.empty() : tensor<128x32xf32>
// CHECK:           hivm.hir.vcast {{.*}} ins(%arg1
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vadd
// CHECK:         } {{{.*}}hivm.independent_preload_prefix{{.*}}hivm.preload_num = 2
// CHECK:         scope.scope
// CHECK:           tensor.empty() : tensor<128x32xf32>
// CHECK:           hivm.hir.vsub
// CHECK:           hivm.hir.vtranspose
// CHECK:         } {{{.*}}hivm.preload_num = 2 : i32, no_inline}
func.func @bare_destination_stays_with_consumer(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>,
    %arg1: tensor<32x128xi8>, %scale: f32)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    %cube = bufferization.to_tensor %arg0 restrict writable :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %empty = tensor.empty() : tensor<32x128xf32>
      %empty16 = tensor.empty() : tensor<32x128xf16>
      %dst = tensor.empty() : tensor<128x32xf32>
      %cast16 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                                hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%arg1 : tensor<32x128xi8>) outs(%empty16 : tensor<32x128xf16>)
          -> tensor<32x128xf16>
      %mask = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                              hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%cast16 : tensor<32x128xf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      %bump = hivm.hir.vadd ins(%mask, %mask : tensor<32x128xf32>, tensor<32x128xf32>)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      %sub = hivm.hir.vsub ins(%cube, %mask : tensor<32x128xf32>, tensor<32x128xf32>)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      %t = hivm.hir.vtranspose ins(%sub : tensor<32x128xf32>)
          outs(%dst : tensor<128x32xf32>) permutation = [1, 0] -> tensor<128x32xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
  }
  return
}

// The affine index cannot be exported (CreatePreload cannot rematerialize
// affine.apply), so it and its dependents (%local_view, %next, %mask_slice)
// stay. That drops %local too (memref export), which in turn drops the
// to_tensor. The surviving independent set is only two vcasts, below the
// three-op hoist threshold, so the entire VECTOR scope stays intact.
// CHECK-LABEL: func.func @keep_affine_index_with_consumer
// CHECK-NOT:     hivm.independent_preload_prefix
// CHECK:         scope.scope
// CHECK:           func.call @cube_touch
// CHECK:         } {{{.*}}hivm.preload_num = 3
// CHECK:         bufferization.to_tensor
// CHECK:         scope.scope
// CHECK:           memref.alloc
// CHECK:           bufferization.to_tensor
// CHECK:           affine.apply
// CHECK:           memref.subview
// CHECK:           hivm.hir.vadd
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vcast
// CHECK:           arith.addi
// CHECK:           tensor.extract_slice
// CHECK:           tensor.extract_slice
// CHECK:           hivm.hir.vmul
// CHECK:         } {{{.*}}hivm.preload_num = 2 : i32, no_inline}
#map = affine_map<()[s0] -> (s0)>
func.func @keep_affine_index_with_consumer(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>,
    %arg1: tensor<32x128xi8>, %scale: f32)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    %cube = bufferization.to_tensor %arg0 restrict writable :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %empty = tensor.empty() : tensor<32x128xf32>
      %empty16 = tensor.empty() : tensor<32x128xf16>
      // This view's affine offset keeps it in the consumer. Exporting the
      // allocation from the hoisted prefix would break CreatePreload.
      %local = memref.alloc() : memref<32x128xf32>
      %local_tensor = bufferization.to_tensor %local restrict writable : memref<32x128xf32>
      %idx = affine.apply #map()[%i]
      %local_view = memref.subview %local[%idx, 0] [1, 128] [1, 1] :
          memref<32x128xf32> to memref<1x128xf32, strided<[128, 1], offset: ?>>
      %local_sum = hivm.hir.vadd ins(%cube, %local_tensor : tensor<32x128xf32>, tensor<32x128xf32>)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      %cast16 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                                hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%arg1 : tensor<32x128xi8>) outs(%empty16 : tensor<32x128xf16>)
          -> tensor<32x128xf16>
      %mask = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                              hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%cast16 : tensor<32x128xf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      // The entire use chain must stay when %idx cannot be exported.
      %next = arith.addi %idx, %c1 : index
      %mask_slice = tensor.extract_slice %mask[%next, 0] [1, 128] [1, 1] :
          tensor<32x128xf32> to tensor<1x128xf32>
      %slice = tensor.extract_slice %cube[%idx, 0] [1, 128] [1, 1] :
          tensor<32x128xf32> to tensor<1x128xf32>
      %scaled = hivm.hir.vmul ins(%cube, %scale : tensor<32x128xf32>, f32)
          outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      func.call @cube_touch(%arg0) {no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>,
       hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
  }
  return
}

// Sink an unregistered pure mask across an unrelated CUBE scope. Retain it
// when preload is disabled, every stage is zero, or an input is overwritten.
// PURE-LABEL: func.func @sink_pure_tensor_mask
// PURE: scope.scope
// PURE-NOT: hivm.hir.vcmp
// PURE: memref.copy
// PURE: scope.scope
// PURE: %[[MASK:.*]] = hivm.hir.vcmp
// PURE-NEXT: %{{.*}} = hivm.hir.vsel ins(%[[MASK]],
// KEEP-LABEL: func.func @sink_pure_tensor_mask
// KEEP: %[[SCOPE:.*]] = scope.scope
// KEEP: hivm.hir.vcmp
// KEEP: scope.return
// KEEP: memref.copy
// KEEP-NOT: hivm.hir.vcmp
// KEEP: hivm.hir.vsel ins(%[[SCOPE]],
func.func @sink_pure_tensor_mask(%input: memref<16x16xf32>, %other: memref<16x16xf32>, %scores: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %scratch = memref.alloc() : memref<16x16xf32>
  %input_tensor = bufferization.to_tensor %input restrict writable : memref<16x16xf32>
  %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %scores) -> tensor<16x16xf32> {
    %mask = scope.scope : () -> tensor<16x16xi1> {
      %empty = tensor.empty() : tensor<16x16xi1>
      %cmp = hivm.hir.vcmp ins(%input_tensor, %zero : tensor<16x16xf32>, f32) outs(%empty : tensor<16x16xi1>) -> tensor<16x16xi1>
      scope.return %cmp : tensor<16x16xi1>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      memref.copy %other, %scratch : memref<16x16xf32> to memref<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
    %next = scope.scope : () -> tensor<16x16xf32> {
      %empty = tensor.empty() : tensor<16x16xf32>
      %selected = hivm.hir.vsel ins(%mask, %acc, %zero : tensor<16x16xi1>, tensor<16x16xf32>, f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return %selected : tensor<16x16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %next : tensor<16x16xf32>
  }
  return %out : tensor<16x16xf32>
}
