// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work="bypass-shape-registry=true" -split-input-file | FileCheck %s

func.func @vf_cast(
    %arg0: memref<32x128xbf16, #hivm.address_space<ub>>,
    %arg1: memref<32x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  return
}

func.func @vf_consume(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  return
}

// No following CUBE: a returned dest stays in the producer even when the
// next VECTOR is its only user.
// CHECK-LABEL: func.func @keep_returned_dest_tensor_without_following_cube
// CHECK:         scope.scope
// CHECK:           %[[SRC:.*]] = bufferization.to_tensor
// CHECK:           %[[EMPTY:.*]] = tensor.empty
// CHECK:           %[[DST:.*]] = hivm.hir.vcast {{.*}} ins(%[[SRC]] {{.*}} outs(%[[EMPTY]]
// CHECK:           scope.return %[[DST]]
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vcast
// CHECK:           hivm.hir.vexp ins(%{{.*}} :
func.func @keep_returned_dest_tensor_without_following_cube(%arg0: memref<32x128xbf16, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<32x128xf32> {
      %src = bufferization.to_tensor %arg0 restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%src : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// No following CUBE: shared and exclusive writers both stay in the
// producer. The later VECTOR only consumes the returned tensor.
// CHECK-LABEL: func.func @keep_dest_without_following_cube
// CHECK:         scope.scope
// CHECK:           bufferization.to_tensor
// CHECK:           %[[SHARED:.*]] = tensor.empty
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vexp {{.*}} outs(%[[SHARED]]
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vcast
// CHECK:           hivm.hir.vexp ins(%{{.*}} :
func.func @keep_dest_without_following_cube(
    %arg0: memref<32x128xbf16, #hivm.address_space<ub>>,
    %arg1: memref<32x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<32x128xf32> {
      %src = bufferization.to_tensor %arg0 restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %keepSrc = bufferization.to_tensor %arg1 restrict writable : memref<32x128xf32, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%src : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      %keep = hivm.hir.vexp ins(%keepSrc : tensor<32x128xf32>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// No following CUBE: the Q load stays with the producer VECTOR.
// CHECK-LABEL: func.func @keep_q_load_cluster_without_following_cube
// CHECK:         memref.alloc
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           bufferization.to_tensor
// CHECK:           hivm.hir.vcast
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK:           hivm.hir.vexp
func.func @keep_q_load_cluster_without_following_cube(
    %src: memref<32x128xbf16, #hivm.address_space<gm>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xbf16, #hivm.address_space<ub>>
    %prod = scope.scope : () -> tensor<32x128xf32> {
      hivm.hir.load ins(%src : memref<32x128xbf16, #hivm.address_space<gm>>)
                    outs(%buf : memref<32x128xbf16, #hivm.address_space<ub>>)
      %q = bufferization.to_tensor %buf restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%q : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Unused CUBE between producer and consumer: the whole cluster including
// MTE2 load follows the VECTOR consumer. Same rule as copy-chain sink.
// CHECK-LABEL: func.func @sink_load_cluster_across_unused_cube
// CHECK:         memref.alloc
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           bufferization.to_tensor
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vexp
func.func @sink_load_cluster_across_unused_cube(
    %src: memref<32x128xbf16, #hivm.address_space<gm>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xbf16, #hivm.address_space<ub>>
    %prod = scope.scope : () -> tensor<32x128xf32> {
      hivm.hir.load ins(%src : memref<32x128xbf16, #hivm.address_space<gm>>)
                    outs(%buf : memref<32x128xbf16, #hivm.address_space<ub>>)
      %q = bufferization.to_tensor %buf restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%q : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// 0216 Q: load writes a subview of a producer-local alloc (mb=2 +
// preload_local). Next CUBE does not read it; the unique VECTOR user is
// after that cube. Move load+alloc+to_tensor past the unused cube and
// drop preload_local. Same rule as copy-chain unused-mmad sink.
// CHECK-LABEL: func.func @sink_q_load_via_subview_across_unused_cube
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           %[[QBUF:.*]] = memref.alloc
// CHECK:           annotation.mark %[[QBUF]] {hivm.multi_buffer = 2 : i32} :
// CHECK:           hivm.hir.load
// CHECK:           bufferization.to_tensor
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vexp
func.func @sink_q_load_via_subview_across_unused_cube(
    %src: memref<32x128xbf16, #hivm.address_space<gm>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<32x128xbf16> {
      %buf = memref.alloc() : memref<32x128xbf16, #hivm.address_space<ub>>
      annotation.mark %buf {hivm.multi_buffer = 2 : i32,
                            hivm.preload_local_buffer = 1 : i32} :
          memref<32x128xbf16, #hivm.address_space<ub>>
      %sv = memref.subview %buf[0, 0] [32, 128] [1, 1] :
          memref<32x128xbf16, #hivm.address_space<ub>> to
          memref<32x128xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%src : memref<32x128xbf16, #hivm.address_space<gm>>)
                    outs(%sv : memref<32x128xbf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %q = bufferization.to_tensor %buf restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      scope.return %q : tensor<32x128xbf16>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %empty = tensor.empty() : tensor<32x128xf32>
      %c = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                           hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%prod : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%c : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Leftover dest-sink shape: producer already returns the alloc memref
// and the consumer only does to_tensor. Still move the load past the
// unused cube.
// CHECK-LABEL: func.func @sink_returned_memref_load_across_unused_cube
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           bufferization.to_tensor
// CHECK:           hivm.hir.vcast
func.func @sink_returned_memref_load_across_unused_cube(
    %src: memref<32x128xbf16, #hivm.address_space<gm>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> memref<32x128xbf16, #hivm.address_space<ub>> {
      %buf = memref.alloc() : memref<32x128xbf16, #hivm.address_space<ub>>
      annotation.mark %buf {hivm.multi_buffer = 2 : i32,
                            hivm.preload_local_buffer = 1 : i32} :
          memref<32x128xbf16, #hivm.address_space<ub>>
      %sv = memref.subview %buf[0, 0] [32, 128] [1, 1] :
          memref<32x128xbf16, #hivm.address_space<ub>> to
          memref<32x128xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%src : memref<32x128xbf16, #hivm.address_space<gm>>)
                    outs(%sv : memref<32x128xbf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      scope.return %buf : memref<32x128xbf16, #hivm.address_space<ub>>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %q = bufferization.to_tensor %prod restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %c = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                           hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%q : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// No CUBE after the producer: the gate chain stays, including across the
// dummy VECTOR scope that does not use the returned tensor.
// CHECK-LABEL: func.func @keep_gate_chain_without_following_cube
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           bufferization.to_tensor
// CHECK:           hivm.hir.vmul
// CHECK:           hivm.hir.vexp
// CHECK:           tensor.expand_shape
// CHECK:           hivm.hir.vmul
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:           memref.load
// CHECK:         } {{{.*}}preload_num = 0
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK:           hivm.hir.vadd
// CHECK:         } {{{.*}}preload_num = 0
func.func @keep_gate_chain_without_following_cube(
    %gate: memref<128xf32, #hivm.address_space<gm>>,
    %h: tensor<128x128xf32>) -> tensor<128x128xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 1.000000e+00 : f32
  %res = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %h) -> (tensor<128x128xf32>) {
    %buf = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %prod = scope.scope : () -> tensor<128x128xf32> {
      hivm.hir.load ins(%gate : memref<128xf32, #hivm.address_space<gm>>)
                    outs(%buf : memref<128xf32, #hivm.address_space<ub>>)
      %g = bufferization.to_tensor %buf restrict writable : memref<128xf32, #hivm.address_space<ub>>
      %e0 = tensor.empty() : tensor<128xf32>
      %m = hivm.hir.vmul ins(%g, %cst : tensor<128xf32>, f32) outs(%e0 : tensor<128xf32>) -> tensor<128xf32>
      %e1 = tensor.empty() : tensor<128xf32>
      %ex = hivm.hir.vexp ins(%m : tensor<128xf32>) outs(%e1 : tensor<128xf32>) -> tensor<128xf32>
      %expanded = tensor.expand_shape %ex [[0, 1]] output_shape [128, 1]
          : tensor<128xf32> into tensor<128x1xf32>
      %eh = tensor.empty() : tensor<128x128xf32>
      %hs = hivm.hir.vmul ins(%h, %expanded : tensor<128x128xf32>, tensor<128x1xf32>)
          outs(%eh : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
      scope.return %hs : tensor<128x128xf32>
    } {hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %dummy = memref.load %gate[%c0] : memref<128xf32, #hivm.address_space<gm>>
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
    %last = scope.scope : () -> tensor<128x128xf32> {
      %out = tensor.empty() : tensor<128x128xf32>
      %s = hivm.hir.vadd ins(%iter, %prod : tensor<128x128xf32>, tensor<128x128xf32>)
          outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
      scope.return %s : tensor<128x128xf32>
    } {hivm.preload_num = 0 : i32, no_inline}
    scf.yield %last : tensor<128x128xf32>
  }
  return %res : tensor<128x128xf32>
}

// Test 5: multi_buffer=1 + preload_local=1 alloc used by one scope is
// sunk into that scope and its mark is deleted. multi_buffer>1 stays
// at loop scope so slots can still rotate.
// CHECK-LABEL: func.func @sink_single_slot_preload_alloc
// CHECK-NOT:     memref.alloc
// CHECK:         scope.scope
// CHECK:           %[[BUF:.*]] = memref.alloc
// CHECK-NOT:       annotation.mark %[[BUF]]
// CHECK:           call @vf_consume(%[[BUF]])
func.func @sink_single_slot_preload_alloc()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 1 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%buf) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// CHECK-LABEL: func.func @keep_rotated_preload_alloc_in_loop
// CHECK:         %[[BUF:.*]] = memref.alloc
// CHECK:         annotation.mark %[[BUF]] {{.*}}multi_buffer = 2{{.*}}preload_local_buffer
// CHECK:         scope.scope
func.func @keep_rotated_preload_alloc_in_loop()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 2 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%buf) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Test 6: TCB + mb=1 + preload_local still sinks. The TCB mark is kept;
// only the single-slot preload marks are deleted.
// CHECK-LABEL: func.func @sink_tcb_single_slot_preload_alloc
// CHECK-NOT:     memref.alloc
// CHECK:         scope.scope
// CHECK:           %[[BUF:.*]] = memref.alloc
// CHECK:           annotation.mark %[[BUF]] {{.*}}tightly_coupled_buffer
// CHECK-NOT:       annotation.mark %[[BUF]] {{.*}}preload_local_buffer
// CHECK:           call @vf_consume(%[[BUF]])
func.func @sink_tcb_single_slot_preload_alloc()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {
      effects = ["write", "read"],
      hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>,
      hivm.tiling_dim = 0 : index, tiledAlloc
    } : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 1 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%buf) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// A loop-level subview of a single-slot preload alloc follows the alloc
// into the unique consumer scope.
// CHECK-LABEL: func.func @sink_preload_alloc_with_subview
// CHECK-NOT:     memref.alloc
// CHECK:         scope.scope
// CHECK:           %[[BUF:.*]] = memref.alloc
// CHECK-NOT:       annotation.mark %[[BUF]]
// CHECK:           %[[SV:.*]] = memref.subview %[[BUF]]
// CHECK:           call @vf_consume(%[[SV]])
func.func @sink_preload_alloc_with_subview()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    %sv = memref.subview %buf[0, 0] [32, 128] [1, 1] :
        memref<32x128xf32, #hivm.address_space<ub>> to
        memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 1 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%sv) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// An unused mmad between load→vcast→copy and the mmad that reads the
// cbuf: move the whole chain (including MTE2 load) into the VECTOR
// immediately before the consumer. The loop-level to_tensor follows.
// CHECK-LABEL: func.func @sink_copy_chain_past_unused_mmad
// CHECK:         memref.alloc
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK-NOT:       hivm.hir.copy
// CHECK:           scope.return
// CHECK:         } {{{.*}}VECTOR
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           bufferization.to_tensor
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.copy
// CHECK:         } {{{.*}}VECTOR
// CHECK:         %[[W:.*]] = bufferization.to_tensor
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1 {{.*}}ins(%[[W]]
func.func @sink_copy_chain_past_unused_mmad(
    %src: memref<16x16xbf16, #hivm.address_space<gm>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %ub = memref.alloc() : memref<16x16xbf16, #hivm.address_space<ub>>
    %cbuf = memref.alloc() : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      hivm.hir.load ins(%src : memref<16x16xbf16, #hivm.address_space<gm>>)
                    outs(%ub : memref<16x16xbf16, #hivm.address_space<ub>>)
      %t = bufferization.to_tensor %ub restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      %emptyF = tensor.empty() : tensor<16x16xf32>
      %cast = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                              hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%t : tensor<16x16xbf16>) outs(%emptyF : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      %emptyB = tensor.empty() : tensor<16x16xbf16>
      %back = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                              hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%cast : tensor<16x16xf32>) outs(%emptyB : tensor<16x16xbf16>)
          -> tensor<16x16xbf16>
      hivm.hir.copy ins(%back : tensor<16x16xbf16>)
                    outs(%cbuf : memref<16x16xbf16, #hivm.address_space<cbuf>>)
                    {"hivm.inserted-copy"}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 3 : i32, no_inline}
    %w = bufferization.to_tensor %cbuf restrict writable : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%w, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// No unused mmad between the copy and its consumer: leave the chain put.
// CHECK-LABEL: func.func @keep_copy_chain_without_intervening_mmad
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           hivm.hir.copy
// CHECK:         } {{{.*}}VECTOR
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK-NOT:       hivm.hir.copy
// CHECK:           hivm.hir.mmadL1
func.func @keep_copy_chain_without_intervening_mmad(
    %src: memref<16x16xbf16, #hivm.address_space<gm>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %ub = memref.alloc() : memref<16x16xbf16, #hivm.address_space<ub>>
    %cbuf = memref.alloc() : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      hivm.hir.load ins(%src : memref<16x16xbf16, #hivm.address_space<gm>>)
                    outs(%ub : memref<16x16xbf16, #hivm.address_space<ub>>)
      %t = bufferization.to_tensor %ub restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      hivm.hir.copy ins(%t : tensor<16x16xbf16>)
                    outs(%cbuf : memref<16x16xbf16, #hivm.address_space<cbuf>>)
                    {"hivm.inserted-copy"}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 3 : i32, no_inline}
    %w = bufferization.to_tensor %cbuf restrict writable : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%w, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Two copies in one VECTOR: the cbuf consumed by the first mmad stays;
// the cbuf consumed only by a later mmad sinks past the unused cube.
// CHECK-LABEL: func.func @sink_only_unused_copy_in_mixed_scope
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           hivm.hir.copy
// CHECK-NOT:       hivm.hir.load
// CHECK:           scope.return
// CHECK:         } {{{.*}}VECTOR
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           hivm.hir.copy
// CHECK:         } {{{.*}}VECTOR
func.func @sink_only_unused_copy_in_mixed_scope(
    %srcA: memref<16x16xbf16, #hivm.address_space<gm>>,
    %srcB: memref<16x16xbf16, #hivm.address_space<gm>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %ubA = memref.alloc() : memref<16x16xbf16, #hivm.address_space<ub>>
    %ubB = memref.alloc() : memref<16x16xbf16, #hivm.address_space<ub>>
    %cbufA = memref.alloc() : memref<16x16xbf16, #hivm.address_space<cbuf>>
    %cbufB = memref.alloc() : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      hivm.hir.load ins(%srcA : memref<16x16xbf16, #hivm.address_space<gm>>)
                    outs(%ubA : memref<16x16xbf16, #hivm.address_space<ub>>)
      %tA = bufferization.to_tensor %ubA restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      hivm.hir.copy ins(%tA : tensor<16x16xbf16>)
                    outs(%cbufA : memref<16x16xbf16, #hivm.address_space<cbuf>>)
                    {"hivm.inserted-copy"}
      hivm.hir.load ins(%srcB : memref<16x16xbf16, #hivm.address_space<gm>>)
                    outs(%ubB : memref<16x16xbf16, #hivm.address_space<ub>>)
      %tB = bufferization.to_tensor %ubB restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      hivm.hir.copy ins(%tB : tensor<16x16xbf16>)
                    outs(%cbufB : memref<16x16xbf16, #hivm.address_space<cbuf>>)
                    {"hivm.inserted-copy"}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    %a = bufferization.to_tensor %cbufA restrict writable : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%a, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 3 : i32, no_inline}
    %b = bufferization.to_tensor %cbufB restrict writable : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%b, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Same as @sink_copy_chain_past_unused_mmad, but the load writes a subview
// of a producer-local alloc (the real kda W chain). The load must move
// with the VF+copy; leaving MTE2 in the producer still blocks the unused
// cube.
// CHECK-LABEL: func.func @sink_copy_chain_load_via_subview
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK-NOT:       hivm.hir.copy
// CHECK:           scope.return
// CHECK:         } {{{.*}}VECTOR
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           hivm.hir.copy
// CHECK:         } {{{.*}}VECTOR
func.func @sink_copy_chain_load_via_subview(
    %src: memref<16x16xbf16, #hivm.address_space<gm>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %cbuf = memref.alloc() : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      %ub = memref.alloc() : memref<16x16xbf16, #hivm.address_space<ub>>
      %sv = memref.subview %ub[0, 0] [16, 16] [1, 1] :
          memref<16x16xbf16, #hivm.address_space<ub>> to
          memref<16x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%src : memref<16x16xbf16, #hivm.address_space<gm>>)
                    outs(%sv : memref<16x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>>)
      %t = bufferization.to_tensor %ub restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      hivm.hir.copy ins(%t : tensor<16x16xbf16>)
                    outs(%cbuf : memref<16x16xbf16, #hivm.address_space<cbuf>>)
                    {"hivm.inserted-copy"}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 3 : i32, no_inline}
    %w = bufferization.to_tensor %cbuf restrict writable : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%w, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// CUBE and a later VECTOR both read the returned tensor: not a unique
// consumer, so the cluster stays in the producer.
// CHECK-LABEL: func.func @keep_returned_cluster_shared_with_cube
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           hivm.hir.vcast
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
func.func @keep_returned_cluster_shared_with_cube(
    %src: memref<16x16xbf16, #hivm.address_space<gm>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<16x16xf32> {
      %ub = memref.alloc() : memref<16x16xbf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%src : memref<16x16xbf16, #hivm.address_space<gm>>)
                    outs(%ub : memref<16x16xbf16, #hivm.address_space<ub>>)
      %t = bufferization.to_tensor %ub restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<16x16xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%t : tensor<16x16xbf16>) outs(%empty : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scope.return %dst : tensor<16x16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%prod : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<16x16xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<16x16xf32>) outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Scalar extra of a sunk dest is `memref.load` of a producer-local view.
// The view stays in the producer (another load still uses it), so clone
// the reinterpret_cast / subview / affine offset into the consumer.
// CHECK-LABEL: func.func @sink_clones_memref_load_view_chain
// CHECK:         scope.scope
// CHECK:           memref.reinterpret_cast
// CHECK:           memref.subview
// CHECK:           memref.load
// CHECK:           hivm.hir.vexp
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           affine.apply
// CHECK:           memref.reinterpret_cast
// CHECK:           memref.subview
// CHECK:           memref.load
// CHECK:           hivm.hir.vmul
// CHECK:           hivm.hir.vadd
// CHECK:           hivm.hir.vexp
func.func @sink_clones_memref_load_view_chain(
    %g: memref<?xf32>,
    %bg: memref<64xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst_m1 = arith.constant -1.000000e+00 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<64xf32> {
      %off = affine.apply affine_map<()[s0] -> (s0)>()[%i]
      %rc = memref.reinterpret_cast %g to offset: [%off], sizes: [1], strides: [1]
          : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %sv = memref.subview %rc[0] [1] [1]
          : memref<1xf32, strided<[1], offset: ?>> to
            memref<1xf32, strided<[1], offset: ?>>
      %scalar = memref.load %sv[%c0] : memref<1xf32, strided<[1], offset: ?>>
      %keep = memref.load %sv[%c0] : memref<1xf32, strided<[1], offset: ?>>
      %one = tensor.empty() : tensor<1xf32>
      %inserted = tensor.insert %keep into %one[%c0] : tensor<1xf32>
      %t = bufferization.to_tensor %bg restrict writable : memref<64xf32, #hivm.address_space<ub>>
      %e0 = tensor.empty() : tensor<64xf32>
      %neg = hivm.hir.vmul ins(%t, %cst_m1 : tensor<64xf32>, f32) outs(%e0 : tensor<64xf32>) -> tensor<64xf32>
      %adj = hivm.hir.vadd ins(%neg, %scalar : tensor<64xf32>, f32) outs(%e0 : tensor<64xf32>) -> tensor<64xf32>
      %exp = hivm.hir.vexp ins(%adj : tensor<64xf32>) outs(%e0 : tensor<64xf32>) -> tensor<64xf32>
      %stay = hivm.hir.vexp ins(%t : tensor<64xf32>) outs(%e0 : tensor<64xf32>) -> tensor<64xf32>
      scope.return %exp : tensor<64xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<64xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<64xf32>) outs(%out : tensor<64xf32>) -> tensor<64xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// The gate is still used by the extract in the next VECTOR, which sits
// before the CUBE, so vexp stays at preload 2. The vmul's next CUBE does
// not read it and the first consumer after that CUBE is the last VECTOR,
// so the vmul sinks and clones the extract.
// CHECK-LABEL: func.func @sink_forwards_extracted_tensor_extra
// CHECK:         scope.scope
// CHECK:           %[[EXP:.*]] = hivm.hir.vexp
// CHECK:           scope.return %[[EXP]]
// CHECK:         } {{{.*}}preload_num = 2
// CHECK:         scope.scope
// CHECK:           tensor.extract
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           tensor.extract
// CHECK:           hivm.hir.vmul
func.func @sink_forwards_extracted_tensor_extra()
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %gate = scope.scope : () -> tensor<1xf32> {
      %empty = tensor.empty() : tensor<1xf32>
      %exp = hivm.hir.vexp ins(%empty : tensor<1xf32>) outs(%empty : tensor<1xf32>) -> tensor<1xf32>
      scope.return %exp : tensor<1xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32, no_inline}
    %prod = scope.scope : () -> tensor<64xf32> {
      %extracted = tensor.extract %gate[%c0] : tensor<1xf32>
      %out = tensor.empty() : tensor<64xf32>
      %scaled = hivm.hir.vmul ins(%out, %extracted : tensor<64xf32>, f32) outs(%out : tensor<64xf32>) -> tensor<64xf32>
      scope.return %scaled : tensor<64xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 3 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<64xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<64xf32>) outs(%out : tensor<64xf32>) -> tensor<64xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// The next CUBE need not be the adjacent scope. It does not read the
// return, so the cluster sinks into the first VECTOR consumer after it.
// CHECK-LABEL: func.func @sink_returned_cluster_past_nonadjacent_cube
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vcast
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 2
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vexp
func.func @sink_returned_cluster_past_nonadjacent_cube(
    %arg0: memref<32x128xbf16, #hivm.address_space<ub>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<32x128xf32> {
      %src = bufferization.to_tensor %arg0 restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%src : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 3 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// The adjacent CUBE consumes the returned tensor, so the cluster stays
// in the producer instead of moving past that CUBE.
// CHECK-LABEL: func.func @keep_returned_cluster_consumed_by_adjacent_cube
// CHECK:         scope.scope
// CHECK:           hivm.hir.vcast
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK-NOT:       hivm.hir.vcast
// CHECK:         } {{{.*}}CUBE
func.func @keep_returned_cluster_consumed_by_adjacent_cube(
    %arg0: memref<16x16xbf16, #hivm.address_space<ub>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<16x16xf32> {
      %src = bufferization.to_tensor %arg0 restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<16x16xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%src : tensor<16x16xbf16>) outs(%empty : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scope.return %dst : tensor<16x16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%prod : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// The first consuming CUBE is not adjacent. Copy-related returned work
// moves into the VECTOR immediately before that CUBE and is returned so
// the CUBE still reads it. Several consuming CUBEs would use the first.
// CHECK-LABEL: func.func @sink_returned_cluster_before_consuming_cube
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vcast
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:           hivm.hir.vcast
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 2
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK-NOT:       hivm.hir.vcast
// CHECK:         } {{{.*}}CUBE
func.func @sink_returned_cluster_before_consuming_cube(
    %arg0: memref<16x16xbf16, #hivm.address_space<ub>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<16x16xf32> {
      %src = bufferization.to_tensor %arg0 restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<16x16xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%src : tensor<16x16xbf16>) outs(%empty : tensor<16x16xf32>)
          -> tensor<16x16xf32>
      scope.return %dst : tensor<16x16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%prod : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 3 : i32, no_inline}
  }
  return
}

// Several VECTOR consumers: the cluster lands in the first one and is
// returned for the later consumer. The next CUBE does not read it.
// CHECK-LABEL: func.func @sink_returned_cluster_to_first_vector_consumer
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vcast
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           hivm.hir.vcast
// CHECK:           hivm.hir.vexp
// CHECK:           scope.return
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vcast
// CHECK:           hivm.hir.vexp
func.func @sink_returned_cluster_to_first_vector_consumer(
    %arg0: memref<32x128xbf16, #hivm.address_space<ub>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<32x128xf32> {
      %src = bufferization.to_tensor %arg0 restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%src : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Later scope first. The vmul (and the expand of the earlier return) sinks
// into the last VECTOR. The earlier exclusive vexp then follows it there.
// CHECK-LABEL: func.func @sink_returned_chain_from_later_scope
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vexp
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vexp
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 0
// CHECK:         scope.scope
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK:           hivm.hir.vexp
// CHECK:           tensor.expand_shape
// CHECK:           hivm.hir.vexp
// CHECK:           hivm.hir.vexp
func.func @sink_returned_chain_from_later_scope()
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %other = tensor.empty() : tensor<16x16xbf16>
  scf.for %i = %c0 to %c4 step %c1 {
    %gate = scope.scope : () -> tensor<64xf32> {
      %empty = tensor.empty() : tensor<64xf32>
      %exp = hivm.hir.vexp ins(%empty : tensor<64xf32>) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
      scope.return %exp : tensor<64xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    %prod = scope.scope : () -> tensor<64x1xf32> {
      %expanded = tensor.expand_shape %gate [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %out = tensor.empty() : tensor<64x1xf32>
      %scaled = hivm.hir.vexp ins(%expanded : tensor<64x1xf32>) outs(%out : tensor<64x1xf32>) -> tensor<64x1xf32>
      scope.return %scaled : tensor<64x1xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<64x1xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<64x1xf32>) outs(%out : tensor<64x1xf32>) -> tensor<64x1xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// -----

// Regression: the preload-1 VECTOR vmul reads a UB buffer that a preload-2
// CUBE fixpipe writes. Sinking it into the preload-0 consumer moved that
// cross-core read from distance 1 to 2, so the buffer's two slots collided.
// CHECK-LABEL: func.func @keep_cluster_reading_cube_written_buffer
// CHECK:         hivm.hir.fixpipe
// CHECK:         %[[T:[0-9a-z_]+]] = bufferization.to_tensor
// CHECK:         %[[R:[0-9a-z_]+]] = scope.scope
// CHECK-NEXT:      hivm.hir.vmul ins(%[[T]], %{{.*}} :
// CHECK-NEXT:      scope.return
// CHECK-NEXT:    } {{.*}}hivm.preload_num = 1 : i32
// CHECK:           hivm.hir.vadd ins(%{{.*}}, %[[R]] :
// CHECK:         } {{.*}}hivm.preload_num = 0 : i32
module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">} {
func.func @keep_cluster_reading_cube_written_buffer(%scale: f32, %lb: index, %ub: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %true = arith.constant true
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x32xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
  %res = scf.for %iv = %lb to %ub step %c1 iter_args(%acc = %init) -> (tensor<64x32xf32>) {
    %lhs = memref.alloc() : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %rhs = memref.alloc() : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
    %tc = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    %a = bufferization.to_tensor %lhs restrict writable : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %b = bufferization.to_tensor %rhs restrict writable : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      %l0c = tensor.empty() : tensor<2x4x16x16xf32>
      %mm = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%a, %b, %true, %c64, %c64, %c32 : tensor<8x4x16x8xf32>, tensor<4x4x16x8xf32>, i1, index, index, index) outs(%l0c : tensor<2x4x16x16xf32>) -> tensor<2x4x16x16xf32>
      annotation.mark %tc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x32xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%mm : tensor<2x4x16x16xf32>) outs(%tc : memref<64x32xf32, #hivm.address_space<ub>>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
    %t = bufferization.to_tensor %tc restrict writable : memref<64x32xf32, #hivm.address_space<ub>>
    %scaled = scope.scope : () -> (tensor<64x32xf32>) {
      %m = hivm.hir.vmul ins(%t, %scale : tensor<64x32xf32>, f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
      scope.return %m : tensor<64x32xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
    %next = scope.scope : () -> (tensor<64x32xf32>) {
      %s = hivm.hir.vadd ins(%acc, %scaled : tensor<64x32xf32>, tensor<64x32xf32>) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
      scope.return %s : tensor<64x32xf32>
    } {hivm.has_loop_carried_dep, hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %next : tensor<64x32xf32>
  } {hivm.cv_pipelined_loop}
  return
}

// Same shape with a VECTOR writer and no following CUBE: the vmul stays.
// CHECK-LABEL: func.func @keep_cluster_reading_vector_written_buffer
// CHECK:         hivm.hir.load
// CHECK:         %[[M:[0-9a-z_]+]] = scope.scope
// CHECK-NEXT:      hivm.hir.vmul
// CHECK-NEXT:      scope.return
// CHECK-NEXT:    } {{.*}}hivm.preload_num = 1 : i32
// CHECK:           hivm.hir.vadd ins(%{{.*}}, %[[M]] :
// CHECK:         } {{.*}}hivm.preload_num = 0 : i32
func.func @keep_cluster_reading_vector_written_buffer(
    %src: memref<64x32xf32, #hivm.address_space<gm>>, %scale: f32, %lb: index, %ub: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x32xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
  %res = scf.for %iv = %lb to %ub step %c1 iter_args(%acc = %init) -> (tensor<64x32xf32>) {
    %buf = memref.alloc() : memref<64x32xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      hivm.hir.load ins(%src : memref<64x32xf32, #hivm.address_space<gm>>)
                    outs(%buf : memref<64x32xf32, #hivm.address_space<ub>>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
    %t = bufferization.to_tensor %buf restrict writable : memref<64x32xf32, #hivm.address_space<ub>>
    %scaled = scope.scope : () -> (tensor<64x32xf32>) {
      %m = hivm.hir.vmul ins(%t, %scale : tensor<64x32xf32>, f32) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
      scope.return %m : tensor<64x32xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
    %next = scope.scope : () -> (tensor<64x32xf32>) {
      %s = hivm.hir.vadd ins(%acc, %scaled : tensor<64x32xf32>, tensor<64x32xf32>) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>
      scope.return %s : tensor<64x32xf32>
    } {hivm.has_loop_carried_dep, hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %next : tensor<64x32xf32>
  } {hivm.cv_pipelined_loop}
  return
}
}
