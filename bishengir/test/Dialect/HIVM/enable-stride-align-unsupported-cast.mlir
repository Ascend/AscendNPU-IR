// RUN: bishengir-opt %s -hivm-enable-stride-align -split-input-file -verify-diagnostics
func.func private @consume_static(memref<4x2xi64, strided<[8, 1]>, #hivm.address_space<ub>>)
func.func @preserve_static_stride_refinement(%src: memref<?x2xi64, strided<[?, 2]>, #hivm.address_space<ub>>) {
  %dst = builtin.unrealized_conversion_cast %src : memref<?x2xi64, strided<[?, 2]>, #hivm.address_space<ub>> to memref<?x2xi64, strided<[?, 1]>, #hivm.address_space<ub>> {PropagateFailure}
  // expected-error @+1 {{cannot translate aligned layout refinement}}
  %cast = memref.cast %dst {PropagateFailure} : memref<?x2xi64, strided<[?, 1]>, #hivm.address_space<ub>> to memref<4x2xi64, strided<[8, 1]>, #hivm.address_space<ub>>
  func.call @consume_static(%cast) : (memref<4x2xi64, strided<[8, 1]>, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func private @consume_offset(memref<4x2xi64, strided<[8, 1], offset: 3>, #hivm.address_space<ub>>)
func.func @preserve_static_offset_refinement(%src: memref<?x2xi64, strided<[8, 2], offset: ?>, #hivm.address_space<ub>>) {
  %dst = builtin.unrealized_conversion_cast %src : memref<?x2xi64, strided<[8, 2], offset: ?>, #hivm.address_space<ub>> to memref<?x2xi64, strided<[8, 1], offset: 3>, #hivm.address_space<ub>> {PropagateFailure}
  // expected-error @+1 {{cannot translate aligned layout refinement}}
  %cast = memref.cast %dst {PropagateFailure} : memref<?x2xi64, strided<[8, 1], offset: 3>, #hivm.address_space<ub>> to memref<4x2xi64, strided<[8, 1], offset: 3>, #hivm.address_space<ub>>
  func.call @consume_offset(%cast) : (memref<4x2xi64, strided<[8, 1], offset: 3>, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func private @consume_negative(memref<2x2xi64, strided<[-2, 1], offset: 2>, #hivm.address_space<ub>>)
func.func @negative_fallback_stride(%src: memref<2x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) {
  // expected-error @+1 {{cannot materialize stride-alignment fallback}}
  %dst = builtin.unrealized_conversion_cast %src : memref<2x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<2x2xi64, strided<[-2, 1], offset: 2>, #hivm.address_space<ub>> {PropagateFailure}
  func.call @consume_negative(%dst) {PropagateFailure} : (memref<2x2xi64, strided<[-2, 1], offset: 2>, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func private @consume_negative_offset(memref<2x2xi64, strided<[2, 1], offset: -1>, #hivm.address_space<ub>>)
func.func @negative_fallback_offset(%src: memref<2x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) {
  // expected-error @+1 {{cannot materialize stride-alignment fallback}}
  %dst = builtin.unrealized_conversion_cast %src : memref<2x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<2x2xi64, strided<[2, 1], offset: -1>, #hivm.address_space<ub>> {PropagateFailure}
  func.call @consume_negative_offset(%dst) {PropagateFailure} : (memref<2x2xi64, strided<[2, 1], offset: -1>, #hivm.address_space<ub>>) -> ()
  return
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
// The regbase materializer must diagnose unsupported layouts too, rather than
// bypass the main fallback's checks for an unmarked conversion.
func.func private @consume_regbase_negative(memref<2x2xi64, strided<[-2, 1], offset: 2>, #hivm.address_space<ub>>)
func.func @regbase_negative_layout(%src: memref<2x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) {
  // expected-error @+1 {{cannot materialize stride-alignment fallback}}
  %dst = builtin.unrealized_conversion_cast %src : memref<2x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<2x2xi64, strided<[-2, 1], offset: 2>, #hivm.address_space<ub>>
  func.call @consume_regbase_negative(%dst) : (memref<2x2xi64, strided<[-2, 1], offset: 2>, #hivm.address_space<ub>>) -> ()
  return
}
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
// Exercise real propagation failure and tagging: the non-unit leading stride
// is refined after a subview, so there is no direct expand group to translate.
func.func private @consume_nonexpand(memref<2x?x?xi64, strided<[8, 4, 1]>, #hivm.address_space<ub>>)
func.func @nonexpand_stride_refinement(%m: index, %n: index, %m2: index, %n2: index) {
  %a = memref.alloc() : memref<1024x2xi64, #hivm.address_space<ub>>
  annotation.mark %a {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1024x2xi64, #hivm.address_space<ub>>
  %s = memref.subview %a[0, 0] [%m, %n] [1, 1] : memref<1024x2xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>>
  %e = memref.expand_shape %s [[0, 1], [2]] output_shape [2, %m2, %n] : memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>> into memref<2x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>>
  %sub = memref.subview %e[0, 0, 0] [2, %m2, %n2] [1, 2, 1] : memref<2x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>> to memref<2x?x?xi64, strided<[?, 4, 1]>, #hivm.address_space<ub>>
  // expected-error @+1 {{cannot translate aligned layout refinement}}
  %c = memref.cast %sub : memref<2x?x?xi64, strided<[?, 4, 1]>, #hivm.address_space<ub>> to memref<2x?x?xi64, strided<[8, 4, 1]>, #hivm.address_space<ub>>
  func.call @consume_nonexpand(%c) : (memref<2x?x?xi64, strided<[8, 4, 1]>, #hivm.address_space<ub>>) -> ()
  return
}
}
