// RUN: bishengir-opt %s -split-input-file -optimize-reduction-loop -canonicalize | FileCheck %s

// Different pointer_cast results and element types can alias.
// CHECK-LABEL: func.func private @same_base(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @same_base(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_same_base() {
  %a0 = arith.constant 0 : i64
  %s = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%a0) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @same_base(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Different pointer_cast addresses keep the store/reduction split.
// CHECK-LABEL: func.func private @disjoint(
// CHECK: splitDepth = 1
// CHECK: return
func.func private @disjoint(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_disjoint() {
  %a0 = arith.constant 0 : i64
  %a8192 = arith.constant 8192 : i64
  %s = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%a8192) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @disjoint(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Only base addresses are compared; partial range overlap is not checked.
// CHECK-LABEL: func.func private @different_bases(
// CHECK: splitDepth = 1
// CHECK: return
func.func private @different_bases(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_different_bases() {
  %a0 = arith.constant 0 : i64
  %a4096 = arith.constant 4096 : i64
  %s = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%a4096) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @different_bases(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// The second multi-buffer addresses are equal.
// CHECK-LABEL: func.func private @later_buffer_overlap(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @later_buffer_overlap(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_later_buffer_overlap() {
  %a0 = arith.constant 0 : i64
  %a65536 = arith.constant 65536 : i64
  %a8192 = arith.constant 8192 : i64
  %s = hivm.hir.pointer_cast(%a0, %a65536) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%a8192, %a65536) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @later_buffer_overlap(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Do not assume that multi-buffer slots always select matching indices.
// CHECK-LABEL: func.func private @cross_buffer_overlap(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @cross_buffer_overlap(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_cross_buffer_overlap() {
  %a0 = arith.constant 0 : i64
  %a65536 = arith.constant 65536 : i64
  %a131072 = arith.constant 131072 : i64
  %s = hivm.hir.pointer_cast(%a0, %a65536) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%a65536, %a131072) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @cross_buffer_overlap(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Check every call site in the same kernel, preserving argument pairs.
// CHECK-LABEL: func.func private @all_call_sites(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @all_call_sites(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_all_call_sites() {
  %c0 = arith.constant 0 : i64
  %c32768 = arith.constant 32768 : i64
  %s0 = hivm.hir.pointer_cast(%c0) : memref<2048xf32, #hivm.address_space<ub>>
  %d0 = hivm.hir.pointer_cast(%c32768) : memref<2048xbf16, #hivm.address_space<ub>>
  %s1 = hivm.hir.pointer_cast(%c0) : memref<2048xf32, #hivm.address_space<ub>>
  %d1 = hivm.hir.pointer_cast(%c0) : memref<2048xbf16, #hivm.address_space<ub>>
  %r0 = func.call @all_call_sites(%s0, %d0) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  %r1 = func.call @all_call_sites(%s1, %d1) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Check every call site in the same kernel, preserving argument pairs.
// CHECK-LABEL: func.func private @swapped_call_sites(
// CHECK: splitDepth = 1
// CHECK: return
func.func private @swapped_call_sites(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_swapped_call_sites() {
  %c0 = arith.constant 0 : i64
  %c32768 = arith.constant 32768 : i64
  %s0 = hivm.hir.pointer_cast(%c0) : memref<2048xf32, #hivm.address_space<ub>>
  %d0 = hivm.hir.pointer_cast(%c32768) : memref<2048xbf16, #hivm.address_space<ub>>
  %s1 = hivm.hir.pointer_cast(%c32768) : memref<2048xf32, #hivm.address_space<ub>>
  %d1 = hivm.hir.pointer_cast(%c0) : memref<2048xbf16, #hivm.address_space<ub>>
  %r0 = func.call @swapped_call_sites(%s0, %d0) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  %r1 = func.call @swapped_call_sites(%s1, %d1) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Loop-local subviews must resolve to the VF arguments.
// CHECK-LABEL: func.func private @subviews(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @subviews(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    %view = memref.subview %dst[%i] [64] [1] : memref<2048xbf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B16> %view[%c0], %mask, %narrow : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_subviews() {
  %a0 = arith.constant 0 : i64
  %s = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%a0) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @subviews(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// An unresolved pointer address is MayAlias.
// CHECK-LABEL: func.func private @dynamic_address(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @dynamic_address(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_dynamic_address(%dynamic: i64) {
  %a0 = arith.constant 0 : i64
  %s = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%dynamic) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @dynamic_address(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Two casts of the same dynamic address must conflict.
// CHECK-LABEL: func.func private @shared_dynamic_address(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @shared_dynamic_address(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_shared_dynamic_address(%dynamic: i64) {
  %s = hivm.hir.pointer_cast(%dynamic) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%dynamic) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @shared_dynamic_address(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Aliasing reads are harmless when the written buffer is disjoint.
// CHECK-LABEL: func.func private @read_alias_write_disjoint(
// CHECK: splitDepth = 1
// CHECK: return
func.func private @read_alias_write_disjoint(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>, %other: memref<2048xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %w = ave.hir.vload <NORM> %other[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vadd %v, %w, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_read_alias_write_disjoint() {
  %a0 = arith.constant 0 : i64
  %a8192 = arith.constant 8192 : i64
  %s = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%a8192) : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @read_alias_write_disjoint(%s, %d, %s) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Unresolved kernel arguments remain conservative without walking callers.
// CHECK-LABEL: func.func private @kernel_arguments(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @kernel_arguments(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_kernel_arguments(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> {
  %r = func.call @kernel_arguments(%src, %dst) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return %r : vector<64xf32>
}

// -----

// No known callers cannot establish disjoint physical storage.
// CHECK-LABEL: func.func private @unknown_callers(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @unknown_callers(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}

// -----

// Follow pointer_cast operands after multi-buffer selection.
// CHECK-LABEL: func.func private @selected_disjoint(
// CHECK: splitDepth = 1
// CHECK: return
func.func private @selected_disjoint(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_selected_disjoint(%cond: i1) {
  %c0 = arith.constant 0 : i64
  %cfirst = arith.constant 8192 : i64
  %c65536 = arith.constant 65536 : i64
  %csecond = arith.constant 73728 : i64
  %s0 = hivm.hir.pointer_cast(%c0) : memref<2048xf32, #hivm.address_space<ub>>
  %s1 = hivm.hir.pointer_cast(%c65536) : memref<2048xf32, #hivm.address_space<ub>>
  %d0 = hivm.hir.pointer_cast(%cfirst) : memref<2048xbf16, #hivm.address_space<ub>>
  %d1 = hivm.hir.pointer_cast(%csecond) : memref<2048xbf16, #hivm.address_space<ub>>
  %s = arith.select %cond, %s0, %s1 : memref<2048xf32, #hivm.address_space<ub>>
  %d = arith.select %cond, %d0, %d1 : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @selected_disjoint(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Follow pointer_cast operands after multi-buffer selection.
// CHECK-LABEL: func.func private @selected_overlap(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @selected_overlap(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_selected_overlap(%cond: i1) {
  %c0 = arith.constant 0 : i64
  %cfirst = arith.constant 8192 : i64
  %c65536 = arith.constant 65536 : i64
  %csecond = arith.constant 65536 : i64
  %s0 = hivm.hir.pointer_cast(%c0) : memref<2048xf32, #hivm.address_space<ub>>
  %s1 = hivm.hir.pointer_cast(%c65536) : memref<2048xf32, #hivm.address_space<ub>>
  %d0 = hivm.hir.pointer_cast(%cfirst) : memref<2048xbf16, #hivm.address_space<ub>>
  %d1 = hivm.hir.pointer_cast(%csecond) : memref<2048xbf16, #hivm.address_space<ub>>
  %s = arith.select %cond, %s0, %s1 : memref<2048xf32, #hivm.address_space<ub>>
  %d = arith.select %cond, %d0, %d1 : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @selected_overlap(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Follow pointer_cast operands after multi-buffer selection.
// CHECK-LABEL: func.func private @selected_swapped(
// CHECK: splitDepth = 1
// CHECK: return
func.func private @selected_swapped(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_selected_swapped(%cond: i1) {
  %c0 = arith.constant 0 : i64
  %cfirst = arith.constant 65536 : i64
  %c65536 = arith.constant 65536 : i64
  %csecond = arith.constant 0 : i64
  %s0 = hivm.hir.pointer_cast(%c0) : memref<2048xf32, #hivm.address_space<ub>>
  %s1 = hivm.hir.pointer_cast(%c65536) : memref<2048xf32, #hivm.address_space<ub>>
  %d0 = hivm.hir.pointer_cast(%cfirst) : memref<2048xbf16, #hivm.address_space<ub>>
  %d1 = hivm.hir.pointer_cast(%csecond) : memref<2048xbf16, #hivm.address_space<ub>>
  %s = arith.select %cond, %s0, %s1 : memref<2048xf32, #hivm.address_space<ub>>
  %d = arith.select %cond, %d0, %d1 : memref<2048xbf16, #hivm.address_space<ub>>
  %r = func.call @selected_swapped(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// VF-local selections can change across the reordered loop iterations.
// CHECK-LABEL: func.func private @loop_variant_select(
// CHECK-NOT: splitDepth
// CHECK: return
func.func private @loop_variant_select() -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %c1024 = arith.constant 1024 : index
  %a0 = arith.constant 0 : i64
  %a8192 = arith.constant 8192 : i64
  %s0 = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %s1 = hivm.hir.pointer_cast(%a8192) : memref<2048xf32, #hivm.address_space<ub>>
  %d0 = hivm.hir.pointer_cast(%a0) : memref<2048xbf16, #hivm.address_space<ub>>
  %d1 = hivm.hir.pointer_cast(%a8192) : memref<2048xbf16, #hivm.address_space<ub>>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %cond = arith.cmpi ult, %i, %c1024 : index
    %src = arith.select %cond, %s0, %s1 : memref<2048xf32, #hivm.address_space<ub>>
    %dst = arith.select %cond, %d1, %d0 : memref<2048xbf16, #hivm.address_space<ub>>
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    %narrow = ave.hir.vtruncf %v, <rint>, false, <part_even>, %mask : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %dst[%i], %mask, %narrow : memref<2048xbf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}

// -----

// Same element types retain splitting even when base addresses may alias.
// CHECK-LABEL: func.func private @same_element_type(
// CHECK: splitDepth = 1
// CHECK: return
func.func private @same_element_type(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %dst[%i], %mask, %v : memref<2048xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
func.func @caller_same_element_type() {
  %a0 = arith.constant 0 : i64
  %s = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %d = hivm.hir.pointer_cast(%a0) : memref<2048xf32, #hivm.address_space<ub>>
  %r = func.call @same_element_type(%s, %d) : (memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) -> vector<64xf32>
  return
}

// -----

// Same element types retain splitting even when base addresses may alias.
// CHECK-LABEL: func.func private @same_element_type_unknown_callers(
// CHECK: splitDepth = 1
// CHECK: return
func.func private @same_element_type_unknown_callers(%src: memref<2048xf32, #hivm.address_space<ub>>, %dst: memref<2048xf32, #hivm.address_space<ub>>) -> vector<64xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant dense<0.0> : vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %r = scf.for %i = %c0 to %c2048 step %c64 iter_args(%acc = %zero) -> vector<64xf32> {
    %v = ave.hir.vload <NORM> %src[%i] : memref<2048xf32, #hivm.address_space<ub>> into vector<64xf32>
    %input = ave.hir.vmul %v, %v, %mask : vector<64xf32>, vector<64xi1>
    %next = ave.hir.vadd %input, %acc, %mask {reductionOp} : vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %dst[%i], %mask, %v : memref<2048xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    scf.yield %next : vector<64xf32>
  } {reductionLoop}
  return %r : vector<64xf32>
}
