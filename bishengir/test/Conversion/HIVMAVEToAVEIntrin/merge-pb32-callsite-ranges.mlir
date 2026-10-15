// RUN: bishengir-opt %s -split-input-file -hacc-append-device-spec=target=Ascend950PR_9579 \
// RUN: -convert-hivmave-to-ave-intrin -canonicalize -cse | FileCheck %s --implicit-check-not=_packed_stores --implicit-check-not=hivm.pdintlv --implicit-check-not=hivm.psts --implicit-check-not=hivm.pstu

// -----

// CHECK-LABEL: func.func @touching_ranges(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[P0:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[P1:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%[[P0]], %[[P1]])
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK-NOT: hivm.vstas
// CHECK: return
func.func @touching_ranges(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [8], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_touching_ranges(
// CHECK: call @touching_ranges(
// CHECK: return
func.func @caller_touching_ranges(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @touching_ranges(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @one_byte_overlap(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @one_byte_overlap(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [7], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_one_byte_overlap(
// CHECK: call @one_byte_overlap(
// CHECK: return
func.func @caller_one_byte_overlap(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @one_byte_overlap(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @unknown_offset(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @unknown_offset(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>, %idx: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [%idx], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_unknown_offset(
// CHECK: call @unknown_offset(
// CHECK: return
func.func @caller_unknown_offset(%idx: index, %a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @unknown_offset(%d, %sbase, %a, %b, %idx) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>, index) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @select_all_safe(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[P0:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[P1:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%[[P0]], %[[P1]])
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK-NOT: hivm.vstas
// CHECK: return
func.func @select_all_safe(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_select_all_safe(
// CHECK: call @select_all_safe(
// CHECK: return
func.func @caller_select_all_safe(%ping: i1, %a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %alt = arith.constant 8192 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  %s1 = hivm.hir.pointer_cast(%alt) : memref<256xi8, #hivm.address_space<ub>>
  %s = arith.select %ping, %sbase, %s1 : memref<256xi8, #hivm.address_space<ub>>
  func.call @select_all_safe(%d, %s, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @select_one_overlap(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @select_one_overlap(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_select_one_overlap(
// CHECK: call @select_one_overlap(
// CHECK: return
func.func @caller_select_one_overlap(%ping: i1, %a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %overlap = arith.constant 64 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  %d1 = hivm.hir.pointer_cast(%overlap) : memref<256xi1, #hivm.address_space<ub>>
  %ds = arith.select %ping, %d, %d1 : memref<256xi1, #hivm.address_space<ub>>
  func.call @select_one_overlap(%ds, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @multiple_safe_calls(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[P0:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[P1:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%[[P0]], %[[P1]])
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK-NOT: hivm.vstas
// CHECK: return
func.func @multiple_safe_calls(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_multiple_safe_calls(
// CHECK: call @multiple_safe_calls(
// CHECK: call @multiple_safe_calls(
// CHECK: return
func.func @caller_multiple_safe_calls(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %alt = arith.constant 8192 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @multiple_safe_calls(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  %d2 = hivm.hir.pointer_cast(%alt) : memref<256xi1, #hivm.address_space<ub>>
  func.call @multiple_safe_calls(%d2, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @multiple_calls_one_overlap(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @multiple_calls_one_overlap(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_multiple_calls_one_overlap(
// CHECK: call @multiple_calls_one_overlap(
// CHECK: call @multiple_calls_one_overlap(
// CHECK: return
func.func @caller_multiple_calls_one_overlap(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %overlap = arith.constant 64 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @multiple_calls_one_overlap(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  %d2 = hivm.hir.pointer_cast(%overlap) : memref<256xi1, #hivm.address_space<ub>>
  func.call @multiple_calls_one_overlap(%d2, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @multiple_calls_one_unknown(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @multiple_calls_one_unknown(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_multiple_calls_one_unknown(
// CHECK: call @multiple_calls_one_unknown(
// CHECK: call @multiple_calls_one_unknown(
// CHECK: return
func.func @caller_multiple_calls_one_unknown(%unknown: i64, %a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @multiple_calls_one_unknown(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  %d2 = hivm.hir.pointer_cast(%unknown) : memref<256xi1, #hivm.address_space<ub>>
  func.call @multiple_calls_one_unknown(%d2, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @wrapped_address(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @wrapped_address(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_wrapped_address(
// CHECK: call @wrapped_address(
// CHECK: return
func.func @caller_wrapped_address(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 4294967296 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @wrapped_address(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @negative_address(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @negative_address(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_negative_address(
// CHECK: call @negative_address(
// CHECK: return
func.func @caller_negative_address(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant -16 : i64
  %sc = arith.constant 1024 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @negative_address(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @offset_unit_overflow(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @offset_unit_overflow(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [2305843009213694016], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_offset_unit_overflow(
// CHECK: call @offset_unit_overflow(
// CHECK: return
func.func @caller_offset_unit_overflow(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  func.call @offset_unit_overflow(%d, %sbase, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @caller_offset_not_added(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @caller_offset_not_added(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [0], sizes: [64], strides: [1] : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_caller_offset_not_added(
// CHECK: call @caller_offset_not_added(
// CHECK: return
func.func @caller_caller_offset_not_added(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 0 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  %s = memref.reinterpret_cast %sbase to offset: [128], sizes: [256], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  func.call @caller_offset_not_added(%d, %s, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// The marked cast folds offset 64 into the source pointer. The VF load
// is at [128, 192), overlapping the first store at [128, 136).
// CHECK-LABEL: func.func @marked_cast_source_offset(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @marked_cast_source_offset(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<256xi8, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_marked_cast_source_offset(
// CHECK: call @marked_cast_source_offset(
// CHECK: return
func.func @caller_marked_cast_source_offset(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 128 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  %off = arith.constant 64 : index
  %sview = memref.reinterpret_cast %sbase to offset: [%off], sizes: [256], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %folded = memref.cast %sview {fold_offset_into_ptr} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<256xi8, #hivm.address_space<ub>>
  func.call @marked_cast_source_offset(%d, %folded, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<256xi8, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}

// -----

// An ordinary cast preserves aligned_ptr. The VF reinterpret_cast replaces
// offset 64, so the load is at [64, 128), disjoint from the store at 128.
// CHECK-LABEL: func.func @ordinary_cast_source_offset(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[P0:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[P1:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%[[P0]], %[[P1]])
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK: return
func.func @ordinary_cast_source_offset(%dst: memref<256xi1, #hivm.address_space<ub>>, %src: memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, %a: vector<64xi8>, %b: vector<64xi8>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %p0 = ave.hir.vcmp <NE> %a, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d0[%c0], %mask, %p0 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %s = memref.reinterpret_cast %src to offset: [64], sizes: [64], strides: [1] : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %x = ave.hir.vload <UNPK4_B8> %s[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi8>
  %p1 = ave.hir.vcmp <NE> %x, %b, %mask : vector<64xi8>, vector<64xi1> -> vector<64xi1>
  %d1 = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %d1[%c0], %mask, %p1 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_ordinary_cast_source_offset(
// CHECK: call @ordinary_cast_source_offset(
// CHECK: return
func.func @caller_ordinary_cast_source_offset(%a: vector<64xi8>, %b: vector<64xi8>) {
  %dc = arith.constant 128 : i64
  %sc = arith.constant 0 : i64
  %d = hivm.hir.pointer_cast(%dc) : memref<256xi1, #hivm.address_space<ub>>
  %sbase = hivm.hir.pointer_cast(%sc) : memref<256xi8, #hivm.address_space<ub>>
  %off = arith.constant 64 : index
  %sview = memref.reinterpret_cast %sbase to offset: [%off], sizes: [256], strides: [1] : memref<256xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %folded = memref.cast %sview : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  func.call @ordinary_cast_source_offset(%d, %folded, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi8>, vector<64xi8>) -> ()
  return
}
