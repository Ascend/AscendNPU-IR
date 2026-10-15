// RUN: bishengir-opt %s -allow-unregistered-dialect -split-input-file -hacc-append-device-spec=target=Ascend950PR_9579 \
// RUN: -convert-hivmave-to-ave-intrin -canonicalize -cse | FileCheck %s --enable-var-scope \
// RUN: --implicit-check-not=_packed_stores --implicit-check-not=hivm.pdintlv --implicit-check-not=hivm.psts --implicit-check-not=hivm.pstu

// Even without intervening memory accesses, upgrading PB32 stores to an
// aligned PB16 store requires checking the actual base at every call site.

// -----

// Every call is checked, while the original VF and call targets stay intact.
// The first base is 16-byte aligned but not 32-byte aligned.
// CHECK-LABEL: func.func @adjacent_multiple_all_aligned(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK: return
func.func @adjacent_multiple_all_aligned(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi1>, %b: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0_dst = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_dst[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_dst = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_dst[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_adjacent_multiple_all_aligned(
// CHECK: call @adjacent_multiple_all_aligned(
// CHECK: call @adjacent_multiple_all_aligned(
// CHECK: return
func.func @caller_adjacent_multiple_all_aligned(%a: vector<64xi1>, %b: vector<64xi1>) {
  %addr0 = arith.constant 16 : i64
  %d0 = hivm.hir.pointer_cast(%addr0) : memref<256xi1, #hivm.address_space<ub>>
  func.call @adjacent_multiple_all_aligned(%d0, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  %addr1 = arith.constant 32 : i64
  %d1 = hivm.hir.pointer_cast(%addr1) : memref<256xi1, #hivm.address_space<ub>>
  func.call @adjacent_multiple_all_aligned(%d1, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  return
}

// -----

// One safe call does not authorize the second unaligned call.
// CHECK-LABEL: func.func @adjacent_multiple_one_unaligned(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @adjacent_multiple_one_unaligned(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi1>, %b: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0_dst = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_dst[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_dst = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_dst[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_adjacent_multiple_one_unaligned(
// CHECK: call @adjacent_multiple_one_unaligned(
// CHECK: call @adjacent_multiple_one_unaligned(
// CHECK: return
func.func @caller_adjacent_multiple_one_unaligned(%a: vector<64xi1>, %b: vector<64xi1>) {
  %addr0 = arith.constant 16 : i64
  %d0 = hivm.hir.pointer_cast(%addr0) : memref<256xi1, #hivm.address_space<ub>>
  func.call @adjacent_multiple_one_unaligned(%d0, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  %addr1 = arith.constant 24 : i64
  %d1 = hivm.hir.pointer_cast(%addr1) : memref<256xi1, #hivm.address_space<ub>>
  func.call @adjacent_multiple_one_unaligned(%d1, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  return
}

// -----

// All pointer choices must preserve alignment.
// CHECK-LABEL: func.func @adjacent_select_all_aligned(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK: return
func.func @adjacent_select_all_aligned(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi1>, %b: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0_dst = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_dst[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_dst = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_dst[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_adjacent_select_all_aligned(
// CHECK: call @adjacent_select_all_aligned(
// CHECK: return
func.func @caller_adjacent_select_all_aligned(%choose: i1, %a: vector<64xi1>, %b: vector<64xi1>) {
  %lo0 = arith.constant 16 : i64
  %hi0 = arith.constant 32 : i64
  %dlo = hivm.hir.pointer_cast(%lo0) : memref<256xi1, #hivm.address_space<ub>>
  %dhi = hivm.hir.pointer_cast(%hi0) : memref<256xi1, #hivm.address_space<ub>>
  %d0 = arith.select %choose, %dlo, %dhi : memref<256xi1, #hivm.address_space<ub>>
  func.call @adjacent_select_all_aligned(%d0, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  return
}

// -----

// All pointer choices must preserve alignment.
// CHECK-LABEL: func.func @adjacent_select_one_unaligned(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @adjacent_select_one_unaligned(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi1>, %b: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0_dst = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_dst[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_dst = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_dst[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_adjacent_select_one_unaligned(
// CHECK: call @adjacent_select_one_unaligned(
// CHECK: return
func.func @caller_adjacent_select_one_unaligned(%choose: i1, %a: vector<64xi1>, %b: vector<64xi1>) {
  %lo0 = arith.constant 16 : i64
  %hi0 = arith.constant 24 : i64
  %dlo = hivm.hir.pointer_cast(%lo0) : memref<256xi1, #hivm.address_space<ub>>
  %dhi = hivm.hir.pointer_cast(%hi0) : memref<256xi1, #hivm.address_space<ub>>
  %d0 = arith.select %choose, %dlo, %dhi : memref<256xi1, #hivm.address_space<ub>>
  func.call @adjacent_select_one_unaligned(%d0, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  return
}

// -----

// A function address use leaves possible additional callers unknown.
// CHECK-LABEL: func.func @adjacent_address_taken(
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: return
func.func @adjacent_address_taken(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi1>, %b: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0_dst = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_dst[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_dst = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_dst[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_adjacent_address_taken(
// CHECK: call @adjacent_address_taken(
// CHECK: return
func.func @caller_adjacent_address_taken(%a: vector<64xi1>, %b: vector<64xi1>) {
  %addr0 = arith.constant 16 : i64
  %d0 = hivm.hir.pointer_cast(%addr0) : memref<256xi1, #hivm.address_space<ub>>
  func.call @adjacent_address_taken(%d0, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  return
}
// CHECK-LABEL: func.func @address_of_adjacent_address_taken(
// CHECK: constant @adjacent_address_taken :
func.func @address_of_adjacent_address_taken() -> ((memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()) {
  %f = func.constant @adjacent_address_taken : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  return %f : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
}

// -----

// An unknown argument base cannot prove alignment. It must not block a
// separate group whose alignment is established by a local allocation.
// CHECK-LABEL: func.func @mixed_local_and_argument(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK: return
func.func @mixed_local_and_argument(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi1>, %b: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0_dst = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_dst[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_dst = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_dst[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %local = memref.alloc() {alignment = 16 : i64} : memref<256xi1, #hivm.address_space<ub>>
  %v0_local = memref.reinterpret_cast %local to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_local[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_local = memref.reinterpret_cast %local to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_local[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_mixed_local_and_argument(
// CHECK: call @mixed_local_and_argument(
// CHECK: return
func.func @caller_mixed_local_and_argument(%unknown: i64, %a: vector<64xi1>, %b: vector<64xi1>) {
  %d0 = hivm.hir.pointer_cast(%unknown) : memref<256xi1, #hivm.address_space<ub>>
  func.call @mixed_local_and_argument(%d0, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  return
}

// -----

// An unregistered region may hide a symbol table, making use enumeration
// unknown. Reject the parameter group, but keep the local allocation merge.
// CHECK-LABEL: func.func @unknown_symbol_scope(
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b32"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK: return
func.func @unknown_symbol_scope(%dst: memref<256xi1, #hivm.address_space<ub>>, %a: vector<64xi1>, %b: vector<64xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  "store_test.unknown_region"() ({
    "store_test.yield"() : () -> ()
  }) : () -> ()
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
  %v0_dst = memref.reinterpret_cast %dst to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_dst[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_dst = memref.reinterpret_cast %dst to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_dst[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %local = memref.alloc() {alignment = 16 : i64} : memref<256xi1, #hivm.address_space<ub>>
  %v0_local = memref.reinterpret_cast %local to offset: [0], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v0_local[%c0], %mask, %a {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  %v1_local = memref.reinterpret_cast %local to offset: [64], sizes: [64], strides: [1] : memref<256xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %v1_local[%c0], %mask, %b {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  return
}
// CHECK-LABEL: func.func @caller_unknown_symbol_scope(
// CHECK: call @unknown_symbol_scope(
// CHECK: return
func.func @caller_unknown_symbol_scope(%a: vector<64xi1>, %b: vector<64xi1>) {
  %addr0 = arith.constant 16 : i64
  %d0 = hivm.hir.pointer_cast(%addr0) : memref<256xi1, #hivm.address_space<ub>>
  func.call @unknown_symbol_scope(%d0, %a, %b) : (memref<256xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>) -> ()
  return
}
