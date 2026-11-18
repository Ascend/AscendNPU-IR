// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -convert-hivmave-to-ave-intrin -cse %s -split-input-file | FileCheck %s \
// RUN: --implicit-check-not=hivm.ppack --implicit-check-not=hivm.pstu \
// RUN: --implicit-check-not=hivm.vstas

// PSTS MaskDist::DIST_PACK (1) selects the even bits of the 256-bit physical
// predicate and writes exactly 16 bytes. Check the distribution operand as well
// as the address: PPACK + PSTS DIST_NORM (0) writes 32 bytes and would clobber
// padding or overrun the allocation in these cases.

// The view starts 16 bytes into the allocation and has only 16 bytes of data.
// As in normalized AVE IR, the bit offset is in the view, with store index zero.
// This covers a 16-byte-aligned address that is not 32-byte aligned.
// CHECK-LABEL: @pb16_at_16_byte_offset
// CHECK: %[[INDEX:.*]] = llvm.mlir.constant(0 : index)
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[DATA:.*]] = builtin.unrealized_conversion_cast %arg1 : vector<128xi1> to vector<256xi1>
// CHECK: %[[BIT_OFFSET:.*]] = llvm.mlir.constant(128 : index)
// CHECK: %[[BITS_PER_BYTE:.*]] = llvm.mlir.constant(8 : index)
// CHECK: %[[BYTE_OFFSET:.*]] = llvm.udiv %[[BIT_OFFSET]], %[[BITS_PER_BYTE]]
// CHECK: %[[START:.*]] = llvm.getelementptr {{.*}}[%[[BYTE_OFFSET]]]
// CHECK: %[[PTR:.*]] = llvm.getelementptr %[[START]][%[[INDEX]]]
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[DATA]], %[[PTR]], %[[ZERO]], %[[PACK]], %[[ZERO]])
// CHECK-NOT: hivm.psts
// CHECK: return
func.func @pb16_at_16_byte_offset(%dst: memref<128xi1, strided<[1], offset: 128>, #hivm.address_space<ub>>, %data: vector<128xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %data {functionType = #ave.func_dist_type<pb16>} : memref<128xi1, strided<[1], offset: 128>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  return
}

// -----

// Each row has 128 data bits and a stride of 256 bits. The first store must
// touch bytes [0, 16), leaving the row's padding [16, 32) intact. The last store
// must touch [32, 48); the memref does not promise padding after the last row.
// CHECK-LABEL: @pb16_preserve_row_padding
// CHECK: %[[ROW0:.*]] = llvm.mlir.constant(0 : index)
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[DATA:.*]] = builtin.unrealized_conversion_cast %arg1 : vector<128xi1> to vector<256xi1>
// CHECK: %[[PTR0:.*]] = llvm.getelementptr {{.*}}[%[[ROW0]]]
// CHECK: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[DATA]], %[[PTR0]], %[[ZERO]], %[[PACK]], %[[ZERO]])
// CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(256 : index)
// CHECK: %[[BITS_PER_BYTE:.*]] = llvm.mlir.constant(8 : index)
// CHECK: %[[ROW1:.*]] = llvm.udiv %[[STRIDE]], %[[BITS_PER_BYTE]]
// CHECK: %[[START1:.*]] = llvm.getelementptr {{.*}}[%[[ROW1]]]
// CHECK: %[[PTR1:.*]] = llvm.getelementptr %[[START1]][%[[ROW0]]]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[DATA]], %[[PTR1]], %[[ZERO]], %[[PACK]], %[[ZERO]])
// CHECK-NOT: hivm.psts
// CHECK: return
func.func @pb16_preserve_row_padding(%dst: memref<2x128xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %data: vector<128xi1>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
  %row0 = memref.reinterpret_cast %dst to offset: [0], sizes: [128], strides: [1] : memref<2x128xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<128xi1, #hivm.address_space<ub>>
  %row1 = memref.reinterpret_cast %dst to offset: [256], sizes: [128], strides: [1] : memref<2x128xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<128xi1, strided<[1], offset: 256>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %row0[%c0], %mask, %data {functionType = #ave.func_dist_type<pb16>} : memref<128xi1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  ave.hir.masked_store <NORM_B8> %row1[%c0], %mask, %data {functionType = #ave.func_dist_type<pb16>} : memref<128xi1, strided<[1], offset: 256>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi1>
  return
}
