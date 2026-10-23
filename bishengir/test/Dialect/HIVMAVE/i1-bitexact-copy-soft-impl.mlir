// RUN: bishengir-opt %s -convert-vector-to-hivmave -ave-i1op-soft-impl | FileCheck %s

// End-to-end lowering of the bit-exact i1 fallback copy emitted by
// EnableStrideAlign (see ../HIVM/hivm-enable-stride-align-i1-bitexact-copy.mlir
// for where this IR comes from): a loop nest of singleton vector<1xi1>
// transfer pairs over non-byte-expressible i1 layouts. The DMA copy library
// would round 19-bit rows up to whole bytes and corrupt the bit packing;
// the transfer pairs instead lower to a predicate-block read-modify-write
// that touches exactly one bit per iteration:
//   linearize [r, c] -> r * 256 + c  (the producer's padded row stride)
//   split into VL-aligned base (div 8 -> bytes) and bit offset (mod 8)
//   one-hot predicate (plt(o+1) xor plt(o)) selects the target bit
//   old <- vload, delta <- preg.and(diff, oneHot), new <- preg.xor(old, delta)
//   masked_store <NORM_B8> writes the containing byte back.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @i1_fallback_rmw
  func.func @i1_fallback_rmw(%src: memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %dst: memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c13 = arith.constant 13 : index
    %c19 = arith.constant 19 : index
    %c1 = arith.constant 1 : index
    %zero_i1 = arith.constant 0 : i1
    scf.for %r = %c0 to %c13 step %c1 {
      scf.for %c = %c0 to %c19 step %c1 {
        %es = memref.subview %src[%r, %c] [1, 1] [1, 1]
          : memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %ed = memref.subview %dst[%r, %c] [1, 1] [1, 1]
          : memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %v = vector.transfer_read %es[%c0], %zero_i1 {in_bounds = [true]} : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>
        vector.transfer_write %v, %ed[%c0] {in_bounds = [true]} : vector<1xi1>, memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
}

// The whole copy is rewritten; the transfer ops never survive.
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write
// Each linearized access is split into a VL-aligned byte base and a bit
// offset inside that byte.
// CHECK: arith.muli %{{.*}}, %{{.*}} : index
// CHECK: %[[BITS:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [1], strides: [1]
// CHECK: %[[BITPOS:.*]] = arith.subi %{{.*}}, %{{.*}} : i32
// Load side: VL-aligned load plus one-hot extraction of the target bit.
// CHECK: ave.hir.vload <NORM> %{{.*}}[%{{.*}}] {"1xi1 processed"} : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<1xi1>
// CHECK: %[[UP:.*]], %{{.*}} = ave.hir.plt %{{.*}} : vector<256xi1>, index
// CHECK: %[[DOWN:.*]], %{{.*}} = ave.hir.plt %{{.*}} : vector<256xi1>, index
// CHECK: %[[ONEHOT_L:.*]] = ave.hir.preg.xor <b8> %[[UP]], %[[DOWN]], %{{.*}} : vector<256xi1>
// Store side: read-modify-write on the containing byte.
// CHECK: ave.hir.vload <NORM> %{{.*}}[%{{.*}}] {"1xi1 processed"} : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi1>
// CHECK: %[[UP2:.*]], %{{.*}} = ave.hir.plt %{{.*}} : vector<256xi1>, index
// CHECK: %[[DOWN2:.*]], %{{.*}} = ave.hir.plt %{{.*}} : vector<256xi1>, index
// CHECK: %[[ONEHOT_S:.*]] = ave.hir.preg.xor <b8> %[[UP2]], %[[DOWN2]], %{{.*}} : vector<256xi1>
// CHECK: %[[DIFF:.*]] = ave.hir.preg.xor <b8> %{{.*}}, %{{.*}}, %{{.*}} : vector<256xi1>
// CHECK: %[[DELTA:.*]] = ave.hir.preg.and <b8> %[[DIFF]], %[[ONEHOT_S]], %{{.*}} : vector<256xi1>
// CHECK: %[[NEW:.*]] = ave.hir.preg.xor <b8> %{{.*}}, %[[DELTA]], %{{.*}} : vector<256xi1>
// CHECK: ave.hir.masked_store <NORM_B8> %{{.*}}[%{{.*}}], %{{.*}}, %[[NEW]] {"1xi1 processed"} : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi1>
