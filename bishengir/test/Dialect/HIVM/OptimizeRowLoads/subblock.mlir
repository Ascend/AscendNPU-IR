// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize | FileCheck %s --check-prefix=SCALAR --implicit-check-not=hivm.hir.custom --implicit-check-not=hivm.hir.varange
// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize -one-shot-bufferize="bufferize-function-boundaries" -hivm-infer-mem-scope -convert-hivm-to-std | FileCheck %s --check-prefix=LIB
// Match row-load windows directly after subblock tiling.
// SCALAR-LABEL: func.func @split_rows
// SCALAR: scf.for
// SCALAR: affine.apply
// SCALAR: scf.for {{.*}} step %c2
// SCALAR: tensor.extract %arg1[
// SCALAR-NOT: scf.if
// SCALAR: sizes: [2, 512]
// SCALAR: scf.for {{.*}} step %c2
// SCALAR-NOT: scf.if
// SCALAR: sizes: [2, 64]
// SCALAR: return
// LIB-LABEL: func.func @split_rows
// LIB: call @load_gm_to_ubuf_2d_bfloat16_t
// LIB-NOT: call @gather_load
// LIB: return
#start = affine_map<()[s0] -> (s0 * 64)>
#tail = affine_map<()[s0] -> (s0 + 512)>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @split_rows(%base: memref<?xbf16> {tt.tensor_kind = 0 : i32},
      %ids: tensor<128xi32>, %pitch: i32, %batch: i64,
      %out: memref<64x512xbf16>, %tailOut: memref<64x64xbf16>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0 : i32
    scf.for %s = %c0 to %c2 step %c1 {
      %lower = affine.apply #start()[%s]
      %end = arith.addi %lower, %c64 : index
      %active = arith.minsi %end, %c128 : index
      %upper = arith.minsi %end, %active : index
      %main = memref.alloc() : memref<64x512xbf16>
      scf.for %i = %lower to %upper step %c1 {
        %id = tensor.extract %ids[%i] : tensor<128xi32>
        %valid = arith.cmpi sge, %id, %zero : i32
        %safe = arith.select %valid, %id, %zero : i32
        %scaled = arith.muli %safe, %pitch : i32
        %wide = arith.extsi %scaled : i32 to i64
        %address = arith.addi %batch, %wide : i64
        %index = arith.index_cast %address : i64 to index
        %src = memref.reinterpret_cast %base to offset: [%index], sizes: [1, 512], strides: [512, 1]
          : memref<?xbf16> to memref<1x512xbf16, strided<[512, 1], offset: ?>>
        %next = arith.addi %i, %c1 : index
        %first = arith.maxsi %i, %lower : index
        %last = arith.minsi %next, %end : index
        %clipped = arith.maxsi %first, %last : index
        %size = arith.subi %clipped, %first : index
        %local = arith.subi %first, %lower : index
        %dst = memref.subview %main[%local, 0] [%size, 512] [1, 1]
          : memref<64x512xbf16> to memref<?x512xbf16, strided<[512, 1], offset: ?>>
        hivm.hir.load ins(%src : memref<1x512xbf16, strided<[512, 1], offset: ?>>)
          outs(%dst : memref<?x512xbf16, strided<[512, 1], offset: ?>>)
          eviction_policy = <EvictFirst> core_type = <VECTOR>
      } {hivm.parallel_loop}
      %main_tensor = bufferization.to_tensor %main restrict writable : memref<64x512xbf16>
      bufferization.materialize_in_destination %main_tensor in writable %out
        : (tensor<64x512xbf16>, memref<64x512xbf16>) -> ()
      %tail = memref.alloc() : memref<64x64xbf16>
      scf.for %i = %lower to %upper step %c1 {
        %id = tensor.extract %ids[%i] : tensor<128xi32>
        %valid = arith.cmpi sge, %id, %zero : i32
        %safe = arith.select %valid, %id, %zero : i32
        %scaled = arith.muli %safe, %pitch : i32
        %wide = arith.extsi %scaled : i32 to i64
        %address = arith.addi %batch, %wide : i64
        %index = arith.index_cast %address : i64 to index
        %tailIndex = affine.apply #tail()[%index]
        %src = memref.reinterpret_cast %base to offset: [%tailIndex], sizes: [1, 64], strides: [64, 1]
          : memref<?xbf16> to memref<1x64xbf16, strided<[64, 1], offset: ?>>
        %next = arith.addi %i, %c1 : index
        %first = arith.maxsi %i, %lower : index
        %last = arith.minsi %next, %end : index
        %clipped = arith.maxsi %first, %last : index
        %size = arith.subi %clipped, %first : index
        %local = arith.subi %first, %lower : index
        %dst = memref.subview %tail[%local, 0] [%size, 64] [1, 1]
          : memref<64x64xbf16> to memref<?x64xbf16, strided<[64, 1], offset: ?>>
        hivm.hir.load ins(%src : memref<1x64xbf16, strided<[64, 1], offset: ?>>)
          outs(%dst : memref<?x64xbf16, strided<[64, 1], offset: ?>>)
          eviction_policy = <EvictFirst> core_type = <VECTOR>
      } {hivm.parallel_loop}
      %tail_tensor = bufferization.to_tensor %tail restrict writable : memref<64x64xbf16>
      bufferization.materialize_in_destination %tail_tensor in writable %tailOut
        : (tensor<64x64xbf16>, memref<64x64xbf16>) -> ()
      %other = memref.alloc() : memref<64x512xbf16>
      scf.for %i = %lower to %upper step %c1 {
        %id = tensor.extract %ids[%i] : tensor<128xi32>
        %valid = arith.cmpi sge, %id, %zero : i32
        %safe = arith.select %valid, %id, %zero : i32
        %scaled = arith.muli %safe, %pitch : i32
        %wide = arith.extsi %scaled : i32 to i64
        %address = arith.addi %batch, %wide : i64
        %index = arith.index_cast %address : i64 to index
        %src = memref.reinterpret_cast %base to offset: [%index], sizes: [1, 512], strides: [512, 1]
          : memref<?xbf16> to memref<1x512xbf16, strided<[512, 1], offset: ?>>
        %next = arith.addi %i, %c1 : index
        %first = arith.maxsi %i, %lower : index
        %last = arith.minsi %next, %end : index
        %clipped = arith.maxsi %first, %last : index
        %size = arith.subi %clipped, %first : index
        %local = arith.subi %first, %lower : index
        %dst = memref.subview %other[%local, 0] [%size, 512] [1, 1]
          : memref<64x512xbf16> to memref<?x512xbf16, strided<[512, 1], offset: ?>>
        hivm.hir.load ins(%src : memref<1x512xbf16, strided<[512, 1], offset: ?>>)
          outs(%dst : memref<?x512xbf16, strided<[512, 1], offset: ?>>)
          eviction_policy = <EvictFirst> core_type = <VECTOR>
      }
      memref.copy %other, %out : memref<64x512xbf16> to memref<64x512xbf16>
    }
    return
  }

  // Four subblocks (256 = 4 * 64 rows) prove the split is not hardcoded to two.
  // SCALAR-LABEL: func.func @split_rows_4
  // SCALAR: scf.for {{.*}} step %c2
  // SCALAR: sizes: [2, 512]
  // SCALAR: return
  func.func @split_rows_4(%base: memref<?xbf16> {tt.tensor_kind = 0 : i32},
      %ids: tensor<256xi32>, %pitch: i32, %batch: i64,
      %out: memref<64x512xbf16>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0 : i32
    scf.for %s = %c0 to %c4 step %c1 {
      %lower = affine.apply #start()[%s]
      %end = arith.addi %lower, %c64 : index
      %active = arith.minsi %end, %c256 : index
      %upper = arith.minsi %end, %active : index
      %main = memref.alloc() : memref<64x512xbf16>
      scf.for %i = %lower to %upper step %c1 {
        %id = tensor.extract %ids[%i] : tensor<256xi32>
        %valid = arith.cmpi sge, %id, %zero : i32
        %safe = arith.select %valid, %id, %zero : i32
        %scaled = arith.muli %safe, %pitch : i32
        %wide = arith.extsi %scaled : i32 to i64
        %address = arith.addi %batch, %wide : i64
        %index = arith.index_cast %address : i64 to index
        %src = memref.reinterpret_cast %base to offset: [%index], sizes: [1, 512], strides: [512, 1]
          : memref<?xbf16> to memref<1x512xbf16, strided<[512, 1], offset: ?>>
        %next = arith.addi %i, %c1 : index
        %first = arith.maxsi %i, %lower : index
        %last = arith.minsi %next, %end : index
        %clipped = arith.maxsi %first, %last : index
        %size = arith.subi %clipped, %first : index
        %local = arith.subi %first, %lower : index
        %dst = memref.subview %main[%local, 0] [%size, 512] [1, 1]
          : memref<64x512xbf16> to memref<?x512xbf16, strided<[512, 1], offset: ?>>
        hivm.hir.load ins(%src : memref<1x512xbf16, strided<[512, 1], offset: ?>>)
          outs(%dst : memref<?x512xbf16, strided<[512, 1], offset: ?>>)
          eviction_policy = <EvictFirst> core_type = <VECTOR>
      } {hivm.parallel_loop}
      %main_tensor = bufferization.to_tensor %main restrict writable : memref<64x512xbf16>
      bufferization.materialize_in_destination %main_tensor in writable %out
        : (tensor<64x512xbf16>, memref<64x512xbf16>) -> ()
    }
    return
  }
}
