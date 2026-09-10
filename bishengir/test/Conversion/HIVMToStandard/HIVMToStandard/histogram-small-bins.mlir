// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 %s -convert-hivm-to-std -split-input-file | FileCheck %s

module {
  // CHECK-LABEL: test_histogram_bins_256
  func.func @test_histogram_bins_256() {
    %c256_i64 = arith.constant 256 : i64
    %alloc_in = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %alloc_out = memref.alloc() : memref<256xi32, #hivm.address_space<ub>>
    // CHECK: %[[CAST_IN:.*]] = memref.cast {{.*}} : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %[[CAST_OUT:.*]] = memref.cast {{.*}} : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: call @histogram_1d_int32_t_small_bins(%[[CAST_IN]], %c256_i64, %[[CAST_OUT]])
    hivm.hir.custom {symbol = "histogram_1d_int32_t", gm_addr_args_indices = array<i32: 0>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
      "__builtin_histogram"
      ins(%alloc_in, %c256_i64 : memref<8xi32, #hivm.address_space<ub>>, i64)
      outs(%alloc_out : memref<256xi32, #hivm.address_space<ub>>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_histogram_bins_257
  func.func @test_histogram_bins_257() {
    %c257_i64 = arith.constant 257 : i64
    %alloc_in = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %alloc_out = memref.alloc() : memref<257xi32, #hivm.address_space<ub>>
    // CHECK: %[[CAST_IN:.*]] = memref.cast {{.*}} : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %[[CAST_OUT:.*]] = memref.cast {{.*}} : memref<257xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: call @histogram_1d_int32_t(%[[CAST_IN]], %c257_i64, %[[CAST_OUT]])
    hivm.hir.custom {symbol = "histogram_1d_int32_t", gm_addr_args_indices = array<i32: 0>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
      "__builtin_histogram"
      ins(%alloc_in, %c257_i64 : memref<8xi32, #hivm.address_space<ub>>, i64)
      outs(%alloc_out : memref<257xi32, #hivm.address_space<ub>>)
    return
  }
}
