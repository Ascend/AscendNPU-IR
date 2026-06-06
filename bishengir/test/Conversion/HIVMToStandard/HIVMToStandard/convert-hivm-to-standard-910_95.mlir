// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 %s -convert-hivm-to-std  -split-input-file| FileCheck %s

module {
  // CHECK-LABEL: test_gather_1d_int8_t
  func.func @test_gather_1d_int8_t() {
    %c96_i64 = arith.constant 96 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64

    %alloc = memref.alloc() : memref<1xi8, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1xi32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xi8, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<1xi32, #hivm.address_space<ub>>
    // CHECK: call @gather_1d_int8_t
    hivm.hir.vgather ins(%alloc : memref<1xi8, #hivm.address_space<ub>>) 
                     indices(%alloc_1 : memref<1xi32, #hivm.address_space<ub>>) 
                     outs(%alloc_2 : memref<1xi8, #hivm.address_space<ub>>) 
                     temp_buffer(%alloc_3 : memref<1xi32, #hivm.address_space<ub>>)
    return
 }
}

// -----
module {
  // CHECK-LABEL: test_histogram_1d_int16_t
  func.func @test_histogram_1d_int16_t() {
    %c4_i64 = arith.constant 4 : i64
    %alloc_in = memref.alloc() : memref<8xi16, #hivm.address_space<ub>>
    %alloc_out = memref.alloc() : memref<4xi32, #hivm.address_space<ub>>
    // CHECK: %[[CAST_IN:.*]] = memref.cast {{.*}} : memref<8xi16, #hivm.address_space<ub>> to memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %[[CAST_OUT:.*]] = memref.cast {{.*}} : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: call @histogram_1d_int16_t(%[[CAST_IN]], %c4_i64, %[[CAST_OUT]])
    hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
      "__builtin_histogram"
      ins(%alloc_in, %c4_i64 : memref<8xi16, #hivm.address_space<ub>>, i64)
      outs(%alloc_out : memref<4xi32, #hivm.address_space<ub>>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_histogram_1d_int32_t
  func.func @test_histogram_1d_int32_t() {
    %c4_i64 = arith.constant 4 : i64
    %alloc_in = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    %alloc_out = memref.alloc() : memref<4xi32, #hivm.address_space<ub>>
    // CHECK: %[[CAST_IN:.*]] = memref.cast {{.*}} : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %[[CAST_OUT:.*]] = memref.cast {{.*}} : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: call @histogram_1d_int32_t(%[[CAST_IN]], %c4_i64, %[[CAST_OUT]])
    hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
      "__builtin_histogram"
      ins(%alloc_in, %c4_i64 : memref<8xi32, #hivm.address_space<ub>>, i64)
      outs(%alloc_out : memref<4xi32, #hivm.address_space<ub>>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_histogram_1d_masked_int8_t
  func.func @test_histogram_1d_masked_int8_t() {
    %c4_i64 = arith.constant 4 : i64
    %alloc_in = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %alloc_mask = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %alloc_out = memref.alloc() : memref<4xi32, #hivm.address_space<ub>>
    // CHECK: %[[CAST_IN:.*]] = memref.cast {{.*}} : memref<8xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %[[CAST_MASK:.*]] = memref.cast {{.*}} : memref<8xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %[[CAST_OUT:.*]] = memref.cast {{.*}} : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: call @histogram_1d_masked_int8_t(%[[CAST_IN]], %[[CAST_MASK]], %c4_i64, %[[CAST_OUT]])
    hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
      "__builtin_histogram"
      ins(%alloc_in, %alloc_mask, %c4_i64 : memref<8xi8, #hivm.address_space<ub>>, memref<8xi8, #hivm.address_space<ub>>, i64)
      outs(%alloc_out : memref<4xi32, #hivm.address_space<ub>>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_histogram_1d_masked_int64_t
  func.func @test_histogram_1d_masked_int64_t() {
    %c4_i64 = arith.constant 4 : i64
    %alloc_in = memref.alloc() : memref<8xi64, #hivm.address_space<ub>>
    %alloc_mask = memref.alloc() : memref<8xi8, #hivm.address_space<ub>>
    %alloc_out = memref.alloc() : memref<4xi32, #hivm.address_space<ub>>
    // CHECK: %[[CAST_IN:.*]] = memref.cast {{.*}} : memref<8xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %[[CAST_MASK:.*]] = memref.cast {{.*}} : memref<8xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: %[[CAST_OUT:.*]] = memref.cast {{.*}} : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    // CHECK: call @histogram_1d_masked_int64_t(%[[CAST_IN]], %[[CAST_MASK]], %c4_i64, %[[CAST_OUT]])
    hivm.hir.custom {gm_addr_args_indices = array<i32: 0>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
      "__builtin_histogram"
      ins(%alloc_in, %alloc_mask, %c4_i64 : memref<8xi64, #hivm.address_space<ub>>, memref<8xi8, #hivm.address_space<ub>>, i64)
      outs(%alloc_out : memref<4xi32, #hivm.address_space<ub>>)
    return
  }
}