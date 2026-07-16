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
    // CHECK: call @gather_simt_1d_int8_t_int32_t
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

// -----
module {
// CHECK-LABEL: test_gather_simt_builtin
  func.func @test_gather_simt_builtin(%src: memref<5x11xf32>,
                                      %offsets: memref<7x11xi32>,
                                      %dst: memref<7x11xf32>) {
    // CHECK: %[[AXIS:.*]] = arith.constant 0 : i32
    // CHECK: call @gather_simt_2d_float_int32_t({{.*}}, {{.*}}, {{.*}}, %[[AXIS]])
    hivm.hir.vgather
        {gather_axis = 0 : i64}
        ins(%src : memref<5x11xf32>)
        indices(%offsets : memref<7x11xi32>)
        outs(%dst : memref<7x11xf32>)
    return
  }
}

// -----
module {
// CHECK-LABEL: test_gather_1d_no_axis
  func.func @test_gather_1d_no_axis(%src: memref<16xf32>,
                                    %indices: memref<16xi32>,
                                    %dst: memref<16xf32>) {
    // CHECK: %[[AXIS:.*]] = arith.constant 0 : i32
    // CHECK: call @gather_simt_1d_float_int32_t({{.*}}, {{.*}}, {{.*}}, %[[AXIS]])
    hivm.hir.vgather ins(%src : memref<16xf32>)
                     indices(%indices : memref<16xi32>)
                     outs(%dst : memref<16xf32>)
    return
  }
}

// -----
module {
// CHECK-LABEL: test_gather_1d_axis_minus1
  func.func @test_gather_1d_axis_minus1(%src: memref<16xf32>,
                                        %indices: memref<16xi32>,
                                        %dst: memref<16xf32>) {
    // CHECK: %[[AXIS:.*]] = arith.constant 0 : i32
    // CHECK: call @gather_simt_1d_float_int32_t({{.*}}, {{.*}}, {{.*}}, %[[AXIS]])
    hivm.hir.vgather {gather_axis = -1 : i64}
        ins(%src : memref<16xf32>)
        indices(%indices : memref<16xi32>)
        outs(%dst : memref<16xf32>)
    return
  }
}

// -----
module {
// CHECK-LABEL: test_gather_1d_axis_tail
  func.func @test_gather_1d_axis_tail(%src: memref<16xf32>,
                                      %indices: memref<16xi32>,
                                      %dst: memref<16xf32>) {
    // CHECK: %[[AXIS:.*]] = arith.constant 0 : i32
    // CHECK: call @gather_simt_1d_float_int32_t({{.*}}, {{.*}}, {{.*}}, %[[AXIS]])
    hivm.hir.vgather {gather_axis = 0 : i64}
        ins(%src : memref<16xf32>)
        indices(%indices : memref<16xi32>)
        outs(%dst : memref<16xf32>)
    return
  }
}

// -----
module {
// CHECK-LABEL: test_gather_2d_tail_axis
  func.func @test_gather_2d_tail_axis(%src: memref<5x11xf32>,
                                      %indices: memref<7x11xi32>,
                                      %dst: memref<7x11xf32>) {
    // CHECK: %[[AXIS:.*]] = arith.constant 1 : i32
    // CHECK: call @gather_simt_2d_float_int32_t({{.*}}, {{.*}}, {{.*}}, %[[AXIS]])
    hivm.hir.vgather {gather_axis = 1 : i64}
        ins(%src : memref<5x11xf32>)
        indices(%indices : memref<7x11xi32>)
        outs(%dst : memref<7x11xf32>)
    return
  }
}

// -----
module {
// CHECK-LABEL: test_gather_simt_2d_half_i16
  func.func @test_gather_simt_2d_half_i16(%src: memref<5x11xf16>,
                                          %indices: memref<7x11xi16>,
                                          %dst: memref<7x11xf16>) {
    // CHECK: %[[AXIS:.*]] = arith.constant 0 : i32
    // CHECK: call @gather_simt_2d_half_int16_t({{.*}}, {{.*}}, {{.*}}, %[[AXIS]])
    hivm.hir.vgather {gather_axis = 0 : i64}
        ins(%src : memref<5x11xf16>)
        indices(%indices : memref<7x11xi16>)
        outs(%dst : memref<7x11xf16>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: func @test_gather_1d_1_size
  func.func @test_gather_1d_1_size() attributes {hacc.entry} {
    %src = memref.alloc() : memref<216x1x1xf32>
    %idx = memref.alloc() : memref<216x1x1xi32>
    %dst = memref.alloc() : memref<216x1x1xf32>
    %tmp = memref.alloc() : memref<216x1x1xi32>
    // CHECK: %[[AXIS:.*]] = arith.constant 2 : i32
    // CHECK: call @gather_simt_3d_float_int32_t({{.*}}, {{.*}}, {{.*}}, %[[AXIS]])
    hivm.hir.vgather ins(%src : memref<216x1x1xf32>)
                     indices(%idx : memref<216x1x1xi32>)
                     outs(%dst : memref<216x1x1xf32>)
                     temp_buffer(%tmp : memref<216x1x1xi32>)
    return
  }
}