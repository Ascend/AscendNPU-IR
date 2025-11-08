// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-alloc-extra-buffer))" -split-input-file %s | FileCheck %s

func.func @test_vbrc_temp_buffer_first_axis_align() {
  %src = memref.alloc() : memref<1x32xf16>
  %dst = memref.alloc() : memref<10x32xf16>

  // CHECK-NO: hivm.hir.vbrc{{.*}}temp_buffer
  hivm.hir.vbrc ins(%src : memref<1x32xf16>)
                outs(%dst : memref<10x32xf16>)
                broadcast_dims = [0]

  return
}

// -----

func.func @test_vbrc_temp_buffer_first_axis_unalign() {
  %src = memref.alloc() : memref<1x3xf16>
  %dst = memref.alloc() : memref<10x3xf16>

  // CHECK: memref.alloc() : memref<160xf16>
  // CHECK: hivm.hir.vbrc{{.*}}temp_buffer({{.*}}memref<160xf16>)
  hivm.hir.vbrc ins(%src : memref<1x3xf16>)
                outs(%dst : memref<10x3xf16>)
                broadcast_dims = [0]

  return
}

// -----

func.func @test_vbrc_temp_buffer_last_axis_align_1() {
  %src = memref.alloc() : memref<10x1xf16>
  %dst = memref.alloc() : memref<10x32xf16>

  // CHECK: memref.alloc() : memref<256xf16>
  // CHECK: hivm.hir.vbrc{{.*}}temp_buffer({{.*}}memref<256xf16>)
  hivm.hir.vbrc ins(%src : memref<10x1xf16>)
                outs(%dst : memref<10x32xf16>)
                broadcast_dims = [1]

  return
}

// -----

func.func @test_vbrc_temp_buffer_last_axis_align_2() {
  %src = memref.alloc() : memref<2x1xf16>
  %dst = memref.alloc() : memref<2x32xf16>

  // CHECK: memref.alloc() : memref<128xf16>
  // CHECK: hivm.hir.vbrc{{.*}}temp_buffer({{.*}}memref<128xf16>)
  hivm.hir.vbrc ins(%src : memref<2x1xf16>)
                outs(%dst : memref<2x32xf16>)
                broadcast_dims = [1]

  return
}

// -----

func.func @test_vbrc_temp_buffer_last_axis_unalign() {
  %src = memref.alloc() : memref<10x1xf16>
  %dst = memref.alloc() : memref<10x33xf16>

  // CHECK: memref.alloc() : memref<768xf16>
  // CHECK: hivm.hir.vbrc{{.*}}temp_buffer({{.*}}memref<768xf16>)
  hivm.hir.vbrc ins(%src : memref<10x1xf16>)
                outs(%dst : memref<10x33xf16>)
                broadcast_dims = [1]

  return
}

// -----

func.func @test_vbrc_temp_buffer_last_axis_align_view() {
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %src = memref.alloca() : memref<16x16x8x32x8x16x8x1xi16>
  %dst = memref.alloca() : memref<16x16x8x32x8x16x8x32xi16>
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      scf.for %arg2 = %c0 to %c8 step %c1 {
        scf.for %arg3 = %c0 to %c32 step %c1 {
          scf.for %arg4 = %c0 to %c8 step %c1 {
            scf.for %arg5 = %c0 to %c16 step %c1 {
              %srcview = memref.subview %src[%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, 0, 0] [1, 1, 1, 1, 1, 1, 8, 1] [1, 1, 1, 1, 1, 1, 1, 1] : memref<16x16x8x32x8x16x8x1xi16> to memref<8x1xi16, strided<[1, 1], offset: ?>>
              %dstview = memref.subview %dst[%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, 0, 0] [1, 1, 1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1, 1, 1] : memref<16x16x8x32x8x16x8x32xi16> to memref<8x32xi16, strided<[32, 1], offset: ?>>

              // CHECK: memref.alloc() : memref<128xi16>
              // CHECK: hivm.hir.vbrc{{.*}}temp_buffer({{.*}}memref<128xi16>)
              hivm.hir.vbrc ins(%srcview : memref<8x1xi16, strided<[1, 1], offset: ?>>)
                            outs(%dstview : memref<8x32xi16, strided<[32, 1], offset: ?>>)
                            broadcast_dims = [1]
            }
          }
        }
      }
    }
  }
  return
}

// -----

func.func @test_vbrc_temp_buffer_last_axis_align_view_2() {
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %src = memref.alloca() : memref<32x16x8x1xi16>
  %dst = memref.alloca() : memref<32x16x8x32xi16>
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      scf.for %arg2 = %c0 to %c8 step %c1 {
        scf.for %arg3 = %c0 to %c32 step %c1 {
          scf.for %arg4 = %c0 to %c8 step %c1 {
            scf.for %arg5 = %c0 to %c16 step %c1
              iter_args(%dst_arg = %dst) -> (memref<32x16x8x32xi16>) {

                // CHECK: memref.alloc() : memref<128xi16>
                // CHECK: hivm.hir.vbrc{{.*}}temp_buffer({{.*}}memref<128xi16>)
                hivm.hir.vbrc ins(%src : memref<32x16x8x1xi16>)
                              outs(%dst_arg : memref<32x16x8x32xi16>)
                              broadcast_dims = [3]

                scf.yield %dst_arg : memref<32x16x8x32xi16>
            }
          }
        }
      }
    }
  }
  return
}

// -----

func.func @test_vreduce_temp_buffer() {
  %src = memref.alloc() : memref<10x32xf16>
  %dst = memref.alloc() : memref<10x1xf16>

  hivm.hir.vreduce <sum> ins(%src : memref<10x32xf16>)
                         outs(%dst : memref<10x1xf16>)
                         reduce_dims = [1]

  return
}

// -----

func.func @test_vreduce_temp_buffer_2() {
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %src = memref.alloca() : memref<32x16x8x32xf16>
  %dst = memref.alloca() : memref<1x16x8x32xf16>
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      scf.for %arg2 = %c0 to %c8 step %c1 {
        scf.for %arg3 = %c0 to %c32 step %c1 {
          scf.for %arg4 = %c0 to %c8 step %c1 {
            scf.for %arg5 = %c0 to %c16 step %c1 {
                %src2 = memref.collapse_shape %src [[0], [1, 2], [3]] : memref<32x16x8x32xf16> into memref<32x128x32xf16>
                %dst2 = memref.collapse_shape %dst [[0], [1, 2], [3]] : memref<1x16x8x32xf16> into memref<1x128x32xf16>

                // CHECK: memref.alloc() : memref<131072xf16>
                // CHECK: hivm.hir.vreduce{{.*}}temp_buffer({{.*}}memref<131072xf16>)
                hivm.hir.vreduce <sum> ins(%src2 : memref<32x128x32xf16>)
                                       outs(%dst2 : memref<1x128x32xf16>)
                                       reduce_dims = [0]
            }
          }
        }
      }
    }
  }
  return
}

// -----
// CHECK-LABEL: func @test_vsel
// CHECK: hivm.hir.vsel{{.*}}temp_buffer({{.*}})
func.func @test_vsel() {
  %src_0 = memref.alloca() : memref<16xi1>
  %src_1 = memref.alloca() : memref<16xf16>
  %src_2 = memref.alloca() : memref<16xf16>
  %dst0 = memref.alloca() : memref<16xf16>
  hivm.hir.vsel ins(%src_0, %src_1, %src_2 : memref<16xi1>, memref<16xf16>, memref<16xf16>)
                outs(%dst0 : memref<16xf16>)
  return
}

// -----
// CHECK-LABEL: func @test_vsel
// CHECK: hivm.hir.vsel{{.*}}temp_buffer({{.*}}memref<12xi64>)
func.func @test_vsel_i64() {
  %src_0 = memref.alloca() : memref<16xi8>
  %src_1 = memref.alloca() : memref<16xi64>
  %src_2 = memref.alloca() : memref<16xi64>
  %dst0 = memref.alloca() : memref<16xi64>
  hivm.hir.vsel ins(%src_0, %src_1, %src_2 : memref<16xi8>, memref<16xi64>, memref<16xi64>)
                outs(%dst0 : memref<16xi64>)
  return
}

// -----

// CHECK-LABEL: func @test_cast_bool
// CHECK: hivm.hir.vcast{{.*}}temp_buffer({{.*}}memref<48xf16>)
func.func @test_cast_bool() {
  %src_0 = memref.alloca() : memref<16xi1>
  %dst0 = memref.alloca() : memref<16xf16>
  hivm.hir.vcast ins(%src_0 : memref<16xi1>)
                 outs(%dst0 : memref<16xf16>)
  return
}

// -----

func.func @test_cast_s642s32_overflow() {
  %src = memref.alloc() : memref<10x32xi64>
  %dst = memref.alloc() : memref<10x32xi32>

  // CHECK: memref.alloc() : memref<0xi64>
  // CHECK: hivm.hir.vcast{{.*}}temp_buffer({{.*}}memref<0xi64>)
  hivm.hir.vcast ins(%src : memref<10x32xi64>)
                 outs(%dst : memref<10x32xi32>)
                 round_mode = <truncwithoverflow>
  return
}

// -----

func.func @test_cast_s322s16_overflow() {
  %src = memref.alloc() : memref<10x32xi32>
  %dst = memref.alloc() : memref<10x32xi16>

  // CHECK: memref.alloc() : memref<0xi32>
  // CHECK: hivm.hir.vcast{{.*}}temp_buffer({{.*}}memref<0xi32>)
  hivm.hir.vcast ins(%src : memref<10x32xi32>)
                 outs(%dst : memref<10x32xi16>)
                 round_mode = <truncwithoverflow>
  return
}

// -----

// CHECK-LABEL: func @test_vinterleave_op
func.func @test_vinterleave_op() {
  %a_f16 = memref.alloc() : memref<2x16xf16>
  %b_f16 = memref.alloc() : memref<2x16xf16>
  %c_f16 = memref.alloc() : memref<2x32xf16>
  // CHECK: memref.alloc() : memref<288xf16>
  // CHECK: hivm.hir.vinterleave{{.*}} interleave_channel_nums = 2 temp_buffer({{.*}}memref<288xf16>)
  hivm.hir.vinterleave ins(%a_f16, %b_f16 : memref<2x16xf16>, memref<2x16xf16>)
                outs(%c_f16 : memref<2x32xf16>)
                interleave_channel_nums = 2
  return
}

// -----

// CHECK-LABEL: func @test_vreduce_op_static
func.func @test_vreduce_op_static() {
  %src_r = memref.alloc() : memref<10xi32>
  %dst_r = memref.alloc() : memref<1xi32>
  %dst_r_index = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce{{.*}} temp_buffer({{.*}}memref<8xi32>)
  hivm.hir.vreduce <max_with_index_left> ins(%src_r : memref<10xi32>)
                                    outs(%dst_r, %dst_r_index : memref<1xi32>, memref<1xi32>)
                                    reduce_dims = [0]
  %src_ar = memref.alloc() : memref<2x10xi32>
  %dst_ar = memref.alloc() : memref<2x1xi32>
  %dst_ar_index = memref.alloc() : memref<2x1xi32>
  // CHECK: hivm.hir.vreduce{{.*}} temp_buffer({{.*}}memref<8xi32>)
  hivm.hir.vreduce <max_with_index_left> ins(%src_ar : memref<2x10xi32>)
                                    outs(%dst_ar, %dst_ar_index : memref<2x1xi32>, memref<2x1xi32>)
                                    reduce_dims = [1]
  %src_ra = memref.alloc() : memref<10x2xi32>
  %dst_ra = memref.alloc() : memref<1x2xi32>
  %dst_ra_index = memref.alloc() : memref<1x2xi32>
  // CHECK: hivm.hir.vreduce{{.*}} temp_buffer({{.*}}memref<24xi32>)
  hivm.hir.vreduce <max_with_index_left> ins(%src_ra : memref<10x2xi32>)
                                    outs(%dst_ra, %dst_ra_index : memref<1x2xi32>, memref<1x2xi32>)
                                    reduce_dims = [0]
  %src_mid = memref.alloc() : memref<2x10x2xi32>
  %dst_mid = memref.alloc() : memref<2x1x2xi32>
  %dst_mid_index = memref.alloc() : memref<2x1x2xi32>
  // CHECK: hivm.hir.vreduce{{.*}} temp_buffer({{.*}}memref<24xi32>)
  hivm.hir.vreduce <max_with_index_left> ins(%src_mid : memref<2x10x2xi32>)
                                    outs(%dst_mid, %dst_mid_index : memref<2x1x2xi32>, memref<2x1x2xi32>)
                                    reduce_dims = [1]
  return
}

// -----

// CHECK-LABEL: func @test_vreduce_op_dynamic
func.func @test_vreduce_op_dynamic(%offset: index, %d: index) {
  %a = memref.alloc() : memref<256xi8>
  %src = memref.view %a[%offset][%d] : memref<256xi8> to memref<?x2xf32>
  %dst = memref.alloc() : memref<1x2xf32>
  %dst_index = memref.alloc() : memref<1x2xi32>
  // CHECK: hivm.hir.vreduce{{.*}} temp_buffer({{.*}}memref<96xf32>)
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<?x2xf32>)
                                    outs(%dst, %dst_index : memref<1x2xf32>, memref<1x2xi32>)
                                    reduce_dims = [0]
  return
}

// -----

// CHECK-LABEL: func @test_vtranspose_int64_op
func.func @test_vtranspose_int64_op() {
  %src = memref.alloc() : memref<8x16xi64>
  %dst = memref.alloc() : memref<16x8xi64>
  // CHECK: hivm.hir.vtranspose{{.*}} temp_buffer({{.*}}memref<512xi64>) permutation = [1, 0]
  hivm.hir.vtranspose 
    ins(%src : memref<8x16xi64>) 
    outs(%dst : memref<16x8xi64>) 
    permutation = [1, 0]
  return
}

// -----

// CHECK-LABEL: func @test_vtranspose_uint64_op
func.func @test_vtranspose_uint64_op() {
  %src = memref.alloc() : memref<8x16xui64>
  %dst = memref.alloc() : memref<16x8xui64>
  // CHECK: hivm.hir.vtranspose{{.*}} temp_buffer({{.*}}memref<512xui64>) permutation = [1, 0]
  hivm.hir.vtranspose 
    ins(%src : memref<8x16xui64>) 
    outs(%dst : memref<16x8xui64>) 
    permutation = [1, 0]
  return
}

// -----

func.func @test_reduce_temp_buffer() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 0 : i32
  %cst_3 = arith.constant 0xFF800000 : f32
  %alloc_1 = memref.alloc() : memref<256x8x1xf32, #hivm.address_space<ub>>
  %subview_1 = memref.subview %alloc_1[0, 0, 0] [256, 1, 1] [1, 1, 1] : memref<256x8x1xf32, #hivm.address_space<ub>> to memref<256x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %27:1 = scf.for %arg1 = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%arg2 = %subview_1) -> (memref<256x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)  : i32 {
    %alloc_2 = memref.alloc() : memref<256x8x1xf32, #hivm.address_space<ub>>
    %subview_2 = memref.subview %alloc_2[0, 0, 0] [256, 1, 1] [1, 1, 1] : memref<256x8x1xf32, #hivm.address_space<ub>> to memref<256x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    scf.yield %subview_2 : memref<256x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  }
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst_3, %alloc_3[%c0] : memref<1xf32, #hivm.address_space<ub>>
  %expand_shape = memref.expand_shape %alloc_3 [[0, 1]] output_shape [1, 1] : memref<1xf32, #hivm.address_space<ub>> into memref<1x1xf32, #hivm.address_space<ub>>
  %subview_3 = memref.subview %27#0[0, 0] [256, 1] [1, 1] : memref<256x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<256xf32, strided<[8]>, #hivm.address_space<ub>>
  %subview_4 = memref.subview %expand_shape[0, 0] [1, 1] [1, 1] : memref<1x1xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1]>, #hivm.address_space<ub>>
  %base_buffer_1, %offset_2, %sizes_3, %strides_4 = memref.extract_strided_metadata %subview_3 : memref<256xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_1 = memref.reinterpret_cast %base_buffer_1 to offset: [%c0], sizes: [%c256, 1], strides: [%c8, 1] : memref<f32, #hivm.address_space<ub>> to memref<256x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %subview_4 : memref<1xf32, strided<[1]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %reinterpret_cast_2 = memref.reinterpret_cast %base_buffer_5 to offset: [%c0], sizes: [%c1, 1], strides: [%c1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vreduce{{.*}}temp_buffer({{.*}}memref<2048xf32>)
  hivm.hir.vreduce <max> ins(%reinterpret_cast_1 : memref<256x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [0]
  return
}

// -----

func.func @test_reducear_temp_buffer() {
  %src = memref.alloc() : memref<8x7x2x15x8x1xi32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<8x7x2x15xi32, #hivm.address_space<ub>>
  %subview = memref.subview %src[0, 0, 0, 0, 0, 0] [8, 7, 2, 15, 4, 1] [1, 1, 1, 1, 1, 1] : memref<8x7x2x15x8x1xi32, #hivm.address_space<ub>> to memref<8x7x2x15x4xi32, strided<[1680, 240, 120, 8, 1]>, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %subview [[0, 1, 2, 3], [4]] : memref<8x7x2x15x4xi32, strided<[1680, 240, 120, 8, 1]>, #hivm.address_space<ub>> into memref<1680x4xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  %expand_shape = memref.expand_shape %dst [[0], [1], [2], [3, 4]] output_shape [8, 7, 2, 15, 1] : memref<8x7x2x15xi32, #hivm.address_space<ub>> into memref<8x7x2x15x1xi32, #hivm.address_space<ub>>
  %collapse_shape_1 = memref.collapse_shape %expand_shape [[0, 1, 2, 3], [4]] : memref<8x7x2x15x1xi32, #hivm.address_space<ub>> into memref<1680x1xi32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vreduce{{.*}}temp_buffer({{.*}}memref<26880xi32>)
  hivm.hir.vreduce <min> ins(%collapse_shape : memref<1680x4xi32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_1 : memref<1680x1xi32, #hivm.address_space<ub>>) reduce_dims = [1]
  return
}

// -----

func.func @test_reducear_xori_temp_buffer() {
  %src = memref.alloc() : memref<8x7x2x15x8x1xi32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<8x7x2x15xi32, #hivm.address_space<ub>>
  %subview = memref.subview %src[0, 0, 0, 0, 0, 0] [8, 7, 2, 15, 4, 1] [1, 1, 1, 1, 1, 1] : memref<8x7x2x15x8x1xi32, #hivm.address_space<ub>> to memref<8x7x2x15x4xi32, strided<[1680, 240, 120, 8, 1]>, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %subview [[0, 1, 2, 3], [4]] : memref<8x7x2x15x4xi32, strided<[1680, 240, 120, 8, 1]>, #hivm.address_space<ub>> into memref<1680x4xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  %expand_shape = memref.expand_shape %dst [[0], [1], [2], [3, 4]] output_shape [8, 7, 2, 15, 1] : memref<8x7x2x15xi32, #hivm.address_space<ub>> into memref<8x7x2x15x1xi32, #hivm.address_space<ub>>
  %collapse_shape_1 = memref.collapse_shape %expand_shape [[0, 1, 2, 3], [4]] : memref<8x7x2x15x1xi32, #hivm.address_space<ub>> into memref<1680x1xi32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vreduce{{.*}}temp_buffer({{.*}}memref<40320xi32>)
  hivm.hir.vreduce <xori> ins(%collapse_shape : memref<1680x4xi32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_1 : memref<1680x1xi32, #hivm.address_space<ub>>) reduce_dims = [1]
  return
}

// -----

// CHECK-LABEL: func @test_vsort_op_float
func.func @test_vsort_op_float() {
  %src = memref.alloc() : memref<32xf32>
  %dst_value = memref.alloc() : memref<32xf32>
  %dst_index = memref.alloc() : memref<32xi32>
  // CHECK: memref.alloc() : memref<192xf32>
  // CHECK: hivm.hir.vsort{{.*}}temp_buffer({{.*}}memref<192xf32>)
  hivm.hir.vsort ins(%src : memref<32xf32>)
                 outs(%dst_value : memref<32xf32>)
                 descending = false
                 sort_axis = 0
  // CHECK: memref.alloc() : memref<160xf32>
  // CHECK: hivm.hir.vsort{{.*}}temp_buffer({{.*}}memref<160xf32>)
  hivm.hir.vsort ins(%src : memref<32xf32>)
                 outs(%dst_value : memref<32xf32>)
                 descending = true
                 sort_axis = 0
  // CHECK: memref.alloc() : memref<160xf32>
  // CHECK: hivm.hir.vsort{{.*}}temp_buffer({{.*}}memref<160xf32>)
  hivm.hir.vsort ins(%src : memref<32xf32>)
                 outs(%dst_value, %dst_index : memref<32xf32>, memref<32xi32>)
                 descending = false
                 sort_axis = 0
  // CHECK: memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.vsort{{.*}}temp_buffer({{.*}}memref<128xf32>)
  hivm.hir.vsort ins(%src : memref<32xf32>)
                 outs(%dst_value, %dst_index : memref<32xf32>, memref<32xi32>)
                 descending = true
                 sort_axis = 0
  return
}

// -----

// CHECK-LABEL: func @test_vsort_op_half
func.func @test_vsort_op_half() {
  %src = memref.alloc() : memref<32xf16>
  %dst_value = memref.alloc() : memref<32xf16>
  %dst_index = memref.alloc() : memref<32xi32>
  // CHECK: memref.alloc() : memref<352xf16>
  // CHECK: hivm.hir.vsort{{.*}}temp_buffer({{.*}}memref<352xf16>)
  hivm.hir.vsort ins(%src : memref<32xf16>)
                 outs(%dst_value : memref<32xf16>)
                 descending = false
                 sort_axis = 0
  // CHECK: memref.alloc() : memref<320xf16>
  // CHECK: hivm.hir.vsort{{.*}}temp_buffer({{.*}}memref<320xf16>)
  hivm.hir.vsort ins(%src : memref<32xf16>)
                 outs(%dst_value : memref<32xf16>)
                 descending = true
                 sort_axis = 0
  // CHECK: memref.alloc() : memref<288xf16>
  // CHECK: hivm.hir.vsort{{.*}}temp_buffer({{.*}}memref<288xf16>)
  hivm.hir.vsort ins(%src : memref<32xf16>)
                 outs(%dst_value, %dst_index : memref<32xf16>, memref<32xi32>)
                 descending = false
                 sort_axis = 0
  // CHECK: memref.alloc() : memref<256xf16>
  // CHECK: hivm.hir.vsort{{.*}}temp_buffer({{.*}}memref<256xf16>)
  hivm.hir.vsort ins(%src : memref<32xf16>)
                 outs(%dst_value, %dst_index : memref<32xf16>, memref<32xi32>)
                 descending = true
                 sort_axis = 0
  return
}

// -----

func.func @test_vadd_1d_last_axis_inline_brc_temp_buffer() {
  %src0 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %src0_inline_brc = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
  %src1 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %src1_inline_brc = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<8xf32>)
  hivm.hir.vadd ins(%src0_inline_brc, %src1 : memref<1xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8xf32, #hivm.address_space<ub>>) broadcast = [0]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<8xf32>)
  hivm.hir.vadd ins(%src0, %src1_inline_brc : memref<8xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8xf32, #hivm.address_space<ub>>) broadcast = [0]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<16xf32>)
  hivm.hir.vadd ins(%src0_inline_brc, %src1_inline_brc : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8xf32, #hivm.address_space<ub>>) broadcast = [0]
  return
}

// -----

func.func @test_vadd_2d_last_axis_inline_brc_temp_buffer() {
  %src0 = memref.alloc() : memref<8x8xf32, #hivm.address_space<ub>>
  %src0_last_inline_brc = memref.alloc() : memref<8x1xf32, #hivm.address_space<ub>>
  %src0_last_first_inline_brc = memref.alloc() : memref<1x1xf32, #hivm.address_space<ub>>
  %src1 = memref.alloc() : memref<8x8xf32, #hivm.address_space<ub>>
  %src1_last_inline_brc = memref.alloc() : memref<8x1xf32, #hivm.address_space<ub>>
  %src1_last_first_inline_brc = memref.alloc() : memref<1x1xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<8x8xf32, #hivm.address_space<ub>>

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<64xf32>)
  hivm.hir.vadd ins(%src0, %src1_last_inline_brc : memref<8x8xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [1]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<8xf32>)
  hivm.hir.vadd ins(%src0, %src1_last_first_inline_brc : memref<8x8xf32, #hivm.address_space<ub>>, memref<1x1xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [0, 1]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<64xf32>)
  hivm.hir.vadd ins(%src0_last_inline_brc, %src1 : memref<8x1xf32, #hivm.address_space<ub>>, memref<8x8xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [1]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<128xf32>)
  hivm.hir.vadd ins(%src0_last_inline_brc, %src1_last_inline_brc : memref<8x1xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [1]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<72xf32>)
  hivm.hir.vadd ins(%src0_last_inline_brc, %src1_last_first_inline_brc : memref<8x1xf32, #hivm.address_space<ub>>, memref<1x1xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [0, 1]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<8xf32>)
  hivm.hir.vadd ins(%src0_last_first_inline_brc, %src1 : memref<1x1xf32, #hivm.address_space<ub>>, memref<8x8xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [0, 1]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<72xf32>)
  hivm.hir.vadd ins(%src0_last_first_inline_brc, %src1_last_inline_brc : memref<1x1xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [0, 1]

  // CHECK: hivm.hir.vadd{{.*}}temp_buffer({{.*}}memref<16xf32>)
  hivm.hir.vadd ins(%src0_last_first_inline_brc, %src1_last_first_inline_brc : memref<1x1xf32, #hivm.address_space<ub>>, memref<1x1xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [0, 1]
  return
}

// -----

func.func @test_vmins_2d_last_axis_inline_brc_temp_buffer() {
  %src0 = memref.alloc() : memref<8x8xf32, #hivm.address_space<ub>>
  %src0_last_inline_brc = memref.alloc() : memref<8x1xf32, #hivm.address_space<ub>>
  %c8 = arith.constant 8.0 : f32
  %dst = memref.alloc() : memref<8x8xf32, #hivm.address_space<ub>>

  // CHECK: hivm.hir.vmin{{.*}}temp_buffer({{.*}}memref<64xf32>)
  hivm.hir.vmin ins(%src0_last_inline_brc, %c8 : memref<8x1xf32, #hivm.address_space<ub>>, f32)
                outs(%dst : memref<8x8xf32, #hivm.address_space<ub>>) broadcast = [1]
  return
}
