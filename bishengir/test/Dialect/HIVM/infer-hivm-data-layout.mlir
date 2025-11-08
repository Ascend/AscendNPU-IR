// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-data-layout -split-input-file | FileCheck %s

// CHECK: #map = affine_map<()[s0, s1] -> ((s0 + 15) floordiv 16)>
// CHECK: #map1 = affine_map<()[s0, s1] -> ((s1 + 15) floordiv 16)>
// CHECK: module {
// CHECK:   func.func @test_infer_data_layout_basic(%arg0: i32, %arg1: i32, %arg2: i32)
// CHECK:     %[[APPLY:.*]] = affine.apply #map()
// CHECK:     %[[APPLY1:.*]] = affine.apply #map1()
// CHECK:     %[[CONST16:.*]] = arith.constant 16 : index
// CHECK:     %[[CONST16_1:.*]] = arith.constant 16 : index
// CHECK:     %[[ALLOC:.*]] = memref.alloc(%[[APPLY1]], %[[APPLY]], %[[CONST16]], %[[CONST16_1]]) {alignment = 64 : i64} : memref<?x?x?x?xf32>
// CHECK:     scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[ARG:.*]] = %[[ALLOC]]) -> (memref<?x?x?x?xf32>)  : i32 {
// CHECK:       %[[APPLY2:.*]] = affine.apply #map()
// CHECK:       %[[APPLY3:.*]] = affine.apply #map1()
// CHECK:       %[[CONST16_2:.*]] = arith.constant 16 : index
// CHECK:       %[[CONST16_3:.*]] = arith.constant 16 : index
// CHECK:       %[[ALLOC1:.*]] = memref.alloc(%[[APPLY3]], %[[APPLY2]], %[[CONST16_2]], %[[CONST16_3]]) : memref<?x?x?x?xf16>
// CHECK:       %[[APPLY4:.*]] = affine.apply #map()
// CHECK:       %[[APPLY5:.*]] = affine.apply #map1()
// CHECK:       %[[CONST16_4:.*]] = arith.constant 16 : index
// CHECK:       %[[CONST16_5:.*]] = arith.constant 16 : index
// CHECK:       %[[ALLOC2:.*]] = memref.alloc(%[[APPLY5]], %[[APPLY4]], %[[CONST16_4]], %[[CONST16_5]]) : memref<?x?x?x?xf16>
// CHECK:       hivm.hir.mmadL1 ins(%[[ALLOC1]], %[[ALLOC2]], {{.*}}, {{.*}}, {{.*}}, {{.*}} : memref<?x?x?x?xf16>, memref<?x?x?x?xf16>, i1, index, index, index)
// CHECK-SAME:                  outs(%[[ARG:.*]] : memref<?x?x?x?xf32>)
// CHECK:       scf.yield {{.*}} : memref<?x?x?x?xf32>
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }

module {
  func.func @test_infer_data_layout_basic(%arg0 : i32,
                                          %arg1 : i32,
                                          %arg2 : i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32>
    %ret = scf.for %iv = %arg0 to %arg1 step %arg2 iter_args(%l0c = %alloc) -> (memref<256x128xf32>) : i32 {
      %l1A = memref.alloc() : memref<256x128xf16>
      %l1B = memref.alloc() : memref<128x128xf16>
      %init_cond = arith.cmpi eq, %iv, %arg1 : i32
      hivm.hir.mmadL1 ins(%l1A, %l1B, %init_cond, %c128, %c128, %c128 :
                             memref<256x128xf16>, memref<128x128xf16>, i1, index, index, index)
                      outs(%l0c : memref<256x128xf32>)
      scf.yield %l0c : memref<256x128xf32>
    }
    return
  }
}

// -----
// CHECK: #map = affine_map<()[s0, s1] -> ((s0 + 15) floordiv 16)>
// CHECK: #map1 = affine_map<()[s0, s1] -> ((s1 + 15) floordiv 16)>
// CHECK: #map2 = affine_map<()[s0, s1] -> ((s1 + 7) floordiv 8)>
// CHECK: module {
// CHECK:   func.func @test_infer_data_layout_basic_f32(%arg0: i32, %arg1: i32, %arg2: i32)
// CHECK:     %[[APPLY:.*]] = affine.apply #map()
// CHECK:     %[[APPLY1:.*]] = affine.apply #map1()
// CHECK:     %[[CONST16:.*]] = arith.constant 16 : index
// CHECK:     %[[CONST16_1:.*]] = arith.constant 16 : index
// CHECK:     %[[ALLOC:.*]] = memref.alloc(%[[APPLY1]], %[[APPLY]], %[[CONST16]], %[[CONST16_1]]) {alignment = 64 : i64} : memref<?x?x?x?xf32>
// CHECK:     scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[ARG:.*]] = %[[ALLOC]]) -> (memref<?x?x?x?xf32>)  : i32 {
// CHECK:       %[[APPLY2:.*]] = affine.apply #map()
// CHECK:       %[[APPLY3:.*]] = affine.apply #map2()
// CHECK:       %[[CONST16_2:.*]] = arith.constant 16 : index
// CHECK:       %[[CONST8:.*]] = arith.constant 8 : index
// CHECK:       %[[ALLOC1:.*]] = memref.alloc(%[[APPLY3]], %[[APPLY2]], %[[CONST16_2]], %[[CONST8]]) : memref<?x?x?x?xf32>
// CHECK:       %[[APPLY4:.*]] = affine.apply #map()
// CHECK:       %[[APPLY5:.*]] = affine.apply #map2()
// CHECK:       %[[CONST16_3:.*]] = arith.constant 16 : index
// CHECK:       %[[CONST8_1:.*]] = arith.constant 8 : index
// CHECK:       %[[ALLOC2:.*]] = memref.alloc(%[[APPLY5]], %[[APPLY4]], %[[CONST16_3]], %[[CONST8_1]]) : memref<?x?x?x?xf32>
// CHECK:       hivm.hir.mmadL1 ins(%[[ALLOC1]], %[[ALLOC2]], {{.*}}, {{.*}}, {{.*}}, {{.*}} : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i1, index, index, index)
// CHECK-SAME:                  outs(%[[ARG:.*]] : memref<?x?x?x?xf32>)
// CHECK:       scf.yield {{.*}} : memref<?x?x?x?xf32>
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }

module {
  func.func @test_infer_data_layout_basic_f32(%arg0 : i32,
                                          %arg1 : i32,
                                          %arg2 : i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32>
    %ret = scf.for %iv = %arg0 to %arg1 step %arg2 iter_args(%l0c = %alloc) -> (memref<256x128xf32>) : i32 {
      %l1A = memref.alloc() : memref<256x128xf32>
      %l1B = memref.alloc() : memref<128x128xf32>
      %init_cond = arith.cmpi eq, %iv, %arg1 : i32
      hivm.hir.mmadL1 ins(%l1A, %l1B, %init_cond, %c128, %c128, %c128 :
                             memref<256x128xf32>, memref<128x128xf32>, i1, index, index, index)
                      outs(%l0c : memref<256x128xf32>)
      scf.yield %l0c : memref<256x128xf32>
    }
    return
  }
}

// -----

module {
  // CHECK-LABEL: test_infer_data_layout_complicated
  func.func @test_infer_data_layout_complicated(%arg0 : i32,
                                                %arg1 : i32,
                                                %arg2 : i32,
                                                %gmA: memref<*xf16, #hivm.address_space<gm>>,
                                                %gmB: memref<*xf16, #hivm.address_space<gm>>,
                                                %gmC: memref<*xf32, #hivm.address_space<gm>>,
                                                %unalignedM : index,
                                                %unalignedK : index,
                                                %unalignedN : index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    %gmC_basic_block = memref.reinterpret_cast %gmC to offset: [%c0], sizes: [128, 128], strides: [1, 1] :
                               memref<*xf32, #hivm.address_space<gm>> to
                               memref<128x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>

    %ret = scf.for %iv = %arg0 to %arg1 step %arg2 iter_args(%l0c = %alloc) -> (memref<128x128xf32>) : i32 {
      %gmA_basic_block = memref.reinterpret_cast %gmA to offset: [%c0], sizes: [128, 128], strides: [1, 1] :
                               memref<*xf16, #hivm.address_space<gm>> to
                               memref<128x128xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %gmB_basic_block = memref.reinterpret_cast %gmB to offset: [%c0], sizes: [128, 128], strides: [1, 1] :
                                   memref<*xf16, #hivm.address_space<gm>> to
                                   memref<128x128xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>

      %l1A = memref.alloc() : memref<128x128xf16>
      %gmA_basic_block_subview = memref.subview %gmA_basic_block[0, 0] [%unalignedM, %unalignedK] [1, 1] :
                                    memref<128x128xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to
                                    memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %l1A_subview = memref.subview %l1A[0, 0] [%unalignedM, %unalignedK] [1, 1] :
                                    memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      // CHECK:     hivm.hir.nd2nz
      // CHECK-NOT: memref.copy
      memref.copy %gmA_basic_block_subview, %l1A_subview :
         memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1]>>

      %l1B = memref.alloc() : memref<128x128xf16>
      %gmB_basic_block_subview = memref.subview %gmB_basic_block[0, 0] [%unalignedK, %unalignedN] [1, 1] :
                                     memref<128x128xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to
                                     memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %l1B_subview = memref.subview %l1B[0, 0] [%unalignedK, %unalignedN] [1, 1] :
                                     memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      // CHECK:     hivm.hir.nd2nz
      // CHECK-NOT: memref.copy
      memref.copy %gmB_basic_block_subview, %l1B_subview :
          memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1]>>

      %init_cond = arith.cmpi eq, %iv, %arg1 : i32
      // CHECK:          hivm.hir.mmadL1 ins({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}} : memref<?x?x?x?xf16>, memref<?x?x?x?xf16>, i1, index, index, index)
      // CHECK-SAME:     outs({{.*}} : memref<?x?x?x?xf32>)
      hivm.hir.mmadL1 ins(%l1A, %l1B, %init_cond, %unalignedM, %unalignedK, %unalignedN : memref<128x128xf16>, memref<128x128xf16>, i1, index, index, index)
                      outs(%l0c : memref<128x128xf32>)
      scf.yield %l0c : memref<128x128xf32>
    }
    hivm.hir.fixpipe {enable_nz2nd} ins(%ret : memref<128x128xf32>)
                                    outs(%gmC_basic_block : memref<128x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>)
  return
  }
}

// -----

module {
// CHECK-LABEL: test_infer_data_layout_hivm_copy
  func.func @test_infer_data_layout_hivm_copy(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32) attributes {func_dyn_memref_args = dense<[true, true, true, false, false, false]> : vector<6xi1>, global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg5, %arg4 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg3 : i32
    scf.for %arg6 = %4 to %c1_i32 step %arg3  : i32 {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
      %5 = arith.divsi %arg6, %c2_i32 : i32
      %6 = arith.remsi %arg6, %c2_i32 : i32
      %7 = arith.cmpi eq, %5, %c0_i32 : i32
      %8 = scf.if %7 -> (i32) {
        %34 = arith.muli %5, %c2_i32 : i32
        %35 = arith.subi %c1_i32, %34 : i32
        scf.yield %35 : i32
      } else {
        scf.yield %c2_i32 : i32
      }
      %9 = arith.divsi %6, %8 : i32
      %10 = arith.muli %5, %c2_i32 : i32
      %11 = arith.remsi %6, %8 : i32
      %12 = arith.addi %10, %11 : i32
      %13 = arith.remsi %5, %c2_i32 : i32
      %14 = arith.cmpi ne, %13, %c0_i32 : i32
      %15 = scf.if %14 -> (i32) {
        %34 = arith.subi %c0_i32, %9 : i32
        scf.yield %34 : i32
      } else {
        scf.yield %9 : i32
      }
      %16 = arith.muli %15, %c16_i32 : i32
      %17 = arith.muli %12, %c16_i32 : i32
      %18 = arith.index_cast %16 : i32 to index
      %19 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%18]
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%19], sizes: [16, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %cast = memref.cast %reinterpret_cast : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %20 = arith.index_cast %17 : i32 to index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [16, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_1 = memref.cast %reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %alloc_2 = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %cast : memref<16x16xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index, index, index
      %21 = affine.apply affine_map<()[s0] -> (-s0 + (s0 floordiv 16) * 16 + 16)>()[%offset]
      %22 = arith.minsi %21, %c16 : index
      %23 = affine.apply affine_map<()[s0] -> (-(s0 floordiv 16) + ((s0 floordiv 16) floordiv 16) * 16 + 16)>()[%offset]
      %24 = arith.minsi %23, %c16 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%24, %22] [1, 1] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %alloc_2[0, 0] [%24, %22] [1, 1] : memref<16x16xf32, #hivm.address_space<cbuf>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cbuf>>
      // CHECK:     hivm.hir.nd2nz
      // CHECK-NOT: hivm.hir.load
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cbuf>>)
      %alloc_4 = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
      %base_buffer_5, %offset_6, %sizes_7:2, %strides_8:2 = memref.extract_strided_metadata %cast_1 : memref<16x16xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index, index, index
      %25 = affine.apply affine_map<()[s0] -> (-s0 + (s0 floordiv 16) * 16 + 16)>()[%offset_6]
      %26 = arith.minsi %25, %c16 : index
      %27 = affine.apply affine_map<()[s0] -> (-(s0 floordiv 16) + ((s0 floordiv 16) floordiv 16) * 16 + 16)>()[%offset_6]
      %28 = arith.minsi %27, %c16 : index
      %subview_9 = memref.subview %reinterpret_cast_0[0, 0] [%28, %26] [1, 1] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %alloc_4[0, 0] [%28, %26] [1, 1] : memref<16x16xf32, #hivm.address_space<cbuf>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cbuf>>
      // CHECK:     hivm.hir.nd2nz
      // CHECK-NOT: hivm.hir.load
      hivm.hir.load ins(%subview_9 : memref<?x?xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cbuf>>)
      hivm.hir.mmadL1 ins(%alloc_2, %alloc_4, %true, %24, %22, %26 : memref<16x16xf32, #hivm.address_space<cbuf>>, memref<16x16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc : memref<16x16xf32, #hivm.address_space<cc>>)
      %29 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 16)>()[%20, %18]
      %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [16, 16], strides: [16, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_12 = memref.cast %reinterpret_cast_11 : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %cast_12 : memref<16x16xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> -> memref<f16, #hivm.address_space<gm>>, index, index, index, index, index
      %30 = affine.apply affine_map<()[s0] -> (-s0 + (s0 floordiv 16) * 16 + 16)>()[%offset_14]
      %31 = arith.minsi %30, %c16 : index
      %32 = affine.apply affine_map<()[s0] -> (-(s0 floordiv 16) + ((s0 floordiv 16) floordiv 16) * 16 + 16)>()[%offset_14]
      %33 = arith.minsi %32, %c16 : index
      %subview_17 = memref.subview %alloc[0, 0] [%33, %31] [1, 1] : memref<16x16xf32, #hivm.address_space<cc>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cc>>
      %subview_18 = memref.subview %reinterpret_cast_11[0, 0] [%33, %31] [1, 1] : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_19 = memref.cast %subview_18 : memref<?x?xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.fixpipe {enable_nz2nd} ins(%subview_17 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<cc>>) outs(%cast_19 : memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----
// CHECK-LABEL: mm_set_init_out_buffer_nd2nz
func.func @mm_set_init_out_buffer_nd2nz(%arg0: memref<256x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, %arg1: memref<128x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, %arg2: i32, %arg3: i1)attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "mix"} {
  %cst = arith.constant 2.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %0 = arith.index_cast %arg2 : i32 to index
  %alloc = memref.alloc() : memref<256x128xf32, #hivm.address_space<cbuf>>
  %alloc_0 = memref.alloc() : memref<128x128xf32, #hivm.address_space<cbuf>>
  %subview = memref.subview %arg0[0, 0] [256, %0] [1, 1] : memref<256x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<256x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %subview_1 = memref.subview %alloc[0, 0] [256, %0] [1, 1] : memref<256x128xf32, #hivm.address_space<cbuf>> to memref<256x?xf32, strided<[128, 1]>, #hivm.address_space<cbuf>>
  // CHECK: hivm.hir.nd2nz {dst_continuous} ins({{.*}} : memref<256x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs({{.*}} : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = {{.*}} : f32 init_condition = {{.*}} : i1
  hivm.hir.load ins(%subview : memref<256x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<256x?xf32, strided<[128, 1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %arg3 : i1
  %1 = arith.minsi %0, %c128 : index
  %subview_2 = memref.subview %arg1[0, 0] [%1, 128] [1, 1] : memref<128x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %subview_3 = memref.subview %alloc_0[0, 0] [%1, 128] [1, 1] : memref<128x128xf32, #hivm.address_space<cbuf>> to memref<?x128xf32, strided<[128, 1]>, #hivm.address_space<cbuf>>
  // CHECK: hivm.hir.nd2nz {dst_continuous} ins({{.*}} : memref<?x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs({{.*}} : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = {{.*}} : f32
  hivm.hir.load ins(%subview_2 : memref<?x128xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x128xf32, strided<[128, 1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true
  %alloc_res = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32, #hivm.address_space<cc>>
  hivm.hir.mmadL1 ins(%alloc, %alloc_0, %true, %c0, %c0, %c0 : memref<256x128xf32, #hivm.address_space<cbuf>>, memref<128x128xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_res : memref<256x128xf32, #hivm.address_space<cc>>)
  return
}

// -----
module {
  // CHECK-LABEL: test_infer_layout_pointer_cast
  func.func @test_infer_layout_pointer_cast(%lb: i32, %ub: i32, %step: i32, %index: i64, %size: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32>
    %cast = hivm.hir.pointer_cast(%index) [%size] : memref<?xf16, #hivm.address_space<gm>>
    %reinterpret_cast = memref.reinterpret_cast %cast to offset: [0], sizes: [256, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x128xf16, #hivm.address_space<gm>>
    %l1A = memref.alloc() : memref<256x128xf16, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.nd2nz
    hivm.hir.copy ins(%reinterpret_cast : memref<256x128xf16, #hivm.address_space<gm>>) outs(%l1A : memref<256x128xf16, #hivm.address_space<cbuf>>)
    %ret = scf.for %iv = %lb to %ub step %step iter_args(%l0c = %alloc) -> (memref<256x128xf32>) : i32 {
      %l1B = memref.alloc() : memref<128x128xf16>
      %init_cond = arith.cmpi eq, %iv, %ub : i32
      hivm.hir.mmadL1 ins(%l1A, %l1B, %init_cond, %c128, %c128, %c128 :
                             memref<256x128xf16, #hivm.address_space<cbuf>>, memref<128x128xf16>, i1, index, index, index)
                      outs(%l0c : memref<256x128xf32>)
      scf.yield %l0c : memref<256x128xf32>
    }
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_infer_layout_memref_cast
  func.func @test_infer_layout_memref_cast(%cast:memref<?xf16, #hivm.address_space<gm>>, %lb: i32, %ub: i32, %step: i32, %index: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32>
    %reinterpret_cast = memref.reinterpret_cast %cast to offset: [0], sizes: [256, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x128xf16, #hivm.address_space<gm>>
    %l1A = memref.alloc() : memref<256x128xf16, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.nd2nz
    hivm.hir.copy ins(%reinterpret_cast : memref<256x128xf16, #hivm.address_space<gm>>) outs(%l1A : memref<256x128xf16, #hivm.address_space<cbuf>>)
    %reinterpret_cast_3 = memref.reinterpret_cast %cast to offset: [8192], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_1 = memref.cast %reinterpret_cast_3 : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %ret = scf.for %iv = %lb to %ub step %step iter_args(%l0c = %alloc) -> (memref<256x128xf32>) : i32 {
      %l1B = memref.alloc() : memref<128x128xf16, #hivm.address_space<cbuf>>
      %cast_6 = memref.cast %l1B: memref<128x128xf16, #hivm.address_space<cbuf>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.nd2nz
      hivm.hir.load ins(%cast_1 : memref<128x128xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%l1B : memref<128x128xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      %init_cond = arith.cmpi eq, %iv, %ub : i32
      hivm.hir.mmadL1 ins(%l1A, %cast_6, %init_cond, %c128, %c128, %c128 :
                             memref<256x128xf16, #hivm.address_space<cbuf>>, memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index)
                      outs(%l0c : memref<256x128xf32>)
      scf.yield %l0c : memref<256x128xf32>
    }
    return
  }
}

// -----
// CHECK-LABEL: triton_batch_nd2nz
func.func @triton_batch_nd2nz(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
  %true = arith.constant true
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [2, 32, 64], strides: [2048, 64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x32x64xf32, strided<[2048, 64, 1]>, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<2x32x64xf32, #hivm.address_space<cbuf>>
  // CHECK: hivm.hir.nd2nz
  hivm.hir.load ins(%reinterpret_cast : memref<2x32x64xf32, strided<[2048, 64, 1]>, #hivm.address_space<gm>>) outs(%alloc : memref<2x32x64xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [2, 64, 16], strides: [1024, 16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x64x16xf32, strided<[1024, 16, 1]>, #hivm.address_space<gm>>
  %alloc_1 = memref.alloc() : memref<2x64x16xf32, #hivm.address_space<cbuf>>
  // CHECK: hivm.hir.nd2nz
  hivm.hir.load ins(%reinterpret_cast_0 : memref<2x64x16xf32, strided<[1024, 16, 1]>, #hivm.address_space<gm>>) outs(%alloc_1 : memref<2x64x16xf32, #hivm.address_space<cbuf>>) init_out_buffer = false
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.index_cast %2 : i64 to index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 4096)>()[%3]
  %view = memref.view %arg2[%4][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x32x16xf32, #hivm.address_space<gm>>
  scf.for %arg10 = %c0 to %c2 step %c1 {
    %subview = memref.subview %alloc[%arg10, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<cbuf>> to memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<cbuf>>
    %subview_2 = memref.subview %alloc_1[%arg10, 0, 0] [1, 64, 16] [1, 1, 1] : memref<2x64x16xf32, #hivm.address_space<cbuf>> to memref<64x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<cbuf>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x16xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%subview, %subview_2, %true, %c32, %c64, %c16 : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<cbuf>>, memref<64x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_3 : memref<32x16xf32, #hivm.address_space<cc>>)
    %subview_4 = memref.subview %view[%arg10, 0, 0] [1, 32, 16] [1, 1, 1] : memref<2x32x16xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_3 : memref<32x16xf32, #hivm.address_space<cc>>) outs(%subview_4 : memref<32x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>)
    annotation.mark %subview_4 : memref<32x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
  }
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
  return
}

// -----
// CHECK-LABEL: invalid_mmadl1_of_tensor_type
func.func @invalid_mmadl1_of_tensor_type(%arg0 : memref<?xf32>, %arg1 : memref<?xf16>, %arg2 : memref<?xf16>) -> tensor<16x16xf32> {
  %cst_1 = arith.constant 2.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %init_condition = arith.constant 0 : i1
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: 0>>
  %2 = bufferization.to_tensor %1  restrict writable : memref<16x16xf32, strided<[16, 1], offset: 0>>
  %3 = tensor.empty() : tensor<16x16xf32>
  // CHECK-NOT: hivm.hir.nd2nz
  %4 = hivm.hir.load ins(%2 : tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %5 = hivm.hir.mmadL1 ins(%4, %4, %init_condition, %c16, %c16, %c16 :
                                tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
                          outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %5 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: test_infer_data_layout_basic_transb_f16
module {
  func.func @test_infer_data_layout_basic_transb_f16(%arg0 : i32,
                                          %arg1 : i32,
                                          %arg2 : i32, %arg3: memref<?xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: 0>, #hivm.address_space<gm>>
  %cast_2 = memref.cast %reinterpret_cast_1 : memref<64x64xf16, strided<[64, 1], offset: 0>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %alloc_7 = memref.alloc() : memref<64x64xf16, #hivm.address_space<cbuf>>
  // CHECK: hivm.hir.nd2nz
  hivm.hir.load ins(%cast_2 : memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<64x64xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
  %alloc = memref.alloc() : memref<64x64xf16, #hivm.address_space<cbuf>>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<cc>>
  hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_7, %true, %c64, %c64, %c64 : memref<64x64xf16, #hivm.address_space<cbuf>>, memref<64x64xf16, #hivm.address_space<cbuf>>, i1, index, index, index)
      outs(%alloc_5 : memref<64x64xf32, #hivm.address_space<cc>>)
  return
  }
}


// -----
// CHECK-LABEL: test_infer_data_layout_basic_transa_i8
module {
  func.func @test_infer_data_layout_basic_transa_i8(%arg0 : i32,
                                          %arg1 : i32,
                                          %arg2 : i32, %arg3: memref<?xi8, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<64x64xi8, strided<[64, 1], offset: 0>, #hivm.address_space<gm>>
  %cast_2 = memref.cast %reinterpret_cast_1 : memref<64x64xi8, strided<[64, 1], offset: 0>, #hivm.address_space<gm>> to memref<64x64xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %alloc_7 = memref.alloc() : memref<64x64xi8, #hivm.address_space<cbuf>>
  // CHECK: hivm.hir.nd2nz
  hivm.hir.load ins(%cast_2 : memref<64x64xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<64x64xi8, #hivm.address_space<cbuf>>) init_out_buffer = false
  %alloc = memref.alloc() : memref<64x64xi8, #hivm.address_space<cbuf>>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xi8, #hivm.address_space<cc>>
  hivm.hir.mmadL1 {a_transpose} ins(%alloc_7, %alloc, %true, %c64, %c64, %c64 : memref<64x64xi8, #hivm.address_space<cbuf>>, memref<64x64xi8, #hivm.address_space<cbuf>>, i1, index, index, index)
      outs(%alloc_5 : memref<64x64xi8, #hivm.address_space<cc>>)
  return
  }
}

// -----
// CHECK-LABEL: test_simple_mixed_cv
module {
  func.func @test_simple_mixed_cv(%arg0 : i32,
                                          %arg1 : i32,
                                          %arg2 : i32, %arg3: memref<?xi8, #hivm.address_space<gm>>,
                                          %arg4: memref<?xi8, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<64x64xi8, strided<[64, 1], offset: 0>, #hivm.address_space<gm>>
  %cast_2 = memref.cast %reinterpret_cast_1 : memref<64x64xi8, strided<[64, 1], offset: 0>, #hivm.address_space<gm>> to memref<64x64xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %alloc_7 = memref.alloc() : memref<64x64xi8, #hivm.address_space<cbuf>>
  // CHECK: hivm.hir.nd2nz
  hivm.hir.load ins(%cast_2 : memref<64x64xi8, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<64x64xi8, #hivm.address_space<cbuf>>) init_out_buffer = false
  %alloc = memref.alloc() : memref<64x64xi8, #hivm.address_space<cbuf>>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xi8, #hivm.address_space<cc>>
  hivm.hir.mmadL1 {a_transpose} ins(%alloc_7, %alloc, %true, %c64, %c64, %c64 : memref<64x64xi8, #hivm.address_space<cbuf>>, memref<64x64xi8, #hivm.address_space<cbuf>>, i1, index, index, index)
      outs(%alloc_5 : memref<64x64xi8, #hivm.address_space<cc>>)
  %view_4 = memref.view %arg4[%c0][] : memref<?xi8, #hivm.address_space<gm>> to memref<64x64xi8, #hivm.address_space<gm>>
  hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_5 : memref<64x64xi8, #hivm.address_space<cc>>) outs(%view_4 : memref<64x64xi8, #hivm.address_space<gm>>)
  return
  }
}

// -----
// CHECK-LABEL: _attn_fwd_mix_aic
func.func @_attn_fwd_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
  %c8_i64 = arith.constant 8 : i64
  %c6_i64 = arith.constant 6 : i64
  %c4_i64 = arith.constant 4 : i64
  %c2_i64 = arith.constant 2 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %c32_i32 = arith.constant 32 : i32
  %c4194304_i64 = arith.constant 4194304 : i64
  %c131072_i64 = arith.constant 131072 : i64
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c2048_i32 = arith.constant 2048 : i32
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %c1 = arith.constant 1 : index
  %c128_i32 = arith.constant 128 : i32
  hivm.hir.set_ffts_base_addr %arg0
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.divsi %1, %arg10 : i32
  %3 = arith.remsi %2, %arg9 : i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.divsi %1, %4 : i32
  %6 = arith.remsi %5, %arg8 : i32
  hivm.hir.set_mask_norm
  %7 = arith.divsi %3, %c32_i32 : i32
  %8 = arith.remsi %3, %c32_i32 : i32
  %9 = arith.extsi %7 : i32 to i64
  %10 = arith.muli %9, %c4194304_i64 : i64
  %11 = arith.extsi %8 : i32 to i64
  %12 = arith.muli %11, %c131072_i64 : i64
  %13 = arith.addi %10, %12 : i64
  %14 = arith.index_cast %13 : i64 to index
  %15 = arith.muli %6, %c64_i32 : i32
  %16 = arith.index_cast %15 : i32 to index
  %17 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%14, %16]
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<64x64xf16, #hivm.address_space<cbuf>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 6
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 7
  // CHECK: hivm.hir.nd2nz
  hivm.hir.load ins(%reinterpret_cast : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<64x64xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
  %cast = memref.cast %reinterpret_cast_0 : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
  %cast_2 = memref.cast %reinterpret_cast_1 : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  %18 = arith.index_cast %0 : i64 to index
  %19 = affine.apply affine_map<()[s0] -> (s0 * 81920 + 32768)>()[%18]
  %view = memref.view %arg1[%19][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x64x64xf16, #hivm.address_space<gm>>
  %20 = affine.apply affine_map<()[s0] -> (s0 * 81920)>()[%18]
  %view_3 = memref.view %arg1[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x64x64xf32, #hivm.address_space<gm>>
  %21 = affine.apply affine_map<()[s0] -> (s0 * 81920 + 49152)>()[%18]
  %view_4 = memref.view %arg1[%21][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x64x64xf32, #hivm.address_space<gm>>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<cc>>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<cc>>
  %22:6 = scf.for %arg11 = %c0_i32 to %c2048_i32 step %c128_i32 iter_args(%arg12 = %cast, %arg13 = %cast_2, %arg14 = %14, %arg15 = %c0, %arg16 = %14, %arg17 = %c0) -> (memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, index, index, index)  : i32 {
    %23 = arith.index_cast %arg11 : i32 to index
    %24 = affine.min affine_map<(d0) -> (2048, d0 + 128)>(%23)
    %25 = affine.apply affine_map<(d0, d1) -> ((d0 - d1) floordiv 64)>(%24, %23)
    %26:5 = scf.for %arg18 = %c0 to %25 step %c1 iter_args(%arg19 = %arg13, %arg20 = %arg16, %arg21 = %c0_i64, %arg22 = %c2_i64, %arg23 = %c4_i64) -> (memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, i64, i64, i64) {
      %alloc_7 = memref.alloc() : memref<64x64xf16, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.nd2nz
      hivm.hir.load ins(%arg19 : memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<64x64xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_7, %true, %c64, %c64, %c64 : memref<64x64xf16, #hivm.address_space<cbuf>>, memref<64x64xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_5 : memref<64x64xf32, #hivm.address_space<cc>>)
      %subview = memref.subview %view_4[%arg18, 0, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x64xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = %arg21
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = %arg22
      hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_5 : memref<64x64xf32, #hivm.address_space<cc>>) outs(%subview : memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = %arg23
      %28 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 4096)>()[%arg17, %arg20]
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_9 = memref.cast %reinterpret_cast_8 : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %29 = arith.addi %arg21, %c1_i64 : i64
      %30 = arith.addi %arg22, %c1_i64 : i64
      %31 = arith.addi %arg23, %c1_i64 : i64
      scf.yield %cast_9, %28, %29, %30, %31 : memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, i64, i64, i64
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
    %27:6 = scf.for %arg18 = %c0 to %25 step %c1 iter_args(%arg19 = %arg12, %arg20 = %arg14, %arg21 = %c4_i64, %arg22 = %c6_i64, %arg23 = %c8_i64, %arg24 = %c4_i64) -> (memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, i64, i64, i64, i64) {
      %alloc_7 = memref.alloc() : memref<64x64xf16, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.nd2nz
      hivm.hir.load ins(%arg19 : memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_7 : memref<64x64xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      %subview = memref.subview %view[%arg18, 0, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x64xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = %arg21
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.nd2nz
      hivm.hir.load ins(%subview : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<64x64xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = %arg22
      hivm.hir.mmadL1 ins(%alloc_8, %alloc_7, %true, %c64, %c64, %c64 : memref<64x64xf16, #hivm.address_space<cbuf>>, memref<64x64xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_6 : memref<64x64xf32, #hivm.address_space<cc>>)
      %subview_9 = memref.subview %view_3[%arg18, 0, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x64xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = %arg23
      hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_6 : memref<64x64xf32, #hivm.address_space<cc>>) outs(%subview_9 : memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = %arg24
      %28 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 4096)>()[%arg15, %arg20]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%28], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_11 = memref.cast %reinterpret_cast_10 : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %29 = arith.addi %arg21, %c1_i64 : i64
      %30 = arith.addi %arg22, %c1_i64 : i64
      %31 = arith.addi %arg23, %c1_i64 : i64
      %32 = arith.addi %arg24, %c1_i64 : i64
      scf.yield %cast_11, %28, %29, %30, %31, %32 : memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, i64, i64, i64, i64
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
    scf.yield %27#0, %26#0, %27#1, %c0, %26#1, %c0 : memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<64x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, index, index, index
  }
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 9
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 8
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 3
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 2
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 1
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
  return
}

