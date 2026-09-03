// RUN: bishengir-opt -hivm-inject-sync -split-input-file %s | FileCheck %s
module{
  memref.global "private" constant @tbl : memref<320xi32, #hivm.address_space<gm>> = dense<"0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000002000000050000000A000000140000002800000051000000A2000000450100008B020000170500002F0A00005F140000BE2800007C510000F9A20000F3450100E68B0200CC170500982F0A00305F140060BE2800C17C510083F9A20006F345010DE68B021BCC170536982F0A6D305F14DB60BE28B7C17C516E83F9A2DC06F345B90DE68B721BCC17E436982FC96D305F93DB60BE27B7C17C4E6E83F99CDC06F339B90DE672721BCCE4E43698C8C96D309193DB602227B7C1444E6E83889CDC061039B90D2072721B41E4E43682C8C96D059193DB0A2227B715444E6E2A889CDC541039B9A92072725241E4E4A582C8C94A059193940A22272915444E532A889CA75410394FA920729F5241E43FA582C87F4A0591FE940A22FC291544F8532A88F0A75410E14FA920C29F5241843FA582097F4A0513FE940A27FC29154EF8532A9DF0A7543AE14FA975C29F52EA843FA5D5097F4AAB13FE945727FC29AF4EF8535F9DF0A7BE3AE14F7D75C29FFAEA843FF4D5097FE8AB13FED15727FCA3AF4EF8475F9DF08FBE3AE11F7D75C23EFAEA847DF4D509FAE8AB13F5D15727EAA3AF4ED4475F9DA98FBE3A531F7D75A63EFAEA4D7DF4D59AFAE8AB34F5D15769EAA3AFD3D4475FA6A98FBE4D531F7D9BA63EFA374D7DF46E9AFAE8DD34F5D1BB69EAA377D3D447EEA6A98FDC4D531FB89BA63E70374D7DE06E9AFAC0DD34F581BB69EA0377D3D406EEA6A90DDC4D531BB89BA63670374D6DE06E9ADBC0DD34B681BB696D0377D3DB06EEA6B60DDC4D6C1BB89BD8367037B16DE06E62DBC0DDC5B681BB8A6D037714DB06EE29B60DDC526C1BB8A5D836704AB16DE09562DBC02BC5B681568A6D03AC14DB065929B60DB3526C1B66A5D836CC4AB16D999562DB322BC5B664568A6DC9AC14DB935929B627B3526C4F66A5D89ECC4AB13C99956278322BC5F164568AE2C9AC14C49359298827B352104F66A5219ECC4A433C99958778322B0EF164561CE2C9AC39C49359728827B3E4104F66C8219ECC90433C9920877832410EF164821CE2C90439C4930872882710E4104F20C8219E"> {alignment = 32 : i64}
  func.func @triton_cos_2d_outlined_vf_0(%arg0: memref<16030xf32, #hivm.address_space<ub>>, %arg1: memref<320xi32, #hivm.address_space<ub>>, %arg2: memref<16030xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst_15 = arith.constant dense<0> : vector<64xi32>
    %c8 = arith.constant dense<8> : vector<64xi32>
    %c8388608 = arith.constant dense<8388608> : vector<64xi32>
    %c8388607 = arith.constant dense<8388607> : vector<64xi32>
    %c255 = arith.constant dense<255> : vector<64xi32>
    %c23 = arith.constant dense<23> : vector<64xi32>
    %cst_22 = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c16030 = arith.constant 16030 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c16030 step %c64 {
      %0 = affine.min affine_map<(d0) -> (-d0 + 16030, 64)>(%arg3)
      %subview = memref.subview %arg0[%arg3] [%0] [1] : memref<16030xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_23 = memref.subview %arg2[%arg3] [%0] [1] : memref<16030xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.create_mask %0 : vector<64xi1>
      %2 = vector.transfer_read %subview[%c0], %cst_22, %1 {in_bounds = [true]} : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
      %3 = arith.bitcast %2 : vector<64xf32> to vector<64xi32>
      %4 = arith.shrui %3, %c23 : vector<64xi32>
      %5 = arith.andi %4, %c255 : vector<64xi32>
      %6 = arith.cmpi eq, %5, %c255 : vector<64xi32>
      %7 = arith.andi %3, %c8388607 : vector<64xi32>
      %8 = arith.addi %7, %c8388608 : vector<64xi32>
      %9 = arith.addi %5, %c8 : vector<64xi32>
      %10 = vector.gather %arg1[%c0] [%9], %1, %cst_15 : memref<320xi32, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1>, vector<64xi32> into vector<64xi32>
      %11 = arith.sitofp %10 : vector<64xi32> to vector<64xf32>
      vector.transfer_write %11, %subview_23[%c0], %1 {in_bounds = [true]} : vector<64xf32>, memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @triton_cos_2d(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c65408_i64 = arith.constant 65408 : i64
    %c64128_i64 = arith.constant 64128 : i64
    %c0_i64 = arith.constant 0 : i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg4, %arg5 : i32
    %1 = arith.muli %0, %arg6 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1145, 14], strides: [14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1145x14xf32, strided<[14, 1]>, #hivm.address_space<gm>>
    %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1]] : memref<1145x14xf32, strided<[14, 1]>, #hivm.address_space<gm>> into memref<16030xf32, strided<[1]>, #hivm.address_space<gm>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<16030xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape : memref<16030xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%2 : memref<16030xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.load ins(%collapse_shape : memref<16030xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%2 : memref<16030xf32, #hivm.address_space<ub>>)
    %3 = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %3 to offset: [0], sizes: [320], strides: [1] : memref<320xi32, #hivm.address_space<gm>> to memref<320xi32, strided<[1]>, #hivm.address_space<gm>>
    %4 = hivm.hir.pointer_cast(%c64128_i64) : memref<320xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<320xi32, strided<[1]>, #hivm.address_space<gm>>) outs(%4 : memref<320xi32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.load ins(%reinterpret_cast_0 : memref<320xi32, strided<[1]>, #hivm.address_space<gm>>) outs(%4 : memref<320xi32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %5 = hivm.hir.pointer_cast(%c65408_i64) : memref<16030xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @triton_cos_2d_outlined_vf_0(%2, %4, %5) {hivm.vector_function} : (memref<16030xf32, #hivm.address_space<ub>>, memref<320xi32, #hivm.address_space<ub>>, memref<16030xf32, #hivm.address_space<ub>>) -> ()
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1145, 14], strides: [14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1145x14xf32, strided<[14, 1]>, #hivm.address_space<gm>>
    %collapse_shape_2 = memref.collapse_shape %reinterpret_cast_1 [[0, 1]] : memref<1145x14xf32, strided<[14, 1]>, #hivm.address_space<gm>> into memref<16030xf32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.store ins(%5 : memref<16030xf32, #hivm.address_space<ub>>) outs(%collapse_shape_2 : memref<16030xf32, strided<[1]>, #hivm.address_space<gm>>)
    return
  }
}
