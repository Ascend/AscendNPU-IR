// RUN: bishengir-opt %s -fix-fused-cat | FileCheck %s

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">} {
  tt.func public @triton_unk_fused_cat_10(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg14: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg15: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg16: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg20: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg21: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
// CHECK-LABEL:   tt.func public @triton_unk_fused_cat_10(
// CHECK-SAME:                                            %[[ARG_0:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_1:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_2:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_3:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_4:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_5:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_6:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_7:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_8:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_9:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_10:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_11:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_12:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_13:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_14:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_15:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[ARG_16:[^:]*]]: !tt.ptr<f32>
// CHECK-SAME:                                            %[[VAL_17:[^:]*]]: i32
// CHECK-SAME:                                            %[[VAL_18:[^:]*]]: i32
// CHECK-SAME:                                            %[[VAL_19:[^:]*]]: i32
// CHECK-SAME:                                            %[[VAL_20:[^:]*]]: i32
// CHECK-SAME:                                            %[[VAL_21:[^:]*]]: i32
    %cst = arith.constant dense<0.000000e+00> : tensor<4x1024xf32>
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<1024> : tensor<4x1xi32>
    %cst_1 = arith.constant dense<32> : tensor<4x1xi32>
    %cst_2 = arith.constant dense<128> : tensor<4x1xi32>
    %cst_3 = arith.constant dense<8> : tensor<4x1xi32>
    %cst_4 = arith.constant dense<64> : tensor<4x1xi32>
    %cst_5 = arith.constant dense<16> : tensor<4x1xi32>
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %5 = arith.addi %1, %c128_i32 : i32
    %6 = arith.minsi %5, %arg17 : i32
    %7 = tt.splat %6 : i32 -> tensor<4x1xi32>
    %8 = tt.expand_dims %3 {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>
    %9 = tt.broadcast %8 : tensor<1x1024xi32> -> tensor<4x1024xi32>
// CHECK:           %[[COLS2:.*]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[COLS1:.*]] = tt.expand_dims %[[COLS2]] {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>
// CHECK:           %[[COLS:.*]] = tt.broadcast %[[COLS1]] : tensor<1x1024xi32> -> tensor<4x1024xi32>

// CHECK:           %[[OFFSET_15_C:.*]] = arith.constant -[[OFFSET_15_I:.*]] : i32
// CHECK:           %[[OFFSET_15_SPLAT:.*]] = tt.splat %[[OFFSET_15_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_15:.*]] = arith.addi %[[COLS]], %[[OFFSET_15_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_14_C:.*]] = arith.constant -[[OFFSET_14_I:.*]] : i32
// CHECK:           %[[OFFSET_14_SPLAT:.*]] = tt.splat %[[OFFSET_14_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_14:.*]] = arith.addi %[[COLS]], %[[OFFSET_14_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_13_C:.*]] = arith.constant -[[OFFSET_13_I:.*]] : i32
// CHECK:           %[[OFFSET_13_SPLAT:.*]] = tt.splat %[[OFFSET_13_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_13:.*]] = arith.addi %[[COLS]], %[[OFFSET_13_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_12_C:.*]] = arith.constant -[[OFFSET_12_I:.*]] : i32
// CHECK:           %[[OFFSET_12_SPLAT:.*]] = tt.splat %[[OFFSET_12_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_12:.*]] = arith.addi %[[COLS]], %[[OFFSET_12_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_11_C:.*]] = arith.constant -[[OFFSET_11_I:.*]] : i32
// CHECK:           %[[OFFSET_11_SPLAT:.*]] = tt.splat %[[OFFSET_11_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_11:.*]] = arith.addi %[[COLS]], %[[OFFSET_11_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_10_C:.*]] = arith.constant -[[OFFSET_10_I:.*]] : i32
// CHECK:           %[[OFFSET_10_SPLAT:.*]] = tt.splat %[[OFFSET_10_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_10:.*]] = arith.addi %[[COLS]], %[[OFFSET_10_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_9_C:.*]] = arith.constant -[[OFFSET_9_I:.*]] : i32
// CHECK:           %[[OFFSET_9_SPLAT:.*]] = tt.splat %[[OFFSET_9_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_9:.*]] = arith.addi %[[COLS]], %[[OFFSET_9_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_8_C:.*]] = arith.constant -[[OFFSET_8_I:.*]] : i32
// CHECK:           %[[OFFSET_8_SPLAT:.*]] = tt.splat %[[OFFSET_8_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_8:.*]] = arith.addi %[[COLS]], %[[OFFSET_8_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_7_C:.*]] = arith.constant -[[OFFSET_7_I:.*]] : i32
// CHECK:           %[[OFFSET_7_SPLAT:.*]] = tt.splat %[[OFFSET_7_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_7:.*]] = arith.addi %[[COLS]], %[[OFFSET_7_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_6_C:.*]] = arith.constant -[[OFFSET_6_I:.*]] : i32
// CHECK:           %[[OFFSET_6_SPLAT:.*]] = tt.splat %[[OFFSET_6_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_6:.*]] = arith.addi %[[COLS]], %[[OFFSET_6_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_5_C:.*]] = arith.constant -[[OFFSET_5_I:.*]] : i32
// CHECK:           %[[OFFSET_5_SPLAT:.*]] = tt.splat %[[OFFSET_5_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_5:.*]] = arith.addi %[[COLS]], %[[OFFSET_5_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_4_C:.*]] = arith.constant -[[OFFSET_4_I:.*]] : i32
// CHECK:           %[[OFFSET_4_SPLAT:.*]] = tt.splat %[[OFFSET_4_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_4:.*]] = arith.addi %[[COLS]], %[[OFFSET_4_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_3_C:.*]] = arith.constant -[[OFFSET_3_I:.*]] : i32
// CHECK:           %[[OFFSET_3_SPLAT:.*]] = tt.splat %[[OFFSET_3_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_3:.*]] = arith.addi %[[COLS]], %[[OFFSET_3_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_2_C:.*]] = arith.constant -[[OFFSET_2_I:.*]] : i32
// CHECK:           %[[OFFSET_2_SPLAT:.*]] = tt.splat %[[OFFSET_2_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_2:.*]] = arith.addi %[[COLS]], %[[OFFSET_2_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_1_C:.*]] = arith.constant -[[OFFSET_1_I:.*]] : i32
// CHECK:           %[[OFFSET_1_SPLAT:.*]] = tt.splat %[[OFFSET_1_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_1:.*]] = arith.addi %[[COLS]], %[[OFFSET_1_SPLAT]] : tensor<4x1024xi32>

// CHECK:           %[[OFFSET_0_C:.*]] = arith.constant 0 : i32
// CHECK:           %[[OFFSET_0_SPLAT:.*]] = tt.splat %[[OFFSET_0_C]] : i32 -> tensor<4x1024xi32>
// CHECK:           %[[OFFSET_0:.*]] = arith.addi %[[COLS]], %[[OFFSET_0_SPLAT]] : tensor<4x1024xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %13 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %14 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %15 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %16 = tt.splat %arg6 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %17 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %18 = tt.splat %arg8 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %19 = tt.splat %arg9 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %20 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %21 = tt.splat %arg11 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %22 = tt.splat %arg12 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %23 = tt.splat %arg13 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %24 = tt.splat %arg14 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %25 = tt.splat %arg15 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    %26 = tt.splat %arg16 : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_0_SPLAT:.*]] = tt.splat %[[ARG_0]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_1_SPLAT:.*]] = tt.splat %[[ARG_1]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_2_SPLAT:.*]] = tt.splat %[[ARG_2]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_3_SPLAT:.*]] = tt.splat %[[ARG_3]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_4_SPLAT:.*]] = tt.splat %[[ARG_4]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_5_SPLAT:.*]] = tt.splat %[[ARG_5]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_6_SPLAT:.*]] = tt.splat %[[ARG_6]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_7_SPLAT:.*]] = tt.splat %[[ARG_7]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_8_SPLAT:.*]] = tt.splat %[[ARG_8]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_9_SPLAT:.*]] = tt.splat %[[ARG_9]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_10_SPLAT:.*]] = tt.splat %[[ARG_10]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_11_SPLAT:.*]] = tt.splat %[[ARG_11]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_12_SPLAT:.*]] = tt.splat %[[ARG_12]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_13_SPLAT:.*]] = tt.splat %[[ARG_13]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_14_SPLAT:.*]] = tt.splat %[[ARG_14]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_15_SPLAT:.*]] = tt.splat %[[ARG_15]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
// CHECK:           %[[ARG_16_SPLAT:.*]] = tt.splat %[[ARG_16]] : !tt.ptr<f32> -> tensor<4x1024x!tt.ptr<f32>>
    scf.for %arg22 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %27 = arith.muli %arg22, %c4_i32 : i32
      %28 = arith.addi %1, %27 : i32
      %29 = tt.splat %28 : i32 -> tensor<4x1xi32>
      %30 = arith.addi %29, %4 : tensor<4x1xi32>
      %31 = arith.cmpi slt, %30, %7 : tensor<4x1xi32>
      %32 = arith.muli %30, %cst_5 : tensor<4x1xi32>
      %33 = tt.broadcast %32 : tensor<4x1xi32> -> tensor<4x1024xi32>
      %34 = arith.addi %9, %33 : tensor<4x1024xi32>
      %35 = tt.addptr %10, %34 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %36 = tt.broadcast %31 : tensor<4x1xi1> -> tensor<4x1024xi1>
      %37 = tt.load %35, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_0:.*]] = arith.addi %[[OFFSET_0]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_0:.*]] = tt.addptr %[[ARG_0_SPLAT]], %[[IDXS_0]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_0:.*]] = tt.load %[[ADDRS_0]], %[[MASK:.*]], %[[DEFAULT:.*]] : tensor<4x1024x!tt.ptr<f32>>
      %38 = arith.muli %30, %cst_4 : tensor<4x1xi32>
      %39 = tt.broadcast %38 : tensor<4x1xi32> -> tensor<4x1024xi32>
      %40 = arith.addi %9, %39 : tensor<4x1024xi32>
      %41 = tt.addptr %11, %40 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %42 = tt.load %41, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_1:.*]] = arith.addi %[[OFFSET_1]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_1:.*]] = tt.addptr %[[ARG_1_SPLAT]], %[[IDXS_1]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_1:.*]] = tt.load %[[ADDRS_1]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %43 = arith.muli %30, %cst_3 : tensor<4x1xi32>
      %44 = tt.broadcast %43 : tensor<4x1xi32> -> tensor<4x1024xi32>
      %45 = arith.addi %9, %44 : tensor<4x1024xi32>
      %46 = tt.addptr %12, %45 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %47 = tt.load %46, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_2:.*]] = arith.addi %[[OFFSET_2]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_2:.*]] = tt.addptr %[[ARG_2_SPLAT]], %[[IDXS_2]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_2:.*]] = tt.load %[[ADDRS_2]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %48 = arith.muli %30, %cst_2 : tensor<4x1xi32>
      %49 = tt.broadcast %48 : tensor<4x1xi32> -> tensor<4x1024xi32>
      %50 = arith.addi %9, %49 : tensor<4x1024xi32>
      %51 = tt.addptr %13, %50 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %52 = tt.load %51, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_3:.*]] = arith.addi %[[OFFSET_3]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_3:.*]] = tt.addptr %[[ARG_3_SPLAT]], %[[IDXS_3]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_3:.*]] = tt.load %[[ADDRS_3]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %53 = arith.muli %30, %cst_1 : tensor<4x1xi32>
      %54 = tt.broadcast %53 : tensor<4x1xi32> -> tensor<4x1024xi32>
      %55 = arith.addi %9, %54 : tensor<4x1024xi32>
      %56 = tt.addptr %14, %55 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %57 = tt.load %56, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_4:.*]] = arith.addi %[[OFFSET_4]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_4:.*]] = tt.addptr %[[ARG_4_SPLAT]], %[[IDXS_4]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_4:.*]] = tt.load %[[ADDRS_4]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %58 = tt.addptr %15, %34 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %59 = tt.load %58, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_5:.*]] = arith.addi %[[OFFSET_5]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_5:.*]] = tt.addptr %[[ARG_5_SPLAT]], %[[IDXS_5]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_5:.*]] = tt.load %[[ADDRS_5]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %60 = tt.addptr %16, %50 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %61 = tt.load %60, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_6:.*]] = arith.addi %[[OFFSET_6]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_6:.*]] = tt.addptr %[[ARG_6_SPLAT]], %[[IDXS_6]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_6:.*]] = tt.load %[[ADDRS_6]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %62 = tt.addptr %17, %55 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %63 = tt.load %62, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_7:.*]] = arith.addi %[[OFFSET_7]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_7:.*]] = tt.addptr %[[ARG_7_SPLAT]], %[[IDXS_7]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_7:.*]] = tt.load %[[ADDRS_7]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %64 = tt.addptr %18, %50 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %65 = tt.load %64, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_8:.*]] = arith.addi %[[OFFSET_8]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_8:.*]] = tt.addptr %[[ARG_8_SPLAT]], %[[IDXS_8]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_8:.*]] = tt.load %[[ADDRS_8]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %66 = tt.addptr %19, %45 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %67 = tt.load %66, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_9:.*]] = arith.addi %[[OFFSET_9]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_9:.*]] = tt.addptr %[[ARG_9_SPLAT]], %[[IDXS_9]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_9:.*]] = tt.load %[[ADDRS_9]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %68 = tt.addptr %20, %50 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %69 = tt.load %68, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_10:.*]] = arith.addi %[[OFFSET_10]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_10:.*]] = tt.addptr %[[ARG_10_SPLAT]], %[[IDXS_10]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_10:.*]] = tt.load %[[ADDRS_10]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %70 = tt.addptr %21, %50 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %71 = tt.load %70, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_11:.*]] = arith.addi %[[OFFSET_11]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_11:.*]] = tt.addptr %[[ARG_11_SPLAT]], %[[IDXS_11]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_11:.*]] = tt.load %[[ADDRS_11]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %72 = tt.addptr %22, %50 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %73 = tt.load %72, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_12:.*]] = arith.addi %[[OFFSET_12]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_12:.*]] = tt.addptr %[[ARG_12_SPLAT]], %[[IDXS_12]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_12:.*]] = tt.load %[[ADDRS_12]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %74 = tt.addptr %23, %55 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %75 = tt.load %74, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_13:.*]] = arith.addi %[[OFFSET_13]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_13:.*]] = tt.addptr %[[ARG_13_SPLAT]], %[[IDXS_13]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_13:.*]] = tt.load %[[ADDRS_13]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %76 = tt.addptr %24, %55 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %77 = tt.load %76, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_14:.*]] = arith.addi %[[OFFSET_14]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_14:.*]] = tt.addptr %[[ARG_14_SPLAT]], %[[IDXS_14]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_14:.*]] = tt.load %[[ADDRS_14]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %78 = tt.addptr %25, %34 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      %79 = tt.load %78, %36, %cst : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             %[[IDXS_15:.*]] = arith.addi %[[OFFSET_15]], %{{.*}} : tensor<4x1024xi32>
// CHECK:             %[[ADDRS_15:.*]] = tt.addptr %[[ARG_15_SPLAT]], %[[IDXS_15]] : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
// CHECK:             %[[T_15:.*]] = tt.load %[[ADDRS_15]], %[[MASK]], %[[DEFAULT]] : tensor<4x1024x!tt.ptr<f32>>
      %extracted_slice = tensor.extract_slice %37[0, 0] [4, 16] [1, 1] : tensor<4x1024xf32> to tensor<4x16xf32>
      %inserted_slice = tensor.insert_slice %extracted_slice into %cst[0, 0] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<4x1024xf32>
      %extracted_slice_6 = tensor.extract_slice %42[0, 0] [4, 64] [1, 1] : tensor<4x1024xf32> to tensor<4x64xf32>
      %inserted_slice_7 = tensor.insert_slice %extracted_slice_6 into %inserted_slice[0, 16] [4, 64] [1, 1] : tensor<4x64xf32> into tensor<4x1024xf32>
      %extracted_slice_8 = tensor.extract_slice %47[0, 0] [4, 8] [1, 1] : tensor<4x1024xf32> to tensor<4x8xf32>
      %inserted_slice_9 = tensor.insert_slice %extracted_slice_8 into %inserted_slice_7[0, 80] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<4x1024xf32>
      %extracted_slice_10 = tensor.extract_slice %52[0, 0] [4, 128] [1, 1] : tensor<4x1024xf32> to tensor<4x128xf32>
      %inserted_slice_11 = tensor.insert_slice %extracted_slice_10 into %inserted_slice_9[0, 88] [4, 128] [1, 1] : tensor<4x128xf32> into tensor<4x1024xf32>
      %extracted_slice_12 = tensor.extract_slice %57[0, 0] [4, 32] [1, 1] : tensor<4x1024xf32> to tensor<4x32xf32>
      %inserted_slice_13 = tensor.insert_slice %extracted_slice_12 into %inserted_slice_11[0, 216] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<4x1024xf32>
      %extracted_slice_14 = tensor.extract_slice %59[0, 0] [4, 16] [1, 1] : tensor<4x1024xf32> to tensor<4x16xf32>
      %inserted_slice_15 = tensor.insert_slice %extracted_slice_14 into %inserted_slice_13[0, 248] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<4x1024xf32>
      %extracted_slice_16 = tensor.extract_slice %61[0, 0] [4, 128] [1, 1] : tensor<4x1024xf32> to tensor<4x128xf32>
      %inserted_slice_17 = tensor.insert_slice %extracted_slice_16 into %inserted_slice_15[0, 264] [4, 128] [1, 1] : tensor<4x128xf32> into tensor<4x1024xf32>
      %extracted_slice_18 = tensor.extract_slice %63[0, 0] [4, 32] [1, 1] : tensor<4x1024xf32> to tensor<4x32xf32>
      %inserted_slice_19 = tensor.insert_slice %extracted_slice_18 into %inserted_slice_17[0, 392] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<4x1024xf32>
      %extracted_slice_20 = tensor.extract_slice %65[0, 0] [4, 128] [1, 1] : tensor<4x1024xf32> to tensor<4x128xf32>
      %inserted_slice_21 = tensor.insert_slice %extracted_slice_20 into %inserted_slice_19[0, 424] [4, 128] [1, 1] : tensor<4x128xf32> into tensor<4x1024xf32>
      %extracted_slice_22 = tensor.extract_slice %67[0, 0] [4, 8] [1, 1] : tensor<4x1024xf32> to tensor<4x8xf32>
      %inserted_slice_23 = tensor.insert_slice %extracted_slice_22 into %inserted_slice_21[0, 552] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<4x1024xf32>
      %extracted_slice_24 = tensor.extract_slice %69[0, 0] [4, 128] [1, 1] : tensor<4x1024xf32> to tensor<4x128xf32>
      %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %inserted_slice_23[0, 560] [4, 128] [1, 1] : tensor<4x128xf32> into tensor<4x1024xf32>
      %extracted_slice_26 = tensor.extract_slice %71[0, 0] [4, 128] [1, 1] : tensor<4x1024xf32> to tensor<4x128xf32>
      %inserted_slice_27 = tensor.insert_slice %extracted_slice_26 into %inserted_slice_25[0, 688] [4, 128] [1, 1] : tensor<4x128xf32> into tensor<4x1024xf32>
      %extracted_slice_28 = tensor.extract_slice %73[0, 0] [4, 128] [1, 1] : tensor<4x1024xf32> to tensor<4x128xf32>
      %inserted_slice_29 = tensor.insert_slice %extracted_slice_28 into %inserted_slice_27[0, 816] [4, 128] [1, 1] : tensor<4x128xf32> into tensor<4x1024xf32>
      %extracted_slice_30 = tensor.extract_slice %75[0, 0] [4, 32] [1, 1] : tensor<4x1024xf32> to tensor<4x32xf32>
      %inserted_slice_31 = tensor.insert_slice %extracted_slice_30 into %inserted_slice_29[0, 944] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<4x1024xf32>
      %extracted_slice_32 = tensor.extract_slice %77[0, 0] [4, 32] [1, 1] : tensor<4x1024xf32> to tensor<4x32xf32>
      %inserted_slice_33 = tensor.insert_slice %extracted_slice_32 into %inserted_slice_31[0, 976] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<4x1024xf32>
      %extracted_slice_34 = tensor.extract_slice %79[0, 0] [4, 16] [1, 1] : tensor<4x1024xf32> to tensor<4x16xf32>
      %inserted_slice_35 = tensor.insert_slice %extracted_slice_34 into %inserted_slice_33[0, 1008] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<4x1024xf32>
// CHECK:             %[[OFFSET_0:.*]] = arith.constant dense<0> : tensor<1x1024xi32>
// CHECK:             %[[MASK_0:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_0]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_1:.*]] = arith.constant dense<[[OFFSET_1_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_1:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_1]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_2:.*]] = arith.constant dense<[[OFFSET_2_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_2:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_2]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_3:.*]] = arith.constant dense<[[OFFSET_3_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_3:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_3]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_4:.*]] = arith.constant dense<[[OFFSET_4_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_4:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_4]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_5:.*]] = arith.constant dense<[[OFFSET_5_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_5:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_5]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_6:.*]] = arith.constant dense<[[OFFSET_6_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_6:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_6]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_7:.*]] = arith.constant dense<[[OFFSET_7_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_7:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_7]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_8:.*]] = arith.constant dense<[[OFFSET_8_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_8:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_8]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_9:.*]] = arith.constant dense<[[OFFSET_9_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_9:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_9]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_10:.*]] = arith.constant dense<[[OFFSET_10_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_10:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_10]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_11:.*]] = arith.constant dense<[[OFFSET_11_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_11:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_11]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_12:.*]] = arith.constant dense<[[OFFSET_12_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_12:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_12]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_13:.*]] = arith.constant dense<[[OFFSET_13_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_13:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_13]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_14:.*]] = arith.constant dense<[[OFFSET_14_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_14:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_14]] : tensor<1x1024xi32>
// CHECK:             %[[OFFSET_15:.*]] = arith.constant dense<[[OFFSET_15_I]]> : tensor<1x1024xi32>
// CHECK:             %[[MASK_15:.*]] = arith.cmpi slt, %[[COLS1]], %[[OFFSET_15]] : tensor<1x1024xi32>

// CHECK:             %[[MASK_15B:.*]] = tt.broadcast %[[MASK_15]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_14:.*]] = arith.select %[[MASK_15B]], %[[T_14]], %[[T_15]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_14B:.*]] = tt.broadcast %[[MASK_14]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_13:.*]] = arith.select %[[MASK_14B]], %[[T_13]], %[[R_14]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_13B:.*]] = tt.broadcast %[[MASK_13]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_12:.*]] = arith.select %[[MASK_13B]], %[[T_12]], %[[R_13]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_12B:.*]] = tt.broadcast %[[MASK_12]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_11:.*]] = arith.select %[[MASK_12B]], %[[T_11]], %[[R_12]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_11B:.*]] = tt.broadcast %[[MASK_11]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_10:.*]] = arith.select %[[MASK_11B]], %[[T_10]], %[[R_11]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_10B:.*]] = tt.broadcast %[[MASK_10]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_9:.*]] = arith.select %[[MASK_10B]], %[[T_9]], %[[R_10]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_9B:.*]] = tt.broadcast %[[MASK_9]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_8:.*]] = arith.select %[[MASK_9B]], %[[T_8]], %[[R_9]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_8B:.*]] = tt.broadcast %[[MASK_8]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_7:.*]] = arith.select %[[MASK_8B]], %[[T_7]], %[[R_8]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_7B:.*]] = tt.broadcast %[[MASK_7]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_6:.*]] = arith.select %[[MASK_7B]], %[[T_6]], %[[R_7]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_6B:.*]] = tt.broadcast %[[MASK_6]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_5:.*]] = arith.select %[[MASK_6B]], %[[T_5]], %[[R_6]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_5B:.*]] = tt.broadcast %[[MASK_5]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_4:.*]] = arith.select %[[MASK_5B]], %[[T_4]], %[[R_5]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_4B:.*]] = tt.broadcast %[[MASK_4]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_3:.*]] = arith.select %[[MASK_4B]], %[[T_3]], %[[R_4]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_3B:.*]] = tt.broadcast %[[MASK_3]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_2:.*]] = arith.select %[[MASK_3B]], %[[T_2]], %[[R_3]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_2B:.*]] = tt.broadcast %[[MASK_2]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_1:.*]] = arith.select %[[MASK_2B]], %[[T_1]], %[[R_2]] : tensor<4x1024xi1>, tensor<4x1024xf32>
// CHECK:             %[[MASK_1B:.*]] = tt.broadcast %[[MASK_1]] : tensor<1x1024xi1> -> tensor<4x1024xi1>
// CHECK:             %[[R_0:.*]] = arith.select %[[MASK_1B]], %[[T_0]], %[[R_1]] : tensor<4x1024xi1>, tensor<4x1024xf32>
      %80 = arith.muli %30, %cst_0 : tensor<4x1xi32>
      %81 = tt.broadcast %80 : tensor<4x1xi32> -> tensor<4x1024xi32>
      %82 = arith.addi %9, %81 : tensor<4x1024xi32>
      %83 = tt.addptr %26, %82 : tensor<4x1024x!tt.ptr<f32>>, tensor<4x1024xi32>
      tt.store %83, %inserted_slice_35, %36 : tensor<4x1024x!tt.ptr<f32>>
// CHECK:             tt.store %{{.*}}, %[[R_0]], %[[MASK]] : tensor<4x1024x!tt.ptr<f32>>
    }
    tt.return
  }
}
