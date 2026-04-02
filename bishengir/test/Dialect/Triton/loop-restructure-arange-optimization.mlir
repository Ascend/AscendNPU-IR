// RUN: bishengir-opt -split-input-file  %s -loop-restructure-arange-optimization | FileCheck %s

// This checks only the major ops, not all the new ops that are created as this pass basically clones all the ops in kernel 

// CHECK-LABEL: tt.func public @triton_unk_fused_cat_12

// the first group (embedding size of 128)

// CHECK-DAG:    %[[TMP0:.*]] = tt.make_range {end = 128 : i32, group_id = 0 : i32, start = 0 : i32} : tensor<128xi32>

// CHECK-DAG:    %[[TMP1:.*]] = tt.expand_dims %[[TMP0:.*]] {axis = 0 : i32, group_id = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>

// CHECK-DAG:    %[[TMP2:.*]] = tt.broadcast %[[TMP1:.*]] {group_id = 0 : i32} : tensor<1x128xi32> -> tensor<16x128xi32>

// CHECK-DAG:    %[[TMP3:.*]] = tt.addptr  %{{.*}}, %[[TMP2:.*]] {group_id = 0 : i32} : tensor<16x128x!tt.ptr<f32>>, tensor<16x128xi32>

// CHECK-DAG:    %[[VAL1:.*]] = tt.load %{{.*}}, %{{.*}} {group_id = 0 : i32} : tensor<16x128x!tt.ptr<f32>>

// CHECK-DAG:    tt.store %[[TMP3:.*]], %[[VAL1:.*]], %{{.*}} {group_id = 0 : i32} : tensor<16x128x!tt.ptr<f32>>


// the second group (embedding size of 16)

// CHECK-DAG:    %[[TMP4:.*]] = tt.make_range {end = 16 : i32, group_id = 1 : i32, start = 0 : i32} : tensor<16xi32>

// CHECK-DAG:     %[[TMP5:.*]] = tt.expand_dims %[[TMP4:.*]] {axis = 0 : i32, group_id = 1 : i32} : tensor<16xi32> -> tensor<1x16xi32>

// CHECK-DAG:    %[[TMP6:.*]] = arith.extsi %[[TMP5:.*]] {group_id = 1 : i32} : tensor<1x16xi32> to tensor<1x16xi64>

// CHECK-DAG:    %[[TMP7:.*]] = tt.broadcast %[[TMP6:.*]] {group_id = 1 : i32} : tensor<1x16xi64> -> tensor<16x16xi64>

// CHECK-DAG:    %[[VAL2:.*]] = tt.load %{{.*}}, %{{.*}} {group_id = 1 : i32} : tensor<16x16x!tt.ptr<f32>>

// CHECK-DAG:    tt.store %{{.*}}, %[[VAL2:.*]], %{{.*}} {group_id = 1 : i32} : tensor<16x16x!tt.ptr<f32>>

// the third group (embedding size of 8)

// CHECK-DAG:    %[[TMP8:.*]] = tt.make_range {end = 8 : i32, group_id = 2 : i32, start = 0 : i32} : tensor<8xi32>

// CHECK-DAG:     %[[TMP9:.*]] = tt.expand_dims  %[[TMP8:.*]] {axis = 0 : i32, group_id = 2 : i32} : tensor<8xi32> -> tensor<1x8xi32>

// CHECK-DAG:    %[[TMP10:.*]] = arith.extsi %[[TMP9:.*]] {group_id = 2 : i32} : tensor<1x8xi32> to tensor<1x8xi64>

// CHECK-DAG:    %[[TMP11:.*]] = tt.broadcast  %[[TMP10:.*]] {group_id = 2 : i32} : tensor<1x8xi64> -> tensor<16x8xi64>

// CHECK-DAG:    %[[VAL3:.*]] = tt.load %{{.*}}, %{{.*}} {group_id = 2 : i32} : tensor<16x8x!tt.ptr<f32>>

// CHECK-DAG:    tt.store %{{.*}}, %[[VAL3:.*]], %{{.*}} {group_id = 2 : i32} : tensor<16x8x!tt.ptr<f32>>

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 920000 : i32} {
  tt.func public @triton_unk_fused_cat_12(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg14: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg15: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg16: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg17: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg18: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg19: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg20: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg24: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg25: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<179390> : tensor<16x512xi64>
    %cst_0 = arith.constant dense<507539> : tensor<16x512xi64>
    %cst_1 = arith.constant dense<0> : tensor<16x1xi64>
    %cst_2 = arith.constant dense<8202> : tensor<16x512xi64>
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant dense<136> : tensor<1x512xi32>
    %cst_4 = arith.constant dense<16> : tensor<16x512xi64>
    %cst_5 = arith.constant dense<8> : tensor<16x512xi64>
    %cst_6 = arith.constant dense<18856> : tensor<16x1xi32>
    %c480_i32 = arith.constant 480 : i32
    %cst_7 = arith.constant dense<128> : tensor<16x512xi64>
    %cst_8 = arith.constant dense<16> : tensor<1x512xi32>
    %cst_9 = arith.constant dense<8> : tensor<1x512xi32>
    %cst_10 = arith.constant dense<128> : tensor<1x512xi32>
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    scf.for %arg26 = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
      %3 = arith.muli %arg26, %c16_i32 : i32
      %4 = arith.addi %0, %3 : i32
      %5 = tt.expand_dims %1 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %6 = tt.splat %4 : i32 -> tensor<16x1xi32>
      %7 = arith.addi %6, %5 : tensor<16x1xi32>
      %8 = arith.addi %0, %c1_i32 : i32
      %9 = arith.minsi %8, %arg21 : i32
      %10 = tt.splat %9 : i32 -> tensor<16x1xi32>
      %11 = arith.cmpi slt, %7, %10 : tensor<16x1xi32>
      %12 = tt.expand_dims %2 {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
      %13 = arith.cmpi slt, %12, %cst_10 : tensor<1x512xi32>
      %14 = arith.cmpi slt, %12, %cst_9 : tensor<1x512xi32>
      %15 = arith.cmpi slt, %12, %cst_8 : tensor<1x512xi32>
      %16 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<16x1x!tt.ptr<i64>>
      %17 = tt.addptr %16, %7 : tensor<16x1x!tt.ptr<i64>>, tensor<16x1xi32>
      %18 = tt.load %17, %11 : tensor<16x1x!tt.ptr<i64>>
      %19 = tt.broadcast %18 : tensor<16x1xi64> -> tensor<16x512xi64>
      %20 = arith.addi %19, %cst_2 : tensor<16x512xi64>
      %21 = arith.cmpi slt, %18, %cst_1 : tensor<16x1xi64>
      %22 = tt.broadcast %18 : tensor<16x1xi64> -> tensor<16x512xi64>
      %23 = tt.broadcast %21 : tensor<16x1xi1> -> tensor<16x512xi1>
      %24 = arith.select %23, %20, %22 : tensor<16x512xi1>, tensor<16x512xi64>
      %25 = arith.muli %24, %cst_7 : tensor<16x512xi64>
      %26 = arith.extsi %12 : tensor<1x512xi32> to tensor<1x512xi64>
      %27 = tt.broadcast %26 : tensor<1x512xi64> -> tensor<16x512xi64>
      %28 = arith.addi %27, %25 : tensor<16x512xi64>
      %29 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x512x!tt.ptr<f32>>
      %30 = tt.addptr %29, %28 : tensor<16x512x!tt.ptr<f32>>, tensor<16x512xi64>
      %31 = tt.broadcast %11 : tensor<16x1xi1> -> tensor<16x512xi1>
      %32 = tt.broadcast %13 : tensor<1x512xi1> -> tensor<16x512xi1>
      %33 = arith.andi %31, %32 : tensor<16x512xi1>
      %34 = tt.addptr %arg20, %c480_i32 : !tt.ptr<f32>, i32
      %35 = arith.muli %7, %cst_6 : tensor<16x1xi32>
      %36 = tt.splat %34 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
      %37 = tt.addptr %36, %35 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
      %38 = tt.broadcast %37 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x512x!tt.ptr<f32>>
      %39 = tt.broadcast %12 : tensor<1x512xi32> -> tensor<16x512xi32>
      %40 = tt.addptr %38, %39 : tensor<16x512x!tt.ptr<f32>>, tensor<16x512xi32>
      %41 = tt.broadcast %13 : tensor<1x512xi1> -> tensor<16x512xi1>
      %42 = tt.broadcast %11 : tensor<16x1xi1> -> tensor<16x512xi1>
      %43 = arith.andi %41, %42 : tensor<16x512xi1>
      %44 = tt.load %30, %33 : tensor<16x512x!tt.ptr<f32>>
      tt.store %40, %44, %43 : tensor<16x512x!tt.ptr<f32>>
      %45 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<16x1x!tt.ptr<i64>>
      %46 = tt.addptr %45, %7 : tensor<16x1x!tt.ptr<i64>>, tensor<16x1xi32>
      %47 = tt.load %46, %11 : tensor<16x1x!tt.ptr<i64>>
      %48 = tt.broadcast %47 : tensor<16x1xi64> -> tensor<16x512xi64>
      %49 = arith.addi %48, %cst_0 : tensor<16x512xi64>
      %50 = arith.cmpi slt, %47, %cst_1 : tensor<16x1xi64>
      %51 = tt.broadcast %47 : tensor<16x1xi64> -> tensor<16x512xi64>
      %52 = tt.broadcast %50 : tensor<16x1xi1> -> tensor<16x512xi1>
      %53 = arith.select %52, %49, %51 : tensor<16x512xi1>, tensor<16x512xi64>
      %54 = arith.muli %53, %cst_5 : tensor<16x512xi64>
      %55 = arith.extsi %12 : tensor<1x512xi32> to tensor<1x512xi64>
      %56 = tt.broadcast %55 : tensor<1x512xi64> -> tensor<16x512xi64>
      %57 = arith.addi %56, %54 : tensor<16x512xi64>
      %58 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<16x512x!tt.ptr<f32>>
      %59 = tt.addptr %58, %57 : tensor<16x512x!tt.ptr<f32>>, tensor<16x512xi64>
      %60 = tt.broadcast %11 : tensor<16x1xi1> -> tensor<16x512xi1>
      %61 = tt.broadcast %14 : tensor<1x512xi1> -> tensor<16x512xi1>
      %62 = arith.andi %60, %61 : tensor<16x512xi1>
      %63 = tt.addptr %arg20, %c480_i32 : !tt.ptr<f32>, i32
      %64 = arith.muli %7, %cst_6 : tensor<16x1xi32>
      %65 = tt.splat %63 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
      %66 = tt.addptr %65, %64 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
      %67 = arith.addi %12, %cst_10 : tensor<1x512xi32>
      %68 = tt.broadcast %66 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x512x!tt.ptr<f32>>
      %69 = tt.broadcast %67 : tensor<1x512xi32> -> tensor<16x512xi32>
      %70 = tt.addptr %68, %69 : tensor<16x512x!tt.ptr<f32>>, tensor<16x512xi32>
      %71 = tt.broadcast %14 : tensor<1x512xi1> -> tensor<16x512xi1>
      %72 = tt.broadcast %11 : tensor<16x1xi1> -> tensor<16x512xi1>
      %73 = arith.andi %71, %72 : tensor<16x512xi1>
      %74 = tt.load %59, %62 : tensor<16x512x!tt.ptr<f32>>
      tt.store %70, %74, %73 : tensor<16x512x!tt.ptr<f32>>
      %75 = tt.splat %arg4 : !tt.ptr<i64> -> tensor<16x1x!tt.ptr<i64>>
      %76 = tt.addptr %75, %7 : tensor<16x1x!tt.ptr<i64>>, tensor<16x1xi32>
      %77 = tt.load %76, %11 : tensor<16x1x!tt.ptr<i64>>
      %78 = tt.broadcast %77 : tensor<16x1xi64> -> tensor<16x512xi64>
      %79 = arith.addi %78, %cst : tensor<16x512xi64>
      %80 = arith.cmpi slt, %77, %cst_1 : tensor<16x1xi64>
      %81 = tt.broadcast %77 : tensor<16x1xi64> -> tensor<16x512xi64>
      %82 = tt.broadcast %80 : tensor<16x1xi1> -> tensor<16x512xi1>
      %83 = arith.select %82, %79, %81 : tensor<16x512xi1>, tensor<16x512xi64>
      %84 = arith.muli %83, %cst_4 : tensor<16x512xi64>
      %85 = arith.extsi %12 : tensor<1x512xi32> to tensor<1x512xi64>
      %86 = tt.broadcast %85 : tensor<1x512xi64> -> tensor<16x512xi64>
      %87 = arith.addi %86, %84 : tensor<16x512xi64>
      %88 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<16x512x!tt.ptr<f32>>
      %89 = tt.addptr %88, %87 : tensor<16x512x!tt.ptr<f32>>, tensor<16x512xi64>
      %90 = tt.broadcast %11 : tensor<16x1xi1> -> tensor<16x512xi1>
      %91 = tt.broadcast %15 : tensor<1x512xi1> -> tensor<16x512xi1>
      %92 = arith.andi %90, %91 : tensor<16x512xi1>
      %93 = tt.addptr %arg20, %c480_i32 : !tt.ptr<f32>, i32
      %94 = arith.muli %7, %cst_6 : tensor<16x1xi32>
      %95 = tt.splat %93 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
      %96 = tt.addptr %95, %94 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
      %97 = arith.addi %12, %cst_3 : tensor<1x512xi32>
      %98 = tt.broadcast %96 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x512x!tt.ptr<f32>>
      %99 = tt.broadcast %97 : tensor<1x512xi32> -> tensor<16x512xi32>
      %100 = tt.addptr %98, %99 : tensor<16x512x!tt.ptr<f32>>, tensor<16x512xi32>
      %101 = tt.broadcast %15 : tensor<1x512xi1> -> tensor<16x512xi1>
      %102 = tt.broadcast %11 : tensor<16x1xi1> -> tensor<16x512xi1>
      %103 = arith.andi %101, %102 : tensor<16x512xi1>
      %104 = tt.load %89, %92 : tensor<16x512x!tt.ptr<f32>>
      tt.store %100, %104, %103 : tensor<16x512x!tt.ptr<f32>>
    }
    tt.return
  }
}


// -----

// CHECK-LABEL: test_get_right_size

// check that the size of the embedding did not shrink

// CHECK-NOT:  tt.store %{{.*}}, %{{.*}}, %{{.*}} {group_id = 0 : i32} : tensor<2x8x8x!tt.ptr<f32>>

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 900101 : i32} {
  tt.func public @test_get_right_size(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg7: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg8: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<2x8x1xi64>
    %cst_0 = arith.constant dense<50> : tensor<2x8x1xi64>
    %cst_1 = arith.constant dense<0> : tensor<2x8x1xi64>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<1024> : tensor<2x1x1xi32>
    %cst_3 = arith.constant dense<128> : tensor<1x8x1xi32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<2x8x128xf32>
    %cst_5 = arith.constant dense<6400> : tensor<2x1x1xi32>
    %cst_6 = arith.constant dense<8> : tensor<2x1x1xi32>
    %cst_7 = arith.constant dense<128> : tensor<1x1x128xi32>
    %cst_8 = arith.constant dense<8> : tensor<1x8x1xi32>
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = tt.expand_dims %2 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %6 = tt.expand_dims %5 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %7 = arith.addi %1, %c4_i32 : i32
    %8 = arith.minsi %7, %arg3 : i32
    %9 = tt.splat %8 : i32 -> tensor<2x1x1xi32>
    %10 = tt.expand_dims %3 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %11 = tt.expand_dims %10 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %12 = arith.cmpi slt, %11, %cst_8 : tensor<1x8x1xi32>
    %13 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x128xi32> -> tensor<1x1x128xi32>
    %15 = arith.cmpi slt, %14, %cst_7 : tensor<1x1x128xi32>
    %16 = tt.broadcast %11 : tensor<1x8x1xi32> -> tensor<2x8x1xi32>
    %17 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<2x8x1x!tt.ptr<i64>>
    %18 = tt.broadcast %12 : tensor<1x8x1xi1> -> tensor<2x8x1xi1>
    %19 = arith.extsi %14 : tensor<1x1x128xi32> to tensor<1x1x128xi64>
    %20 = tt.broadcast %19 : tensor<1x1x128xi64> -> tensor<2x8x128xi64>
    %21 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x8x128x!tt.ptr<f32>>
    %22 = tt.broadcast %15 : tensor<1x1x128xi1> -> tensor<2x1x128xi1>
    %23 = tt.broadcast %12 : tensor<1x8x1xi1> -> tensor<2x8x128xi1>
    %24 = arith.muli %11, %cst_3 : tensor<1x8x1xi32>
    %25 = tt.broadcast %14 : tensor<1x1x128xi32> -> tensor<1x8x128xi32>
    %26 = tt.broadcast %24 : tensor<1x8x1xi32> -> tensor<1x8x128xi32>
    %27 = arith.addi %25, %26 : tensor<1x8x128xi32>
    %28 = tt.broadcast %27 : tensor<1x8x128xi32> -> tensor<2x8x128xi32>
    %29 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x8x128x!tt.ptr<f32>>
    scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %30 = arith.muli %arg9, %c2_i32 : i32
      %31 = arith.addi %1, %30 : i32
      %32 = tt.splat %31 : i32 -> tensor<2x1x1xi32>
      %33 = arith.addi %32, %6 : tensor<2x1x1xi32>
      %34 = arith.cmpi slt, %33, %9 : tensor<2x1x1xi32>
      %35 = arith.muli %33, %cst_6 : tensor<2x1x1xi32>
      %36 = tt.broadcast %35 : tensor<2x1x1xi32> -> tensor<2x8x1xi32>
      %37 = arith.addi %16, %36 : tensor<2x8x1xi32>
      %38 = tt.addptr %17, %37 : tensor<2x8x1x!tt.ptr<i64>>, tensor<2x8x1xi32>
      %39 = tt.broadcast %34 : tensor<2x1x1xi1> -> tensor<2x8x1xi1>
      %40 = arith.andi %39, %18 : tensor<2x8x1xi1>
      %41 = tt.load %38, %40, %cst_1 : tensor<2x8x1x!tt.ptr<i64>>
      %42 = arith.addi %41, %cst_0 : tensor<2x8x1xi64>
      %43 = arith.cmpi slt, %41, %cst_1 : tensor<2x8x1xi64>
      %44 = arith.select %43, %42, %41 : tensor<2x8x1xi1>, tensor<2x8x1xi64>
      %45 = arith.muli %44, %cst : tensor<2x8x1xi64>
      %46 = tt.broadcast %45 : tensor<2x8x1xi64> -> tensor<2x8x128xi64>
      %47 = arith.addi %20, %46 : tensor<2x8x128xi64>
      %48 = arith.muli %33, %cst_5 : tensor<2x1x1xi32>
      %49 = arith.extsi %48 : tensor<2x1x1xi32> to tensor<2x1x1xi64>
      %50 = tt.broadcast %49 : tensor<2x1x1xi64> -> tensor<2x8x128xi64>
      %51 = arith.addi %47, %50 : tensor<2x8x128xi64>
      %52 = tt.addptr %21, %51 : tensor<2x8x128x!tt.ptr<f32>>, tensor<2x8x128xi64>
      %53 = tt.broadcast %34 : tensor<2x1x1xi1> -> tensor<2x1x128xi1>
      %54 = arith.andi %22, %53 : tensor<2x1x128xi1>
      %55 = tt.broadcast %54 : tensor<2x1x128xi1> -> tensor<2x8x128xi1>
      %56 = arith.andi %55, %23 : tensor<2x8x128xi1>
      %57 = tt.load %52, %56, %cst_4 : tensor<2x8x128x!tt.ptr<f32>>
      %58 = arith.muli %33, %cst_2 : tensor<2x1x1xi32>
      %59 = tt.broadcast %58 : tensor<2x1x1xi32> -> tensor<2x8x128xi32>
      %60 = arith.addi %28, %59 : tensor<2x8x128xi32>
      %61 = tt.addptr %29, %60 : tensor<2x8x128x!tt.ptr<f32>>, tensor<2x8x128xi32>
      tt.store %61, %57, %56 : tensor<2x8x128x!tt.ptr<f32>>
    }
    tt.return
  }
} 