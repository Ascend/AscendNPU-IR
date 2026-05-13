// RUN: bishengir-opt %s -convert-hacc-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @Fused_Add_fusion_hfusion(
// CHECK: %[[PARAM:.*]] = llvm.load %[[ADDR:.*]] : !llvm.ptr -> i32
// CHECK-NOT: %[[CONS:.*]] = llvm.mlir.constant(20
// CHECK: llvm.call @_Z21__cce_rtConfigureCalljPvS_
// CHECK: llvm.call @example_stub
// CHECK-LABEL: llvm.func internal @rtRegisterGlobals
// CHECK: llvm.mlir.addressof @example_stub
// CHECK: llvm.call @rtFunctionRegister

module {
  llvm.func @example(!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, hacc.always_inline, hacc.entry}
  llvm.func @Fused_Add_fusion_hfusion(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.mlir.constant(16 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg1, %9[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.constant(16 : index) : i64
    %15 = llvm.insertvalue %14, %13[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %arg2, %18[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.insertvalue %arg2, %19[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.insertvalue %21, %20[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.mlir.constant(16 : index) : i64
    %24 = llvm.insertvalue %23, %22[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.insertvalue %25, %24[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @example(%arg0, %arg1, %arg2) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>) -> ()
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @Fused_Add_fusion_hfusion_2(
// CHECK: %[[CONS:.*]] = llvm.mlir.constant(20
// CHECK: llvm.call @_Z21__cce_rtConfigureCalljPvS_(%[[CONS]]
// CHECK: llvm.call @example_stub
// CHECK-LABEL: llvm.func internal @rtRegisterGlobals
// CHECK: llvm.mlir.addressof @example_stub
// CHECK: llvm.call @rtFunctionRegister


module {
  llvm.func @example(!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, hacc.always_inline, hacc.entry, hacc.block_dim = 20 : i64}
  llvm.func @Fused_Add_fusion_hfusion_2(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.mlir.constant(16 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg1, %9[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.constant(16 : index) : i64
    %15 = llvm.insertvalue %14, %13[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %arg2, %18[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.insertvalue %arg2, %19[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.insertvalue %21, %20[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.mlir.constant(16 : index) : i64
    %24 = llvm.insertvalue %23, %22[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.insertvalue %25, %24[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @example(%arg0, %arg1, %arg2) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>) -> ()
    llvm.return
  }
}

// -----

// CHECK: llvm.func @main_multi_LAST_AXIS_PBR_2_tiling_func(
// CHECK: undef
// CHECK: insertvalue
// CHECK: insertvalue
// CHECK: return
// CHECK: llvm.func @main_multi_LAST_AXIS_PBR_2_infershape_func_outs_0(
// CHECK: insertvalue
// CHECK: insertvalue
// CHECK: return
module {
  llvm.func @main_multi_LAST_AXIS_PBR_2_tiling_func(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<1>, %arg15: !llvm.ptr<1>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr<1>, %arg20: !llvm.ptr<1>, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr<1>, %arg27: !llvm.ptr<1>, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: !llvm.ptr<1>, %arg34: !llvm.ptr<1>, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: !llvm.ptr<1>, %arg39: !llvm.ptr<1>, %arg40: i64, %arg41: i64, %arg42: i64, %arg43: i64, %arg44: i64, %arg45: !llvm.ptr<1>, %arg46: !llvm.ptr<1>, %arg47: i64, %arg48: i64, %arg49: i64) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg45, %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg46, %1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg47, %2[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg48, %3[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.insertvalue %arg49, %4[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.mlir.constant(5 : index) : i64
    %7 = llvm.mlir.constant(4 : index) : i64
    %8 = llvm.mlir.constant(3 : index) : i64
    %9 = llvm.mlir.constant(2 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(0 : i64) : i64
    %12 = llvm.mlir.constant(14 : i64) : i64
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.mlir.constant(256 : i64) : i64
    %15 = llvm.mlir.constant(15104 : i64) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    llvm.store %11, %arg46 : i64, !llvm.ptr<1>
    %17 = llvm.getelementptr %arg46[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
    llvm.store %12, %17 : i64, !llvm.ptr<1>
    %18 = llvm.getelementptr %arg46[2] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
    llvm.store %13, %18 : i64, !llvm.ptr<1>
    %19 = llvm.getelementptr %arg46[3] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
    llvm.store %13, %19 : i64, !llvm.ptr<1>
    %20 = llvm.getelementptr %arg46[4] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
    llvm.store %14, %20 : i64, !llvm.ptr<1>
    %21 = llvm.getelementptr %arg46[5] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
    llvm.store %15, %21 : i64, !llvm.ptr<1>
    llvm.return
  }
  llvm.func @main_multi_LAST_AXIS_PBR_2(!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@main_multi_LAST_AXIS_PBR_2_tiling_func>, hacc.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>}
  llvm.func @main_multi_LAST_AXIS_PBR_2_infershape_func_outs_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<1>, %arg15: !llvm.ptr<1>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr<1>, %arg20: !llvm.ptr<1>, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr<1>, %arg27: !llvm.ptr<1>, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: !llvm.ptr<1>, %arg34: !llvm.ptr<1>, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: !llvm.ptr<1>, %arg39: !llvm.ptr<1>, %arg40: i64, %arg41: i64, %arg42: i64) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_output_shape_function>} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg38, %8[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg39, %9[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %arg40, %10[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %arg41, %11[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg42, %12[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(256 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    llvm.store %arg3, %arg39 : i64, !llvm.ptr<1>
    %17 = llvm.getelementptr %arg39[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
    llvm.store %15, %17 : i64, !llvm.ptr<1>
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @mlir_fused_sum_0_0_0_stub
module {
  llvm.func @mlir_fused_sum_0_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
    %0 = llvm.mlir.constant(3 : i64) : i64
    llvm.return %0 : i64
  }
  llvm.func @mlir_fused_sum_0_0_tiling_function(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
    %0 = llvm.mlir.constant(-8193 : index) : i64
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(-1 : index) : i64
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.mlir.constant(7 : index) : i64
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(8192 : index) : i64
    %8 = llvm.add %arg4, %4  : i64
    %9 = llvm.icmp "slt" %8, %5 : i64
    %10 = llvm.sub %2, %8  : i64
    %11 = llvm.select %9, %10, %8 : i1, i64
    %12 = llvm.sdiv %11, %3  : i64
    %13 = llvm.sub %2, %12  : i64
    %14 = llvm.select %9, %13, %12 : i1, i64
    %15 = llvm.mul %14, %3  : i64
    %16 = llvm.icmp "sge" %15, %7 : i64
    %17 = llvm.zext %16 : i1 to i64
    %18 = llvm.icmp "sgt" %17, %6 : i64
    %19 = llvm.zext %18 : i1 to i64
    %20 = llvm.icmp "eq" %17, %6 : i64
    %21 = llvm.zext %20 : i1 to i64
    %22 = llvm.icmp "sgt" %17, %5 : i64
    %23 = llvm.zext %22 : i1 to i64
    %24 = llvm.icmp "eq" %17, %5 : i64
    %25 = llvm.zext %24 : i1 to i64
    %26 = llvm.mul %19, %2  : i64
    %27 = llvm.add %26, %6  : i64
    %28 = llvm.mul %21, %7  : i64
    %29 = llvm.mul %21, %2  : i64
    %30 = llvm.add %29, %6  : i64
    %31 = llvm.add %arg4, %4  : i64
    %32 = llvm.icmp "slt" %31, %5 : i64
    %33 = llvm.sub %2, %31  : i64
    %34 = llvm.select %32, %33, %31 : i1, i64
    %35 = llvm.sdiv %34, %3  : i64
    %36 = llvm.sub %2, %35  : i64
    %37 = llvm.select %32, %36, %35 : i1, i64
    %38 = llvm.mul %37, %3  : i64
    %39 = llvm.mul %30, %38  : i64
    %40 = llvm.add %28, %39  : i64
    %41 = llvm.mul %27, %40  : i64
    %42 = llvm.add %19, %41  : i64
    %43 = llvm.mul %23, %2  : i64
    %44 = llvm.add %43, %6  : i64
    %45 = llvm.add %arg4, %4  : i64
    %46 = llvm.icmp "slt" %45, %5 : i64
    %47 = llvm.sub %2, %45  : i64
    %48 = llvm.select %46, %47, %45 : i1, i64
    %49 = llvm.sdiv %48, %3  : i64
    %50 = llvm.sub %2, %49  : i64
    %51 = llvm.select %46, %50, %49 : i1, i64
    %52 = llvm.mul %51, %3  : i64
    %53 = llvm.select %1, %0, %7 : i1, i64
    %54 = llvm.sdiv %53, %52  : i64
    %55 = llvm.sub %2, %54  : i64
    %56 = llvm.select %1, %55, %54 : i1, i64
    %57 = llvm.mul %25, %56  : i64
    %58 = llvm.mul %25, %2  : i64
    %59 = llvm.add %58, %6  : i64
    %60 = llvm.mul %59, %arg3  : i64
    %61 = llvm.add %57, %60  : i64
    %62 = llvm.mul %44, %61  : i64
    %63 = llvm.add %23, %62  : i64
    llvm.store %17, %arg15 : i64, !llvm.ptr
    %64 = llvm.getelementptr %arg15[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %63, %64 : i64, !llvm.ptr
    %65 = llvm.getelementptr %arg15[2] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %42, %65 : i64, !llvm.ptr
    llvm.return
  }
  llvm.func @mlir_fused_sum_0_0_1(!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@mlir_fused_sum_0_get_tiling_struct_size_function>, hacc.tiling_function = #hacc.tiling_function<@mlir_fused_sum_0_0_tiling_function>, hacc.block_dim = 48 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, hivm.func_core_type = #hivm.func_core_type<AIV>}
  llvm.func @mlir_fused_sum_0_0_0(!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@mlir_fused_sum_0_get_tiling_struct_size_function>, hacc.tiling_function = #hacc.tiling_function<@mlir_fused_sum_0_0_tiling_function>, hacc.block_dim = 48 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, hivm.func_core_type = #hivm.func_core_type<AIV>}
  llvm.func @mlir_fused_sum_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<1>, %arg15: !llvm.ptr<1>, %arg16: i64, %arg17: i64, %arg18: i64) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %0 = llvm.load %arg15 : !llvm.ptr<1> -> i64
    %1 = llvm.trunc %0 : i64 to i32
    llvm.switch %1 : i32, ^bb3 [
      1: ^bb1,
      0: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    llvm.call @mlir_fused_sum_0_0_1(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> ()
    llvm.br ^bb4
  ^bb2:  // pred: ^bb0
    llvm.call @mlir_fused_sum_0_0_0(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> ()
    llvm.br ^bb4
  ^bb3:  // pred: ^bb0
    llvm.br ^bb4
  ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
    llvm.return
  }
  // CHECK: llvm.func @mlir_fused_sum_0_tiling_function
  // CHECK-NOT: llvm.func @mlir_fused_sum_0_0_tiling_function_stub
  // CHECK: llvm.call @mlir_fused_sum_0_0_tiling_function
  llvm.func @mlir_fused_sum_0_tiling_function(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    llvm.call @mlir_fused_sum_0_0_tiling_function(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }
}

// -----

// CHECK: @rtConfigureCall(

// CHECK: llvm.call @mlir_fused_threshold_backward_11_0_0_stub(
// CHECK: llvm.call @puts(
// CHECK: llvm.call @abort(
// CHECK: llvm.unreachable
// CHECK: return
module {
  llvm.mlir.global private constant @assert_msg_1(dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 118, 97, 108, 117, 101, 115, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 32, 110, 101, 119, 32, 100, 105, 109, 101, 110, 115, 105, 111, 110, 115, 0]> : tensor<46xi8>) {addr_space = 0 : i32} : !llvm.array<46 x i8>
  llvm.mlir.global private constant @assert_msg_0(dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 118, 97, 108, 117, 101, 115, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 32, 110, 101, 119, 32, 100, 105, 109, 101, 110, 115, 105, 111, 110, 115, 0]> : tensor<46xi8>) {addr_space = 0 : i32} : !llvm.array<46 x i8>
  llvm.func @abort()
  llvm.func @puts(!llvm.ptr)
  llvm.func @mlir_fused_threshold_backward_11_0_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = llvm.mlir.constant(3 : i64) : i64
    llvm.return %0 : i64
  }
  llvm.func @mlir_fused_threshold_backward_11_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
    %0 = llvm.call @mlir_fused_threshold_backward_11_0_get_tiling_struct_size_function() : () -> i64
    llvm.return %0 : i64
  }
  llvm.func @mlir_fused_threshold_backward_11_0_1(!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>}, i64 {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<3>}, i64 {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<4>}, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) attributes {enable_auto_mark_buffer_size, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@mlir_fused_threshold_backward_11_tiling_function>, hacc.block_dim = 48 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, hivm.func_core_type = #hivm.func_core_type<AIV>}
  llvm.func @mlir_fused_threshold_backward_11_0_0(!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>}, i64 {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<3>}, i64 {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<4>}, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) attributes {enable_auto_mark_buffer_size, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@mlir_fused_threshold_backward_11_tiling_function>, hacc.block_dim = 48 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, hivm.func_core_type = #hivm.func_core_type<AIV>}
  llvm.func @mlir_fused_threshold_backward_11(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<1>, %arg15: !llvm.ptr<1>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: !llvm.ptr<1>, %arg26: !llvm.ptr<1>, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}, %arg37: !llvm.ptr<1>, %arg38: !llvm.ptr<1>, %arg39: i64, %arg40: i64, %arg41: i64) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.load %arg36 : !llvm.ptr -> i64
    %2 = llvm.icmp "sge" %arg11, %0 : i64
    llvm.cond_br %2, ^bb1, ^bb11
  ^bb1:  // pred: ^bb0
    %3 = llvm.icmp "sge" %arg12, %0 : i64
    llvm.cond_br %3, ^bb2, ^bb11
  ^bb2:  // pred: ^bb1
    %4 = llvm.icmp "sge" %arg13, %0 : i64
    llvm.cond_br %4, ^bb3, ^bb11
  ^bb3:  // pred: ^bb2
    %5 = llvm.trunc %1 : i64 to i32
    llvm.switch %5 : i32, ^bb11 [
      1: ^bb5,
      0: ^bb6
    ]
  ^bb5:  // pred: ^bb3
    llvm.call @mlir_fused_threshold_backward_11_0_1(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg11, %arg12, %arg13, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> ()
    llvm.br ^bb11
  ^bb6:  // pred: ^bb3
    llvm.call @mlir_fused_threshold_backward_11_0_0(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg11, %arg12, %arg13, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> ()
    llvm.br ^bb11
  ^bb11:  // pred: ^bb2
    %8 = llvm.mlir.addressof @assert_msg_1 : !llvm.ptr
    llvm.call @puts(%8) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
}
