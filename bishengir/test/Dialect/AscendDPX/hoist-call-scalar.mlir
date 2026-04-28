// RUN: bishengir-opt -hoist-call-scalar-to-caller %s | FileCheck %s

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 900101 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 116 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @__assertfail(!llvm.ptr {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, !llvm.ptr {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, !llvm.ptr {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, i64 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}) attributes {passthrough = ["noreturn"]}
  llvm.func @_mlir_ciface_simt_div_magic_mul_uint32_t(i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}) -> i32
  llvm.func @_mlir_ciface_simt_div_magic_shift_uint32_t(i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}) -> i32
  // CHECK-LABEL: @_gather_kernel_4
  llvm.func @_gather_kernel_4(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg22: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg23: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}, %arg24: !llvm.ptr<1>, %arg25: !llvm.ptr<1>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.kernel, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
    %0 = llvm.mlir.constant(1024 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
    %4 = llvm.zext %arg21 : i32 to i64
    %5 = llvm.zext %arg22 : i32 to i64
    %6 = llvm.zext %arg23 : i32 to i64
    %7 = llvm.mlir.constant(128 : i64) : i64
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.urem %3, %7  : i64
    %11 = llvm.udiv %3, %7  : i64
    %12 = llvm.urem %11, %8  : i64
    %13 = llvm.udiv %11, %8  : i64
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.inttoptr %14 {hivm.shared_memory} : i64 to !llvm.ptr<6>
    // CHECK: llvm.call
    // CHECK: llvm.call
    // CHECK: llvm.store
    hivm_regbaseintrins.intrins.launch_func @_gather_kernel_4_vf_simt threads in (%0, %1, %2) args(%arg0, %arg1, %arg2, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %7, %10, %15) : !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, !llvm.ptr<6>
    llvm.return
  }
  // CHECK-LABEL: @_gather_kernel_4_vf_simt
  // CHECK-NOT: call_scalar
  llvm.func @_gather_kernel_4_vf_simt(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr<6> {hivm.shared_memory}) attributes {hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<1024>, noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 1024>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1024 : i32) : i32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(-2147483648 : i64) : i64
    %6 = llvm.mlir.constant(2147483647 : i64) : i64
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(false) : i1
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(32 : i32) : i32
    %11 = llvm.mlir.constant(5 : i32) : i32
    %12 = llvm.mlir.constant(1023 : i32) : i32
    %13 = llvm.mlir.undef : vector<1xi32>
    %14 = llvm.mlir.constant(0 : index) : i32
    %15 = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg7) {use_shmem_offset = 0 : i32} : (i32) -> i32
    %16 = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_mul_uint32_t(%arg7, %15) {use_shmem_offset = 16 : i32} : (i32, i32) -> i32
    %17 = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg6) {use_shmem_offset = 32 : i32} : (i32) -> i32
    %18 = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_mul_uint32_t(%arg6, %17) {use_shmem_offset = 48 : i32} : (i32, i32) -> i32
    %19 = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg5) {use_shmem_offset = 64 : i32} : (i32) -> i32
    %20 = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_mul_uint32_t(%arg5, %19) {use_shmem_offset = 80 : i32} : (i32, i32) -> i32
    %21 = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg4) {use_shmem_offset = 96 : i32} : (i32) -> i32
    %22 = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_mul_uint32_t(%arg4, %21) {use_shmem_offset = 112 : i32} : (i32, i32) -> i32
    %23 = llvm.trunc %arg19 : i64 to i32
    %24 = llvm.trunc %arg18 : i64 to i32
    %25 = llvm.sext %24 : i32 to i64
    %26 = llvm.sub %25, %2 : i64
    %27 = llvm.icmp "sle" %26, %6 : i64
    %28 = llvm.icmp "sge" %26, %5 : i64
    %29 = llvm.and %27, %28  : i1
    %30 = llvm.icmp "eq" %29, %8 : i1
    %31 = llvm.or %8, %30  : i1
    llvm.cond_br %31, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %32 = llvm.sub %24, %7 : i32
    %33 = llvm.sext %arg17 : i32 to i64
    %34 = llvm.sext %32 : i32 to i64
    %35 = llvm.add %33, %34 : i64
    %36 = llvm.icmp "sle" %35, %6 : i64
    %37 = llvm.icmp "sge" %35, %5 : i64
    %38 = llvm.and %36, %37  : i1
    %39 = llvm.icmp "eq" %38, %8 : i1
    %40 = llvm.or %8, %39  : i1
    llvm.cond_br %40, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %41 = llvm.add %arg17, %32 : i32
    %42 = ascend_dpx.div %41, %24 : (i32, i32) -> i32
    %43 = llvm.mul %23, %42 : i32
    %44 = llvm.add %43, %42 : i32
    %45 = llvm.intr.smin(%44, %arg17)  : (i32, i32) -> i32
    %46 = ascend_dpx.thread_id_x
    %47 = llvm.urem %46, %10  : i32
    %48 = llvm.udiv %46, %10  : i32
    %49 = llvm.shl %47, %4 : i32
    %50 = llvm.or %4, %49  : i32
    %51 = llvm.shl %48, %11 : i32
    %52 = llvm.or %50, %51  : i32
    %53 = llvm.and %52, %12  : i32
    %54 = llvm.lshr %53, %4  : i32
    %55 = llvm.or %54, %4  : i32
    %56 = llvm.xor %4, %55  : i32
    %57 = llvm.xor %56, %4  : i32
    %58 = llvm.add %57, %14 : i32
    llvm.br ^bb5(%43 : i32)
  ^bb5(%59: i32):  // 2 preds: ^bb4, ^bb6
    %60 = llvm.icmp "slt" %59, %45 : i32
    llvm.cond_br %60, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %61 = llvm.add %59, %58 : i32
    %62 = llvm.icmp "slt" %61, %arg17 : i32
    %63 = ascend_dpx.umulhi %61, %16 : (i32, i32) -> i32
    %64 = llvm.add %63, %61 : i32
    %65 = llvm.ashr %64, %15  : i32
    %66 = llvm.mul %65, %arg7 : i32
    %67 = llvm.sub %61, %66 : i32
    %68 = ascend_dpx.umulhi %65, %18 : (i32, i32) -> i32
    %69 = llvm.add %68, %65 : i32
    %70 = llvm.ashr %69, %17  : i32
    %71 = llvm.mul %70, %arg6 : i32
    %72 = llvm.sub %65, %71 : i32
    %73 = llvm.mul %72, %arg13 : i32
    %74 = llvm.add %67, %73 : i32
    %75 = llvm.mul %72, %arg16 : i32
    %76 = llvm.add %67, %75 : i32
    %77 = llvm.mul %72, %arg10 : i32
    %78 = ascend_dpx.umulhi %70, %20 : (i32, i32) -> i32
    %79 = llvm.add %78, %70 : i32
    %80 = llvm.ashr %79, %19  : i32
    %81 = llvm.mul %80, %arg5 : i32
    %82 = llvm.sub %70, %81 : i32
    %83 = llvm.mul %82, %arg12 : i32
    %84 = llvm.add %74, %83 : i32
    %85 = llvm.mul %82, %arg15 : i32
    %86 = llvm.add %76, %85 : i32
    %87 = llvm.mul %82, %arg9 : i32
    %88 = llvm.add %77, %87 : i32
    %89 = ascend_dpx.umulhi %80, %22 : (i32, i32) -> i32
    %90 = llvm.add %89, %80 : i32
    %91 = llvm.ashr %90, %21  : i32
    %92 = llvm.mul %91, %arg4 : i32
    %93 = llvm.sub %80, %92 : i32
    %94 = llvm.mul %93, %arg11 : i32
    %95 = llvm.add %84, %94 : i32
    %96 = llvm.mul %93, %arg14 : i32
    %97 = llvm.add %86, %96 : i32
    %98 = llvm.mul %93, %arg8 : i32
    %99 = llvm.add %88, %98 : i32
    %100 = llvm.getelementptr %arg1[%95] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i64
    %101 = ascend_dpx.load %100, %62, %0 : (!llvm.ptr<1>, i1, i64) -> i64
    %102 = llvm.trunc %101 : i64 to i32
    %103 = llvm.icmp "slt" %102, %4 : i32
    %104 = llvm.add %102, %arg3 : i32
    %105 = llvm.select %103, %104, %102 : i1, i32
    %106 = llvm.add %99, %105 : i32
    %107 = llvm.getelementptr %arg0[%106] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %108 = llvm.bitcast %3 : f32 to i32
    %109 = ascend_dpx.load %107, %62, %108 : (!llvm.ptr<1>, i1, i32) -> i32
    %110 = llvm.getelementptr %arg2[%97] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %111 = llvm.insertelement %109, %13[%4 : i32] : vector<1xi32>
    ascend_dpx.store %110, %111, %62 : <1>, vector<1xi32>
    %112 = llvm.add %59, %1 : i32
    llvm.br ^bb5(%112 : i32)
  ^bb7:  // pred: ^bb5
    llvm.return
  }
}