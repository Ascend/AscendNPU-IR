// RUN: bishengir-opt %s --dpx-div-optimization -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 1900101 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 100 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @__assertfail(!llvm.ptr {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, !llvm.ptr {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, i32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, !llvm.ptr {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, i64 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}) attributes {passthrough = ["noreturn"]}
  llvm.func @_gather_kernel_4(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg22: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg23: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}, %arg24: !llvm.ptr<1>, %arg25: !llvm.ptr<1>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.kernel, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
    %0 = llvm.mlir.constant(1024 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
    %4 = llvm.zext %arg21 : i32 to i64
    %5 = llvm.zext %arg22 : i32 to i64
    %6 = llvm.zext %arg23 : i32 to i64
    %7 = llvm.mlir.constant(64 : i64) : i64
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.urem %3, %7  : i64
    %11 = llvm.udiv %3, %7  : i64
    %12 = llvm.urem %11, %8  : i64
    %13 = llvm.udiv %11, %8  : i64
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.inttoptr %14 {hivm.shared_memory} : i64 to !llvm.ptr<6>
    // CHECK: _mlir_ciface_simt_div_magic_shift_uint32_t(%arg10)
    // CHECK-NEXT: _mlir_ciface_simt_div_magic_mul_uint32_t(%arg10,
    // CHECK: ascend_dpx.store
    // CHECK-NEXT: ascend_dpx.store
    // CHECK: _mlir_ciface_simt_div_magic_shift_uint32_t(%arg9)
    // CHECK-NEXT: _mlir_ciface_simt_div_magic_mul_uint32_t(%arg9,
    // CHECK: ascend_dpx.store
    // CHECK-NEXT: ascend_dpx.store
    // CHECK: _mlir_ciface_simt_div_magic_shift_uint32_t(%arg8)
    // CHECK-NEXT: _mlir_ciface_simt_div_magic_mul_uint32_t(%arg8,
    // CHECK: ascend_dpx.store
    // CHECK-NEXT: ascend_dpx.store
    // CHECK: _mlir_ciface_simt_div_magic_shift_uint32_t(%arg7)
    // CHECK-NEXT: _mlir_ciface_simt_div_magic_mul_uint32_t(%arg7,
    // CHECK: ascend_dpx.store
    // CHECK-NEXT: ascend_dpx.store
    hivm_regbaseintrins.intrins.launch_func @_gather_kernel_4_vf_simt threads in (%0, %1, %2) args(%arg0, %arg1, %arg2, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %7, %10, %15) : !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, !llvm.ptr<6>
    llvm.return
  }
  llvm.func @_gather_kernel_4_vf_simt(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i64 {tt.divisibility = 16 : i32}, %arg19: i64, %arg20: !llvm.ptr<6> {hivm.shared_memory, tt.divisibility = 16 : i32}) attributes {hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<1024>, noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 1024>} {
    %0 = llvm.mlir.constant(0 : i32) : i32

    // CHECK: ascend_dpx.load
    // CHECK-NEXT: ascend_dpx.load
    // CHECK-NEXT: ascend_dpx.umulhi
    // CHECK-NEXT: llvm.lshr
    %1 = ascend_dpx.div %0, %arg7 : (i32, i32) -> i32

    // CHECK: ascend_dpx.load
    // CHECK-NEXT: ascend_dpx.load
    // CHECK-NEXT: ascend_dpx.umulhi
    // CHECK-NEXT: llvm.lshr
    %2 = llvm.sdiv %0, %arg6 : i32

    // CHECK: ascend_dpx.load
    // CHECK-NEXT: ascend_dpx.load
    // CHECK-NEXT: ascend_dpx.umulhi
    // CHECK-NEXT: llvm.lshr
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.sub
    %3 = llvm.urem %0, %arg5 : i32
    // CHECK: ascend_dpx.load
    // CHECK-NEXT: ascend_dpx.load
    // CHECK-NEXT: ascend_dpx.umulhi
    // CHECK-NEXT: llvm.lshr
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.sub
    %4 = llvm.srem %0, %arg4 : i32
    llvm.return
  }

  llvm.func @_gather_kernel_4_2(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg22: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg23: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}, %arg24: !llvm.ptr<1>, %arg25: !llvm.ptr<1>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.kernel, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
    %0 = llvm.mlir.constant(1024 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
    %4 = llvm.zext %arg21 : i32 to i64
    %5 = llvm.zext %arg22 : i32 to i64
    %6 = llvm.zext %arg23 : i32 to i64
    %7 = llvm.mlir.constant(64 : i64) : i64
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.urem %3, %7  : i64
    %11 = llvm.udiv %3, %7  : i64
    %12 = llvm.urem %11, %8  : i64
    %13 = llvm.udiv %11, %8  : i64
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.inttoptr %14 {hivm.shared_memory} : i64 to !llvm.ptr<6>
    // CHECK: _mlir_ciface_simt_div_magic_shift_uint32_t(%arg14)
    // CHECK-NEXT: _mlir_ciface_simt_div_magic_mul_uint32_t(%arg14,
    // CHECK: ascend_dpx.store
    // CHECK-NEXT: ascend_dpx.store
    // CHECK: _mlir_ciface_simt_div_magic_shift_uint32_t(%arg13)
    // CHECK-NEXT: _mlir_ciface_simt_div_magic_mul_uint32_t(%arg13,
    // CHECK: ascend_dpx.store
    // CHECK-NEXT: ascend_dpx.store
    // CHECK: _mlir_ciface_simt_div_magic_shift_uint32_t(%arg12)
    // CHECK-NEXT: _mlir_ciface_simt_div_magic_mul_uint32_t(%arg12,
    // CHECK: ascend_dpx.store
    // CHECK-NEXT: ascend_dpx.store
    // CHECK: _mlir_ciface_simt_div_magic_shift_uint32_t(%arg11)
    // CHECK-NEXT: _mlir_ciface_simt_div_magic_mul_uint32_t(%arg11,
    // CHECK: ascend_dpx.store
    // CHECK-NEXT: ascend_dpx.store
    hivm_regbaseintrins.intrins.launch_func @_gather_kernel_4_vf_simt threads in (%0, %1, %2) args(%arg0, %arg1, %arg2, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %7, %10, %15) : !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, !llvm.ptr<6>
    llvm.return
  }
  llvm.func @_gather_kernel_4_2_vf_simt(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i64 {tt.divisibility = 16 : i32}, %arg19: i64, %arg20: !llvm.ptr<6> {hivm.shared_memory, tt.divisibility = 16 : i32}) attributes {hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<1024>, noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 1024>} {
    %0 = llvm.mlir.constant(0 : i32) : i32

    %1 = ascend_dpx.div %0, %arg11 : (i32, i32) -> i32

    %2 = llvm.sdiv %0, %arg10 : i32

    %3 = llvm.urem %0, %arg9 : i32
    %4 = llvm.srem %0, %arg8 : i32

    %22 = llvm.udiv %0, %arg10 : i32
    llvm.return
  }

  llvm.func @_gather_kernel_4_vf_not_simt(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i64 {tt.divisibility = 16 : i32}, %arg19: i64, %arg20: !llvm.ptr<6> {tt.divisibility = 16 : i32}) attributes {hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<1024>, noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 1024>} {
    %0 = llvm.mlir.constant(0 : i32) : i32

    // CHECK: ascend_dpx.div
    %1 = ascend_dpx.div %0, %arg7 : (i32, i32) -> i32

    // CHECK: llvm.sdiv
    %2 = llvm.sdiv %0, %arg6 : i32

    // CHECK: llvm.udiv
    %22 = llvm.udiv %0, %arg6 : i32

    // CHECK: llvm.urem
    %3 = llvm.urem %0, %arg5 : i32

    // CHECK: llvm.srem
    %4 = llvm.srem %0, %arg4 : i32
    llvm.return
  }

  llvm.func @_gather_kernel_4_3(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg22: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg23: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}, %arg24: !llvm.ptr<1>, %arg25: !llvm.ptr<1>, %arg26: !llvm.ptr<6> {hivm.shared_memory}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.kernel, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
    %0 = llvm.mlir.constant(1024 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
    %4 = llvm.zext %arg21 : i32 to i64
    %5 = llvm.zext %arg22 : i32 to i64
    %6 = llvm.zext %arg23 : i32 to i64
    %7 = llvm.mlir.constant(64 : i64) : i64
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.urem %3, %7  : i64
    %11 = llvm.udiv %3, %7  : i64
    %12 = llvm.urem %11, %8  : i64
    %13 = llvm.udiv %11, %8  : i64
    // CHECK-LABEL: _gather_kernel_4_3(
    // CHECK-SAME: %[[ARGUMENT:arg[0-9]+]]: !llvm.ptr<6> {hivm.shared_memory}
    // CHECK: %{{.*}} = llvm.getelementptr %[[ARGUMENT]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
    // CHECK: %{{.*}} = llvm.getelementptr %[[ARGUMENT]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
    // CHECK: %{{.*}} = llvm.getelementptr %[[ARGUMENT]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
    // CHECK: %{{.*}} = llvm.getelementptr %[[ARGUMENT]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
    // CHECK: %{{.*}} = llvm.getelementptr %[[ARGUMENT]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
    // CHECK: %{{.*}} = llvm.getelementptr %[[ARGUMENT]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
    // CHECK: %{{.*}} = llvm.getelementptr %[[ARGUMENT]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
    // CHECK: %{{.*}} = llvm.getelementptr %[[ARGUMENT]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
    hivm_regbaseintrins.intrins.launch_func @_gather_kernel_4_3_vf_simt threads in (%0, %1, %2) args(%arg0, %arg1, %arg2, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %7, %10, %arg26) : !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, !llvm.ptr<6>
    llvm.return
  }
  // CHECK-LABEL: llvm.func @_gather_kernel_4_3_vf_simt
  // CHECK-SAME: %[[PARAM:arg[0-9]+]]: !llvm.ptr<6> {hivm.shared_memory
  // CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
  // CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
  // CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
  // CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
  // CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
  // CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
  // CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
  // CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]][%{{.*}}] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
  llvm.func @_gather_kernel_4_3_vf_simt(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i64 {tt.divisibility = 16 : i32}, %arg19: i64, %arg20: !llvm.ptr<6> {hivm.shared_memory, tt.divisibility = 16 : i32}) attributes {hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<1024>, noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 1024>} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = ascend_dpx.div %0, %arg7 : (i32, i32) -> i32
    %2 = llvm.sdiv %0, %arg6 : i32
    %3 = llvm.urem %0, %arg5 : i32
    %4 = llvm.srem %0, %arg4 : i32
    llvm.return
  }
}
