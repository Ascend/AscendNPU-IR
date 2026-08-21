// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins

// CHECK-NOT: ascend_dpx.atomic_and

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 111 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @atomic_and(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.undef : vector<2xi16>
    %1 = llvm.mlir.undef : vector<2xi32>
    %2 = llvm.mlir.constant(3 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(127 : i32) : i32
    %6 = llvm.mlir.constant(5 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mlir.constant(0 : index) : i32
    %10 = llvm.mlir.constant(512 : i32) : i32
    %11 = ascend_dpx.block_idx_x
    %12 = llvm.mul %11, %10 : i32
    %13 = ascend_dpx.thread_id_x
    %14 = llvm.urem %13, %8  : i32
    %15 = llvm.udiv %13, %8  : i32
    %16 = llvm.shl %14, %7 : i32
    %17 = llvm.or %7, %16  : i32
    %18 = llvm.shl %15, %6 : i32
    %19 = llvm.or %17, %18  : i32
    %20 = llvm.and %19, %5  : i32
    %21 = llvm.shl %20, %4 : i32
    %22 = llvm.or %21, %7  : i32
    %23 = llvm.xor %7, %22  : i32
    %24 = llvm.xor %23, %7  : i32
    %25 = llvm.xor %23, %3  : i32
    %26 = llvm.xor %23, %4  : i32
    %27 = llvm.xor %23, %2  : i32
    %28 = llvm.add %24, %9 : i32
    %29 = llvm.add %25, %9 : i32
    %30 = llvm.add %26, %9 : i32
    %31 = llvm.add %27, %9 : i32
    %32 = llvm.add %12, %28 : i32
    %33 = llvm.add %12, %30 : i32
    %34 = llvm.icmp "slt" %32, %arg3 : i32
    %35 = llvm.getelementptr %arg0[%32] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    %36 = llvm.getelementptr %arg0[%33] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    %37 = ascend_dpx.load %35, %34, %7 : (!llvm.ptr<1>, i1, i32) -> i32
    %38 = llvm.bitcast %37 : i32 to vector<2xi16>
    %39 = llvm.extractelement %38[%7 : i32] : vector<2xi16>
    %40 = llvm.extractelement %38[%3 : i32] : vector<2xi16>
    %41 = ascend_dpx.load %36, %34, %7 : (!llvm.ptr<1>, i1, i32) -> i32
    %42 = llvm.bitcast %41 : i32 to vector<2xi16>
    %43 = llvm.extractelement %42[%7 : i32] : vector<2xi16>
    %44 = llvm.extractelement %42[%3 : i32] : vector<2xi16>
    %45 = llvm.getelementptr %arg1[%28] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    %46 = llvm.getelementptr %arg1[%29] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    %47 = llvm.getelementptr %arg1[%30] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    %48 = llvm.getelementptr %arg1[%31] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    %49 = ascend_dpx.atomic_and %45, %39 : (!llvm.ptr<1>, i16) -> i16
    %50 = ascend_dpx.atomic_and %46, %40 : (!llvm.ptr<1>, i16) -> i16
    %51 = ascend_dpx.atomic_and %47, %43 : (!llvm.ptr<1>, i16) -> i16
    %52 = ascend_dpx.atomic_and %48, %44 : (!llvm.ptr<1>, i16) -> i16
    %53 = llvm.getelementptr %arg2[%28] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    %54 = llvm.insertelement %49, %0[%7 : i32] : vector<2xi16>
    %55 = llvm.insertelement %50, %54[%3 : i32] : vector<2xi16>
    %56 = llvm.bitcast %55 : vector<2xi16> to i32
    %57 = llvm.insertelement %56, %1[%7 : i32] : vector<2xi32>
    %58 = llvm.insertelement %51, %0[%7 : i32] : vector<2xi16>
    %59 = llvm.insertelement %52, %58[%3 : i32] : vector<2xi16>
    %60 = llvm.bitcast %59 : vector<2xi16> to i32
    %61 = llvm.insertelement %60, %57[%3 : i32] : vector<2xi32>
    ascend_dpx.store %53, %61, %34 : <1>, vector<2xi32>
    llvm.return
  }
}
