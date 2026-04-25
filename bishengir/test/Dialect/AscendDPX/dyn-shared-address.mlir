// RUN: bishengir-opt --adapt-gpu-kernel -split-input-file %s | FileCheck %s

// CHECK: %[[ARGUMENT:arg[0-9]+]]: !llvm.ptr<6> {hivm.shared_memory}
// CHECK: hivm_regbaseintrins.intrins.launch_func @triton_unk_reduce_vf_simt
// CHECK-SAME: %[[ARGUMENT]]
// CHECK: llvm.func @triton_unk_reduce_vf_simt
// CHECK-SAME: %[[PARAM:arg[0-9]+]]: !llvm.ptr<6>
// CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]]
// CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]]
// CHECK: %{{.*}} = llvm.load %[[PARAM]] : !llvm.ptr<6> -> f32
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 111 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 128 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_unk_reduce(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>, %arg9: !llvm.ptr<6> {hivm.shared_memory}) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 1024>} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(4096 : i32) : i32
    %2 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32)>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(4095 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mlir.constant(5 : i32) : i32
    %8 = llvm.mlir.constant(1023 : i32) : i32
    %9 = llvm.mlir.constant(2 : i32) : i32
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.mlir.constant(16 : i32) : i32
    %12 = llvm.mlir.constant(31 : i32) : i32
    %13 = llvm.mlir.constant(8 : i32) : i32
    %14 = llvm.mlir.constant(4 : i32) : i32
    %15 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %16 = llvm.mlir.constant(true) : i1
    %17 = llvm.mlir.undef : vector<1xf32>
    %18 = llvm.mlir.undef : vector<1xi32>
    %19 = llvm.mlir.constant(-1 : i32) : i32
    %20 = llvm.mlir.constant(0 : index) : i32
    %21 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f32, f32, f32)> 
    %22 = llvm.insertvalue %0, %21[1] : !llvm.struct<(f32, f32, f32, f32)> 
    %23 = llvm.insertvalue %0, %22[2] : !llvm.struct<(f32, f32, f32, f32)> 
    %24 = llvm.insertvalue %0, %23[3] : !llvm.struct<(f32, f32, f32, f32)> 
    %25 = ascend_dpx.block_idx_x
    %26 = ascend_dpx.grid_dim_x
    %27 = ascend_dpx.thread_id_x
    %28 = llvm.urem %27, %6  : i32
    %29 = llvm.udiv %27, %6  : i32
    %30 = llvm.shl %28, %3 : i32
    %31 = llvm.or %3, %30  : i32
    %32 = llvm.shl %29, %7 : i32
    %33 = llvm.or %31, %32  : i32
    %34 = llvm.and %33, %8  : i32
    %35 = llvm.shl %34, %9 : i32
    %36 = llvm.or %35, %3  : i32
    %37 = llvm.xor %3, %36  : i32
    %38 = llvm.xor %37, %3  : i32
    %39 = llvm.xor %37, %5  : i32
    %40 = llvm.xor %37, %9  : i32
    %41 = llvm.xor %37, %10  : i32
    %42 = llvm.add %38, %20 : i32
    %43 = llvm.add %39, %20 : i32
    %44 = llvm.add %40, %20 : i32
    %45 = llvm.add %41, %20 : i32
    %46 = llvm.add %arg3, %4 : i32
    %47 = ascend_dpx.div %46, %1 : (i32, i32) -> i32
    llvm.br ^bb1(%25 : i32)
  ^bb1(%48: i32):  // 2 preds: ^bb0, ^bb5
    %49 = llvm.icmp "slt" %48, %arg2 : i32
    llvm.cond_br %49, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %50 = llvm.getelementptr %arg1[%48] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %51 = llvm.mul %arg3, %48 : i32
    %52 = llvm.getelementptr %arg0[%51] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.br ^bb3(%3, %24 : i32, !llvm.struct<(f32, f32, f32, f32)>)
  ^bb3(%53: i32, %54: !llvm.struct<(f32, f32, f32, f32)>):  // 2 preds: ^bb2, ^bb4
    %55 = llvm.icmp "slt" %53, %47 : i32
    llvm.cond_br %55, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %56 = llvm.mul %53, %1 : i32
    %57 = llvm.add %56, %42 : i32
    %58 = llvm.add %56, %43 : i32
    %59 = llvm.add %56, %44 : i32
    %60 = llvm.add %56, %45 : i32
    %61 = llvm.getelementptr %52[%57] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %62 = llvm.getelementptr %52[%58] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %63 = llvm.getelementptr %52[%59] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %64 = llvm.getelementptr %52[%60] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %65 = llvm.icmp "slt" %57, %arg3 : i32
    %66 = llvm.bitcast %0 : f32 to i32
    %67 = ascend_dpx.load %61, %65, %66 : (!llvm.ptr<1>, i1, i32) -> i32
    %68 = llvm.bitcast %67 : i32 to f32
    %69 = ascend_dpx.load %62, %65, %66 : (!llvm.ptr<1>, i1, i32) -> i32
    %70 = llvm.bitcast %69 : i32 to f32
    %71 = ascend_dpx.load %63, %65, %66 : (!llvm.ptr<1>, i1, i32) -> i32
    %72 = llvm.bitcast %71 : i32 to f32
    %73 = ascend_dpx.load %64, %65, %66 : (!llvm.ptr<1>, i1, i32) -> i32
    %74 = llvm.bitcast %73 : i32 to f32
    %75 = llvm.extractvalue %54[0] : !llvm.struct<(f32, f32, f32, f32)> 
    %76 = llvm.extractvalue %54[1] : !llvm.struct<(f32, f32, f32, f32)> 
    %77 = llvm.extractvalue %54[2] : !llvm.struct<(f32, f32, f32, f32)> 
    %78 = llvm.extractvalue %54[3] : !llvm.struct<(f32, f32, f32, f32)> 
    %79 = llvm.fadd %75, %68  : f32
    %80 = llvm.fadd %76, %70  : f32
    %81 = llvm.fadd %77, %72  : f32
    %82 = llvm.fadd %78, %74  : f32
    %83 = llvm.insertvalue %79, %2[0] : !llvm.struct<(f32, f32, f32, f32)> 
    %84 = llvm.insertvalue %80, %83[1] : !llvm.struct<(f32, f32, f32, f32)> 
    %85 = llvm.insertvalue %81, %84[2] : !llvm.struct<(f32, f32, f32, f32)> 
    %86 = llvm.insertvalue %82, %85[3] : !llvm.struct<(f32, f32, f32, f32)> 
    %87 = llvm.add %53, %5 : i32
    llvm.br ^bb3(%87, %86 : i32, !llvm.struct<(f32, f32, f32, f32)>)
  ^bb5:  // pred: ^bb3
    ascend_dpx.sync_threads
    %88 = llvm.extractvalue %54[0] : !llvm.struct<(f32, f32, f32, f32)> 
    %89 = llvm.extractvalue %54[1] : !llvm.struct<(f32, f32, f32, f32)> 
    %90 = llvm.extractvalue %54[2] : !llvm.struct<(f32, f32, f32, f32)> 
    %91 = llvm.extractvalue %54[3] : !llvm.struct<(f32, f32, f32, f32)> 
    %92 = llvm.fadd %88, %89  : f32
    %93 = llvm.fadd %92, %90  : f32
    %94 = llvm.fadd %93, %91  : f32
    %95 = llvm.bitcast %94 : f32 to i32
    %96 = ascend_dpx.shfl.bfly %95, %3, %12, %11 : (i32, i32, i32, i32) -> i32
    %97 = llvm.bitcast %96 : i32 to f32
    %98 = llvm.fadd %94, %97  : f32
    %99 = llvm.bitcast %98 : f32 to i32
    %100 = ascend_dpx.shfl.bfly %99, %3, %12, %13 : (i32, i32, i32, i32) -> i32
    %101 = llvm.bitcast %100 : i32 to f32
    %102 = llvm.fadd %98, %101  : f32
    %103 = llvm.bitcast %102 : f32 to i32
    %104 = ascend_dpx.shfl.bfly %103, %3, %12, %14 : (i32, i32, i32, i32) -> i32
    %105 = llvm.bitcast %104 : i32 to f32
    %106 = llvm.fadd %102, %105  : f32
    %107 = llvm.bitcast %106 : f32 to i32
    %108 = ascend_dpx.shfl.bfly %107, %3, %12, %9 : (i32, i32, i32, i32) -> i32
    %109 = llvm.bitcast %108 : i32 to f32
    %110 = llvm.fadd %106, %109  : f32
    %111 = llvm.bitcast %110 : f32 to i32
    %112 = ascend_dpx.shfl.bfly %111, %3, %12, %5 : (i32, i32, i32, i32) -> i32
    %113 = llvm.bitcast %112 : i32 to f32
    %114 = llvm.fadd %110, %113  : f32
    %115 = llvm.urem %28, %6  : i32
    %116 = llvm.urem %29, %6  : i32
    %117 = llvm.icmp "eq" %115, %3 : i32
    %118 = llvm.and %16, %16  : i1
    %119 = llvm.and %118, %117  : i1
    %120 = llvm.getelementptr %15[%116] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %121 = llvm.insertelement %114, %17[%3 : i32] : vector<1xf32>
    %122 = llvm.extractelement %121[%3 : i32] : vector<1xf32>
    %123 = llvm.bitcast %122 : f32 to i32
    %124 = llvm.insertelement %123, %18[%3 : i32] : vector<1xi32>
    ascend_dpx.store %120, %124, %119 : <3>, vector<1xi32>
    ascend_dpx.sync_threads
    %125 = llvm.icmp "slt" %27, %6 : i32
    %126 = llvm.getelementptr %15[%27] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %127 = ascend_dpx.load %126, %125, %3 : (!llvm.ptr<3>, i1, i32) -> i32
    %128 = llvm.bitcast %127 : i32 to f32
    %129 = llvm.insertelement %128, %17[%3 : i32] : vector<1xf32>
    %130 = llvm.extractelement %129[%3 : i32] : vector<1xf32>
    %131 = llvm.bitcast %130 : f32 to i32
    %132 = ascend_dpx.shfl.bfly %131, %3, %12, %11 : (i32, i32, i32, i32) -> i32
    %133 = llvm.bitcast %132 : i32 to f32
    %134 = llvm.fadd %130, %133  : f32
    %135 = llvm.bitcast %134 : f32 to i32
    %136 = ascend_dpx.shfl.bfly %135, %3, %12, %13 : (i32, i32, i32, i32) -> i32
    %137 = llvm.bitcast %136 : i32 to f32
    %138 = llvm.fadd %134, %137  : f32
    %139 = llvm.bitcast %138 : f32 to i32
    %140 = ascend_dpx.shfl.bfly %139, %3, %12, %14 : (i32, i32, i32, i32) -> i32
    %141 = llvm.bitcast %140 : i32 to f32
    %142 = llvm.fadd %138, %141  : f32
    %143 = llvm.bitcast %142 : f32 to i32
    %144 = ascend_dpx.shfl.bfly %143, %3, %12, %9 : (i32, i32, i32, i32) -> i32
    %145 = llvm.bitcast %144 : i32 to f32
    %146 = llvm.fadd %142, %145  : f32
    %147 = llvm.bitcast %146 : f32 to i32
    %148 = ascend_dpx.shfl.bfly %147, %3, %12, %5 : (i32, i32, i32, i32) -> i32
    %149 = llvm.bitcast %148 : i32 to f32
    %150 = llvm.fadd %146, %149  : f32
    %151 = llvm.and %125, %117  : i1
    %152 = llvm.insertelement %150, %17[%3 : i32] : vector<1xf32>
    %153 = llvm.extractelement %152[%3 : i32] : vector<1xf32>
    %154 = llvm.bitcast %153 : f32 to i32
    %155 = llvm.insertelement %154, %18[%3 : i32] : vector<1xi32>
    ascend_dpx.store %126, %155, %151 : <3>, vector<1xi32>
    ascend_dpx.sync_threads
    %156 = llvm.load %15 : !llvm.ptr<3> -> f32
    %157 = llvm.and %28, %19  : i32
    %158 = llvm.icmp "eq" %157, %3 : i32
    %159 = llvm.and %29, %19  : i32
    %160 = llvm.icmp "eq" %159, %3 : i32
    %161 = llvm.and %158, %160  : i1
    %162 = llvm.and %3, %19  : i32
    %163 = llvm.icmp "eq" %162, %3 : i32
    %164 = llvm.and %161, %163  : i1
    %165 = llvm.bitcast %156 : f32 to i32
    %166 = llvm.insertelement %165, %18[%3 : i32] : vector<1xi32>
    ascend_dpx.store %50, %166, %164 : <1>, vector<1xi32>
    %167 = llvm.add %48, %26 : i32
    llvm.br ^bb1(%167 : i32)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}
