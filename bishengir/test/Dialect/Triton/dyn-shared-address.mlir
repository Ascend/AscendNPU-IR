// RUN: bishengir-opt --triton-remap -split-input-file %s | FileCheck %s

// CHECK: %[[ARGUMENT:arg[0-9]+]]: !llvm.ptr<6> {hivm.shared_memory}
// CHECK: hivm_regbaseintrins.intrins.launch_func @triton_unk_reduce_vf_simt
// CHECK-SAME: %[[ARGUMENT]]
// CHECK: llvm.func @triton_unk_reduce_vf_simt
// CHECK-SAME: %[[PARAM:arg[0-9]+]]: !llvm.ptr<6>
// CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]]
// CHECK: %{{.*}} = llvm.getelementptr %[[PARAM]]
// CHECK: %{{.*}} = llvm.load %[[PARAM]] : !llvm.ptr<6> -> f32
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 900001 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 128 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
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
    %10 = llvm.mlir.undef : vector<1xf32>
    %11 = llvm.mlir.constant(16 : i32) : i32
    %12 = llvm.mlir.constant(31 : i32) : i32
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(8 : i32) : i32
    %15 = llvm.mlir.constant(4 : i32) : i32
    %16 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %17 = llvm.mlir.constant(true) : i1
    %18 = llvm.mlir.undef : vector<1xi32>
    %19 = llvm.mlir.constant(0 : index) : i32
    %20 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f32, f32, f32)> 
    %21 = llvm.insertvalue %0, %20[1] : !llvm.struct<(f32, f32, f32, f32)> 
    %22 = llvm.insertvalue %0, %21[2] : !llvm.struct<(f32, f32, f32, f32)> 
    %23 = llvm.insertvalue %0, %22[3] : !llvm.struct<(f32, f32, f32, f32)> 
    %24 = nvvm.read.ptx.sreg.ctaid.x : i32
    %25 = nvvm.read.ptx.sreg.nctaid.x : i32
    %26 = nvvm.read.ptx.sreg.tid.x : i32
    %27 = llvm.urem %26, %6  : i32
    %28 = llvm.udiv %26, %6  : i32
    %29 = llvm.shl %27, %3 : i32
    %30 = llvm.or %3, %29  : i32
    %31 = llvm.shl %28, %7 : i32
    %32 = llvm.or %30, %31  : i32
    %33 = llvm.or %32, %3  : i32
    %34 = llvm.and %33, %8  : i32
    %35 = llvm.shl %34, %9 : i32
    %36 = llvm.or %35, %3  : i32
    %37 = llvm.xor %3, %36  : i32
    %38 = llvm.xor %37, %3  : i32
    %39 = llvm.add %38, %19 : i32
    %40 = llvm.add %arg3, %4 : i32
    %41 = llvm.sdiv %40, %1  : i32
    llvm.br ^bb1(%24 : i32)
  ^bb1(%42: i32):  // 2 preds: ^bb0, ^bb5
    %43 = llvm.icmp "slt" %42, %arg2 : i32
    llvm.cond_br %43, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %44 = llvm.getelementptr %arg1[%42] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %45 = llvm.mul %arg3, %42 : i32
    %46 = llvm.getelementptr %arg0[%45] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.br ^bb3(%3, %23 : i32, !llvm.struct<(f32, f32, f32, f32)>)
  ^bb3(%47: i32, %48: !llvm.struct<(f32, f32, f32, f32)>):  // 2 preds: ^bb2, ^bb4
    %49 = llvm.icmp "slt" %47, %41 : i32
    llvm.cond_br %49, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %50 = llvm.mul %47, %1 : i32
    %51 = llvm.add %50, %39 : i32
    %52 = llvm.getelementptr %46[%51] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %53 = llvm.icmp "slt" %51, %arg3 : i32
    %54 = llvm.insertelement %0, %10[%19 : i32] : vector<1xf32>
    %55 = llvm.bitcast %54 : vector<1xf32> to i32
    %56 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b" %55, %55, %55, %55, %52, %53 : (i32, i32, i32, i32, !llvm.ptr<1>, i1) -> !llvm.struct<(i32, i32, i32, i32)>
    %57 = llvm.extractvalue %56[0] : !llvm.struct<(i32, i32, i32, i32)> 
    %58 = llvm.bitcast %57 : i32 to vector<1xf32>
    %59 = llvm.extractvalue %56[1] : !llvm.struct<(i32, i32, i32, i32)> 
    %60 = llvm.bitcast %59 : i32 to vector<1xf32>
    %61 = llvm.extractvalue %56[2] : !llvm.struct<(i32, i32, i32, i32)> 
    %62 = llvm.bitcast %61 : i32 to vector<1xf32>
    %63 = llvm.extractvalue %56[3] : !llvm.struct<(i32, i32, i32, i32)> 
    %64 = llvm.bitcast %63 : i32 to vector<1xf32>
    %65 = llvm.extractelement %58[%19 : i32] : vector<1xf32>
    %66 = llvm.extractelement %60[%19 : i32] : vector<1xf32>
    %67 = llvm.extractelement %62[%19 : i32] : vector<1xf32>
    %68 = llvm.extractelement %64[%19 : i32] : vector<1xf32>
    %69 = llvm.extractvalue %48[0] : !llvm.struct<(f32, f32, f32, f32)> 
    %70 = llvm.extractvalue %48[1] : !llvm.struct<(f32, f32, f32, f32)> 
    %71 = llvm.extractvalue %48[2] : !llvm.struct<(f32, f32, f32, f32)> 
    %72 = llvm.extractvalue %48[3] : !llvm.struct<(f32, f32, f32, f32)> 
    %73 = llvm.fadd %69, %65  : f32
    %74 = llvm.fadd %70, %66  : f32
    %75 = llvm.fadd %71, %67  : f32
    %76 = llvm.fadd %72, %68  : f32
    %77 = llvm.insertvalue %73, %2[0] : !llvm.struct<(f32, f32, f32, f32)> 
    %78 = llvm.insertvalue %74, %77[1] : !llvm.struct<(f32, f32, f32, f32)> 
    %79 = llvm.insertvalue %75, %78[2] : !llvm.struct<(f32, f32, f32, f32)> 
    %80 = llvm.insertvalue %76, %79[3] : !llvm.struct<(f32, f32, f32, f32)> 
    %81 = llvm.add %47, %5 : i32
    llvm.br ^bb3(%81, %80 : i32, !llvm.struct<(f32, f32, f32, f32)>)
  ^bb5:  // pred: ^bb3
    nvvm.barrier0
    %82 = llvm.extractvalue %48[0] : !llvm.struct<(f32, f32, f32, f32)> 
    %83 = llvm.extractvalue %48[1] : !llvm.struct<(f32, f32, f32, f32)> 
    %84 = llvm.extractvalue %48[2] : !llvm.struct<(f32, f32, f32, f32)> 
    %85 = llvm.extractvalue %48[3] : !llvm.struct<(f32, f32, f32, f32)> 
    %86 = llvm.fadd %82, %83  : f32
    %87 = llvm.fadd %86, %84  : f32
    %88 = llvm.fadd %87, %85  : f32
    %89 = llvm.bitcast %88 : f32 to i32
    %90 = nvvm.shfl.sync  bfly %13, %89, %11, %12 : i32 -> i32
    %91 = llvm.bitcast %90 : i32 to f32
    %92 = llvm.fadd %88, %91  : f32
    %93 = llvm.bitcast %92 : f32 to i32
    %94 = nvvm.shfl.sync  bfly %13, %93, %14, %12 : i32 -> i32
    %95 = llvm.bitcast %94 : i32 to f32
    %96 = llvm.fadd %92, %95  : f32
    %97 = llvm.bitcast %96 : f32 to i32
    %98 = nvvm.shfl.sync  bfly %13, %97, %15, %12 : i32 -> i32
    %99 = llvm.bitcast %98 : i32 to f32
    %100 = llvm.fadd %96, %99  : f32
    %101 = llvm.bitcast %100 : f32 to i32
    %102 = nvvm.shfl.sync  bfly %13, %101, %9, %12 : i32 -> i32
    %103 = llvm.bitcast %102 : i32 to f32
    %104 = llvm.fadd %100, %103  : f32
    %105 = llvm.bitcast %104 : f32 to i32
    %106 = nvvm.shfl.sync  bfly %13, %105, %5, %12 : i32 -> i32
    %107 = llvm.bitcast %106 : i32 to f32
    %108 = llvm.fadd %104, %107  : f32
    %109 = llvm.urem %27, %6  : i32
    %110 = llvm.urem %28, %6  : i32
    %111 = llvm.icmp "eq" %109, %3 : i32
    %112 = llvm.and %17, %17  : i1
    %113 = llvm.and %112, %111  : i1
    %114 = llvm.getelementptr %16[%110] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %115 = llvm.insertelement %108, %10[%3 : i32] : vector<1xf32>
    %116 = llvm.extractelement %115[%3 : i32] : vector<1xf32>
    %117 = llvm.bitcast %116 : f32 to i32
    %118 = llvm.insertelement %117, %18[%3 : i32] : vector<1xi32>
    %119 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b" %114, %118, %113 : (!llvm.ptr<3>, vector<1xi32>, i1) -> !llvm.void
    nvvm.barrier0
    %120 = llvm.icmp "slt" %26, %6 : i32
    %121 = llvm.getelementptr %16[%26] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %122 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b" %121, %120 : (!llvm.ptr<3>, i1) -> i32
    %123 = llvm.insertelement %122, %18[%3 : i32] : vector<1xi32>
    %124 = llvm.extractelement %123[%3 : i32] : vector<1xi32>
    %125 = llvm.bitcast %124 : i32 to f32
    %126 = llvm.insertelement %125, %10[%3 : i32] : vector<1xf32>
    %127 = llvm.extractelement %126[%3 : i32] : vector<1xf32>
    %128 = llvm.bitcast %127 : f32 to i32
    %129 = nvvm.shfl.sync  bfly %13, %128, %11, %12 : i32 -> i32
    %130 = llvm.bitcast %129 : i32 to f32
    %131 = llvm.fadd %127, %130  : f32
    %132 = llvm.bitcast %131 : f32 to i32
    %133 = nvvm.shfl.sync  bfly %13, %132, %14, %12 : i32 -> i32
    %134 = llvm.bitcast %133 : i32 to f32
    %135 = llvm.fadd %131, %134  : f32
    %136 = llvm.bitcast %135 : f32 to i32
    %137 = nvvm.shfl.sync  bfly %13, %136, %15, %12 : i32 -> i32
    %138 = llvm.bitcast %137 : i32 to f32
    %139 = llvm.fadd %135, %138  : f32
    %140 = llvm.bitcast %139 : f32 to i32
    %141 = nvvm.shfl.sync  bfly %13, %140, %9, %12 : i32 -> i32
    %142 = llvm.bitcast %141 : i32 to f32
    %143 = llvm.fadd %139, %142  : f32
    %144 = llvm.bitcast %143 : f32 to i32
    %145 = nvvm.shfl.sync  bfly %13, %144, %5, %12 : i32 -> i32
    %146 = llvm.bitcast %145 : i32 to f32
    %147 = llvm.fadd %143, %146  : f32
    %148 = llvm.and %120, %111  : i1
    %149 = llvm.insertelement %147, %10[%3 : i32] : vector<1xf32>
    %150 = llvm.extractelement %149[%3 : i32] : vector<1xf32>
    %151 = llvm.bitcast %150 : f32 to i32
    %152 = llvm.insertelement %151, %18[%3 : i32] : vector<1xi32>
    %153 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b" %121, %152, %148 : (!llvm.ptr<3>, vector<1xi32>, i1) -> !llvm.void
    nvvm.barrier0
    %154 = llvm.load %16 : !llvm.ptr<3> -> f32
    %155 = llvm.and %27, %13  : i32
    %156 = llvm.icmp "eq" %155, %3 : i32
    %157 = llvm.and %28, %13  : i32
    %158 = llvm.icmp "eq" %157, %3 : i32
    %159 = llvm.and %156, %158  : i1
    %160 = llvm.and %3, %13  : i32
    %161 = llvm.icmp "eq" %160, %3 : i32
    %162 = llvm.and %159, %161  : i1
    %163 = llvm.insertelement %154, %10[%3 : i32] : vector<1xf32>
    %164 = llvm.bitcast %163 : vector<1xf32> to i32
    %165 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %164, %44, %162 : (i32, !llvm.ptr<1>, i1) -> !llvm.void
    %166 = llvm.add %42, %25 : i32
    llvm.br ^bb1(%166 : i32)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}
