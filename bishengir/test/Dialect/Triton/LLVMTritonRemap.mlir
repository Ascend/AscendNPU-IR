// RUN: bishengir-opt %s -triton-remap | FileCheck %s


// CHECK-LABEL: test_inline_asm_triton_3_2
// CHECK: "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
// CHECK:  llvm.intr.exp
// CHECK:  llvm.fdiv
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_inline_asm_triton_3_2(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg3: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(4 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(32 : i32) : i32
    %5 = llvm.mlir.constant(128 : i32) : i32
    %6 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(2 : i32) : i32
    %8 = llvm.mlir.constant(8 : i32) : i32
    %9 = llvm.mlir.constant(16 : i32) : i32
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.mlir.undef : vector<1xf32>
    %12 = llvm.mlir.constant(1.44269502 : f32) : f32
    %13 = llvm.mlir.constant(0 : index) : i32
    %14 = llvm.inline_asm asm_dialect = att operand_attrs = [] "mov.u32 $0, %ctaid.x;", "=r"  : () -> i32
    %15 = llvm.mul %14, %5 : i32
    %16 = nvvm.read.ptx.sreg.tid.x : i32
    %17 = llvm.urem %16, %4  : i32
    %18 = llvm.and %17, %1  : i32
    %19 = llvm.icmp "eq" %18, %3 : i32
    %20 = llvm.select %19, %3, %1 : i1, i32
    %21 = llvm.xor %3, %20  : i32
    %22 = llvm.and %17, %7  : i32
    %23 = llvm.icmp "eq" %22, %3 : i32
    %24 = llvm.select %23, %3, %7 : i1, i32
    %25 = llvm.xor %21, %24  : i32
    %26 = llvm.and %17, %2  : i32
    %27 = llvm.icmp "eq" %26, %3 : i32
    %28 = llvm.select %27, %3, %2 : i1, i32
    %29 = llvm.xor %25, %28  : i32
    %30 = llvm.and %17, %8  : i32
    %31 = llvm.icmp "eq" %30, %3 : i32
    %32 = llvm.select %31, %3, %8 : i1, i32
    %33 = llvm.xor %29, %32  : i32
    %34 = llvm.and %17, %9  : i32
    %35 = llvm.icmp "eq" %34, %3 : i32
    %36 = llvm.select %35, %3, %9 : i1, i32
    %37 = llvm.xor %33, %36  : i32
    %38 = llvm.xor %37, %3  : i32
    %39 = llvm.add %38, %13 : i32
    %40 = llvm.add %15, %5 : i32
    %41 = llvm.intr.smin(%40, %arg4)  : (i32, i32) -> i32
    llvm.br ^bb1(%3 : i32)
  ^bb1(%42: i32):  // 2 preds: ^bb0, ^bb5
    %43 = llvm.icmp "slt" %42, %2 : i32
    llvm.cond_br %43, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %44 = llvm.mul %42, %4 : i32
    %45 = llvm.add %15, %44 : i32
    %46 = llvm.add %45, %39 : i32
    %47 = llvm.icmp "slt" %46, %41 : i32
    %48 = llvm.getelementptr %arg0[%46] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %49 = llvm.getelementptr %arg2[%46] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %50 = llvm.mul %46, %7 : i32
    llvm.br ^bb3(%3 : i32)
  ^bb3(%51: i32):  // 2 preds: ^bb2, ^bb4
    %52 = llvm.icmp "slt" %51, %arg5 : i32
    llvm.cond_br %52, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %53 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %arg1, %10 : (!llvm.ptr<1>, i1) -> i32
    %54 = llvm.bitcast %53 : i32 to vector<1xf32>
    %55 = llvm.extractelement %54[%13 : i32] : vector<1xf32>
    %56 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %48, %47 : (!llvm.ptr<1>, i1) -> i32
    %57 = llvm.bitcast %56 : i32 to vector<1xf32>
    %58 = llvm.extractelement %57[%13 : i32] : vector<1xf32>
    %59 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %49, %47 : (!llvm.ptr<1>, i1) -> i32
    %60 = llvm.bitcast %59 : i32 to vector<1xf32>
    %61 = llvm.extractelement %60[%13 : i32] : vector<1xf32>
    %62 = llvm.icmp "slt" %51, %1 : i32
    %63 = llvm.and %62, %47  : i1
    %64 = llvm.insertelement %0, %11[%13 : i32] : vector<1xf32>
    %65 = llvm.bitcast %64 : vector<1xf32> to i32
    %66 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];\0A\09@!$4 mov.u32 $0, $3;", "=r,l,b,r,b" %48, %63, %65, %63 : (!llvm.ptr<1>, i1, i32, i1) -> i32
    %67 = llvm.bitcast %66 : i32 to vector<1xf32>
    %68 = llvm.extractelement %67[%13 : i32] : vector<1xf32>
    %69 = llvm.fadd %68, %55  : f32
    %70 = llvm.fsub %0, %69  : f32
    %71 = llvm.fmul %70, %12  : f32
    %72 = llvm.inline_asm asm_dialect = att operand_attrs = [] "ex2.approx.f32 $0, $1;", "=f,f" %71 : (f32) -> f32
    %73 = llvm.fadd %72, %6  : f32
    %74 = llvm.inline_asm asm_dialect = att operand_attrs = [] "div.full.f32 $0, $1, $2;", "=r,r,r" %6, %73 : (f32, f32) -> f32
    %75 = llvm.fadd %58, %55  : f32
    %76 = llvm.fsub %0, %75  : f32
    %77 = llvm.fmul %76, %12  : f32
    %78 = llvm.inline_asm asm_dialect = att operand_attrs = [] "ex2.approx.f32 $0, $1;", "=f,f" %77 : (f32) -> f32
    %79 = llvm.fadd %78, %6  : f32
    %80 = llvm.inline_asm asm_dialect = att operand_attrs = [] "div.full.f32 $0, $1, $2;", "=r,r,r" %6, %79 : (f32, f32) -> f32
    %81 = llvm.fadd %61, %55  : f32
    %82 = llvm.fsub %0, %81  : f32
    %83 = llvm.fmul %82, %12  : f32
    %84 = llvm.call_intrinsic "llvm.nvvm.ex2.approx.f"(%83) : (f32) -> f32
    %85 = llvm.fadd %84, %6  : f32
    %86 = llvm.call_intrinsic "llvm.nvvm.div.full"(%6, %85) : (f32, f32) -> f32
    %87 = llvm.fmul %80, %86  : f32
    %88 = llvm.select %62, %74, %87 : i1, f32
    %89 = llvm.add %51, %50 : i32
    %90 = llvm.getelementptr %arg3[%89] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %91 = llvm.and %52, %47  : i1
    %92 = llvm.udiv %16, %4  : i32
    %93 = llvm.udiv %92, %1  : i32
    %94 = llvm.urem %93, %2  : i32
    %95 = llvm.udiv %17, %4  : i32
    %96 = llvm.urem %95, %1  : i32
    %97 = llvm.mul %94, %1 : i32
    %98 = llvm.add %97, %96 : i32
    %99 = llvm.mul %98, %1 : i32
    %100 = llvm.icmp "slt" %99, %1 : i32
    %101 = llvm.and %10, %100  : i1
    %102 = llvm.insertelement %88, %11[%3 : i32] : vector<1xf32>
    %103 = llvm.bitcast %102 : vector<1xf32> to i32
    %104 = llvm.and %101, %91  : i1
    %105 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %103, %90, %104 : (i32, !llvm.ptr<1>, i1) -> !llvm.void
    %106 = llvm.add %51, %1 : i32
    llvm.br ^bb3(%106 : i32)
  ^bb5:  // pred: ^bb3
    %107 = llvm.add %42, %1 : i32
    llvm.br ^bb1(%107 : i32)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}

