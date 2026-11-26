// REQUIRES: hivmc, enable-lir-compile
// RUN: split-file %s %t
// RUN: hivmc %t/input.mlir --target=Ascend960DT_968B7 --enable-hivm-compile=false --link-aicore-bitcode=%t/support.ll --save-temps=%t -o %t/kernel
// RUN: FileCheck %s --check-prefix=A6 --implicit-check-not=@llvm.hivm.vintlvv2.v256s8. --input-file=%t/kernel.ll
// RUN: sed '/^  llvm.func @llvm.hivm.vintlvv2.v256s8(/s/i32) ->/i64) ->/' %t/input.mlir > %t/conflict.mlir
// RUN: not hivmc %t/conflict.mlir --target=Ascend960DT_968B7 --enable-hivm-compile=false --link-aicore-bitcode=%t/support.ll -o %t/conflict 2>&1 | FileCheck %s --check-prefix=CONFLICT

// CONFLICT: incompatible declaration of CANN intrinsic llvm.hivm.vintlvv2.v256s8

//--- support.ll
source_filename = "support"

//--- input.mlir
module attributes {hacc.target = #hacc.target<"Ascend960DT_968B7">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // A declaration from linked MLIR may already have the CANN spelling.
  llvm.func @llvm.hivm.vintlvv2.v256s8(vector<256xi8>, vector<256xi8>, i32) -> vector<256xi8>

  llvm.func @shuffle(%src: !llvm.ptr<6>, %dst: !llvm.ptr<6>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-920r1">, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %lo = llvm.mlir.constant(0 : i32) : i32
    %hi = llvm.mlir.constant(1 : i32) : i32
    %dist = llvm.mlir.constant(3 : i32) : i32
    %a = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%src, %lo, %lo, %lo) : (!llvm.ptr<6>, i32, i32, i32) -> vector<256xi8>
    // A6: call <256 x i8> @llvm.hivm.vintlvv2.v256s8({{.*}}i32 0)
    %b = "hivm_regbaseintrins.intr.hivm.vintlvv2"(%a, %a, %lo) : (vector<256xi8>, vector<256xi8>, i32) -> vector<256xi8>
    %h = llvm.bitcast %b : vector<256xi8> to vector<128xi16>
    // A6: call <128 x i16> @llvm.hivm.vdintlvv2.v128s16({{.*}}i32 1)
    %half = "hivm_regbaseintrins.intr.hivm.vdintlvv2"(%h, %h, %hi) : (vector<128xi16>, vector<128xi16>, i32) -> vector<128xi16>
    %i = llvm.bitcast %half : vector<128xi16> to vector<64xi32>
    // A6: call <64 x i32> @llvm.hivm.vdintlvv2.v64s32({{.*}}i32 0)
    // A6: call <64 x i32> @llvm.hivm.vdintlvv2.v64s32({{.*}}i32 1)
    %pair = "hivm_regbaseintrins.intr.hivm.vdintlv"(%i, %i) : (vector<64xi32>, vector<64xi32>) -> !llvm.struct<(vector<64xi32>, vector<64xi32>)>
    %w = llvm.extractvalue %pair[1] : !llvm.struct<(vector<64xi32>, vector<64xi32>)>
    %f = llvm.bitcast %w : vector<64xi32> to vector<64xf32>
    // A6: call <64 x float> @llvm.hivm.vintlvv2.v64f32({{.*}}i32 1)
    %res = "hivm_regbaseintrins.intr.hivm.vintlvv2"(%f, %f, %hi) : (vector<64xf32>, vector<64xf32>, i32) -> vector<64xf32>
    %out = llvm.bitcast %res : vector<64xf32> to vector<256xi8>
    %mask = "hivm_regbaseintrins.intr.hivm.pge.b8"(%lo, %lo) {mask_bit_width = 8 : i32} : (i32, i32) -> vector<256xi1>
    "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"(%out, %dst, %lo, %dist, %lo, %mask) : (vector<256xi8>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    llvm.return
  }

  llvm.func @kernel(%in: i64, %out: i64) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-920r1">, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %src = llvm.inttoptr %in : i64 to !llvm.ptr<6>
    %dst = llvm.inttoptr %out : i64 to !llvm.ptr<6>
    llvm.call @shuffle(%src, %dst) {hivm.vector_function} : (!llvm.ptr<6>, !llvm.ptr<6>) -> ()
    llvm.return
  }
}

// A6: ; Function Attrs: nounwind readnone
// A6-NEXT: declare <256 x i8> @llvm.hivm.vintlvv2.v256s8(
