// RUN: bishengir-opt %s -decompose-frem | FileCheck %s

module {
  // CHECK: %0 = llvm.call @_mlir_ciface_simt_divrn_float(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: %1 = llvm.bitcast %0 : f32 to i32
  // CHECK: %2 = llvm.mlir.constant(23 : i32) : i32
  // CHECK: %3 = llvm.mlir.constant(127 : i32) : i32
  // CHECK: %4 = llvm.mlir.constant(255 : i32) : i32
  // CHECK: %5 = llvm.lshr %1, %2  : i32
  // CHECK: %6 = llvm.and %5, %4  : i32
  // CHECK: %7 = llvm.sub %6, %3 : i32
  // CHECK: %8 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %9 = llvm.icmp "slt" %7, %8 : i32
  // CHECK: %10 = llvm.icmp "slt" %7, %2 : i32
  // CHECK: %11 = llvm.mlir.constant(-2147483648 : i32) : i32
  // CHECK: %12 = llvm.and %1, %11  : i32
  // CHECK: %13 = llvm.bitcast %12 : i32 to f32
  // CHECK: %14 = llvm.mlir.constant(8388607 : i32) : i32
  // CHECK: %15 = llvm.lshr %14, %7  : i32
  // CHECK: %16 = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %17 = llvm.xor %15, %16  : i32
  // CHECK: %18 = llvm.and %1, %17  : i32
  // CHECK: %19 = llvm.bitcast %18 : i32 to f32
  // CHECK: %20 = llvm.select %10, %19, %0 : i1, f32
  // CHECK: %21 = llvm.select %9, %13, %20 : i1, f32
  // CHECK: %22 = llvm.fmul %21, %arg1  : f32
  // CHECK: %23 = llvm.fsub %arg0, %22  : f32
  llvm.func @foo1(%arg0: f32, %arg1: f32) {
    %0 = llvm.frem %arg0, %arg1  : f32
    llvm.return
  }

  // CHECK: %0 = llvm.fdiv %arg0, %arg1  : f16
  // CHECK: %1 = llvm.fpext %0 : f16 to f32
  // CHECK: %2 = llvm.bitcast %1 : f32 to i32
  // CHECK: %3 = llvm.mlir.constant(23 : i32) : i32
  // CHECK: %4 = llvm.mlir.constant(127 : i32) : i32
  // CHECK: %5 = llvm.mlir.constant(255 : i32) : i32
  // CHECK: %6 = llvm.lshr %2, %3  : i32
  // CHECK: %7 = llvm.and %6, %5  : i32
  // CHECK: %8 = llvm.sub %7, %4 : i32
  // CHECK: %9 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %10 = llvm.icmp "slt" %8, %9 : i32
  // CHECK: %11 = llvm.icmp "slt" %8, %3 : i32
  // CHECK: %12 = llvm.mlir.constant(-2147483648 : i32) : i32
  // CHECK: %13 = llvm.and %2, %12  : i32
  // CHECK: %14 = llvm.bitcast %13 : i32 to f32
  // CHECK: %15 = llvm.mlir.constant(8388607 : i32) : i32
  // CHECK: %16 = llvm.lshr %15, %8  : i32
  // CHECK: %17 = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %18 = llvm.xor %16, %17  : i32
  // CHECK: %19 = llvm.and %2, %18  : i32
  // CHECK: %20 = llvm.bitcast %19 : i32 to f32
  // CHECK: %21 = llvm.select %11, %20, %1 : i1, f32
  // CHECK: %22 = llvm.select %10, %14, %21 : i1, f32
  // CHECK: %23 = llvm.fptrunc %22 : f32 to f16
  // CHECK: %24 = llvm.fmul %23, %arg1  : f16
  // CHECK: %25 = llvm.fsub %arg0, %24  : f16
  llvm.func @foo2(%arg0: f16, %arg1: f16) {
    %0 = llvm.frem %arg0, %arg1  : f16
    llvm.return
  }

  // CHECK: %0 = llvm.fdiv %arg0, %arg1  : bf16
  // CHECK: %1 = llvm.fpext %0 : bf16 to f32
  // CHECK: %2 = llvm.bitcast %1 : f32 to i32
  // CHECK: %3 = llvm.mlir.constant(23 : i32) : i32
  // CHECK: %4 = llvm.mlir.constant(127 : i32) : i32
  // CHECK: %5 = llvm.mlir.constant(255 : i32) : i32
  // CHECK: %6 = llvm.lshr %2, %3  : i32
  // CHECK: %7 = llvm.and %6, %5  : i32
  // CHECK: %8 = llvm.sub %7, %4 : i32
  // CHECK: %9 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %10 = llvm.icmp "slt" %8, %9 : i32
  // CHECK: %11 = llvm.icmp "slt" %8, %3 : i32
  // CHECK: %12 = llvm.mlir.constant(-2147483648 : i32) : i32
  // CHECK: %13 = llvm.and %2, %12  : i32
  // CHECK: %14 = llvm.bitcast %13 : i32 to f32
  // CHECK: %15 = llvm.mlir.constant(8388607 : i32) : i32
  // CHECK: %16 = llvm.lshr %15, %8  : i32
  // CHECK: %17 = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %18 = llvm.xor %16, %17  : i32
  // CHECK: %19 = llvm.and %2, %18  : i32
  // CHECK: %20 = llvm.bitcast %19 : i32 to f32
  // CHECK: %21 = llvm.select %11, %20, %1 : i1, f32
  // CHECK: %22 = llvm.select %10, %14, %21 : i1, f32
  // CHECK: %23 = llvm.fptrunc %22 : f32 to bf16
  // CHECK: %24 = llvm.fmul %23, %arg1  : bf16
  // CHECK: %25 = llvm.fsub %arg0, %24  : bf16
  llvm.func @foo3(%arg0: bf16, %arg1: bf16) {
    %0 = llvm.frem %arg0, %arg1  : bf16
    llvm.return
  }
}