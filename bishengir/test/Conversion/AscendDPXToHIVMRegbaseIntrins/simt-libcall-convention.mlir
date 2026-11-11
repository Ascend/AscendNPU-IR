// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins | FileCheck %s --check-prefix=MLIR
// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins | bishengir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// Check both creation and reuse of the declaration: the calling convention
// must survive LLVM export on every call, not just on the callee declaration.
// MLIR: llvm.func @_mlir_ciface_simt_umulhi_uint32_t
// MLIR-SAME: hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_callable
// LLVM: declare {{.*}}simt_callable i32 @_mlir_ciface_simt_umulhi_uint32_t(i32, i32)

llvm.func @two_umulhi_calls(%lhs: i32, %rhs: i32) -> i32 {
  // MLIR-LABEL: llvm.func @two_umulhi_calls
  // MLIR: llvm.call @_mlir_ciface_simt_umulhi_uint32_t
  // MLIR-SAME: hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_callable
  // MLIR: llvm.call @_mlir_ciface_simt_umulhi_uint32_t
  // MLIR-SAME: hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_callable
  // LLVM-LABEL: define i32 @two_umulhi_calls(
  // LLVM: %[[FIRST:.*]] = call simt_callable i32 @_mlir_ciface_simt_umulhi_uint32_t(i32 %0, i32 %1)
  // LLVM: %[[SECOND:.*]] = call simt_callable i32 @_mlir_ciface_simt_umulhi_uint32_t(i32 %[[FIRST]], i32 %1)
  // LLVM: ret i32 %[[SECOND]]
  %first = ascend_dpx.umulhi %lhs, %rhs : (i32, i32) -> i32
  %second = ascend_dpx.umulhi %first, %rhs : (i32, i32) -> i32
  llvm.return %second : i32
}

llvm.func @ordinary(i32) -> i32

llvm.func @ordinary_call(%arg: i32) -> i32 {
  // LLVM-LABEL: define i32 @ordinary_call(
  // LLVM: %[[RESULT:.*]] = call i32 @ordinary(i32 %0)
  // LLVM: ret i32 %[[RESULT]]
  %result = llvm.call @ordinary(%arg) : (i32) -> i32
  llvm.return %result : i32
}
