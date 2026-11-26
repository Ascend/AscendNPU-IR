// RUN: mkdir -p %t.debug %t.line-only
// RUN: cp %s %t.ttir
// RUN: bishengir-compile %t.ttir --target=Ascend950PR_9589 --enable-triton-ir-compile --pure-simt --num-warps=4 --threads-per-warp=32 --enable-debug-info=true --enable-debug-variables=true --enable-optimized-metaop=true --save-temps=%t.debug -o %t.debug/kernel.o
// RUN: FileCheck %s --implicit-check-not="call simt_callable i32 @_mlir_ciface_simt_umulhi_uint32_t" --input-file=%t.debug/kernel.ll
// RUN: bishengir-compile %t.ttir --target=Ascend950PR_9589 --enable-triton-ir-compile --pure-simt --num-warps=4 --threads-per-warp=32 --enable-debug-info=true --enable-debug-variables=false --enable-optimized-metaop=true --save-temps=%t.line-only -o %t.line-only/kernel.o
// RUN: FileCheck %s --check-prefix=LINE-ONLY --implicit-check-not=llvm.dbg.value --input-file=%t.line-only/kernel.ll

// Expand the generated unsigned multiply-high in the caller when emitting
// local-variable records, so the incoming arguments can keep their locations.
// CHECK-LABEL: define {{.*}} @argument_locations_vf_simt(
// CHECK-SAME: ptr addrspace(1) {{[^%]*}}[[OUT:%[0-9]+]], i32 {{[^%]*}}[[DIVIDEND:%[0-9]+]], i32 {{[^%]*}}[[DIVISOR:%[0-9]+]],
// CHECK: @llvm.dbg.value(metadata ptr addrspace(1) [[OUT]], metadata [[OUT_VAR:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT: @llvm.dbg.value(metadata i32 [[DIVIDEND]], metadata [[DIVIDEND_VAR:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT: @llvm.dbg.value(metadata i32 [[DIVISOR]], metadata [[DIVISOR_VAR:![0-9]+]], metadata !DIExpression())
// CHECK: [[LHS:%[0-9]+]] = zext i32 {{%[0-9]+}} to i64
// CHECK-NEXT: [[RHS:%[0-9]+]] = zext i32 {{%[0-9]+}} to i64
// CHECK-NEXT: [[PRODUCT:%[0-9]+]] = mul i64 [[LHS]], [[RHS]]
// CHECK-NEXT: [[HIGH:%[0-9]+]] = lshr i64 [[PRODUCT]], 32
// CHECK-NEXT: trunc i64 [[HIGH]] to i32
// CHECK-DAG: [[OUT_VAR]] = !DILocalVariable(name: "out", arg: 1,
// CHECK-DAG: [[DIVIDEND_VAR]] = !DILocalVariable(name: "dividend", arg: 2,
// CHECK-DAG: [[DIVISOR_VAR]] = !DILocalVariable(name: "divisor", arg: 3,
// LINE-ONLY: call simt_callable i32 @_mlir_ciface_simt_umulhi_uint32_t
// LINE-ONLY: !DILocation(line: 12, column: 20,

module {
  tt.func public @argument_locations(
      %out: !tt.ptr<i32> loc("out"("params.py":10:0)),
      %dividend: i32 loc("dividend"("params.py":10:0)),
      %divisor: i32 loc("divisor"("params.py":10:0))) {
    %quotient = arith.divsi %dividend, %divisor : i32 loc("params.py":12:20)
    %result = arith.addi %quotient, %divisor : i32 loc("params.py":13:14)
    tt.store %out, %result : !tt.ptr<i32> loc("params.py":14:8)
    tt.return
  } loc("params.py":10:0)
}
