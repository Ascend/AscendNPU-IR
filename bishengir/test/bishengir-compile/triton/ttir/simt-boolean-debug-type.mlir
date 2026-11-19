// RUN: mkdir -p %t.saved
// RUN: cp %s %t.ttir
// RUN: bishengir-compile %t.ttir --target=Ascend950PR_9589 --enable-triton-ir-compile --pure-simt --num-warps=4 --threads-per-warp=32 --enable-debug-info=true --enable-debug-variables=true --save-temps=%t.saved -o %t.saved/kernel.o
// RUN: FileCheck %s --input-file=%t.saved/kernel.ll

// CCEC emits the storage size in bytes. Keep the SSA predicate as i1.
// CHECK-LABEL: define {{.*}} @boolean_debug_type_vf_simt(
// CHECK-SAME: ptr addrspace(1) {{%[0-9]+}}, i1 [[Z:%[0-9]+]], i32
// CHECK: @llvm.dbg.value(metadata i1 [[Z]], metadata [[ARG:![0-9]+]], metadata !DIExpression())
// CHECK: [[CMP:%[0-9]+]] = icmp slt i32 {{%[0-9]+}}, 0
// CHECK: @llvm.dbg.value(metadata i1 [[CMP]], metadata [[LOCAL:![0-9]+]], metadata !DIExpression())
// CHECK-DAG: [[ARG]] = !DILocalVariable(name: "z", arg: 2, {{.*}}type: [[TYPE:![0-9]+]]
// CHECK-DAG: [[LOCAL]] = !DILocalVariable(name: "negative", {{.*}}type: [[TYPE]]
// CHECK-DAG: [[TYPE]] = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)

module {
  tt.func public @boolean_debug_type(%out: !tt.ptr<i32>, %z: i1 loc("z"("boolean.py":11:0)), %input: i32) {
    %zero = arith.constant 0 : i32
    %negative = arith.cmpi slt, %input, %zero : i32 loc("negative"("boolean.py":12:9))
    %masked = arith.ori %z, %negative : i1 loc("boolean.py":13:8)
    %result = arith.extui %masked : i1 to i32 loc("boolean.py":13:13)
    tt.store %out, %result : !tt.ptr<i32> loc("boolean.py":14:8)
    tt.return
  } loc("boolean.py":11:0)
}
