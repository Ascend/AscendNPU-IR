// RUN: mkdir -p %t.saved
// RUN: cp %s %t.ttir
// RUN: bishengir-compile %t.ttir --target=Ascend950PR_9589 --enable-triton-ir-compile --pure-simt --num-warps=4 --threads-per-warp=32 --enable-debug-info=true --enable-debug-variables=false --save-temps=%t.saved -o %t.saved/kernel.o
// RUN: FileCheck %s --input-file=%t.saved/kernel.ll

// Both levels of a cross-file inline chain must retain their own files.
// CHECK-LABEL: define {{.*}} @inlined_files_vf_simt(
// CHECK: mul i32 {{.*}}, !dbg [[CALLEE:![0-9]+]]
// CHECK: add i32 {{.*}}, !dbg [[LOCAL:![0-9]+]]
// CHECK: store <1 x i32> {{.*}}, !dbg [[STORE:![0-9]+]]
// CHECK-DAG: [[ROOT:![0-9]+]] = distinct !DISubprogram(name: "inlined_files_vf_simt",
// CHECK-DAG: [[CALLEE]] = !DILocation(line: 302, column: 36, scope: [[CALLEE_SCOPE:![0-9]+]], inlinedAt: [[BRIDGE:![0-9]+]])
// CHECK-DAG: [[CALLEE_SCOPE]] = distinct !DILexicalBlockFile(scope: [[ROOT]], file: [[CALLEE_FILE:![0-9]+]], discriminator: 0)
// CHECK-DAG: [[CALLEE_FILE]] = !DIFile(filename: "standard.py", directory: "/lib/triton")
// CHECK-DAG: [[BRIDGE]] = !DILocation(line: 47, column: 12, scope: [[BRIDGE_SCOPE:![0-9]+]], inlinedAt: [[SITE:![0-9]+]])
// CHECK-DAG: [[BRIDGE_SCOPE]] = distinct !DILexicalBlockFile(scope: [[ROOT]], file: [[BRIDGE_FILE:![0-9]+]], discriminator: 0)
// CHECK-DAG: [[BRIDGE_FILE]] = !DIFile(filename: "bridge.py", directory: "/lib/triton")
// CHECK-DAG: [[SITE]] = !DILocation(line: 12, column: 9, scope: [[ROOT]])
// CHECK-DAG: [[LOCAL]] = !DILocation(line: 20, column: 7, scope: [[ROOT]], inlinedAt: [[LOCAL_SITE:![0-9]+]])
// CHECK-DAG: [[LOCAL_SITE]] = !DILocation(line: 13, column: 4, scope: [[ROOT]])
// CHECK-DAG: [[STORE]] = !DILocation(line: 303, column: 8, scope: [[CALLEE_SCOPE]], inlinedAt: [[STORE_SITE:![0-9]+]])
// CHECK-DAG: [[STORE_SITE]] = !DILocation(line: 14, column: 8, scope: [[ROOT]])

module {
  tt.func public @inlined_files(%out: !tt.ptr<i32>, %x: i32, %offset: i32) {
    %three = arith.constant 3 : i32
    %scaled = arith.muli %x, %three : i32 loc(callsite("/lib/triton/standard.py":302:36 at callsite("/lib/triton/bridge.py":47:12 at "/cases/kernel.py":12:9)))
    %shifted = arith.addi %scaled, %offset : i32 loc(callsite("/cases/kernel.py":20:7 at "/cases/kernel.py":13:4))
    tt.store %out, %shifted : !tt.ptr<i32> loc(callsite("/lib/triton/standard.py":303:8 at "/cases/kernel.py":14:8))
    tt.return
  } loc("/cases/kernel.py":10:0)
}
