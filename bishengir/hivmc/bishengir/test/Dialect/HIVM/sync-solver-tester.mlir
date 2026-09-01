// REQUIRES: sync-tester

// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-tester-mode=true sync-tester-options="10,0,5,5,1,1"}))" | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-tester-mode=true sync-tester-options="10,0,10,5,1,1"}))" | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-tester-mode=true sync-tester-options="10,0,20,5,1,1"}))" | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-tester-mode=true sync-tester-options="10,0,20,20,1,1"}))" | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-tester-mode=true sync-tester-options="10,0,40,5,1,1"}))" | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-tester-mode=true sync-tester-options="10,0,40,20,1,1"}))" | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-tester-mode=true sync-tester-options="10,0,50,5,1,1"}))" | FileCheck %s
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-tester-mode=true sync-tester-options="10,0,50,20,1,1"}))" | FileCheck %s

// CHECK: succeeded
// CHECK-NOT: failed
module {
  func.func @kernel() {
    return
  }
}
