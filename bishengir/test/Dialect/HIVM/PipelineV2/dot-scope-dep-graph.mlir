// REQUIRES: asserts
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: cd %t.dir && bishengir-opt -merge-cv-scopes -debug-only=merge-cv-scopes -split-input-file %s -o /dev/null 2> %t.dir/stderr.log
// RUN: ls -1 %t.dir/*.dot | FileCheck %s --check-prefix=FILES
// RUN: FileCheck %s --input-file=%t.dir/merged.2-lcd-grouping.nested_mixed_cores.dot --check-prefix=MIXED-DOT
// RUN: FileCheck %s --input-file=%t.dir/merged.2-lcd-grouping.multi_rule_progression.dot --check-prefix=NESTED-DOT
// RUN: FileCheck %s --input-file=%t.dir/merged.3-merged-scope-same-core.multi_rule_progression.dot --check-prefix=COLLAPSED-DOT

// Verify that DOT files are emitted ONLY for rules that actually mutate IR, in
// alphabetical/execution order:
// FILES-NOT: unchanged_scopes
// FILES:     merged.2-lcd-grouping.multi_rule_progression.dot
// FILES-NEXT: merged.2-lcd-grouping.nested_mixed_cores.dot
// FILES-NEXT: merged.3-merged-scope-same-core.multi_rule_progression.dot
// FILES-NEXT: merged.5-graph-same-core.multi_rule_progression.dot
// FILES-NEXT: merged.6-sink-alloca.sink_alloca_only.dot
// FILES-NOT: .dot

// Verify nested scope cluster and SSA edge names in @nested_mixed_cores:
// MIXED-DOT-LABEL: digraph "Scope Dependence Graph for 'nested_mixed_cores'"
// MIXED-DOT-NOT: res#
// MIXED-DOT-NOT: arg#
// MIXED-DOT: [[N0:Node0x[0-9a-f]+]] [shape=record,{{.*}}label="{Scope 0 (VECTOR) -\> %2:\l
// MIXED-DOT: [[N0]] -> [[N1:Node0x[0-9a-f]+]][tooltip="Scope 0 -> Scope 1" label="%2"];
// MIXED-DOT: [[N1]] [shape=record,{{.*}}label="{Scope 1 (CUBE) -\> %3:\l
// MIXED-DOT: [[N1]] -> [[N2:Node0x[0-9a-f]+]][tooltip="Scope 1 -> Scope 2" label="%3"];
// MIXED-DOT: [[N2]] [shape=record,{{.*}}label="{Scope 2 (VECTOR) -\> %4:\l
// MIXED-DOT: [[N2]] -> [[N0]][tooltip="Scope 2 -> Scope 0" label="LCD(%1-\>%arg2)" style=dashed color=red penwidth=2 constraint=false];
// MIXED-DOT: subgraph cluster_scope_0 {
// MIXED-DOT:   label = "Outer Scope 0 (cv_merged_scope)\n%1 = %4";
// MIXED-DOT:   [[N0]];
// MIXED-DOT:   [[N1]];
// MIXED-DOT:   [[N2]];
// MIXED-DOT: }

// Verify cross-cluster multi-result SSA edges (%1#0, LCD(%1#1->%arg2)) and
// external consumer placement outside cluster_scope_0:
// NESTED-DOT-LABEL: digraph "Scope Dependence Graph for 'multi_rule_progression'"
// NESTED-DOT-NOT: res#
// NESTED-DOT-NOT: arg#
// NESTED-DOT: [[M0:Node0x[0-9a-f]+]] [shape=record,{{.*}}label="{Scope 0 (VECTOR) -\> %2:\l
// NESTED-DOT: [[M0]] -> [[M1:Node0x[0-9a-f]+]][tooltip="Scope 0 -> Scope 1" label="%2"];
// NESTED-DOT: [[M0]] -> [[M2:Node0x[0-9a-f]+]][tooltip="Scope 0 -> Scope 2" label="%1#0"];
// NESTED-DOT: [[M1]] [shape=record,{{.*}}label="{Scope 1 (VECTOR) -\> %3:\l
// NESTED-DOT: [[M1]] -> [[M0]][tooltip="Scope 1 -> Scope 0" label="LCD(%1#1-\>%arg2)" style=dashed color=red penwidth=2 constraint=false];
// NESTED-DOT: [[M2]] [shape=record,{{.*}}label="{Scope 2 (VECTOR):\l
// NESTED-DOT: subgraph cluster_scope_0 {
// NESTED-DOT:   label = "Outer Scope 0 (cv_merged_scope)\n%1#0 = %2, %1#1 = %3";
// NESTED-DOT:   [[M0]];
// NESTED-DOT:   [[M1]];
// NESTED-DOT-NOT: [[M2]];
// NESTED-DOT: }

// Verify DOT after Rule 3 collapses {cv_merged_scope} into a multi-result leaf:
// COLLAPSED-DOT-LABEL: digraph "Scope Dependence Graph for 'multi_rule_progression'"
// COLLAPSED-DOT-NOT: res#
// COLLAPSED-DOT-NOT: arg#
// COLLAPSED-DOT-NOT: subgraph cluster_scope_
// COLLAPSED-DOT: [[C0:Node0x[0-9a-f]+]] [shape=record,{{.*}}label="{Scope 0 (VECTOR) -\> %1#0, %1#1:\l
// COLLAPSED-DOT-DAG: [[C0]] -> [[C1:Node0x[0-9a-f]+]][tooltip="Scope 0 -> Scope 1" label="%1#0"];
// COLLAPSED-DOT-DAG: [[C0]] -> [[C0]][tooltip="Scope 0 -> Scope 0" label="LCD(%1#1-\>%arg2)" style=dashed color=red penwidth=2 constraint=false];
// COLLAPSED-DOT: [[C1]] [shape=record,{{.*}}label="{Scope 1 (VECTOR):\l

func.func @unchanged_scopes(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %0 = scope.scope : () -> f32 {
      %x = arith.addf %a, %a : f32
      scope.return %x : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      %y = arith.mulf %0, %0 : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

func.func @nested_mixed_cores(%init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %init) -> (f32) {
    %0 = scope.scope : () -> f32 {
      %a = arith.addf %acc, %acc : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %1 = scope.scope : () -> f32 {
      %b = arith.mulf %0, %0 : f32
      scope.return %b : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %2 = scope.scope : () -> f32 {
      %c = arith.subf %1, %1 : f32
      scope.return %c : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %2 : f32
  } {hivm.cv_pipelined_loop}
  return %r : f32
}

// -----

func.func @multi_rule_progression(%init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %init) -> (f32) {
    %0 = scope.scope : () -> f32 {
      %a = arith.addf %acc, %acc : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %n = arith.negf %0 : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %2 = scope.scope : () -> f32 {
      %b = arith.mulf %0, %0 : f32
      scope.return %b : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %2 : f32
  } {hivm.cv_pipelined_loop}
  return %r : f32
}

// -----

func.func @sink_alloca_only(%val: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %buf = scope.scope : () -> memref<i32> {
      %alloca = memref.alloca() : memref<i32>
      scope.return %alloca : memref<i32>
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
    scope.scope : () -> () {
      memref.store %val, %buf[] : memref<i32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  } {hivm.cv_pipelined_loop}
  return
}
