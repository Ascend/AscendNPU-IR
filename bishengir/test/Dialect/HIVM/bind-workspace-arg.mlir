// RUN: bishengir-opt --hivm-bind-workspace-arg -split-input-file -verify-diagnostics %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @bind_workspace_arg
// CHECK-SAME: %[[ARG:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}
func.func @bind_workspace_arg(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}){
  // CHECK: memref_ext.alloc_workspace() from %[[ARG]]
  memref_ext.alloc_workspace() : memref<100xi32>
  return
}

// -----

func.func @bind_workspace_arg_error(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}){
  // expected-error@+1 {{failed to bind workspace argument}}
  memref_ext.alloc_workspace() : memref<100xi32>
  return
}