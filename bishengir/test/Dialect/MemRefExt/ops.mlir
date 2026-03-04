// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect -mlir-print-op-generic %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s

// -----
module {
  func.func @alloc_workspace(%dynamic: index, %workspaceArg: memref<?xi8>){
    %offset = arith.constant 1 : index
    // CHECK: memref_ext.alloc_workspace({{.*}}) offset = [{{.*}}]
    memref_ext.alloc_workspace(%dynamic) offset = [%offset] : memref<?x100xi8, strided<[?, 1], offset: 1>>
    // CHECK: memref_ext.alloc_workspace()
    memref_ext.alloc_workspace() : memref<100xi8>
    // CHECK: memref_ext.alloc_workspace() from
    memref_ext.alloc_workspace() from %workspaceArg : from memref<?xi8> to memref<100xi32>
    return
  }
}