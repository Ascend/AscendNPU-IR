// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew" -allow-unregistered-dialect %s | FileCheck %s

// Test that CVPipelining with preload mode correctly handles nested scf.for
// operations with iter_args when cloning operations into scope bodies.
//
// This test verifies the fix for the issue where IRMapping containing
// BlockArgument mappings from outer ForOps caused Region::cloneInto() to
// skip adding block arguments to cloned inner ForOps. The problem manifested
// as ForOps with 0 block arguments when they should have at least the
// induction variable plus any iter_args.
//
// The structure mirrors hstu_fwd.mlir before cv-pipelining:
// - Outer scf.for loop (no iter_args) - this will be pipelined
// - Inner scf.if with nested scf.for that has iter_args
// - CUBE and VECTOR operations that trigger scope creation
//
// Before the fix:
// When createNewLoopsForPreloadWithScopes() cloned operations into scope
// bodies, the globalIRMap contained mappings for the outer loop's induction
// variable. When an inner ForOp was cloned, Region::cloneInto() saw the
// BlockArgument mapping and skipped creating block arguments for the cloned
// ForOp, resulting in a malformed ForOp with 0 arguments.
//
// After the fix:
// All BlockArgument mappings are removed from scopeMap before cloning,
// ensuring cloned ForOps receive fresh block arguments with correct structure.

// CHECK-LABEL: func.func @test_preload_nested_for_with_if
// CHECK: scf.for
// CHECK: scf.if
// CHECK: scf.for {{%.*}} = {{%.*}} to {{%.*}} step {{%.*}} iter_args({{%.*}} = {{%.*}}, {{%.*}} = {{%.*}}) -> (i32, i32)
// CHECK: scope.scope
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scope.scope
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
func.func @test_preload_nested_for_with_if(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<?xi32>) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true, true]> : vector<2xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %a_mem = "some_op"() : () -> memref<256x64xf16>
  %a = bufferization.to_tensor %a_mem : memref<256x64xf16>
  %k_mem = "some_op"() : () -> memref<64x64xf16>

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %bound = "some_op"() : () -> i64
  %true = arith.constant true
  %cond = "some_op"() : () -> i1

  %init = tensor.empty() : tensor<256x64xf32>
  %cst_f32 = arith.constant 0.0 : f32
  %vbrc_init = hivm.hir.vbrc ins(%cst_f32 : f32) outs(%init : tensor<256x64xf32>) -> tensor<256x64xf32>

  // Outer loop - will be pipelined with preload mode
  // This loop's induction variable %arg21 will be in globalIRMap
  %final = scf.for %arg21 = %c0_i64 to %bound step %c1_i64 iter_args(%acc = %vbrc_init) -> tensor<256x64xf32> : i64 {
    %idx = arith.index_cast %arg21 : i64 to index

    // scf.if containing a nested ForOp with iter_args
    // This structure comes from hstu_fwd.mlir line 71-99
    %result:2 = scf.if %cond -> (i64, i64) {
      scf.yield %c0_i64, %c0_i64 : i64, i64
    } else {
      // Inner ForOp with iter_args - this is the critical test case
      // When this ForOp is cloned into a scope body with globalIRMap containing
      // a mapping for %arg21, Region::cloneInto() must NOT skip creating block
      // arguments for the cloned ForOp.
      %inner:2 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32
                 iter_args(%arg23 = %c0_i32, %arg24 = %c3_i32) -> (i32, i32) : i32 {
        %sum = arith.addi %arg23, %arg24 : i32
        %div = arith.divsi %sum, %c2_i32 : i32
        %div_idx = arith.index_cast %div : i32 to index

        %reinterpret = memref.reinterpret_cast %arg1 to offset: [%div_idx], sizes: [1], strides: [1]
          : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %load_val = memref.load %reinterpret[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %ext = arith.extsi %load_val : i32 to i64
        %cmp = arith.cmpi sle, %ext, %c0_i64 : i64

        %select1 = arith.select %cmp, %arg24, %div : i32
        %select2 = scf.if %cmp -> (i32) {
          %add = arith.addi %div, %c1_i32 : i32
          scf.yield %add : i32
        } else {
          scf.yield %arg23 : i32
        }
        scf.yield %select2, %select1 : i32, i32
      }
      %ext_inner = arith.extsi %inner#0 : i32 to i64
      scf.yield %ext_inner, %c0_i64 : i64, i64
    }

    // CUBE operations that trigger scope creation
    %k_alloc = memref.alloc() : memref<64x64xf16>
    hivm.hir.load ins(%k_mem : memref<64x64xf16>) outs(%k_alloc : memref<64x64xf16>)
    %k_tensor = bufferization.to_tensor %k_alloc : memref<64x64xf16>

    %dot_init = tensor.empty() : tensor<256x64xf32>
    %dot = hivm.hir.mmadL1 ins(%a, %k_tensor, %true, %c256, %c64, %c64
      : tensor<256x64xf16>, tensor<64x64xf16>, i1, index, index, index)
      outs(%dot_init : tensor<256x64xf32>) -> tensor<256x64xf32>

    %ws = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<256x64xf32>
    annotation.mark %ws {hivm.multi_buffer = 4 : i32} : memref<256x64xf32>
    %ws_tensor = bufferization.to_tensor %ws restrict writable : memref<256x64xf32>
    %fix = hivm.hir.fixpipe ins(%dot : tensor<256x64xf32>)
      outs(%ws_tensor : tensor<256x64xf32>) -> tensor<256x64xf32>

    // VECTOR operations
    %load_init = tensor.empty() : tensor<256x64xf32>
    %load = hivm.hir.load ins(%fix : tensor<256x64xf32>)
      outs(%load_init : tensor<256x64xf32>) -> tensor<256x64xf32>
    %exp_init = tensor.empty() : tensor<256x64xf32>
    %exp = hivm.hir.vexp ins(%load : tensor<256x64xf32>)
      outs(%exp_init : tensor<256x64xf32>) -> tensor<256x64xf32>

    %add = arith.addf %exp, %acc : tensor<256x64xf32>
    scf.yield %add : tensor<256x64xf32>
  }

  "some_consume"(%final) : (tensor<256x64xf32>) -> ()
  return
}
