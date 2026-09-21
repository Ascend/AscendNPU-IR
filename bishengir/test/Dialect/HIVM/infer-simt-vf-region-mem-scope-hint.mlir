// RUN: bishengir-opt %s --infer-simt-vf-memory-scope-hint --split-input-file | FileCheck %s

// A tensor region can expose a new memref view through to_tensor. The outer
// traceback stops at to_memref, so the local traversal must still trace the
// subview to its allocation rather than treating all incoming roots as anchors.
// CHECK-LABEL: func.func @simt_from_region_view(
// CHECK-SAME: %arg0: memref<8xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}
module {
  func.func @simt_from_region_view(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @view_caller(%cond: i1) {
    %allocation = memref.alloc() : memref<16xi64>
    %view = memref.subview %allocation[0] [8] [1] : memref<16xi64> to memref<8xi64, strided<[1]>>
    %from_view = bufferization.to_tensor %view restrict writable : memref<8xi64, strided<[1]>>
    %local = tensor.empty() : tensor<8xi64>
    %result = scf.if %cond -> tensor<8xi64> {
      scf.yield %from_view : tensor<8xi64>
    } else {
      scf.yield %local : tensor<8xi64>
    }
    %buffer = bufferization.to_memref %result : memref<8xi64>
    call @simt_from_region_view(%buffer) : (memref<8xi64>) -> ()
    return
  }
}

// -----

// Non-region alternatives remain supported when every source is proven UB.
// CHECK-LABEL: func.func @simt_from_known_select(
// CHECK-SAME: %arg0: memref<8xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}
module {
  func.func @simt_from_known_select(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @known_caller(%cond: i1) {
    %a = memref.alloc() : memref<8xi64>
    %b = memref.alloc() : memref<8xi64>
    %selected = arith.select %cond, %a, %b : memref<8xi64>
    call @simt_from_known_select(%selected) : (memref<8xi64>) -> ()
    return
  }
}

// -----

// Dynamic slice results stay in SIMD even when returned through an if/loop.
// Trace both branch values and the loop's initial/carried values, terminating
// the cycle through the loop block argument.
// CHECK-LABEL: func.func @simt_from_ub_region(
// CHECK-SAME: %arg0: memref<8xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}
module {
  func.func @simt_from_ub_region(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @ub_caller(%src: tensor<?xi64>, %size: index, %cond: i1, %count: index) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<8xi64>
    %result = scf.for %i = %c0 to %count step %c1 iter_args(%acc = %empty) -> tensor<8xi64> {
      %next = scf.if %cond -> tensor<8xi64> {
        %inserted = tensor.insert_slice %src into %acc[0] [%size] [1] : tensor<?xi64> into tensor<8xi64>
        scf.yield %inserted : tensor<8xi64>
      } else {
        scf.yield %acc : tensor<8xi64>
      }
      scf.yield %next : tensor<8xi64>
    }
    %buffer = bufferization.to_memref %result : memref<8xi64>
    call @simt_from_ub_region(%buffer) : (memref<8xi64>) -> ()
    return
  }
}

// -----

// A tensor-valued conditional is not necessarily backed by UB: inverse
// bufferization pairs can forward global buffers through either branch.
// CHECK-LABEL: func.func @simt_from_gm_region(
// CHECK-SAME: %arg0: memref<8xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<gm>}
module {
  func.func @simt_from_gm_region(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @gm_caller(%a: memref<8xi64>, %b: memref<8xi64>, %cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %ta = bufferization.to_tensor %a restrict writable : memref<8xi64>
    %tb = bufferization.to_tensor %b restrict writable : memref<8xi64>
    %result = scf.if %cond -> tensor<8xi64> {
      scf.yield %ta : tensor<8xi64>
    } else {
      scf.yield %tb : tensor<8xi64>
    }
    %buffer = bufferization.to_memref %result : memref<8xi64>
    call @simt_from_gm_region(%buffer) : (memref<8xi64>) -> ()
    return
  }
}

// -----

// Follow the selected result's yields, not every tensor produced by the region.
// An unrelated GM result must not conflict with the requested UB result.
// CHECK-LABEL: func.func @simt_from_second_result(
// CHECK-SAME: %arg0: memref<8xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}
module {
  func.func @simt_from_second_result(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @multiple_results_caller(%gm: memref<8xi64>, %cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %global = bufferization.to_tensor %gm restrict writable : memref<8xi64>
    %local = tensor.empty() : tensor<8xi64>
    %results:2 = scf.if %cond -> (tensor<8xi64>, tensor<8xi64>) {
      scf.yield %global, %local : tensor<8xi64>, tensor<8xi64>
    } else {
      scf.yield %global, %local : tensor<8xi64>, tensor<8xi64>
    }
    %buffer = bufferization.to_memref %results#1 : memref<8xi64>
    call @simt_from_second_result(%buffer) : (memref<8xi64>) -> ()
    return
  }
}
