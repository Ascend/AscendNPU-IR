// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

// Test that memref.extract_aligned_pointer_as_index on a function argument is
// lowered to tt.ptr_to_int + arith.index_cast, and the bridge
// unrealized_conversion_cast folds away after Stage2 + reconcile (the source
// argument becomes a !tt.ptr, so the memref->!tt.ptr cast round-trips).

module attributes {hacc.simt_module, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: tt.func @extract_aligned_pointer_as_index
  // CHECK-SAME: %arg0: !tt.ptr<i32>
  // CHECK: %[[P2I:.*]] = tt.ptr_to_int %arg0 : !tt.ptr<i32> -> i64
  // CHECK: arith.index_cast %[[P2I]] : i64 to index
  // CHECK-NOT: memref.extract_aligned_pointer_as_index
  // CHECK-NOT: unrealized_conversion_cast
  func.func @extract_aligned_pointer_as_index(%arg0: memref<?xi32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %p = memref.extract_aligned_pointer_as_index %arg0 : memref<?xi32, #hivm.address_space<gm>> -> index
    return
  }
}

// -----

// The op is typically used to inspect whether an optional pointer argument is
// null. Lower the full idiom (extract -> index_cast to i64 -> cmpi) and check
// the bridge cast folds.

module attributes {hacc.simt_module, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: tt.func @extract_aligned_pointer_null_check
  // CHECK-SAME: %arg0: !tt.ptr<i32>
  // CHECK: %[[P2I:.*]] = tt.ptr_to_int %arg0 : !tt.ptr<i32> -> i64
  // CHECK: arith.cmpi
  // CHECK-NOT: memref.extract_aligned_pointer_as_index
  // CHECK-NOT: unrealized_conversion_cast
  func.func @extract_aligned_pointer_null_check(%arg0: memref<?xi32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %c0 = arith.constant 0 : i64
    %p = memref.extract_aligned_pointer_as_index %arg0 : memref<?xi32, #hivm.address_space<gm>> -> index
    %addr = arith.index_cast %p : index to i64
    %isnull = arith.cmpi eq, %addr, %c0 : i64
    return
  }
}
