// RUN: bishengir-opt %s -allow-unregistered-dialect -auto-blockify-parallel-loop -split-input-file -verify-diagnostics | FileCheck %s

// Regression: invariant GM-to-UB loads from the blockify body are hoisted,
// while an inner-loop load that depends on its induction variable stays put.
// CHECK-LABEL: func.func @blockify_invariant_load(
// CHECK:         %[[INV_SRC:.*]] = memref.reinterpret_cast %arg0 to offset: [{{.*}}], sizes: [512], strides: [1]
// CHECK:         %[[INV_DST:.*]] = memref.alloc() : memref<512xf16>
// CHECK:         hivm.hir.load ins(%[[INV_SRC]] : memref<512xf16, strided<[1]>>) outs(%[[INV_DST]] : memref<512xf16>)
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
// CHECK-NOT:       hivm.hir.load
// CHECK:           %[[INNER_SRC:.*]] = memref.reinterpret_cast %arg1 to offset: [%{{.*}}]
// CHECK:           hivm.hir.load ins(%[[INNER_SRC]] : memref<512xf16, strided<[1], offset: ?>>) outs({{.*}})
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @blockify_invariant_load(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: i32, %arg4: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.muli %arg3, %arg4 : i32
    annotation.mark %0 {logical_block_num} : i32
    %1 = hivm.hir.get_block_idx -> i64
    %block_i32 = arith.trunci %1 : i64 to i32
    scf.for %iv = %block_i32 to %arg3 step %c1_i32 : i32 {
      %source = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1]>>
      %destination = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%source : memref<512xf16, strided<[1]>>) outs(%destination : memref<512xf16>)
      %source_tensor = bufferization.to_tensor %destination restrict writable : memref<512xf16>
      %offset = arith.index_cast %iv : i32 to index
      %inner_source = memref.reinterpret_cast %arg1 to offset: [%offset], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1], offset: ?>>
      %inner_destination = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%inner_source : memref<512xf16, strided<[1], offset: ?>>) outs(%inner_destination : memref<512xf16>)
      %inner_tensor = bufferization.to_tensor %inner_destination restrict writable : memref<512xf16>
    }
    return
  }
}

// -----

// Regression: a load nested in an inner loop must be hoisted to the blockify
// loop body first, then hoisted again before the generated blockify loop.
// CHECK-LABEL: func.func @blockify_nested_invariant_load_fixpoint(
// CHECK:         %[[INV_SRC:.*]] = memref.reinterpret_cast %arg0 to offset: [{{.*}}], sizes: [512], strides: [1]
// CHECK:         %[[INV_DST:.*]] = memref.alloc() : memref<512xf16>
// CHECK:         hivm.hir.load ins(%[[INV_SRC]] : memref<512xf16, strided<[1]>>) outs(%[[INV_DST]] : memref<512xf16>)
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
// CHECK-NOT:       hivm.hir.load ins(%[[INV_SRC]]
// CHECK:           scf.for %arg6 = %c0_i32 to %arg3 step %c1_i32_0 : i32
// CHECK-NOT:         hivm.hir.load ins(%[[INV_SRC]]
// CHECK:             %[[DEP_SRC:.*]] = memref.reinterpret_cast %arg1 to offset: [%{{.*}}]
// CHECK:             hivm.hir.load ins(%[[DEP_SRC]] : memref<256xf32, strided<[1], offset: ?>>) outs({{.*}})
// CHECK:           }
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @blockify_nested_invariant_load_fixpoint(%arg0: memref<?xf16>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.muli %arg2, %arg3 : i32
    annotation.mark %0 {logical_block_num} : i32
    %1 = hivm.hir.get_block_idx -> i64
    %block_i32 = arith.trunci %1 : i64 to i32
    scf.for %iv = %block_i32 to %arg2 step %c1_i32 : i32 {
      scf.for %inner_iv = %c0_i32 to %arg3 step %c1_i32 : i32 {
        %source = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1]>>
        %destination = memref.alloc() : memref<512xf16>
        hivm.hir.load ins(%source : memref<512xf16, strided<[1]>>) outs(%destination : memref<512xf16>)
        %source_tensor = bufferization.to_tensor %destination restrict writable : memref<512xf16>
        %offset = arith.index_cast %inner_iv : i32 to index
        %inner_source = memref.reinterpret_cast %arg1 to offset: [%offset], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
        %inner_destination = memref.alloc() : memref<256xf32>
        hivm.hir.load ins(%inner_source : memref<256xf32, strided<[1], offset: ?>>) outs(%inner_destination : memref<256xf32>)
        %inner_tensor = bufferization.to_tensor %inner_destination restrict writable : memref<256xf32>
      }
    }
    return
  }
}

// -----

// Regression: a region-bearing producer may capture the enclosing loop IV
// without listing it among the producer's explicit operands.
// CHECK-LABEL: func.func @region_capture(
// CHECK-NOT:     hivm.hir.load
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}{{ *}}: i32 {
// CHECK:           scf.for %[[CAPTURE_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             scf.for %[[INNER_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK:               %[[NEXT:.*]] = arith.addi %[[CAPTURE_IV]], %[[INNER_IV]] : index
// CHECK:               scf.yield %[[NEXT]] : index
// CHECK:             }
// CHECK:             %[[CAPTURED_SRC:.*]] = memref.reinterpret_cast %arg0 to offset: [%{{.*}}]
// CHECK:             hivm.hir.load ins(%[[CAPTURED_SRC]] : memref<1xf32, strided<[1], offset: ?>>) outs({{.*}})
// CHECK:           }
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @region_capture(%arg0: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %offset = scf.for %j = %c0 to %c1 step %c1 iter_args(%acc = %c0) -> index {
        %next = arith.addi %iv, %j : index
        scf.yield %next : index
      }
      %view = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %dst = memref.alloc() : memref<1xf32>
      hivm.hir.load ins(%view : memref<1xf32, strided<[1], offset: ?>>) outs(%dst : memref<1xf32>)
      %tensor = bufferization.to_tensor %dst restrict writable : memref<1xf32>
    }
    return
  }
}

// -----

// Regression: a load depending on the current loop's block argument must not be
// hoisted, even when its other producers are speculatable.
// CHECK-LABEL: func.func @blockify_block_argument_dependent_load(
// CHECK:         scf.for %[[BLOCK_LOOP_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}{{ *}}: i32 {
// CHECK:           scf.for %[[ORIG_IV:.*]] = %{{.*}} to %arg1 step %{{.*}} : i32 {
// CHECK:             %[[OFFSET:.*]] = arith.index_cast %[[ORIG_IV]] : i32 to index
// CHECK:             %[[SRC:.*]] = memref.reinterpret_cast %arg0 to offset: [%[[OFFSET]]], sizes: [128], strides: [1]
// CHECK:             hivm.hir.load ins(%[[SRC]] : memref<128xf16, strided<[1]>>) outs({{.*}})
// CHECK:           }
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @blockify_block_argument_dependent_load(%arg0: memref<?xf16>, %arg1: i32, %arg2: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.muli %arg1, %arg2 : i32
    annotation.mark %0 {logical_block_num} : i32
    %1 = hivm.hir.get_block_idx -> i64
    %block_i32 = arith.trunci %1 : i64 to i32
    scf.for %iv = %block_i32 to %arg1 step %c1_i32 : i32 {
      %offset = arith.index_cast %iv : i32 to index
      %source = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1]>>
      %destination = memref.alloc() : memref<128xf16>
      hivm.hir.load ins(%source : memref<128xf16, strided<[1]>>) outs(%destination : memref<128xf16>)
      %source_tensor = bufferization.to_tensor %destination restrict writable : memref<128xf16>
    }
    return
  }
}

// -----

// Safety: an scf.if result may be the source view, so the store through it must keep the source load in the inner loop.
// CHECK-LABEL: func.func @if_result_aliases_source(
// CHECK-NOT:     hivm.hir.load
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}{{ *}}: i32 {
// CHECK:           scf.for %[[ORIG_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             hivm.hir.load ins(%[[SRC:.*]] :
// CHECK:           }
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @if_result_aliases_source(%arg0: memref<?xf16>, %cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1]>>
      %other = memref.alloc() : memref<512xf16>
      %other_view = memref.cast %other : memref<512xf16> to memref<512xf16, strided<[1]>>
      %res = scf.if %cond -> (memref<512xf16, strided<[1]>>) {
        scf.yield %src : memref<512xf16, strided<[1]>>
      } else {
        scf.yield %other_view : memref<512xf16, strided<[1]>>
      }
      %val = memref.alloc() : memref<512xf16>
      hivm.hir.store ins(%val : memref<512xf16>) outs(%res : memref<512xf16, strided<[1]>>)
      %dst = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%src : memref<512xf16, strided<[1]>>) outs(%dst : memref<512xf16>)
    }
    return
  }
}

// -----

// Safety: a loop iter_arg initialized with the source view is stored through, so the invariant source load must stay in the inner loop.
// CHECK-LABEL: func.func @iter_arg_aliases_source(
// CHECK-NOT:     hivm.hir.load
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}{{ *}}: i32 {
// CHECK:           scf.for %[[ORIG_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             hivm.hir.load ins(%[[SRC:.*]] :
// CHECK:           }
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @iter_arg_aliases_source(%arg0: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1]>>
      %res = scf.for %k = %c0 to %c2 step %c1 iter_args(%acc = %src) -> (memref<512xf16, strided<[1]>>) {
        %val = memref.alloc() : memref<512xf16>
        hivm.hir.store ins(%val : memref<512xf16>) outs(%acc : memref<512xf16, strided<[1]>>)
        scf.yield %acc : memref<512xf16, strided<[1]>>
      }
      %dst = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%src : memref<512xf16, strided<[1]>>) outs(%dst : memref<512xf16>)
    }
    return
  }
}

// -----

// Safety: distinct function arguments may alias, so reading one and storing to another must not hoist the load.
// CHECK-LABEL: func.func @different_args_may_alias(
// CHECK-NOT:     hivm.hir.load
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}{{ *}}: i32 {
// CHECK:           scf.for %[[ORIG_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             hivm.hir.load ins(%[[SRC:.*]] :
// CHECK:           }
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @different_args_may_alias(%arg0: memref<?xf16>, %arg1: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1]>>
      %out = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1]>>
      %val = memref.alloc() : memref<512xf16>
      hivm.hir.store ins(%val : memref<512xf16>) outs(%out : memref<512xf16, strided<[1]>>)
      %dst = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%src : memref<512xf16, strided<[1]>>) outs(%dst : memref<512xf16>)
    }
    return
  }
}

// -----

// Safety: a write through a view of the source memref aliases the source load and blocks hoisting.
// CHECK-LABEL: func.func @subview_write_aliases_source(
// CHECK-NOT:     hivm.hir.load
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}{{ *}}: i32 {
// CHECK:           scf.for %[[ORIG_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             hivm.hir.load ins(%[[SRC:.*]] :
// CHECK:           }
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @subview_write_aliases_source(%arg0: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1]>>
      %view = memref.subview %src[0] [128] [1] : memref<512xf16, strided<[1]>> to memref<128xf16, strided<[1]>>
      %val = memref.alloc() : memref<128xf16>
      hivm.hir.store ins(%val : memref<128xf16>) outs(%view : memref<128xf16, strided<[1]>>)
      %dst = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%src : memref<512xf16, strided<[1]>>) outs(%dst : memref<512xf16>)
    }
    return
  }
}

// -----

// Safety: a write to an independent allocation does not alias the source, so the
// source load can still be hoisted out of the inner loop and blockify loop.
// CHECK-LABEL: func.func @independent_alloc_write_hoists(
// CHECK:         hivm.hir.load ins(%{{.*}} : memref<512xf16, strided<[1]>>)
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}{{ *}}: i32 {
// CHECK-NOT:       hivm.hir.load ins(%{{.*}} : memref<512xf16, strided<[1]>>)
// CHECK:           scf.for
// CHECK-NOT:         hivm.hir.load ins(%{{.*}} : memref<512xf16, strided<[1]>>)
// CHECK:             hivm.hir.store
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @independent_alloc_write_hoists(%arg0: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1]>>
      %own = memref.alloc() : memref<512xf16>
      %val = memref.alloc() : memref<512xf16>
      hivm.hir.store ins(%val : memref<512xf16>) outs(%own : memref<512xf16>)
      %dst = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%src : memref<512xf16, strided<[1]>>) outs(%dst : memref<512xf16>)
    }
    return
  }
}

// -----

// Performance: a shared addi DAG (x + x, 28 levels) feeds the view offset. The
// hoisting analysis must visit each producer once instead of exponentially.
// CHECK-LABEL: func.func @shared_producer_dag(
// CHECK:         arith.addi %{{.*}}, %{{.*}} : index
// CHECK:         scf.for %[[BLOCK_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}{{ *}}: i32 {
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @shared_producer_dag(%arg0: memref<?xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %x0 = arith.constant 1 : index
      %x1 = arith.addi %x0, %x0 : index
      %x2 = arith.addi %x1, %x1 : index
      %x3 = arith.addi %x2, %x2 : index
      %x4 = arith.addi %x3, %x3 : index
      %x5 = arith.addi %x4, %x4 : index
      %x6 = arith.addi %x5, %x5 : index
      %x7 = arith.addi %x6, %x6 : index
      %x8 = arith.addi %x7, %x7 : index
      %x9 = arith.addi %x8, %x8 : index
      %x10 = arith.addi %x9, %x9 : index
      %x11 = arith.addi %x10, %x10 : index
      %x12 = arith.addi %x11, %x11 : index
      %x13 = arith.addi %x12, %x12 : index
      %x14 = arith.addi %x13, %x13 : index
      %x15 = arith.addi %x14, %x14 : index
      %x16 = arith.addi %x15, %x15 : index
      %x17 = arith.addi %x16, %x16 : index
      %x18 = arith.addi %x17, %x17 : index
      %x19 = arith.addi %x18, %x18 : index
      %x20 = arith.addi %x19, %x19 : index
      %x21 = arith.addi %x20, %x20 : index
      %x22 = arith.addi %x21, %x21 : index
      %x23 = arith.addi %x22, %x22 : index
      %x24 = arith.addi %x23, %x23 : index
      %x25 = arith.addi %x24, %x24 : index
      %x26 = arith.addi %x25, %x25 : index
      %x27 = arith.addi %x26, %x26 : index
      %x28 = arith.addi %x27, %x27 : index
      %src = memref.reinterpret_cast %arg0 to offset: [%x28], sizes: [512], strides: [1] : memref<?xf16> to memref<512xf16, strided<[1], offset: ?>>
      %dst = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%src : memref<512xf16, strided<[1], offset: ?>>) outs(%dst : memref<512xf16>)
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @source_alloc_dealloc(
// CHECK-NOT: hivm.hir.load
// CHECK: scf.for
// CHECK-NOT: hivm.hir.load
// CHECK: scf.for
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<512xf16>
// CHECK: hivm.hir.load ins(%[[SRC]] : memref<512xf16>)
// CHECK: memref.dealloc %[[SRC]] : memref<512xf16>
// CHECK: } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @source_alloc_dealloc(%arg0: memref<512xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %src = memref.alloc() : memref<512xf16>
      %dst = memref.alloc() : memref<512xf16>
      hivm.hir.load ins(%src : memref<512xf16>) outs(%dst : memref<512xf16>)
      memref.dealloc %src : memref<512xf16>
    }
    return
  }
}

// -----

// Regression: reading the destination before the load makes the iteration
// order observable. With src[0] = 2 and initial dst[0] = 1, hoisting changes
// the result from [1, 2] to [2, 2], so the load must remain in the inner loop.
// CHECK-LABEL: func.func @destination_read_before_load(
// CHECK:         scf.for
// CHECK-NOT:       hivm.hir.load
// CHECK:           scf.for
// CHECK:             bufferization.to_tensor
// CHECK:             tensor.extract
// CHECK:             memref.store
// CHECK:             hivm.hir.load
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @destination_read_before_load(%src: memref<1xf32>, %out: memref<2xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %one = arith.constant 1.0 : f32
    %dst = memref.alloc() : memref<1xf32>
    memref.store %one, %dst[%c0] : memref<1xf32>
    %dst_view = memref.cast %dst : memref<1xf32> to memref<?xf32>
    %result = memref.alloc() : memref<2xf32>
    scf.for %iv = %c0 to %c2 step %c1 {
      %tensor = bufferization.to_tensor %dst_view restrict writable : memref<?xf32>
      %value = tensor.extract %tensor[%c0] : tensor<?xf32>
      memref.store %value, %result[%iv] : memref<2xf32>
      hivm.hir.load ins(%src : memref<1xf32>) outs(%dst_view : memref<?xf32>)
    }
    memref.copy %result, %out : memref<2xf32> to memref<2xf32>
    return
  }
}

// -----

// Regression: a destination to_tensor used outside the load's loop must not be
// moved after its existing tensor consumers. Moving it without preserving
// dominance produces an invalid SSA use.
// CHECK-LABEL: func.func @preserve_external_to_tensor_dominance(
// CHECK:         bufferization.to_tensor
// CHECK:         tensor.extract
// CHECK:         scf.for
// CHECK:           hivm.hir.load
// CHECK:         } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @preserve_external_to_tensor_dominance(%src: memref<?xf32>, %observed: memref<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    %one = arith.constant 1.0 : f32
    %dst = memref.alloc() : memref<1xf32>
    memref.store %one, %dst[%c0] : memref<1xf32>
    %dst_view = memref.cast %dst : memref<1xf32> to memref<?xf32>
    %tensor = bufferization.to_tensor %dst_view restrict writable : memref<?xf32>
    %value = tensor.extract %tensor[%c0] : tensor<?xf32>
    memref.store %value, %observed[%c0] : memref<1xf32>
    %offset = arith.index_cast %block : i64 to index
    %view = memref.reinterpret_cast %src to offset: [%offset], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
    scf.for %iv = %c0 to %c2 step %c1 {
      hivm.hir.load ins(%view : memref<1xf32, strided<[1], offset: ?>>) outs(%dst_view : memref<?xf32>)
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @unknown_effect_blocks_hoist(
// CHECK-NOT: hivm.hir.load
// CHECK: scf.for
// CHECK-NOT: hivm.hir.load
// CHECK: scf.for
// CHECK: "test.unknown_effect"
// CHECK: hivm.hir.load
// CHECK: } {autoblockify.subloop}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @unknown_effect_blocks_hoist(%arg0: memref<512xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %blocks = arith.constant 56 : i32
    annotation.mark %blocks {logical_block_num} : i32
    %block = hivm.hir.get_block_idx -> i64
    scf.for %iv = %c0 to %c2 step %c1 {
      %dst = memref.alloc() : memref<512xf16>
      "test.unknown_effect"(%arg0) : (memref<512xf16>) -> ()
      hivm.hir.load ins(%arg0 : memref<512xf16>) outs(%dst : memref<512xf16>)
    }
    return
  }
}
