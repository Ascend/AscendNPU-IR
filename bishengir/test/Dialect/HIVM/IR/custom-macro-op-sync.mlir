// RUN: bishengir-opt %s -split-input-file | bishengir-opt -split-input-file | FileCheck %s

// -----

// sync_event_slots round-trip and sync_related_args syntax.

func.func @custom_macro_sync_event_slots(%arg0: memref<4xf32>, %arg1: tensor<4xf32>,
                                         %evt0: i64)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       symbol = "k_custom_macro_sync",
       sync_event_slots = [
         #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait, <EVENT_ID1>>
       ]}
      "user.macro_sync"
      ins(%arg0, %arg1 : memref<4xf32>, tensor<4xf32>)
      outs(%empty : tensor<4xf32>) -> tensor<4xf32>
      sync_related_args(%evt0 : i64)
  return
}

// CHECK-LABEL: func.func @custom_macro_sync_event_slots
// CHECK:       sync_event_slots = [
// CHECK-SAME:    #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait, <EVENT_ID1>>
// CHECK-SAME:  ]
// CHECK:       sync_related_args(%{{.*}} : i64)

// -----

// Default internal (no explicit macro_sync keyword) and set round-trip.

func.func @custom_macro_sync_default_internal(%arg0: memref<4xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       symbol = "k_default_internal",
       sync_event_slots = [
         #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>>
       ]}
      "user.default_internal"
      ins(%arg0 : memref<4xf32>)
      outs(%arg0 : memref<4xf32>)
  return
}

// CHECK-LABEL: func.func @custom_macro_sync_default_internal
// CHECK:       sync_event_slots = [
// CHECK-SAME:    #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>>
// CHECK-SAME:  ]

// -----

func.func @custom_macro_sync_set_slot(%arg0: memref<4xf32>, %evt0: i64, %evt1: i64)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       symbol = "k_set_slot",
       sync_event_slots = [
         #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, set>,
         #hivm.sync_event_slot<#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE2>, set, <EVENT_ID2>>
       ]}
      "user.set_slot"
      ins(%arg0 : memref<4xf32>)
      outs(%arg0 : memref<4xf32>)
      sync_related_args(%evt0, %evt1 : i64, i64)
  return
}

// CHECK-LABEL: func.func @custom_macro_sync_set_slot
// CHECK:       sync_event_slots = [
// CHECK-SAME:    #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, set>,
// CHECK-SAME:    #hivm.sync_event_slot<#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE2>, set, <EVENT_ID2>>
// CHECK-SAME:  ]
// CHECK:       sync_related_args(%{{.*}}, %{{.*}} : i64, i64)

// -----

func.func @custom_macro_sync_internal_slot(%arg0: memref<4xf32>, %evt0: i64, %evt1: i64)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       symbol = "k_internal",
       sync_event_slots = [
         #hivm.sync_event_slot<internal>,
         #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, internal, <EVENT_ID1>>
       ]}
      "user.internal"
      ins(%arg0 : memref<4xf32>)
      outs(%arg0 : memref<4xf32>)
      sync_related_args(%evt0, %evt1 : i64, i64)
  return
}

// CHECK-LABEL: func.func @custom_macro_sync_internal_slot
// CHECK:       sync_event_slots = [
// CHECK-SAME:    #hivm.sync_event_slot<internal>,
// CHECK-SAME:    #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, internal, <EVENT_ID1>>
// CHECK-SAME:  ]
// CHECK:       sync_related_args(%{{.*}}, %{{.*}} : i64, i64)
