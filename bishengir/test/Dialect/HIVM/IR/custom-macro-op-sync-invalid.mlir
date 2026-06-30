// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @custom_macro_sync_args_wrong_size(%arg0: memref<4xf32>, %evt: i64)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{sync_related_args should be empty}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       symbol = "k_bad"}
      "user.bad"
      ins(%arg0 : memref<4xf32>)
      outs(%arg0 : memref<4xf32>)
      sync_related_args(%evt : i64)
  return
}

// -----

func.func @custom_macro_sync_pipe_slot_in_attr(%arg0: memref<4xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       symbol = "k_pipe_slot",
       sync_event_slots = [
         #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>>
       ]}
      "user.pipe_slot"
      ins(%arg0 : memref<4xf32>)
      outs(%arg0 : memref<4xf32>)
  return
}

// -----

func.func @custom_macro_sync_args_too_few(%arg0: memref<4xf32>, %evt: i64)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{sync_related_args should be of size 2}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       symbol = "k_too_few",
       sync_event_slots = [
         #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait>,
         #hivm.sync_event_slot<#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE2>, set>
       ]}
      "user.too_few"
      ins(%arg0 : memref<4xf32>)
      outs(%arg0 : memref<4xf32>)
      sync_related_args(%evt : i64)
  return
}

// -----

func.func @custom_macro_sync_args_too_many(%arg0: memref<4xf32>, %evt0: i64, %evt1: i64)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{sync_related_args should be of size 1}}
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       symbol = "k_too_many",
       sync_event_slots = [
         #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait>
       ]}
      "user.too_many"
      ins(%arg0 : memref<4xf32>)
      outs(%arg0 : memref<4xf32>)
      sync_related_args(%evt0, %evt1 : i64, i64)
  return
}
