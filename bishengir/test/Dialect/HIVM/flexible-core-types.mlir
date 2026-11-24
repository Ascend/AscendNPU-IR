// RUN: bishengir-opt %s -hivm-infer-func-core-type | FileCheck %s --check-prefix=INFER
// RUN: bishengir-opt %s -hivm-infer-func-core-type -hivm-split-mix-kernel | FileCheck %s --check-prefix=SPLIT
// RUN: bishengir-opt %s -hivm-split-mix-kernel | FileCheck %s --check-prefix=DIRECT
// RUN: bishengir-opt %s -hivm-infer-func-core-type -hivm-split-mix-kernel -hivm-infer-func-core-type -hivm-split-mix-kernel | FileCheck %s --check-prefix=SPLIT

// No distributed marker and no concrete constraint: default to AIV.
// INFER-LABEL: func.func @unmarked(
// INFER-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// INFER: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// SPLIT-LABEL: func.func @unmarked(
// SPLIT: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// DIRECT-LABEL: func.func @unmarked(
// DIRECT: hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>
func.func @unmarked() {
  hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "query"} "query"
  return
}

// Native tcoretype fields: a flexible debug follows the function.
// INFER-LABEL: func.func @debug_cube(
// INFER: tcoretype = #hivm.tcore_type<CUBE>
// SPLIT-LABEL: func.func @debug_cube(
// SPLIT: tcoretype = #hivm.tcore_type<CUBE>
// DIRECT-LABEL: func.func @debug_cube(
// DIRECT: tcoretype = #hivm.tcore_type<CUBE>
func.func @debug_cube(%x: i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  hivm.hir.debug {debugtype = "print", hex = false, prefix = "", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %x : i32
  return
}

// Load inference based on memory spaces wins over the MIX fallback.
// INFER-LABEL: func.func @load_vector(
// INFER: core_type = <VECTOR>
// SPLIT-LABEL: func.func @load_vector_mix_aic(
// SPLIT-NOT: hivm.hir.load
// SPLIT: return
// SPLIT-LABEL: func.func @load_vector_mix_aiv(
// SPLIT: hivm.hir.load
// SPLIT-SAME: core_type = <VECTOR>
// DIRECT-LABEL: func.func @load_vector_mix_aic(
// DIRECT-NOT: hivm.hir.load
// DIRECT: return
// DIRECT-LABEL: func.func @load_vector_mix_aiv(
// DIRECT: hivm.hir.load
// DIRECT-SAME: core_type = <VECTOR>
func.func @load_vector(%src: memref<16xf32, #hivm.address_space<gm>>, %dst: memref<16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%dst : memref<16xf32, #hivm.address_space<ub>>) core_type = <CUBE_OR_VECTOR>
  return
}

// A query used only by a vector operation still exists in both halves.
// SPLIT-LABEL: func.func @one_consumer_mix_aic(
// SPLIT: symbol = "rank"
// SPLIT-NOT: symbol = "consume"
// SPLIT: return
// SPLIT-LABEL: func.func @one_consumer_mix_aiv(
// SPLIT: symbol = "rank"
// SPLIT: symbol = "consume"
func.func @one_consumer() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %rank = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>, symbol = "rank"} "rank" -> i32
  hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "consume"} "consume" ins(%rank : i32)
  return
}

// A producer-specific debug choice must survive splitting.
// INFER-LABEL: func.func @debug_producer(
// INFER: tcoretype = #hivm.tcore_type<CUBE>
// SPLIT-LABEL: func.func @debug_producer_mix_aic(
// SPLIT: hivm.hir.debug
// SPLIT-SAME: tcoretype = #hivm.tcore_type<CUBE>
// SPLIT-LABEL: func.func @debug_producer_mix_aiv(
// SPLIT-NOT: hivm.hir.debug
// SPLIT: return
func.func @debug_producer() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %x = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "cube_value"} "cube_value" -> i32
  hivm.hir.debug {debugtype = "print", hex = false, prefix = "", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %x : i32
  return
}

// Native synchronization fields remain present and become concrete.
// INFER-LABEL: func.func @native_sync(
// INFER: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>]
// SPLIT-LABEL: func.func @native_sync(
// SPLIT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>]
// DIRECT-LABEL: func.func @native_sync(
// DIRECT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>]
func.func @native_sync() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  hivm.hir.sync_block_set[<CUBE_OR_VECTOR>, <PIPE_S>, <PIPE_S>] flag = 0
  return
}
