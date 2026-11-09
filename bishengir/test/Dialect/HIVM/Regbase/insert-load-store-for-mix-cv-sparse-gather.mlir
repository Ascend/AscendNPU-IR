// RUN: bishengir-opt -hivm-insert-load-store-for-mix-cv %s -split-input-file -verify-diagnostics --canonicalize | FileCheck %s

// sparse_gather_load_to_L1buffer declares its output address space (cbuf) in
// arg_attrs, so the pass must leave the gathered tile on L1 and feed it to the
// cube mmad directly. Only the plain-GM B operand needs a load (GM -> L1); no
// load/store is inserted around the gather result.

// CHECK-LABEL: @sparse_gather_l1_output_stays_cbuf
// CHECK: %[[B:.+]] = hivm.hir.load ins(%arg2 {{.*}}"hivm.inserted-load"
// CHECK: %[[G:.+]] = hivm.hir.custom {{.*}}"__builtin_sparse_gather_load_to_L1buffer"
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.store
// CHECK: hivm.hir.mmadL1 {{.*}}ins(%[[G]], %[[B]]
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @sparse_gather_l1_output_stays_cbuf(%base: tensor<1xbf16>,
                                                %index: tensor<1xi32>,
                                                %b: tensor<2x1x16x16xbf16>)
      -> tensor<16x16xf32>
      attributes {hacc.entry,
                  hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c192_i32 = arith.constant 192 : i32
    %c24_i32 = arith.constant 24 : i32
    %c0_i32 = arith.constant 0 : i32
    %mask = arith.constant 16777215 : i32
    %c0_i64 = arith.constant 0 : i64
    %init = tensor.empty() : tensor<2x1x16x16xbf16>
    %gather = hivm.hir.custom
        {arg_attrs = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                      {address_space = #hivm.address_space<cbuf>,
                       data_layout = #hivm.data_layout<Fractal>}],
         gm_addr_args_indices = array<i32: 0, 1>,
         hivm.pipe = #hivm.pipe<PIPE_MTE2>,
         hivm.tcore_type = #hivm.tcore_type<CUBE>,
         symbol = "sparse_gather_load_to_L1buffer_bfloat16_t_int32_t"}
        "__builtin_sparse_gather_load_to_L1buffer"
        ins(%base, %index, %c16_i32, %c16_i32, %c8_i32, %c192_i32, %c0_i64,
            %c24_i32, %mask, %c0_i32
            : tensor<1xbf16>, tensor<1xi32>, i32, i32, i32, i32, i64, i32, i32,
              i32)
        outs(%init : tensor<2x1x16x16xbf16>) -> tensor<2x1x16x16xbf16>
    %acc = tensor.empty() : tensor<16x16xf32>
    %res = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true}
        ins(%gather, %b, %true, %c16, %c16, %c16
            : tensor<2x1x16x16xbf16>, tensor<2x1x16x16xbf16>, i1, index, index,
              index)
        outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %res : tensor<16x16xf32>
  }
}

// -----

// Without the address_space declaration the pass falls back to the pipe
// (MTE2 -> GM), so the gather result is treated as a GM value and the cube mmad
// forces a GM -> L1 load on it. This is the behaviour the arg_attrs lookup
// avoids above.

// CHECK-LABEL: @sparse_gather_without_cbuf_attr_loads_from_gm
// CHECK: %[[G:.+]] = hivm.hir.custom {{.*}}"__builtin_sparse_gather_load_to_L1buffer"
// CHECK: %[[L:.+]] = hivm.hir.load ins(%[[G]] {{.*}}"hivm.inserted-load"
// CHECK: hivm.hir.mmadL1 {{.*}}ins(%[[L]],
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @sparse_gather_without_cbuf_attr_loads_from_gm(%base: tensor<1xbf16>,
                                                           %index: tensor<1xi32>,
                                                           %b: tensor<2x1x16x16xbf16>)
      -> tensor<16x16xf32>
      attributes {hacc.entry,
                  hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c192_i32 = arith.constant 192 : i32
    %c24_i32 = arith.constant 24 : i32
    %c0_i32 = arith.constant 0 : i32
    %mask = arith.constant 16777215 : i32
    %c0_i64 = arith.constant 0 : i64
    %init = tensor.empty() : tensor<2x1x16x16xbf16>
    %gather = hivm.hir.custom
        {arg_attrs = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                      {data_layout = #hivm.data_layout<Fractal>}],
         gm_addr_args_indices = array<i32: 0, 1>,
         hivm.pipe = #hivm.pipe<PIPE_MTE2>,
         hivm.tcore_type = #hivm.tcore_type<CUBE>,
         symbol = "sparse_gather_load_to_L1buffer_bfloat16_t_int32_t"}
        "__builtin_sparse_gather_load_to_L1buffer"
        ins(%base, %index, %c16_i32, %c16_i32, %c8_i32, %c192_i32, %c0_i64,
            %c24_i32, %mask, %c0_i32
            : tensor<1xbf16>, tensor<1xi32>, i32, i32, i32, i32, i64, i32, i32,
              i32)
        outs(%init : tensor<2x1x16x16xbf16>) -> tensor<2x1x16x16xbf16>
    %acc = tensor.empty() : tensor<16x16xf32>
    %res = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true}
        ins(%gather, %b, %true, %c16, %c16, %c16
            : tensor<2x1x16x16xbf16>, tensor<2x1x16x16xbf16>, i1, index, index,
              index)
        outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %res : tensor<16x16xf32>
  }
}
