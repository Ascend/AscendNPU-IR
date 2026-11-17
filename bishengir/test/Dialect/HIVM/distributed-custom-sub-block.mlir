// RUN: bishengir-opt %s -hivm-bind-sub-block | FileCheck %s

// Remote writes use an empty GM-address list but must have one queue producer.
// CHECK-LABEL: func.func @remote_put
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[LANE:.*]] = hivm.hir.get_sub_block_idx
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[LANE]]
// CHECK: %[[COND:.*]] = arith.cmpi eq, %[[INDEX]], %[[ZERO]]
// CHECK: scf.if %[[COND]]
// CHECK: hivm.hir.custom {{.*}}hivm.is_distributed{{.*}}symbol = "aclshmem_putmem_half"
// CHECK: } {limit_sub_block_id0}
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @remote_put(%dst: memref<1xf16>, %src: memref<1xf16>, %bytes: i32, %peer: i32, %local: memref<1024xf16>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed,
        hivm.pipe = #hivm.pipe<PIPE_S>,
        hivm.tcore_type = #hivm.tcore_type<VECTOR>,
        hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device",
        libpath = "", pure = false, symbol = "aclshmem_putmem_half"}
        "dist.aclshmem_putmem_half" ins(%dst, %src, %bytes, %peer : memref<1xf16>, memref<1xf16>, i32, i32)
    %value = arith.constant dense<1.0> : tensor<1024xf16>
    hivm.hir.store ins(%value : tensor<1024xf16>) outs(%local : memref<1024xf16>)
    return
  }
}
