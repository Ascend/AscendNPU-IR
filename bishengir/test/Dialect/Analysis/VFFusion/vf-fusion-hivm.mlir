// Tests for --vf-fusion-hivm with default fusion-mode=linear-scan
//
// HIVM-level input after ConvertToHIVM: DMA is already hir.load/store;
// compute is tensor hivm.hir.* and not yet a VF. Load/store stay in the
// caller; compute is outlined into a VF, then --hivm-vectorize-ops
// lowers the outlined VF.
//
// RUN: bishengir-opt %s --split-input-file --vf-fusion-hivm | FileCheck %s
// RUN: bishengir-opt %s --split-input-file --vf-fusion-hivm --hivm-vectorize-ops | FileCheck %s --check-prefix=CHECK-VEC

// CHECK: func.func private @toy_kernel_fused_{{[0-9]+}}
// CHECK: hivm.vector_function
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.vcmp
// CHECK: hivm.hir.vsel
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.store

// CHECK-LABEL: func.func @toy_kernel(
// CHECK: hivm.hir.load
// CHECK: hivm.hir.load
// CHECK: call @toy_kernel_fused_{{[0-9]+}}
// CHECK: hivm.hir.store
// CHECK-NOT: hivm.hir.vexp

// CHECK-VEC: func.func private @toy_kernel_fused_{{[0-9]+}}
// CHECK-VEC: hivm.vector_function
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: math.exp
// CHECK-VEC: arith.addf
// CHECK-VEC: arith.cmpf
// CHECK-VEC: arith.select
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-NOT: hivm.hir.vexp
// CHECK-VEC-LABEL: func.func @toy_kernel(
// CHECK-VEC: hivm.hir.load
// CHECK-VEC: call @toy_kernel_fused_{{[0-9]+}}
// CHECK-VEC: hivm.hir.store
func.func @toy_kernel(%src: tensor<8xf32>, %bound: tensor<8xf32>,
                      %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %exp_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %cmp_init = tensor.empty() : tensor<8xi1>
  %sel_init = tensor.empty() : tensor<8xf32>
  %store_init = tensor.empty() : tensor<8xf32>

  %a = hivm.hir.load ins(%src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%bound : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>

  %e = hivm.hir.vexp ins(%a : tensor<8xf32>)
      outs(%exp_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%e, %a : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %m = hivm.hir.vcmp ins(%s, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%cmp_init : tensor<8xi1>) compare_mode = <lt> -> tensor<8xi1>
  %r = hivm.hir.vsel ins(%m, %s, %b : tensor<8xi1>, tensor<8xf32>, tensor<8xf32>)
      outs(%sel_init : tensor<8xf32>) -> tensor<8xf32>

  %out = hivm.hir.store ins(%r : tensor<8xf32>)
      outs(%store_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// A single vectorizable op must still become a VF.
// CHECK: func.func private @single_vexp_fused_{{[0-9]+}}
// CHECK: hivm.vector_function
// CHECK: hivm.hir.vexp
// CHECK-LABEL: func.func @single_vexp(
// CHECK: hivm.hir.load
// CHECK: call @single_vexp_fused_{{[0-9]+}}
// CHECK: hivm.hir.store
// CHECK-NOT: hivm.hir.vexp

// CHECK-VEC: func.func private @single_vexp_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: math.exp
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-NOT: hivm.hir.vexp
// CHECK-VEC-LABEL: func.func @single_vexp(
// CHECK-VEC: call @single_vexp_fused_{{[0-9]+}}
func.func @single_vexp(%src: tensor<8xf32>, %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %e_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %e = hivm.hir.vexp ins(%a : tensor<8xf32>)
      outs(%e_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%e : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// Consecutive binary compute is one VF.
// CHECK: func.func private @vadd_vmul_fused_{{[0-9]+}}
// CHECK: hivm.vector_function
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.vmul
// CHECK-LABEL: func.func @vadd_vmul(
// CHECK: hivm.hir.load
// CHECK: call @vadd_vmul_fused_{{[0-9]+}}
// CHECK: hivm.hir.store
// CHECK-NOT: hivm.hir.vadd

// CHECK-VEC: func.func private @vadd_vmul_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: arith.addf
// CHECK-VEC: arith.mulf
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-NOT: hivm.hir.vadd
// CHECK-VEC-LABEL: func.func @vadd_vmul(
// CHECK-VEC: call @vadd_vmul_fused_{{[0-9]+}}
func.func @vadd_vmul(%a_src: tensor<8xf32>, %b_src: tensor<8xf32>,
                     %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %mul_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%b_src : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %p = hivm.hir.vmul ins(%s, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%mul_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%p : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// A store between compute runs is a barrier: two VFs.
// CHECK: func.func private @store_splits_fused_{{[0-9]+}}
// CHECK: hivm.hir.vadd
// CHECK: func.func private @store_splits_fused_{{[0-9]+}}
// CHECK: hivm.hir.vmul
// CHECK-LABEL: func.func @store_splits(
// CHECK: call @store_splits_fused_{{[0-9]+}}
// CHECK: hivm.hir.store
// CHECK: call @store_splits_fused_{{[0-9]+}}
// CHECK: hivm.hir.store

// CHECK-VEC: func.func private @store_splits_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: arith.addf
// CHECK-VEC: vector.transfer_write
// CHECK-VEC: func.func private @store_splits_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: arith.mulf
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-LABEL: func.func @store_splits(
// CHECK-VEC: call @store_splits_fused_{{[0-9]+}}
// CHECK-VEC: hivm.hir.store
// CHECK-VEC: call @store_splits_fused_{{[0-9]+}}
func.func @store_splits(%a_src: tensor<8xf32>, %b_src: tensor<8xf32>,
                       %c_src: tensor<8xf32>, %dst: tensor<8xf32>)
    -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %c_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %mul_init = tensor.empty() : tensor<8xf32>
  %s0_init = tensor.empty() : tensor<8xf32>
  %s1_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%b_src : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %mid = hivm.hir.store ins(%s : tensor<8xf32>)
      outs(%s0_init : tensor<8xf32>) -> tensor<8xf32>
  %c = hivm.hir.load ins(%c_src : tensor<8xf32>)
      outs(%c_init : tensor<8xf32>) -> tensor<8xf32>
  %p = hivm.hir.vmul ins(%c, %c : tensor<8xf32>, tensor<8xf32>)
      outs(%mul_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%p : tensor<8xf32>)
      outs(%s1_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// Already a VF: do not outline again. Vectorize still lowers the body.
// CHECK-LABEL: func.func @already_vf(
// CHECK-NOT: call @already_vf_fused
// CHECK: hivm.hir.vadd

// CHECK-VEC-LABEL: func.func @already_vf(
// CHECK-VEC-NOT: call @already_vf_fused
// CHECK-VEC-NOT: hivm.hir.vadd
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: arith.addf
// CHECK-VEC: vector.transfer_write
func.func @already_vf(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd",
      hivm.vector_function
    } {
  %add_init = tensor.empty() : tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  return %s : tensor<8xf32>
}

// -----

// Cube/AIC kernels are not fusion candidates.
// CHECK-LABEL: func.func @skip_aic(
// CHECK-NOT: call @skip_aic_fused
// CHECK: hivm.hir.vadd

// CHECK-VEC-LABEL: func.func @skip_aic(
// CHECK-VEC-NOT: call @skip_aic_fused
// CHECK-VEC-NOT: vector.transfer_read
// CHECK-VEC: hivm.hir.vadd
func.func @skip_aic(%a_src: tensor<8xf32>, %b_src: tensor<8xf32>,
                    %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIC>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%b_src : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%s : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// Non-SIMD parallel_mode is skipped.
// CHECK-LABEL: func.func @skip_simt(
// CHECK-NOT: call @skip_simt_fused
// CHECK: hivm.hir.vadd

// CHECK-VEC-LABEL: func.func @skip_simt(
// CHECK-VEC-NOT: call @skip_simt_fused
// CHECK-VEC-NOT: vector.transfer_read
// CHECK-VEC: hivm.hir.vadd
func.func @skip_simt(%a_src: tensor<8xf32>, %b_src: tensor<8xf32>,
                     %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simt"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%b_src : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%s : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// Ill-formed host funcs are not fusion candidates even with AIV SIMD attrs.
// CHECK-LABEL: func.func @skip_host(
// CHECK-NOT: call @skip_host_fused
// CHECK: hivm.hir.vadd

// CHECK-VEC-LABEL: func.func @skip_host(
// CHECK-VEC-NOT: call @skip_host_fused
// CHECK-VEC-NOT: vector.transfer_read
// CHECK-VEC: hivm.hir.vadd
func.func @skip_host(%a_src: tensor<8xf32>, %b_src: tensor<8xf32>,
                     %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<HOST>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%b_src : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%s : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// Constants are glue: they do not split a compute run.
// CHECK: func.func private @constant_glue_fused_{{[0-9]+}}
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.vmul
// CHECK-LABEL: func.func @constant_glue(
// CHECK: arith.constant
// CHECK: call @constant_glue_fused_{{[0-9]+}}
// CHECK-NOT: hivm.hir.vadd

// CHECK-VEC: func.func private @constant_glue_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: arith.addf
// CHECK-VEC: arith.mulf
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-NOT: hivm.hir.vadd
// CHECK-VEC-LABEL: func.func @constant_glue(
// CHECK-VEC: call @constant_glue_fused_{{[0-9]+}}
func.func @constant_glue(%a_src: tensor<8xf32>, %dst: tensor<8xf32>)
    -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %mul_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %cst = arith.constant dense<1.000000e+00> : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %cst : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %p = hivm.hir.vmul ins(%s, %cst : tensor<8xf32>, tensor<8xf32>)
      outs(%mul_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%p : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// Unary + binary still one VF.
// CHECK: func.func private @vabs_vmax_fused_{{[0-9]+}}
// CHECK: hivm.hir.vabs
// CHECK: hivm.hir.vmax
// CHECK-LABEL: func.func @vabs_vmax(
// CHECK: call @vabs_vmax_fused_{{[0-9]+}}
// CHECK: hivm.hir.store
// CHECK-NOT: hivm.hir.vabs

// CHECK-VEC: func.func private @vabs_vmax_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: math.absf
// CHECK-VEC: arith.maximumf
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-NOT: hivm.hir.vabs
// CHECK-VEC-LABEL: func.func @vabs_vmax(
// CHECK-VEC: call @vabs_vmax_fused_{{[0-9]+}}
func.func @vabs_vmax(%a_src: tensor<8xf32>, %b_src: tensor<8xf32>,
                     %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %abs_init = tensor.empty() : tensor<8xf32>
  %max_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%b_src : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>
  %abs = hivm.hir.vabs ins(%a : tensor<8xf32>)
      outs(%abs_init : tensor<8xf32>) -> tensor<8xf32>
  %m = hivm.hir.vmax ins(%abs, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%max_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%m : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// Two AIV SIMD funcs in one module each get their own VF.
// Outlined funcs are inserted before their caller:
//   kernel_a_fused, kernel_a, kernel_b_fused, kernel_b
// CHECK: func.func private @kernel_a_fused_{{[0-9]+}}
// CHECK: hivm.hir.vsub
// CHECK-LABEL: func.func @kernel_a(
// CHECK: call @kernel_a_fused_{{[0-9]+}}
// CHECK: func.func private @kernel_b_fused_{{[0-9]+}}
// CHECK: hivm.hir.vexp
// CHECK-LABEL: func.func @kernel_b(
// CHECK: call @kernel_b_fused_{{[0-9]+}}

// CHECK-VEC: func.func private @kernel_a_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: arith.subf
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-LABEL: func.func @kernel_a(
// CHECK-VEC: call @kernel_a_fused_{{[0-9]+}}
// CHECK-VEC: func.func private @kernel_b_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: math.exp
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-LABEL: func.func @kernel_b(
// CHECK-VEC: call @kernel_b_fused_{{[0-9]+}}
func.func @kernel_a(%a_src: tensor<16xf16>, %b_src: tensor<16xf16>,
                     %dst: tensor<16xf16>) -> tensor<16xf16>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<16xf16>
  %b_init = tensor.empty() : tensor<16xf16>
  %sub_init = tensor.empty() : tensor<16xf16>
  %s_init = tensor.empty() : tensor<16xf16>
  %a = hivm.hir.load ins(%a_src : tensor<16xf16>)
      outs(%a_init : tensor<16xf16>) -> tensor<16xf16>
  %b = hivm.hir.load ins(%b_src : tensor<16xf16>)
      outs(%b_init : tensor<16xf16>) -> tensor<16xf16>
  %d = hivm.hir.vsub ins(%a, %b : tensor<16xf16>, tensor<16xf16>)
      outs(%sub_init : tensor<16xf16>) -> tensor<16xf16>
  %out = hivm.hir.store ins(%d : tensor<16xf16>)
      outs(%s_init : tensor<16xf16>) -> tensor<16xf16>
  return %out : tensor<16xf16>
}

func.func @kernel_b(%src: tensor<16xf16>, %dst: tensor<16xf16>) -> tensor<16xf16>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.func_core_type = #hivm.func_core_type<AIV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<16xf16>
  %e_init = tensor.empty() : tensor<16xf16>
  %s_init = tensor.empty() : tensor<16xf16>
  %a = hivm.hir.load ins(%src : tensor<16xf16>)
      outs(%a_init : tensor<16xf16>) -> tensor<16xf16>
  %e = hivm.hir.vexp ins(%a : tensor<16xf16>)
      outs(%e_init : tensor<16xf16>) -> tensor<16xf16>
  %out = hivm.hir.store ins(%e : tensor<16xf16>)
      outs(%s_init : tensor<16xf16>) -> tensor<16xf16>
  return %out : tensor<16xf16>
}

// -----

// No core-type attributes at all: not provably cube, so it is fused
// (same fail-open AIV gate as the HFusion-level VFFusion).
// CHECK: func.func private @no_core_type_attrs_fused_{{[0-9]+}}
// CHECK: hivm.vector_function
// CHECK: hivm.hir.vadd
// CHECK-LABEL: func.func @no_core_type_attrs(
// CHECK: call @no_core_type_attrs_fused_{{[0-9]+}}
// CHECK-NOT: hivm.hir.vadd

// CHECK-VEC: func.func private @no_core_type_attrs_fused_{{[0-9]+}}
// CHECK-VEC: vector.transfer_read
// CHECK-VEC: arith.addf
// CHECK-VEC: vector.transfer_write
// CHECK-VEC-LABEL: func.func @no_core_type_attrs(
// CHECK-VEC: call @no_core_type_attrs_fused_{{[0-9]+}}
func.func @no_core_type_attrs(%a_src: tensor<8xf32>, %b_src: tensor<8xf32>,
                              %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%b_src : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%s : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// A cube fusion_kind alone marks the function as cube, even without a
// core-type attribute.
// CHECK-LABEL: func.func @skip_shallow_cv(
// CHECK-NOT: call @skip_shallow_cv_fused
// CHECK: hivm.hir.vadd

// CHECK-VEC-LABEL: func.func @skip_shallow_cv(
// CHECK-VEC-NOT: call @skip_shallow_cv_fused
// CHECK-VEC-NOT: vector.transfer_read
// CHECK-VEC: hivm.hir.vadd
func.func @skip_shallow_cv(%a_src: tensor<8xf32>, %b_src: tensor<8xf32>,
                           %dst: tensor<8xf32>) -> tensor<8xf32>
    attributes {
      hacc.entry,
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>,
      parallel_mode = "simd"
    } {
  %a_init = tensor.empty() : tensor<8xf32>
  %b_init = tensor.empty() : tensor<8xf32>
  %add_init = tensor.empty() : tensor<8xf32>
  %s_init = tensor.empty() : tensor<8xf32>
  %a = hivm.hir.load ins(%a_src : tensor<8xf32>)
      outs(%a_init : tensor<8xf32>) -> tensor<8xf32>
  %b = hivm.hir.load ins(%b_src : tensor<8xf32>)
      outs(%b_init : tensor<8xf32>) -> tensor<8xf32>
  %s = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%add_init : tensor<8xf32>) -> tensor<8xf32>
  %out = hivm.hir.store ins(%s : tensor<8xf32>)
      outs(%s_init : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}
