// REQUIRES: bishengir-hifloat8
// RUN: not bishengir-opt %s --convert-hivmave-to-ave-intrin 2>&1 | FileCheck %s
// A direct HiF8 -> FP16 AVE extension is not a native conversion. The HIVM
// vectorizer expands it through FP32 before reaching this layer.
// CHECK: error: {{.*}}HiFloat8 extension requires an FP32 result
func.func @unsupported_direct_half(%x: vector<256xf8HIF>, %mask: vector<256xi1>) -> vector<128xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %y = ave.hir.vextf %x, <part_even>, %mask : vector<256xf8HIF>, vector<128xf16>, vector<256xi1>
  return %y : vector<128xf16>
}
