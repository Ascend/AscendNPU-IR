// RUN: bishengir-opt %s -convert-hivm-to-std -expand-strided-metadata -lower-affine -canonicalize -convert-hivm-to-llvm -canonicalize -triton-globalkernelargs-to-llvm -reconcile-unrealized-casts | bishengir-translate -mlir-to-llvmir | FileCheck %s
// Raw-pointer launcher ABI: ptr, offset i32, width i32, step i32.
// Check the entire final LLVM entry, not just argument count or MLIR descriptors.
// Unknown allocation capacity is NOT an additional launcher argument.
// The two memrefs flatten to 14 operands in LLVM; with six scalars this is
// a 20-operand wrapper call. Its C-interface call below has exactly 8 operands
// (two descriptor pointers + six scalars), with no capacity operand at either layer.
// CHECK-LABEL: define dso_local void @span_pointer(ptr addrspace(1) %0, i32 %1, i32 %2, i32 %3)
// CHECK: call void @load_gm_to_ubuf_2d_float_span(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 {{%[0-9]+}}, i64 1, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 1, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i64 4, i64 32, i64 32, i64 1, i32 2, float 7.000000e+00, i64 0, i64 {{%[0-9]+}}, i32 0, i32 1){{$}}
// CHECK-LABEL: define dso_local void @span_subview(ptr addrspace(1) %0, i32 %1, i32 %2, i32 %3)
// CHECK: call void @load_gm_to_ubuf_2d_float_span(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 {{%[0-9]+}}, i64 1, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 1, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i64 4, i64 32, i64 32, i64 1, i32 2, float 7.000000e+00, i64 0, i64 {{%[0-9]+}}, i32 0, i32 1){{$}}
// CHECK-LABEL: define dso_local void @span_mix_aic(ptr addrspace(1) %0, i32 %1, i32 %2, i32 %3)
// CHECK-LABEL: define private void @load_gm_to_ubuf_2d_float_span(
// CHECK-SAME: ptr addrspace(1) {{%[0-9]+}}, ptr addrspace(1) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 [[MODE:%[0-9]+]], float [[PAD:%[0-9]+]], i64 [[LEFT:%[0-9]+]], i64 [[RIGHT:%[0-9]+]], i32 [[CACHE:%[0-9]+]], i32 [[AXES:%[0-9]+]])
// CHECK: call void @_mlir_ciface_load_gm_to_ubuf_2d_float_span(ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, i32 [[MODE]], float [[PAD]], i64 [[LEFT]], i64 [[RIGHT]], i32 [[CACHE]], i32 [[AXES]]){{$}}
// CHECK: declare dso_local void @_mlir_ciface_load_gm_to_ubuf_2d_float_span(ptr, ptr, i32, float, i64, i64, i32, i32)
// CHECK-LABEL: define dso_local void @span_mix_aiv(ptr addrspace(1) %0, i32 %1, i32 %2, i32 %3)
// CHECK: call void @load_gm_to_ubuf_2d_float_span(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 {{%[0-9]+}}, i64 1, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 1, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i64 4, i64 32, i64 32, i64 1, i32 2, float 7.000000e+00, i64 0, i64 {{%[0-9]+}}, i32 0, i32 1){{$}}
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span_pointer(%src: memref<?xf32, #hivm.address_space<gm>>, %off: i32, %width: i32, %step: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, func_dyn_memref_args = dense<[true, false, false, false]> : vector<4xi1>} {
  %o = arith.index_cast %off : i32 to index
  %w = arith.index_cast %width : i32 to index
  %s = arith.index_cast %step : i32 to index
  %v = memref.reinterpret_cast %src to offset: [%o], sizes: [1, %w], strides: [%s, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  %z = arith.constant 0 : index
  %d = arith.constant 32 : index
  %rp = arith.subi %d, %w : index
  %p = arith.constant 7.0 : f32
  hivm.hir.load ins(%v : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %z : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
 func.func @span_subview(%src: memref<?xf32, #hivm.address_space<gm>>, %off: i32, %width: i32, %step: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, func_dyn_memref_args = dense<[true, false, false, false]> : vector<4xi1>} {
  %o = arith.index_cast %off : i32 to index
  %w = arith.index_cast %width : i32 to index
  %s = arith.index_cast %step : i32 to index
  %v = memref.reinterpret_cast %src to offset: [%o], sizes: [2, %w], strides: [%s, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %sv = memref.subview %v[1, 0] [1, %w] [1, 1] : memref<2x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  %z = arith.constant 0 : index
  %d = arith.constant 32 : index
  %rp = arith.subi %d, %w : index
  %p = arith.constant 7.0 : f32
  hivm.hir.load ins(%sv : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %z : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
 func.func @span_mix_aic(%src: memref<?xf32, #hivm.address_space<gm>>, %off: i32, %width: i32, %step: i32) attributes {hivm.part_of_mix, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, func_dyn_memref_args = dense<[true, false, false, false]> : vector<4xi1>} {
  return
 }

 func.func @span_mix_aiv(%src: memref<?xf32, #hivm.address_space<gm>>, %off: i32, %width: i32, %step: i32) attributes {hivm.part_of_mix, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, func_dyn_memref_args = dense<[true, false, false, false]> : vector<4xi1>} {
  %o = arith.index_cast %off : i32 to index
  %w = arith.index_cast %width : i32 to index
  %s = arith.index_cast %step : i32 to index
  %v = memref.reinterpret_cast %src to offset: [%o], sizes: [1, %w], strides: [%s, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  %z = arith.constant 0 : index
  %d = arith.constant 32 : index
  %rp = arith.subi %d, %w : index
  %p = arith.constant 7.0 : f32
  hivm.hir.load ins(%v : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%dst : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %z : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}
