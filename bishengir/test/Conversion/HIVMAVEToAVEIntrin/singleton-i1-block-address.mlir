// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960DT_968B7 -ave-i1op-soft-impl -hivm-complex-reduction-intermediate-lowering -canonicalize -analyze-vector-layout -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse | bishengir-translate -mlir-to-llvmir | FileCheck %s --check-prefixes=CHECK,A6
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -ave-i1op-soft-impl -hivm-complex-reduction-intermediate-lowering -canonicalize -analyze-vector-layout -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts -canonicalize -cse | bishengir-translate -mlir-to-llvmir | FileCheck %s --check-prefixes=CHECK,A5

// Source element 259 with a five-bit descriptor offset selects bit 264.
// Copy it to bit 261. Both predicate blocks start at bit 256 (byte 32).
// The destination read and write must use the same block to preserve its other bits.
// CHECK-LABEL: define internal ptc_simdvf void @copy_singleton(
// A6: %[[INPUT:[0-9]+]] = getelementptr i8, {{.*}}, i64 32
// A5: %[[INPUT:[0-9]+]] = getelementptr i1, {{.*}}, i32 32
// CHECK: call <256 x i1> @llvm.hivm.plds.b8(ptr addrspace(6) %[[INPUT]],
// A6: @llvm.hivm.plt.b8.v300(i32 9)
// A6: @llvm.hivm.plt.b8.v300(i32 8)
// A5: @llvm.hivm.plt.b8.v300(i32 4)
// A5: @llvm.hivm.plt.b8.v300(i32 3)
// A6: %[[OUTPUT:[0-9]+]] = getelementptr i8, {{.*}}, i64 32
// A5: %[[OUTPUT:[0-9]+]] = getelementptr i1, {{.*}}, i32 32
// CHECK: call <256 x i1> @llvm.hivm.plds.b8(ptr addrspace(6) %[[OUTPUT]],
// CHECK: call void @llvm.hivm.psts.b8({{.*}}, ptr addrspace(6) %[[OUTPUT]],
func.func @copy_singleton(%input: memref<768xi1, strided<[1], offset: 5>, 6>, %output: memref<768xi1, 6>) attributes {hivm.vector_function} {
  %zero = arith.constant 0 : index
  %one = ave.hir.pge <VL1> : vector<256xi1>
  %in = memref.subview %input[259] [1] [1] : memref<768xi1, strided<[1], offset: 5>, 6> to memref<1xi1, strided<[1], offset: 264>, 6>
  %out = memref.subview %output[261] [1] [1] : memref<768xi1, 6> to memref<1xi1, strided<[1], offset: 261>, 6>
  %v = ave.hir.vload <NORM> %in[%zero] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xi1, strided<[1], offset: 264>, 6> into vector<256xi1>
  ave.hir.masked_store <NORM_B8> %out[%zero], %one, %v {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xi1, strided<[1], offset: 261>, 6>, vector<256xi1>, vector<256xi1>
  return
}
