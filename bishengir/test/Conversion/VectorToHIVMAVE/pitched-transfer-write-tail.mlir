// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960DT_968B7 -convert-vector-to-hivmave -ave-plt-to-pltm -analyze-vector-layout -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -canonicalize | FileCheck %s --check-prefixes=COMMON,A6
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-vector-to-hivmave -ave-plt-to-pltm -analyze-vector-layout -ave-normalize-ops -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -canonicalize | FileCheck %s --check-prefixes=COMMON,A5
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960DT_968B7 -ave-plt-to-pltm | FileCheck %s --check-prefix=PACKED-A6
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -ave-plt-to-pltm | FileCheck %s --check-prefix=PACKED-A5

// With end = 2, only lane 0 may write. PLTM would round the initial
// offset of 1 down to 0 and enable lane 1 as well.
#tail = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
// COMMON-LABEL: func.func @unaligned_scatter
// COMMON: scf.for
// COMMON: %[[PLT:.*]] = "hivm_regbaseintrins.intr.hivm.plt.b16.v300"
// A6: %[[MASK:.*]] = llvm.extractvalue %[[PLT]][0]
// A6: "hivm_regbaseintrins.intr.hivm.vscatter.v128s16_v300"({{.*}}, %[[MASK]])
// A5: "hivm_regbaseintrins.intr.hivm.vstus.post.f16"
func.func @unaligned_scatter(%dst: memref<?xf16, strided<[2]>, #hivm.address_space<ub>>, %value: vector<128xf16>, %end: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c1 to %end step %c128 {
    %count = affine.min #tail(%i)[%end]
    %mask = vector.create_mask %count : vector<128xi1>
    vector.transfer_write %value, %dst[%i], %mask {in_bounds = [true]} : vector<128xf16>, memref<?xf16, strided<[2]>, #hivm.address_space<ub>>
  }
  return
}

// PACKED-A6-LABEL: func.func @packed_tail
// PACKED-A6: %[[MASK:.*]], %{{.*}} = ave.hir.plt
// PACKED-A6: ave.hir.masked_store <NORM_B8> {{.*}}, %[[MASK]],
// PACKED-A5-LABEL: func.func @packed_tail
// PACKED-A5: ave.hir.pltm
func.func @packed_tail(%dst: memref<128xi1, #hivm.address_space<ub>>, %data: vector<64xi1>, %end: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %vl = arith.constant 64 : index
  scf.for %i = %one to %end step %vl {
    %count = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%i)[%end]
    %mask, %remaining = ave.hir.plt %count {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>, index
    ave.hir.masked_store <NORM_B8> %dst[%zero], %mask, %data {functionType = #ave.func_dist_type<pb32>} : memref<128xi1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
  }
  return
}
