// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-vector-to-hivmave | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-vector-to-hivmave -convert-arith-to-hivmave -analyze-vector-layout -remove-vector-layout-attr -ave-normalize-ops -canonicalize -convert-hivmave-to-ave-intrin -canonicalize | FileCheck %s --check-prefix=INTRIN

// A rank-1 static-strided f32 column write on A5 (Ascend950) lowers to a scatter
// whose per-lane element offsets are lane*stride, not a continuous NORM_B32
// store. Before the fix the 4 active lanes silently landed at elements 0,1,2,3
// instead of 0,8192,16384,24576 (a silent miscompile, not a precision error).
// CHECK-LABEL: @column_mask4_a5
// CHECK-DAG: %[[MASK:.*]] = ave.hir.pge <VL4>
// CHECK-DAG: %[[LANES:.*]] = ave.hir.vci %{{.*}}, <INCREASE>
// CHECK: %[[IDX:.*]] = ave.hir.vmuls %[[LANES]], %{{.*}}, %[[MASK]]
// CHECK: ave.hir.vscatter %arg0[%{{.*}}] [%[[IDX]]], %[[MASK]], %{{.*}}
// CHECK-NOT: ave.hir.masked_store
// INTRIN-LABEL: @column_mask4_a5
// INTRIN: "hivm_regbaseintrins.intr.hivm.vscatter_v64s32_v300"
func.func @column_mask4_a5(%dst: memref<4xf32, strided<[8192], offset: ?>, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %value = arith.constant dense<0xFF800000> : vector<64xf32>
  %m = vector.constant_mask [4] : vector<64xi1>
  vector.transfer_write %value, %dst[%c0], %m {in_bounds = [true]} : vector<64xf32>, memref<4xf32, strided<[8192], offset: ?>, #hivm.address_space<ub>>
  return
}
