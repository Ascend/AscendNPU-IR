// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s

// i1 ave.hir.vload <US>/<DS> must lower to hivm.plds.b8 with CCE MaskDist
// encodings (US=1, DS=2), not HIVMAVE enum values 24/25, and not hardcoded 0.

// CHECK-LABEL: @i1_plds_norm
func.func @i1_plds_norm(%arg0: memref<256xi1, #hivm.address_space<ub>>) -> vector<256xi1>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  // CHECK: "hivm_regbaseintrins.intr.hivm.plds.b8"
  %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<256xi1, #hivm.address_space<ub>> into vector<256xi1>
  return %0 : vector<256xi1>
}

// -----

// CHECK-LABEL: @i1_plds_us
func.func @i1_plds_us(%arg0: memref<256xi1, #hivm.address_space<ub>>) -> vector<256xi1>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIST:.*]] = llvm.mlir.constant(1 : i32)
  // CHECK: "hivm_regbaseintrins.intr.hivm.plds.b8"({{.*}}, {{.*}}, %[[DIST]], {{.*}})
  %0 = ave.hir.vload <US> %arg0[%c0] : memref<256xi1, #hivm.address_space<ub>> into vector<256xi1>
  return %0 : vector<256xi1>
}

// -----

// CHECK-LABEL: @i1_plds_ds
func.func @i1_plds_ds(%arg0: memref<256xi1, #hivm.address_space<ub>>) -> vector<256xi1>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIST:.*]] = llvm.mlir.constant(2 : i32)
  // CHECK: "hivm_regbaseintrins.intr.hivm.plds.b8"({{.*}}, {{.*}}, %[[DIST]], {{.*}})
  %0 = ave.hir.vload <DS> %arg0[%c0] : memref<256xi1, #hivm.address_space<ub>> into vector<256xi1>
  return %0 : vector<256xi1>
}
