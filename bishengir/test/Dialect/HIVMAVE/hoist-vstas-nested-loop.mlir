// RUN: bishengir-opt -hoist-vstas %s | FileCheck %s

// HoistVstas must not hoist a vstas out of a loop that is itself nested in
// another scf.for: the hoisted commit would run once per outer iteration at
// the outer back edge, while the outer loop's next iteration may re-read the
// committed buffer before the store queue drains (a loop-carried memory
// dependence invisible in SSA form). The nested vstas must stay inside the
// inner loop, while a vstas in a non-nested loop is still hoisted.
// Reduced from the fp8 SERIAL tree-reduce sum kernel (triton_sum_3D_dim0_
// dimm1, Ascend950PR) in which this hoist made lane 0 lose every per-m
// commit except the last one.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {

// CHECK-LABEL: func.func @vstas_nested_loops
func.func @vstas_nested_loops() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32
  %data = arith.constant dense<0.000000e+00> : vector<64xf32>
  %ptr = llvm.mlir.undef : !llvm.ptr<6>
  %init = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"() : () -> vector<32xi8>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK-NOT: iter_args
  // CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.f32"
  // CHECK: llvm.extractvalue
  // CHECK: llvm.extractvalue
  // CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
  // CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vstas"
  // CHECK: return
  scf.for %arg0 = %c0 to %c8 step %c1 {
    scf.for %arg1 = %c0 to %c8 step %c1 {
      %vstus = "hivm_regbaseintrins.intr.hivm.vstus.post.f32"(%data, %ptr, %c0_i32, %init) {hivm.is_continuous} : (vector<64xf32>, !llvm.ptr<6>, i32, vector<32xi8>) -> !llvm.struct<(vector<32xi8>, ptr<6>)>
      %vec = llvm.extractvalue %vstus[0] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      %ptr1 = llvm.extractvalue %vstus[1] : !llvm.struct<(vector<32xi8>, ptr<6>)>
      "hivm_regbaseintrins.intr.hivm.vstas"(%vec, %ptr1, %c0_i32, %c0_i32) {hivm.is_continuous} : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> ()
    }
  }
  return
}

// CHECK-LABEL: func.func @vstas_single_loop
func.func @vstas_single_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32
  %data = arith.constant dense<0.000000e+00> : vector<64xf32>
  %ptr = llvm.mlir.undef : !llvm.ptr<6>
  %init = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"() : () -> vector<32xi8>
  // CHECK: scf.for {{.*}} iter_args
  // CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.f32"
  // CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vstas"
  // CHECK: scf.yield
  // CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
  // CHECK: return
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %vstus = "hivm_regbaseintrins.intr.hivm.vstus.post.f32"(%data, %ptr, %c0_i32, %init) {hivm.is_continuous} : (vector<64xf32>, !llvm.ptr<6>, i32, vector<32xi8>) -> !llvm.struct<(vector<32xi8>, ptr<6>)>
    %vec = llvm.extractvalue %vstus[0] : !llvm.struct<(vector<32xi8>, ptr<6>)>
    %ptr1 = llvm.extractvalue %vstus[1] : !llvm.struct<(vector<32xi8>, ptr<6>)>
    "hivm_regbaseintrins.intr.hivm.vstas"(%vec, %ptr1, %c0_i32, %c0_i32) {hivm.is_continuous} : (vector<32xi8>, !llvm.ptr<6>, i32, i32) -> ()
  }
  return
}

}
