// RUN: split-file %s %t
// RUN: bishengir-opt %t/unseparated.mlir -o %t/unseparated-before.mlir
// RUN: bishengir-opt %t/unseparated.mlir -hivm-mark-buffer-pool -o %t/unseparated-after.mlir
// RUN: diff %t/unseparated-before.mlir %t/unseparated-after.mlir

//--- unseparated.mlir

// A normal compute scope is not an independent Load candidate. No target
// specification is needed when Mark has no work.
func.func @unseparated(%src: memref<16xf32, #hivm.address_space<gm>>,
                       %dst: memref<16xf32, #hivm.address_space<ub>>) {
  scope.scope : () -> () {
    hivm.hir.load ins(%src : memref<16xf32, #hivm.address_space<gm>>) outs(%dst : memref<16xf32, #hivm.address_space<ub>>)
    scope.return
  } {hivm.preload_num = 2 : i32}
  return
}
