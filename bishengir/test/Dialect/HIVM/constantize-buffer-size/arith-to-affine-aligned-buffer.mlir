// RUN: bishengir-opt %s --hivm-constantize-buffer-size -allow-unregistered-dialect | FileCheck %s --check-prefix=ARITH
// RUN: bishengir-opt %s --convert-arith-to-affine --hivm-constantize-buffer-size -allow-unregistered-dialect | FileCheck %s --check-prefix=AFFINE

// Stride alignment rounds the tile size up to eight f32 elements. The
// remainder expression cannot be bounded directly by ConstantizeBufferSize.
// ArithToAffine exposes the rounding expression and enables static backing
// storage, while the view preserves the runtime size (including the tail).
// ARITH-LABEL: func.func @aligned_temporary
// AFFINE-LABEL: func.func @aligned_temporary
func.func @aligned_temporary(%iv: index) {
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %n = affine.min affine_map<(d0) -> (280 - d0, 64)>(%iv)
  %padded = arith.addi %n, %c7 : index
  %rem = arith.remsi %padded, %c8 : index
  // ARITH: %[[SIZE:.*]] = arith.subi
  %size = arith.subi %padded, %rem : index
  // ARITH: %[[BUF:.*]] = memref.alloc(%[[SIZE]]) : memref<1x?x1xf32, #hivm.address_space<ub>>
  // ARITH: "test.use"(%[[BUF]])
  // Require static backing storage without depending on bound tightness.
  // AFFINE: %[[STORAGE:.*]] = memref.alloc() : memref<{{[0-9]+}}xi8, #hivm.address_space<ub>>
  // AFFINE: %[[VIEW:.*]] = memref.view %[[STORAGE]][{{.*}}][{{.*}}]
  // AFFINE: "test.use"(%[[VIEW]])
  %buf = memref.alloc(%size) : memref<1x?x1xf32, #hivm.address_space<ub>>
  "test.use"(%buf) : (memref<1x?x1xf32, #hivm.address_space<ub>>) -> ()
  return
}
