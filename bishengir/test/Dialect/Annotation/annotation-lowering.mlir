// RUN: bishengir-opt %s -annotation-lowering | FileCheck %s

func.func @lowering(%arg1: f32, %arg2: f32, %arg3: f32) -> f32 {
  %1 = arith.mulf %arg1, %arg2 : f32
  %2 = arith.addf %1, %arg3 : f32
  // CHECK-NOT: annotation.mark
  annotation.mark %2 {attr = 2 : i32} : f32
  return %2 : f32
}