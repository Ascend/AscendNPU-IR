// REQUIRES: execution-engine
// RUN: bishengir-opt %s \
// RUN:   --execution-engine-convert-hfusion-to-upstream \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --lower-affine \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%S/../../../../build/lib/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%S/../../../../build/lib/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

module {
 	 
 	 func.func @deinterleave_even(%arg0: memref<8xf32>) -> tensor<4xf32> {
 	   %0 = bufferization.to_tensor %arg0 restrict writable : memref<8xf32>
 	   %1 = hfusion.deinterleave %0 channel<0> : tensor<8xf32> -> tensor<4xf32>
 	   return %1 : tensor<4xf32>
 	 }
 	 
 	 func.func @deinterleave_odd(%arg0: memref<8xf32>) -> tensor<4xf32> {
 	   %0 = bufferization.to_tensor %arg0 restrict writable : memref<8xf32>
 	   %1 = hfusion.deinterleave %0 channel<1> : tensor<8xf32> -> tensor<4xf32>
 	   return %1 : tensor<4xf32>
 	 }
 	 
 	 func.func @main() {
 	   %c0 = arith.constant 0 : index
 	   %c1 = arith.constant 1 : index
 	   %c2 = arith.constant 2 : index
 	   %c3 = arith.constant 3 : index
 	   %c4 = arith.constant 4 : index
 	   %c5 = arith.constant 5 : index
 	   %c6 = arith.constant 6 : index
 	   %c7 = arith.constant 7 : index
 	 
 	   %m = memref.alloc() : memref<8xf32>
 	 
 	   // input: [1, 2, 3, 4, 5, 6, 7, 8]
 	   %f1 = arith.constant 1.0 : f32
 	   %f2 = arith.constant 2.0 : f32
 	   %f3 = arith.constant 3.0 : f32
 	   %f4 = arith.constant 4.0 : f32
 	   %f5 = arith.constant 5.0 : f32
 	   %f6 = arith.constant 6.0 : f32
 	   %f7 = arith.constant 7.0 : f32
 	   %f8 = arith.constant 8.0 : f32
 	   memref.store %f1, %m[%c0] : memref<8xf32>
 	   memref.store %f2, %m[%c1] : memref<8xf32>
 	   memref.store %f3, %m[%c2] : memref<8xf32>
 	   memref.store %f4, %m[%c3] : memref<8xf32>
 	   memref.store %f5, %m[%c4] : memref<8xf32>
 	   memref.store %f6, %m[%c5] : memref<8xf32>
 	   memref.store %f7, %m[%c6] : memref<8xf32>
 	   memref.store %f8, %m[%c7] : memref<8xf32>
 	 
 	   // print
 	   %in_u = memref.cast %m : memref<8xf32> to memref<*xf32>
 	   func.call @printMemrefF32(%in_u) : (memref<*xf32>) -> ()
 	 
 	   // even: 0,2,4,6 → [1, 3, 5, 7]
 	   %even_t = func.call @deinterleave_even(%m) : (memref<8xf32>) -> tensor<4xf32>
 	   %even_m = bufferization.to_memref %even_t : memref<4xf32>
 	   %even_u = memref.cast %even_m : memref<4xf32> to memref<*xf32>
 	   func.call @printMemrefF32(%even_u) : (memref<*xf32>) -> ()
 	 
 	   // odd: 1,3,5,7 → [2, 4, 6, 8]
 	   %odd_t = func.call @deinterleave_odd(%m) : (memref<8xf32>) -> tensor<4xf32>
 	   %odd_m = bufferization.to_memref %odd_t : memref<4xf32>
 	   %odd_u = memref.cast %odd_m : memref<4xf32> to memref<*xf32>
 	   func.call @printMemrefF32(%odd_u) : (memref<*xf32>) -> ()
 	 
 	   memref.dealloc %m : memref<8xf32>
 	   return
 	 }
 	 
 	 func.func private @printMemrefF32(memref<*xf32>)
 	 }
 	 // CHECK: [1,  2,  3,  4,  5,  6,  7,  8]
 	 // CHECK: [1,  3,  5,  7]
 	 // CHECK: [2,  4,  6,  8]
