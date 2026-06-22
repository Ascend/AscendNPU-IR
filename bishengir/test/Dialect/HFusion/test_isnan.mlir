// RUN: bishengir-opt %s \
// RUN:   --execution-engine-convert-hfusion-to-upstream \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%S/../../../../build/lib/libmlir_runner_utils.so \
// RUN:   -shared-libs=%S/../../../../build/lib/libmlir_c_runner_utils.so | \
// RUN: FileCheck %s

module {
func.func @isnan_kernel(%arg0: memref<4xf32>) -> tensor<4xi1> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<4xf32>
    %1 = hfusion.isnan %0 : tensor<4xf32> -> tensor<4xi1>
    return %1 : tensor<4xi1>
}

func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %m = memref.alloc() : memref<4xf32>

    %v0   = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    %nan  = arith.divf %zero, %zero : f32
    %inf  = arith.constant 0x7F800000 : f32
    %ninf = arith.constant 0xFF800000 : f32

    memref.store %v0,   %m[%c0] : memref<4xf32>
    memref.store %nan,  %m[%c1] : memref<4xf32>
    memref.store %inf,  %m[%c2] : memref<4xf32>
    memref.store %ninf, %m[%c3] : memref<4xf32>

    // 打印输入
    %unranked_in = memref.cast %m : memref<4xf32> to memref<*xf32>
    func.call @printMemrefF32(%unranked_in) : (memref<*xf32>) -> ()

    // 运行 isnan kernel
    %result_tensor = func.call @isnan_kernel(%m) : (memref<4xf32>) -> tensor<4xi1>

    // 转为 i32 memref 打印
    %res_m = bufferization.to_memref %result_tensor : memref<4xi1>
    %res_i32 = memref.alloc() : memref<4xi32>
    %r0 = memref.load %res_m[%c0] : memref<4xi1>
    %r1 = memref.load %res_m[%c1] : memref<4xi1>
    %r2 = memref.load %res_m[%c2] : memref<4xi1>
    %r3 = memref.load %res_m[%c3] : memref<4xi1>
    %b0 = arith.extui %r0 : i1 to i32
    %b1 = arith.extui %r1 : i1 to i32
    %b2 = arith.extui %r2 : i1 to i32
    %b3 = arith.extui %r3 : i1 to i32
    memref.store %b0, %res_i32[%c0] : memref<4xi32>
    memref.store %b1, %res_i32[%c1] : memref<4xi32>
    memref.store %b2, %res_i32[%c2] : memref<4xi32>
    memref.store %b3, %res_i32[%c3] : memref<4xi32>
    %unranked_res = memref.cast %res_i32 : memref<4xi32> to memref<*xi32>
    func.call @printMemrefI32(%unranked_res) : (memref<*xi32>) -> ()

    memref.dealloc %m : memref<4xf32>
    memref.dealloc %res_i32 : memref<4xi32>
    return
}

func.func private @printMemrefF32(memref<*xf32>)
func.func private @printMemrefI32(memref<*xi32>)
}
// CHECK: [1, nan, inf, -inf]
// CHECK: [0, 1, 0, 0]