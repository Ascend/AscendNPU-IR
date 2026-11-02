// REQUIRES: execution-engine
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
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_950z --hfusion-normalize-ops | \
// RUN:   sed 's/^module attributes {.*} {$/module {/' | \
// RUN: bishengir-opt --execution-engine-convert-hfusion-to-upstream \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

module {
func.func @isnan_kernel(%arg0: memref<8xf32>) -> tensor<8xi1> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<8xf32>
    %1 = hfusion.isnan %0 : tensor<8xf32> -> tensor<8xi1>
    return %1 : tensor<8xi1>
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

    %v0   = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    %nan  = arith.divf %zero, %zero : f32
    %inf  = arith.constant 0x7F800000 : f32
    %ninf = arith.constant 0xFF800000 : f32
    %negative_nan = arith.constant 0xFFC00001 : f32
    %signaling_nan = arith.constant 0x7F800001 : f32
    %subnormal = arith.constant 0x00000001 : f32
    %negative_zero = arith.constant -0.0 : f32

    memref.store %v0,   %m[%c0] : memref<8xf32>
    memref.store %nan,  %m[%c1] : memref<8xf32>
    memref.store %inf,  %m[%c2] : memref<8xf32>
    memref.store %ninf, %m[%c3] : memref<8xf32>
    memref.store %negative_nan, %m[%c4] : memref<8xf32>
    memref.store %signaling_nan, %m[%c5] : memref<8xf32>
    memref.store %subnormal, %m[%c6] : memref<8xf32>
    memref.store %negative_zero, %m[%c7] : memref<8xf32>

    // 打印输入
    %unranked_in = memref.cast %m : memref<8xf32> to memref<*xf32>
    func.call @printMemrefF32(%unranked_in) : (memref<*xf32>) -> ()

    // 运行 isnan kernel
    %result_tensor = func.call @isnan_kernel(%m) : (memref<8xf32>) -> tensor<8xi1>

    // 转为 i32 memref 打印
    %res_m = bufferization.to_memref %result_tensor : memref<8xi1>
    %res_i32 = memref.alloc() : memref<8xi32>
    %r0 = memref.load %res_m[%c0] : memref<8xi1>
    %r1 = memref.load %res_m[%c1] : memref<8xi1>
    %r2 = memref.load %res_m[%c2] : memref<8xi1>
    %r3 = memref.load %res_m[%c3] : memref<8xi1>
    %r4 = memref.load %res_m[%c4] : memref<8xi1>
    %r5 = memref.load %res_m[%c5] : memref<8xi1>
    %r6 = memref.load %res_m[%c6] : memref<8xi1>
    %r7 = memref.load %res_m[%c7] : memref<8xi1>
    %b0 = arith.extui %r0 : i1 to i32
    %b1 = arith.extui %r1 : i1 to i32
    %b2 = arith.extui %r2 : i1 to i32
    %b3 = arith.extui %r3 : i1 to i32
    %b4 = arith.extui %r4 : i1 to i32
    %b5 = arith.extui %r5 : i1 to i32
    %b6 = arith.extui %r6 : i1 to i32
    %b7 = arith.extui %r7 : i1 to i32
    memref.store %b0, %res_i32[%c0] : memref<8xi32>
    memref.store %b1, %res_i32[%c1] : memref<8xi32>
    memref.store %b2, %res_i32[%c2] : memref<8xi32>
    memref.store %b3, %res_i32[%c3] : memref<8xi32>
    memref.store %b4, %res_i32[%c4] : memref<8xi32>
    memref.store %b5, %res_i32[%c5] : memref<8xi32>
    memref.store %b6, %res_i32[%c6] : memref<8xi32>
    memref.store %b7, %res_i32[%c7] : memref<8xi32>
    %unranked_res = memref.cast %res_i32 : memref<8xi32> to memref<*xi32>
    func.call @printMemrefI32(%unranked_res) : (memref<*xi32>) -> ()

    memref.dealloc %m : memref<8xf32>
    memref.dealloc %res_i32 : memref<8xi32>
    return
}

func.func private @printMemrefF32(memref<*xf32>)
func.func private @printMemrefI32(memref<*xi32>)
}
// CHECK: [1, nan, inf, -inf,{{.*}}]
// CHECK: [0, 1, 0, 0, 1, 1, 0, 0]
