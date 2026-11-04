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
    %c8 = arith.constant 8 : index

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
    scf.for %i = %c0 to %c8 step %c1 {
      %r = memref.load %res_m[%i] : memref<8xi1>
      %b = arith.extui %r : i1 to i32
      memref.store %b, %res_i32[%i] : memref<8xi32>
    }
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
