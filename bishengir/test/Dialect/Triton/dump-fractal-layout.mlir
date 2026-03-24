// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 probe-offset=1536" %s | FileCheck %s --check-prefix=CHECK-1536
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 probe-offset=0"    %s | FileCheck %s --check-prefix=CHECK-0
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 probe-offset=256"  %s | FileCheck %s --check-prefix=CHECK-256

// Verify fractal NZ shared memory layout for a 64x64 tensor with 16x16 fractal
// blocks (fractalM0=16, fractalN0=16).  Blocks are ordered column-major
// (M direction varies fastest):
//
//   block index = block_col * outerM + block_row   (outerM = 64/16 = 4)
//
//   offset 0    -> block 0 (block_row=0, block_col=0) -> row=0,  col=0
//   offset 256  -> block 1 (block_row=1, block_col=0) -> row=16, col=0
//   offset 1536 -> block 6 (block_row=2, block_col=1) -> row=32, col=16
//     (= byte offset 3072 for float16, 2 bytes/element)

// CHECK-1536: fractal-layout: offset=1536 -> dim0=32 dim1=16
// CHECK-0:    fractal-layout: offset=0 -> dim0=0 dim1=0
// CHECK-256:  fractal-layout: offset=256 -> dim0=16 dim1=0

module {}
