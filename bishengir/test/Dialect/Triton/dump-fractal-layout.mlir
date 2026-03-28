// === zN layout tests ===
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=1536" %s | FileCheck %s --check-prefix=CHECK-ZN-1536
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=0"    %s | FileCheck %s --check-prefix=CHECK-ZN-0
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=256"  %s | FileCheck %s --check-prefix=CHECK-ZN-256

// Verify fractal zN shared memory layout for a 64x64 tensor with 16x16 fractal
// blocks (fractalM0=16, fractalN0=16).
// zN: inner Z-shape (cols first), outer N-shape (M-blocks first, column-major).
//
//   block index = block_col * outerM + block_row   (outerM = 64/16 = 4)
//
//   offset 0    -> block 0 (block_row=0, block_col=0) -> row=0,  col=0
//   offset 256  -> block 1 (block_row=1, block_col=0) -> row=16, col=0
//   offset 1536 -> block 6 (block_row=2, block_col=1) -> row=32, col=16

// CHECK-ZN-1536: fractal-layout: offset=1536 -> dim0=32 dim1=16
// CHECK-ZN-0:    fractal-layout: offset=0 -> dim0=0 dim1=0
// CHECK-ZN-256:  fractal-layout: offset=256 -> dim0=16 dim1=0

// === nZ layout tests ===
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=0"   %s | FileCheck %s --check-prefix=CHECK-NZ-0
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=1"   %s | FileCheck %s --check-prefix=CHECK-NZ-1
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=256" %s | FileCheck %s --check-prefix=CHECK-NZ-256

// Verify fractal nZ shared memory layout for a 64x64 tensor with 16x16 fractal
// blocks (fractalM0=16, fractalN0=16).
// nZ: inner N-shape (rows first), outer Z-shape (N-blocks first, row-major).
//
//   Within a block: offset increments go down rows first, then across columns.
//   Across blocks: N-blocks (column direction) vary fastest.
//
//   offset 0   -> block 0, within-block offset 0 -> row=0, col=0
//   offset 1   -> block 0, within-block offset 1 -> row=1, col=0 (next row)
//   offset 256 -> block 1 (N-blocks first): block_col=1, block_row=0
//                 -> row=0, col=16

// CHECK-NZ-0:   fractal-layout: offset=0 -> dim0=0 dim1=0
// CHECK-NZ-1:   fractal-layout: offset=1 -> dim0=1 dim1=0
// CHECK-NZ-256: fractal-layout: offset=256 -> dim0=0 dim1=16

module {}
