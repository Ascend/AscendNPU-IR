//===- fractal-layout-zN.mlir - zN fractal layout tests ---------*- mlir -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verify the fractal zN shared memory layout for various tensor shapes,
// fractal block sizes, and element offsets.
//
// zN layout: inner Z-shape (cols first), outer N-shape (M-blocks first,
//            column-major block ordering).
//
// Within a 16x16 fractal block (256 elements):
//   offset increments go across columns first, then down rows.
//   offset 0  -> (0,0), offset 1  -> (0,1), ..., offset 15 -> (0,15)
//   offset 16 -> (1,0), offset 17 -> (1,1), ...
//
// Across blocks (outer N-shape, M-blocks first):
//   block index = block_col * outerM + block_row
//
//===----------------------------------------------------------------------===//

// --- 64x64 tensor, 16x16 fractal blocks (4x4 outer blocks) ---
// outerM=4, outerN=4.  block_idx = block_col*4 + block_row.
//
// offset 0: block 0, within-block 0 -> (0, 0)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=0" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-64x64-OFF0
// ZN-64x64-OFF0: fractal-layout: offset=0 -> dim0=0 dim1=0

// offset 1: block 0, within-block 1 -> (0, 1)  [cols first]
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=1" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-64x64-OFF1
// ZN-64x64-OFF1: fractal-layout: offset=1 -> dim0=0 dim1=1

// offset 16: block 0, within-block 16 -> (1, 0)  [wrapped to next row]
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=16" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-64x64-OFF16
// ZN-64x64-OFF16: fractal-layout: offset=16 -> dim0=1 dim1=0

// offset 255: last element of block 0 -> (15, 15)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=255" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-64x64-OFF255
// ZN-64x64-OFF255: fractal-layout: offset=255 -> dim0=15 dim1=15

// offset 256: block 1 (block_row=1, block_col=0) -> (16, 0)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=256" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-64x64-OFF256
// ZN-64x64-OFF256: fractal-layout: offset=256 -> dim0=16 dim1=0

// offset 1024: block 4 (block_row=0, block_col=1) -> (0, 16)
// block_idx=4: block_col = 4/4 = 1, block_row = 4%4 = 0
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=1024" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-64x64-OFF1024
// ZN-64x64-OFF1024: fractal-layout: offset=1024 -> dim0=0 dim1=16

// offset 1536: block 6 (block_row=2, block_col=1) -> (32, 16)
// block_idx=6: block_col = 6/4 = 1, block_row = 6%4 = 2
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=1536" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-64x64-OFF1536
// ZN-64x64-OFF1536: fractal-layout: offset=1536 -> dim0=32 dim1=16

// offset 4095: last element of tensor -> (63, 63)
// block_idx=15 (last block): block_col=3, block_row=3 -> base (48,48)
// within-block 255 -> (15,15) -> total (63, 63)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=4095" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-64x64-OFF4095
// ZN-64x64-OFF4095: fractal-layout: offset=4095 -> dim0=63 dim1=63

// --- 32x64 tensor, 16x16 fractal blocks (outerM=2, outerN=4) ---
// This matches the original gtest shape.
// block_idx = block_col * 2 + block_row
//
// offset 0: block 0 -> (0, 0)
// RUN: bishengir-opt --dump-fractal-layout="shape=32,64 layout-type=zN probe-offset=0" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-32x64-OFF0
// ZN-32x64-OFF0: fractal-layout: offset=0 -> dim0=0 dim1=0

// offset 256: block 1 (block_row=1, block_col=0) -> (16, 0)
// RUN: bishengir-opt --dump-fractal-layout="shape=32,64 layout-type=zN probe-offset=256" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-32x64-OFF256
// ZN-32x64-OFF256: fractal-layout: offset=256 -> dim0=16 dim1=0

// offset 512: block 2 (block_row=0, block_col=1) -> (0, 16)
// block_idx=2: block_col = 2/2 = 1, block_row = 2%2 = 0
// RUN: bishengir-opt --dump-fractal-layout="shape=32,64 layout-type=zN probe-offset=512" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-32x64-OFF512
// ZN-32x64-OFF512: fractal-layout: offset=512 -> dim0=0 dim1=16

// offset 2047: last element -> (31, 63)
// RUN: bishengir-opt --dump-fractal-layout="shape=32,64 layout-type=zN probe-offset=2047" %s \
// RUN:   | FileCheck %s --check-prefix=ZN-32x64-OFF2047
// ZN-32x64-OFF2047: fractal-layout: offset=2047 -> dim0=31 dim1=63

module {}
