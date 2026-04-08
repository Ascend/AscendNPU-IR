//===- fractal-layout-nZ.mlir - nZ fractal layout tests ---------*- mlir -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verify the fractal nZ shared memory layout for various tensor shapes,
// fractal block sizes, and element offsets.
//
// nZ layout: inner N-shape (rows first), outer Z-shape (N-blocks first,
//            row-major block ordering).
//
// Within a 16x16 fractal block (256 elements):
//   offset increments go down rows first, then across columns.
//   offset 0  -> (0,0), offset 1  -> (1,0), ..., offset 15 -> (15,0)
//   offset 16 -> (0,1), offset 17 -> (1,1), ...
//
// Across blocks (outer Z-shape, N-blocks first):
//   block index = block_row * outerN + block_col
//
//===----------------------------------------------------------------------===//

// --- 64x64 tensor, 16x16 fractal blocks (4x4 outer blocks) ---
// outerM=4, outerN=4.  block_idx = block_row*4 + block_col.
//
// offset 0: block 0, within-block 0 -> (0, 0)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=0" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF0
// NZ-64x64-OFF0: fractal-layout: offset=0 -> dim0=0 dim1=0

// offset 1: block 0, within-block 1 -> (1, 0)  [rows first]
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=1" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF1
// NZ-64x64-OFF1: fractal-layout: offset=1 -> dim0=1 dim1=0

// offset 16: block 0, within-block 16 -> (0, 1)  [wrapped to next col]
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=16" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF16
// NZ-64x64-OFF16: fractal-layout: offset=16 -> dim0=0 dim1=1

// offset 255: last element of block 0 -> (15, 15)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=255" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF255
// NZ-64x64-OFF255: fractal-layout: offset=255 -> dim0=15 dim1=15

// offset 256: block 1 (N-blocks first: block_col=1, block_row=0) -> (0, 16)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=256" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF256
// NZ-64x64-OFF256: fractal-layout: offset=256 -> dim0=0 dim1=16

// offset 1024: block 4 (block_col=0, block_row=1) -> (16, 0)
// block_idx=4: block_row = 4/4 = 1, block_col = 4%4 = 0
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=1024" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF1024
// NZ-64x64-OFF1024: fractal-layout: offset=1024 -> dim0=16 dim1=0

// offset 1280: block 5 (block_row=1, block_col=1) -> (16, 16)
// block_idx=5: block_row = 5/4 = 1, block_col = 5%4 = 1
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=1280" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF1280
// NZ-64x64-OFF1280: fractal-layout: offset=1280 -> dim0=16 dim1=16

// offset 4095: last element of tensor -> (63, 63)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=4095" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF4095
// NZ-64x64-OFF4095: fractal-layout: offset=4095 -> dim0=63 dim1=63

// --- 64x64 tensor, nZ: verify block boundary crossing ---
// offset 257: block 1, within-block 1 -> (1, 16)
// block_idx=1 (block_col=1, block_row=0), within (1,0)
// RUN: bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=nZ probe-offset=257" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-64x64-OFF257
// NZ-64x64-OFF257: fractal-layout: offset=257 -> dim0=1 dim1=16

// --- 32x64 tensor, 16x16 fractal blocks (outerM=2, outerN=4) ---
// This matches the original gtest shape.
// nZ outer Z-shape: block_idx = block_row * 4 + block_col
//
// offset 0: block 0 -> (0, 0)
// RUN: bishengir-opt --dump-fractal-layout="shape=32,64 layout-type=nZ probe-offset=0" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-32x64-OFF0
// NZ-32x64-OFF0: fractal-layout: offset=0 -> dim0=0 dim1=0

// offset 256: block 1 (N-blocks first: block_col=1, block_row=0) -> (0, 16)
// RUN: bishengir-opt --dump-fractal-layout="shape=32,64 layout-type=nZ probe-offset=256" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-32x64-OFF256
// NZ-32x64-OFF256: fractal-layout: offset=256 -> dim0=0 dim1=16

// offset 1024: block 4 (block_row=1, block_col=0) -> (16, 0)
// block_idx=4: block_row = 4/4 = 1, block_col = 4%4 = 0
// RUN: bishengir-opt --dump-fractal-layout="shape=32,64 layout-type=nZ probe-offset=1024" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-32x64-OFF1024
// NZ-32x64-OFF1024: fractal-layout: offset=1024 -> dim0=16 dim1=0

// offset 2047: last element -> (31, 63)
// RUN: bishengir-opt --dump-fractal-layout="shape=32,64 layout-type=nZ probe-offset=2047" %s \
// RUN:   | FileCheck %s --check-prefix=NZ-32x64-OFF2047
// NZ-32x64-OFF2047: fractal-layout: offset=2047 -> dim0=31 dim1=63

module {}
