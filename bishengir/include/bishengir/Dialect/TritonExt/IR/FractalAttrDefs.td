//===- FractalAttrDefs.td - Fractal shared encoding attrs ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FractalSharedEncodingAttr and FractalLayoutType enum
// for Ascend Cube fractal shared memory layouts (zN/nZ).
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRITONEXT_FRACTAL_ATTRDEFS
#define BISHENGIR_TRITONEXT_FRACTAL_ATTRDEFS

include "mlir/IR/EnumAttr.td"
include "bishengir/Dialect/TritonExt/IR/TritonExtDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

//===----------------------------------------------------------------------===//
// FractalLayoutType enum
//===----------------------------------------------------------------------===//

def FractalLayoutType_zN : I32EnumAttrCase<"zN", 0, "zN">;
def FractalLayoutType_nZ : I32EnumAttrCase<"nZ", 1, "nZ">;
def FractalLayoutType : I32EnumAttr<
    "FractalLayoutType", "Fractal layout ordering",
    [FractalLayoutType_zN, FractalLayoutType_nZ]> {
  let cppNamespace = "::bishengir::triton_ext";
}

//===----------------------------------------------------------------------===//
// FractalSharedEncodingAttr
//===----------------------------------------------------------------------===//

def FractalSharedEncodingAttr
    : TritonExt_Attr<"FractalSharedEncoding", "fractal_shared",
                     [SharedEncodingTrait, LayoutEncodingTrait]> {

  let description = [{
    Custom fractal shared memory layout for Huawei Ascend Cube emulation.

    Supports two layout types:
      - "zN": inner Z-shape (cols then rows), outer N-shape (M-blocks then N-blocks)
      - "nZ": inner N-shape (rows then cols), outer Z-shape (N-blocks then M-blocks)
  }];

  let parameters = (
    ins
    "int64_t":$fractalM0,
    "int64_t":$fractalN0,
    "FractalLayoutType":$layoutType,
    "::mlir::triton::gpu::CTALayoutAttr":$CTALayout
  );

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

#endif
