//===- AscendDPX.h - Ascend David SIMT Dialect -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ASCENDDPX_IR_ASCENDDPX_H
#define BISHENGIR_DIALECT_ASCENDDPX_IR_ASCENDDPX_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// AscendDPX Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/AscendDPX/IR/AscendDPXDialect.h.inc"

//===----------------------------------------------------------------------===//
// AscendDPX Enums
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/AscendDPX/IR/AscendDPXEnums.h.inc"

//===----------------------------------------------------------------------===//
// AscendDPX Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXAttrs.h.inc"

//===----------------------------------------------------------------------===//
// AscendDPX Dialect Operations
//===----------------------------------------------------------------------===//

// generated regbased vector operation declarations
#define GET_OP_CLASSES
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXOps.h.inc"

#endif // BISHENGIR_DIALECT_ASCENDDPX_IR_ASCENDDPX_H