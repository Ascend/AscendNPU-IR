//===- AVE.h - Ascend Vector Extension Dialect -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_AVE_IR_AVE_H
#define BISHENGIR_DIALECT_AVE_IR_AVE_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// AVE Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/IR/AVEDialect.h.inc"

//===----------------------------------------------------------------------===//
// AVE Enums
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/IR/AVEEnums.h.inc"

//===----------------------------------------------------------------------===//
// AVE Types
//===----------------------------------------------------------------------===//

// generated type declarations
#define GET_TYPEDEF_CLASSES
#include "bishengir/Dialect/HIVMAVE/IR/AVETypes.h.inc"

//===----------------------------------------------------------------------===//
// AVE Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVMAVE/IR/AVEAttrs.h.inc"

//===----------------------------------------------------------------------===//
// AVE Interface
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/Interfaces/AVELibraryFunctionOpInterface.h"

//===----------------------------------------------------------------------===//
// AVE Dialect Operations
//===----------------------------------------------------------------------===//

// generated regbased vector operation declarations
#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVMAVE/IR/AVEOps.h.inc"

#endif // BISHENGIR_DIALECT_AVE_IR_AVE_H