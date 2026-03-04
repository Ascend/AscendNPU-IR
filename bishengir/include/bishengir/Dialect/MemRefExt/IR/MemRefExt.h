//==========- MemRefExt.h - MemRefExt dialect ----------------*- C++ -*-======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXT_H_
#define BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/ShapedOpInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include <optional>

#include "mlir/Dialect/MemRef/IR/MemRef.h"

//===----------------------------------------------------------------------===//
// MemRefExt Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRefExt/IR/MemRefExtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// MemRefExt Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/MemRefExt/IR/MemRefExtOps.h.inc"

#endif // BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXT_H_
