//==========- MemRefExt.h - MemRefExt dialect ----------------*- C++ -*-======//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
