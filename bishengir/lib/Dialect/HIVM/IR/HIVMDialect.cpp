//===- HIVMDialect.cpp - HIVM Dialect -------------------------------------===//
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

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

//For function inliner support
#include "mlir/Transforms/InliningUtils.h"

#include <numeric>

#include "bishengir/Dialect/HIVM/IR/HIVMEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMAttrs.cpp.inc"

#include "bishengir/Dialect/HIVM/IR/HIVMDialect.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// HIVMInlinerInterface Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct HIVMInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in HIVM dialect are always legal to inline.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToRepl) const final {}
};

} // namespace

//===----------------------------------------------------------------------===//
// HIVMDialect
//===----------------------------------------------------------------------===//

void mlir::hivm::HIVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.cpp.inc"
      >();
  // uncomment when adding types
  addTypes<
#define GET_TYPEDEF_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMAttrs.cpp.inc"
      >();

  //Add function inliner interfaces
  addInterfaces<HIVMInlinerInterface>();
}
