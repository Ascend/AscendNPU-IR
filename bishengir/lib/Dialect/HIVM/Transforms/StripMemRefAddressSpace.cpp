//===- StripMemRefAddressSpace.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_STRIPMEMREFADDRESSSPACE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
TypeConverter createStripMemRefAddressSpaceTypeConverter() {
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });

  auto convertTypes = [&converter](ArrayRef<Type> types) {
    return llvm::map_to_vector(types, [&converter](Type type) {
      return converter.convertType(type);
    });
  };

  converter.addConversion([&convertTypes](FunctionType type) -> Type {
    return FunctionType::get(type.getContext(), convertTypes(type.getInputs()),
                             convertTypes(type.getResults()));
  });
  converter.addConversion([&convertTypes](TupleType type) -> Type {
    return TupleType::get(type.getContext(), convertTypes(type.getTypes()));
  });
  converter.addConversion([](MemRefType type) -> Type {
    if (!type.getMemorySpace())
      return type;
    return MemRefType::get(type.getShape(), type.getElementType(),
                           type.getLayout(), Attribute());
  });
  converter.addConversion([](UnrankedMemRefType type) -> Type {
    if (!type.getMemorySpace())
      return type;
    return UnrankedMemRefType::get(type.getElementType(), Attribute());
  });
  return converter;
}

LogicalResult rewriteTypesInOp(Operation *op, TypeConverter &converter) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    // Function signatures are not covered by result/block-argument rewrites
    // below, so update them explicitly before walking into the body.
    auto newFuncType = dyn_cast<FunctionType>(
        converter.convertType(funcOp.getFunctionType()));
    if (!newFuncType)
      return failure();
    funcOp.setFunctionType(newFuncType);
  }

  for (OpResult result : op->getResults()) {
    Type oldType = result.getType();
    Type newType = converter.convertType(oldType);
    if (!newType)
      return failure();
    if (newType != oldType)
      result.setType(newType);
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      auto signatureConversion = converter.convertBlockSignature(&block);
      if (!signatureConversion)
        return failure();
      for (auto [arg, newType] :
           llvm::zip_equal(block.getArguments(),
                           signatureConversion->getConvertedTypes())) {
        Type oldType = arg.getType();
        if (newType != oldType)
          arg.setType(newType);
      }

      for (Operation &nestedOp : block.getOperations())
        if (failed(rewriteTypesInOp(&nestedOp, converter)))
          return failure();
    }
  }
  return success();
}

struct StripMemRefAddressSpacePass
    : public impl::StripMemRefAddressSpaceBase<StripMemRefAddressSpacePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    TypeConverter converter = createStripMemRefAddressSpaceTypeConverter();
    if (failed(rewriteTypesInOp(module.getOperation(), converter))) {
      signalPassFailure();
      return;
    }

    SmallVector<memref::MemorySpaceCastOp> noOpCasts;
    module.walk([&](memref::MemorySpaceCastOp castOp) {
      if (castOp.getSource().getType() == castOp.getType())
        noOpCasts.push_back(castOp);
    });
    // After all address spaces are stripped, the memory_space_cast ops are
    // meaningless, so clean them up after all types have been rewritten.
    for (memref::MemorySpaceCastOp castOp : noOpCasts) {
      castOp.getResult().replaceAllUsesWith(castOp.getSource());
      castOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createStripMemRefAddressSpacePass() {
  return std::make_unique<StripMemRefAddressSpacePass>();
}
