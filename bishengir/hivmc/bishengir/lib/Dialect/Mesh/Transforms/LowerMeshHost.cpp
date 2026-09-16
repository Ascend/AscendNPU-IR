//===--------- LowerMeshHost.cpp - Lower mesh into proper calls --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the lowering logic for host code to clean up remaining
// bufferization operations and allocs that deal with memrefs in different
// address spaces.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Mesh/Transforms/Passes.h"
#include "bishengir/Dialect/Mesh/Util.h"

#define DEBUG_TYPE "mesh-host-lower"

namespace mlir {
#define GEN_PASS_DEF_LOWERMESHHOSTPASS
#include "bishengir/Dialect/Mesh/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using bufferization::ToMemrefOp;
using bufferization::ToTensorOp;
using hivm::AddressSpace;
using llvm::dbgs;

namespace {

struct LowerMeshHost : public impl::LowerMeshHostPassBase<LowerMeshHost> {
  using Base::Base;
  void runOnOperation() override;

private:
  LogicalResult lowerAlloc(ToMemrefOp toMemref, ToTensorOp toTensor,
                           memref::AllocOp alloc, OpBuilder &builder,
                           LLVMTypeConverter &converter);
  LogicalResult lowerCopy(ToMemrefOp toMemref, ToTensorOp toTensor,
                          memref::CopyOp ret, OpBuilder &builder,
                          LLVMTypeConverter &converter, bool hostToDevice);
  LogicalResult getBufferizeChain(ToMemrefOp toMemref, ToTensorOp &toTensor);

  // Data members
  SmallVector<std::tuple<ToMemrefOp, ToTensorOp, memref::AllocOp>>
      allocWorklist;
  DenseMap<Value, Value> gmAllocMap;
  SmallVector<std::tuple<ToMemrefOp, ToTensorOp, memref::CopyOp>>
      copyH2DWorklist;
  SmallVector<std::tuple<ToMemrefOp, ToTensorOp, memref::CopyOp>>
      copyD2HWorklist;
  DenseSet<Operation *> toErase;
};
} // namespace

static bool isGM(Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  auto as = dyn_cast<hivm::AddressSpaceAttr>(type.getMemorySpace());
  if (!as)
    return false;
  return as.getAddressSpace() == AddressSpace::GM;
}

/// Lower the alloc->toTensor->toMemref<GM> chain.
LogicalResult LowerMeshHost::lowerAlloc(ToMemrefOp toMemref,
                                        ToTensorOp toTensor,
                                        memref::AllocOp alloc,
                                        OpBuilder &builder,
                                        LLVMTypeConverter &converter) {
  // NOTE: For now we assume alloc will not have any uses apart from the cases
  // we handle in this pass. However if this is not the case, then we would need
  // to insert a memcpy to copy from device to host.

  // If memref being converted to isn't GM, then this isn't something we're
  // expecting to see.
  if (!isGM(toMemref.getMemref())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Unexpected memory space of to_tensor->to_memref->return");
    return failure();
  }

  Location loc = alloc->getLoc();
  MLIRContext *ctx = alloc->getContext();
  builder.setInsertionPoint(toMemref);
  auto targetTy = toMemref.getType();
  auto elTy = targetTy.getElementType();

  // replace the toMemref with the allocated GM memref and erase the
  // now-redundant bufferization conversion ops
  auto moduleOp = alloc->getParentOfType<ModuleOp>();
  func::FuncOp newFunc = bishengir::getCustomFunction(
      "_mlir_device_malloc", moduleOp, loc, builder, builder.getIndexType(),
      LLVM::LLVMPointerType::get(ctx));

  // Get size of memory being allocated from the static shapes
  size_t allocSize = 1;
  for (auto dim : targetTy.getShape()) {
    if (dim == ShapedType::kDynamic)
      llvm_unreachable("Currently not supporting dynamic shape");
    allocSize *= static_cast<size_t>(dim);
  }
  // Bits to bytes
  allocSize *= elTy.getIntOrFloatBitWidth() / 8;

  // Create the constant in IR
  Value size = builder.create<arith::ConstantIndexOp>(loc, allocSize);

  // Call the malloc function
  auto newOp = builder.create<func::CallOp>(loc, newFunc, size);

  // Convert the pointer into the memref we need
  auto desc = MemRefDescriptor::fromStaticShape(
      builder, loc, converter, cast<MemRefType>(targetTy), newOp->getResult(0));
  auto cast = builder.create<UnrealizedConversionCastOp>(loc, targetTy,
                                                         ValueRange{desc});

  // Replace the result of toMemref with the allocated memref
  toMemref.replaceAllUsesWith(cast.getResult(0));

  // We don't erase the alloc op here, since canonicalization should be able to
  // clean it up if it has no additional uses. If it does, then we can't safely
  // erase anyway.
  toErase.insert(toMemref);
  toErase.insert(toTensor);

  // Map new gm alloc to the old memref alloc
  LLVM_DEBUG(dbgs() << "Mapping: "; alloc->dump(); dbgs() << "\tto: ";
             cast->dump());
  gmAllocMap[alloc] = cast.getResult(0);

  return success();
}

/// Lowers memref.copy into proper calls to external functions. We should only
/// see device/host copys here.
LogicalResult LowerMeshHost::lowerCopy(ToMemrefOp toMemref, ToTensorOp toTensor,
                                       memref::CopyOp copy, OpBuilder &builder,
                                       LLVMTypeConverter &converter,
                                       bool hostToDevice) {
  LLVM_DEBUG(dbgs() << "Lowering copy op: "; copy->dump();
             dbgs() << "\thostToDevice=" << hostToDevice << '\n');
  Location loc = copy.getLoc();
  builder.setInsertionPointAfter(toMemref);
  Value dest;
  Value src;

  // Set the source and destination corresponding to the direction of movement
  if (hostToDevice) {
    src = copy.getSource();
    dest = gmAllocMap.lookup(copy.getTarget());
    if (!dest) {
      LLVM_DEBUG(dbgs() << "Unexpected memref as destination of ";
                 copy.getTarget().dump());
      return failure();
    }
    assert(isGM(dest));
  } else {
    src = toTensor.getMemref();
    dest = copy.getTarget();
    assert(isGM(src));
  }

  // Set up the function to be called. Here we want a memcpy of
  auto srcType = cast<MemRefType>(src.getType());
  auto destType = cast<MemRefType>(dest.getType());

  assert(srcType.getNumDynamicDims() == 0 && destType.getNumDynamicDims() == 0);

  auto moduleOp = copy->getParentOfType<ModuleOp>();
  auto ptrTy = LLVM::LLVMPointerType::get(copy.getContext());

  func::FuncOp func = bishengir::getCustomFunction(
      "_mlir_device_memcpy", moduleOp, loc, builder,
      {ptrTy, ptrTy, builder.getI1Type()});
  LLVM_DEBUG(dbgs() << "Converting src "; srcType.dump());
  auto srcDesc =
      MemRefDescriptor(builder
                           .create<UnrealizedConversionCastOp>(
                               loc, converter.convertType(srcType), src)
                           .getResult(0));
  LLVM_DEBUG(dbgs() << "Converting dest"; destType.dump());
  auto destDesc =
      MemRefDescriptor(builder
                           .create<UnrealizedConversionCastOp>(
                               loc, converter.convertType(destType), dest)
                           .getResult(0));

  Value srcPtr = srcDesc.allocatedPtr(builder, loc);
  Value destPtr = destDesc.allocatedPtr(builder, loc);
  Value direction = builder.create<arith::ConstantIntOp>(loc, hostToDevice, 1);
  if (hostToDevice)
    destPtr = builder.create<LLVM::BitcastOp>(loc, ptrTy, destPtr);
  else
    srcPtr = builder.create<LLVM::BitcastOp>(loc, ptrTy, srcPtr);

  builder.create<func::CallOp>(loc, func,
                               ValueRange{destPtr, srcPtr, direction});

  toErase.insert(toMemref);
  toErase.insert(toTensor);
  copy->erase();
  return success();
}

/// Obtains the ToTensorOp from the ToMemrefOp, checks for assumptions.
LogicalResult LowerMeshHost::getBufferizeChain(ToMemrefOp toMemref,
                                               ToTensorOp &toTensor) {
  toTensor = dyn_cast<ToTensorOp>(toMemref.getTensor().getDefiningOp());
  if (!toTensor || !toTensor->hasOneUse())
    return failure();
  return success();
}

/// Returns a memref of the same shape and type in the default memory space.
static MemRefType clearAddrSpace(MemRefType memref, Attribute ms) {
  // Only used for assertion check
  auto as = dyn_cast<hivm::AddressSpaceAttr>(ms);
  (void)as;
  assert(as && as.getAddressSpace() == AddressSpace::GM &&
         "Expecting only address space in host function to be GM or default");

  auto typeBuilder = MemRefType::Builder(memref);
  return typeBuilder.setMemorySpace(Attribute());
}

/// ============================================================================
/// Main runOnOperation method
/// ============================================================================
void LowerMeshHost::runOnOperation() {
  // Initialize lists
  allocWorklist.clear();
  gmAllocMap.clear();
  copyH2DWorklist.clear();
  copyD2HWorklist.clear();
  toErase.clear();

  ModuleOp op = getOperation();

  SmallVector<func::FuncOp> worklist;
  op->walk([&worklist](func::FuncOp func) { worklist.push_back(func); });

  for (auto func : worklist) {
    // Only work on host functions
    auto funcAttr = func->getAttrOfType<hacc::HACCFuncTypeAttr>(
        hacc::HACCFuncTypeAttr::name);
    if (!funcAttr || funcAttr.getFunctionKind() != hacc::HACCFuncType::HOST)
      continue;

    // Check remaining ToTensor->ToMemref ops
    auto result = func.walk([this](ToMemrefOp toMemref) {
      ToTensorOp toTensor;
      if (failed(getBufferizeChain(toMemref, toTensor))) {
        LLVM_DEBUG(llvm::dbgs() << "Unexpected usage of to_memref\n");
        return WalkResult::interrupt();
      }

      bool matched = false;
      // Case 1: We want to check for alloc's generated by one-shot-bufferize to
      // see if they are feeding into toTensor->toMemref in GM
      Operation *maybeAlloc = toTensor.getMemref().getDefiningOp();
      if (auto alloc = dyn_cast<memref::AllocOp>(maybeAlloc)) {
        allocWorklist.push_back(std::make_tuple(toMemref, toTensor, alloc));
        matched = true;
      }
      // Case 2:
      // Check for memref.copy -> toTensor -> toMemref<gm>, or
      // <gm>toTensor -> toMemref -> memref.copy, corresponding to device to
      // host and host to device respectively.
      for (auto usr : toMemref->getUsers()) {
        if (auto copy = dyn_cast<memref::CopyOp>(usr)) {
          LLVM_DEBUG(dbgs() << "d2h\n");
          copyD2HWorklist.push_back(std::make_tuple(toMemref, toTensor, copy));
          return WalkResult::advance();
        }
      }
      for (auto usr : toTensor.getMemref().getUsers()) {
        if (auto copy = dyn_cast<memref::CopyOp>(usr)) {
          LLVM_DEBUG(dbgs() << "h2d\n");
          copyH2DWorklist.push_back(std::make_tuple(toMemref, toTensor, copy));
          return WalkResult::advance();
        }
      }

      if (!matched)
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    // If there are unmatched matterns, then we missed something and should fix
    // it in this pass
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(func->getContext());
    LLVMTypeConverter typeConverter(func->getContext());
    // Since we're dealing with Host functions here, address space doesn't
    // really have any meaning. It can be converted to default, i.e. empty
    // attribute
    typeConverter.addTypeAttributeConversion(
        [](BaseMemRefType type, hivm::AddressSpaceAttr attr) {
          return Attribute();
        });

    for (auto item : allocWorklist) {
      ToMemrefOp toMemref = std::get<0>(item);
      ToTensorOp toTensor = std::get<1>(item);
      memref::AllocOp alloc = std::get<2>(item);
      if (failed(
              lowerAlloc(toMemref, toTensor, alloc, builder, typeConverter))) {
        signalPassFailure();
        return;
      }
    }
    for (auto item : copyH2DWorklist) {
      ToMemrefOp toMemref = std::get<0>(item);
      ToTensorOp toTensor = std::get<1>(item);
      memref::CopyOp copy = std::get<2>(item);
      if (failed(lowerCopy(toMemref, toTensor, copy, builder, typeConverter,
                           true))) {
        signalPassFailure();
        return;
      }
    }
    for (auto item : copyD2HWorklist) {
      ToMemrefOp toMemref = std::get<0>(item);
      ToTensorOp toTensor = std::get<1>(item);
      memref::CopyOp copy = std::get<2>(item);
      if (failed(lowerCopy(toMemref, toTensor, copy, builder, typeConverter,
                           false))) {
        signalPassFailure();
        return;
      }
    }
  }

  // This is sort of ghetto, but I'm not sure if there's a better way of doing
  // it. Registering a global TypeAttributeConversion wouldn't work since
  // it would not be able to differentiate context in host vs device
  SmallVector<Type, 16> inputs;
  SmallVector<Type, 1> outputs;
  op->walk([&inputs, &outputs](Operation *inst) {
    // If this is a func operation, then we have to change the function type
    // associated instead of the operands.
    if (auto func = dyn_cast<func::FuncOp>(inst)) {
      inputs.clear();
      outputs.clear();
      auto ft = func.getFunctionType();
      for (Type ty : ft.getInputs()) {
        auto memref = dyn_cast<MemRefType>(ty);
        if (!memref) {
          inputs.push_back(ty);
          continue;
        }
        if (Attribute ms = memref.getMemorySpace())
          inputs.push_back(clearAddrSpace(memref, ms));
        else
          inputs.push_back(memref);
      }
      for (Type ty : ft.getResults()) {
        auto memref = dyn_cast<MemRefType>(ty);
        if (!memref) {
          outputs.push_back(ty);
          continue;
        }
        if (Attribute ms = memref.getMemorySpace())
          outputs.push_back(clearAddrSpace(memref, ms));
        else
          outputs.push_back(memref);
      }
      auto newFuncTy = FunctionType::get(inst->getContext(), inputs, outputs);
      func.setFunctionType(newFuncTy);

      return;
    }

    // Otherwise we can just change the operand type directly.
    for (auto operand : inst->getOperands()) {
      auto memref = dyn_cast<MemRefType>(operand.getType());
      if (!memref)
        continue;
      if (Attribute ms = memref.getMemorySpace())
        operand.setType(clearAddrSpace(memref, ms));
    }
    for (auto result : inst->getResults()) {
      auto memref = dyn_cast<MemRefType>(result.getType());
      if (!memref)
        continue;
      if (Attribute ms = memref.getMemorySpace())
        result.setType(clearAddrSpace(memref, ms));
    }
  });
}

std::unique_ptr<Pass> mesh::createLowerMeshHostPass() {
  return std::make_unique<LowerMeshHost>();
}
