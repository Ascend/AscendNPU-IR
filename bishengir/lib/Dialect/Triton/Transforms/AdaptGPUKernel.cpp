//===- AdaptGPUKernel.cpp - Generate attr and wrapper for kernel -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Targets/NPUTargetSpec.h.inc"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define DEBUG_TYPE "dpx-outline-kernel-func"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_ADAPTGPUKERNEL
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

struct FuncInfo {
  LLVM::LLVMFunctionType funcType;
  StringAttr funcName;
  ArrayAttr funcArgAttr;
};

static Value castValueToType(OpBuilder &builder, Location loc, Value val,
                             Type targetType) {
  OpBuilder::InsertionGuard g(builder);
  if (!val)
    return nullptr;
  Type srcType = val.getType();
  if (srcType == targetType)
    return val;
  // int
  if (auto srcInt = dyn_cast<IntegerType>(srcType)) {
    if (auto dstInt = dyn_cast<IntegerType>(targetType)) {
      unsigned s = srcInt.getWidth();
      unsigned d = dstInt.getWidth();
      if (s == d)
        return builder.create<LLVM::BitcastOp>(loc, targetType, val);
      if (s > d)
        return builder.create<LLVM::TruncOp>(loc, targetType, val);
      return builder.create<LLVM::ZExtOp>(loc, targetType, val);
    }
  }
  // vector types: cast elementwise
  if (auto srcVec = dyn_cast<VectorType>(srcType)) {
    if (auto dstVec = dyn_cast<VectorType>(targetType)) {
      int ne = srcVec.getNumElements();
      Value out = builder.create<LLVM::UndefOp>(loc, targetType);
      for (int i = 0; i < ne; ++i) {
        Value idx = builder.create<LLVM::ConstantOp>(
            loc, builder.getI32Type(), builder.getI32IntegerAttr(i));
        Value elt = builder.create<LLVM::ExtractElementOp>(
            loc, srcVec.getElementType(), val, idx);
        Value castedElt =
            castValueToType(builder, loc, elt, dstVec.getElementType());
        out = builder.create<LLVM::InsertElementOp>(loc, targetType, out,
                                                    castedElt, idx);
      }
      return out;
    }
  }
  return builder.create<LLVM::BitcastOp>(loc, targetType, val);
}

namespace {
struct AdaptGPUKernelPass
    : public impl::AdaptGPUKernelBase<AdaptGPUKernelPass> {
  bishengir::TritonRemapOptions options;
  AdaptGPUKernelPass(bishengir::TritonRemapOptions opts) : options{opts} {}
  using GridDims = std::tuple<Value, Value, Value>;
  GridDims getGridDims(LLVM::LLVMFuncOp funcOp, OpBuilder builder,
                       Location loc) {
    OpBuilder::InsertionGuard g(builder);
    Value gridX, gridY, gridZ;
    Type int64Ty = builder.getI64Type();
    for (auto [idx, value] : llvm::enumerate(funcOp.getArguments())) {
      auto dictAttr = funcOp.getArgAttrDict(idx);
      if (!dictAttr)
        continue;

      Attribute attr = dictAttr.get(gpu::GPUBlockMappingAttr::name);
      auto blockMappingAttr = dyn_cast_or_null<gpu::GPUBlockMappingAttr>(attr);
      if (!blockMappingAttr)
        continue;

      gpu::MappingId mapping = blockMappingAttr.getBlock();
      Value gridDim = value;
      // Target type is i64, cast if needed
      if (value.getType() != int64Ty) {
        gridDim = castValueToType(builder, loc, value, int64Ty);
      }
      switch (mapping) {
      case gpu::MappingId::DimX:
        gridX = gridDim;
        break;
      case gpu::MappingId::DimY:
        gridY = gridDim;
        break;
      case gpu::MappingId::DimZ:
        gridZ = gridDim;
        break;
      default:
        llvm_unreachable("Unknown mapping");
      }
    }
    if ((!gridX || !gridY || !gridZ) || options.useGridFlag) {
      LDBG("Falling back to use compile options in remapper: gridDimX="
           << options.gridDimX << ", gridDimY=" << options.gridDimY
           << ", gridDimZ=" << options.gridDimZ);
      gridX = builder.create<LLVM::ConstantOp>(
          loc, int64Ty, builder.getI64IntegerAttr(options.gridDimX));
      gridY = builder.create<LLVM::ConstantOp>(
          loc, int64Ty, builder.getI64IntegerAttr(options.gridDimY));
      gridZ = builder.create<LLVM::ConstantOp>(
          loc, int64Ty, builder.getI64IntegerAttr(options.gridDimZ));
    }
    return {gridX, gridY, gridZ};
  }

  // Create the LaunchFuncOp to call the simt kernel
  void populateEntry(LLVM::LLVMFuncOp funcOp, MLIRContext *context,
                     OpBuilder &builder, Location loc,
                     LLVM::LLVMFuncOp wrapperFuncOp) {
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();

    // Calculate the block dim
    int64_t numWarp = -1;
    if (auto numWarpAttr = moduleOp->getAttr(triton::gpu::AttrNumWarpsName))
      if (auto intAttr = dyn_cast<IntegerAttr>(numWarpAttr))
        numWarp = intAttr.getInt();

    int64_t numThreadPerWarp = -1;
    if (auto numThreadAttr =
            moduleOp->getAttr(triton::gpu::AttrNumThreadsPerWarp))
      if (auto intAttr = dyn_cast<IntegerAttr>(numThreadAttr))
        numThreadPerWarp = intAttr.getInt();

    if (numWarp <= 0 || numThreadPerWarp <= 0)
      llvm::report_fatal_error(
          "Cannot determine num-warps or threads-per-warp! num-warps: " +
          Twine(numWarp) + ", threads-per-warp: " + Twine(numThreadPerWarp));

    int64_t vfLaunchBound = numWarp * numThreadPerWarp;
    constexpr int64_t VF_LAUNCH_BOUND_THRESHOLD = 2048;
    if (vfLaunchBound > VF_LAUNCH_BOUND_THRESHOLD)
      llvm::report_fatal_error(
          "vf LAUNCH_BOUND exceeds maximum number: " + Twine(vfLaunchBound) +
          " > " + Twine(VF_LAUNCH_BOUND_THRESHOLD));
    else if (vfLaunchBound == VF_LAUNCH_BOUND_THRESHOLD)
      funcOp.emitWarning("vfLaunchBound of " + Twine(vfLaunchBound) +
                         " may trigger too much register spilling!");

    funcOp->setAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName,
                    hivm_regbaseintrins::SIMT_EntryAttr::get(
                        funcOp->getContext(), vfLaunchBound));

    SmallVector<Value> args(wrapperFuncOp.getArguments().begin(),
                            wrapperFuncOp.getArguments().end());
    Type int64Ty = builder.getI64Type();
    // launch block sizes (threads per block)
    Value blockX = builder.create<LLVM::ConstantOp>(
        loc, int64Ty, builder.getI64IntegerAttr(vfLaunchBound));
    Value blockY = builder.create<LLVM::ConstantOp>(
        loc, int64Ty, builder.getI64IntegerAttr(1));
    Value blockZ = builder.create<LLVM::ConstantOp>(
        loc, int64Ty, builder.getI64IntegerAttr(1));

    Value newIdx = builder.create<hivm::GetBlockIdxInstrOp>(loc, int64Ty);
    auto [gridX, gridY, gridZ] = getGridDims(wrapperFuncOp, builder, loc);
    Value px;
    Value py;
    Value pz;
    if (options.isSimdSimtMixCompile) {
      // In the SIMD-SIMT mixed path, HIVMToTriton lowers the one-dimensional
      // HIVM block id to tt.get_program_id x and the VF body performs the
      // logical 1D -> 3D decomposition itself.  Pass the raw linear NPU block id
      // as program_id x so the decomposition only happens once.
      px = newIdx;
      py = builder.create<LLVM::ConstantOp>(
          loc, int64Ty, builder.getI64IntegerAttr(0));
      pz = builder.create<LLVM::ConstantOp>(
          loc, int64Ty, builder.getI64IntegerAttr(0));
    } else {
      // get grid x,y,z from linear grid id using x
      // px = pid % Gx
      // tmp = pid / Gx
      // py = tmp % Gy
      // pz = tmp / Gy
      Value tmp0 = newIdx;
      px = builder.create<LLVM::URemOp>(loc, int64Ty, tmp0, gridX);
      Value tmp1 = builder.create<LLVM::UDivOp>(loc, int64Ty, tmp0, gridX);
      py = builder.create<LLVM::URemOp>(loc, int64Ty, tmp1, gridY);
      Value tmp2 = builder.create<LLVM::UDivOp>(loc, int64Ty, tmp1, gridY);
      pz = tmp2;
    }
    auto ub = triton::util::allocateSharedMemory(wrapperFuncOp, builder, loc);

    args.push_back(gridX);
    args.push_back(gridY);
    args.push_back(gridZ);
    args.push_back(px);
    args.push_back(py);
    args.push_back(pz);
    args.push_back(ub);

    // The launch func op needs the 3D block dim (number of threads)
    // For triton, the block dim is 1D, so the y dim and the z dim are always 1
    builder.create<hivm_regbaseintrins::LaunchFuncOp>(
        funcOp->getLoc(), SymbolRefAttr::get(funcOp), blockX, blockY, blockZ,
        args);

    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }

  void createLaunchFunction(LLVM::LLVMFuncOp vf, MLIRContext *context,
                            const FuncInfo &originalFuncInfo) {
    OpBuilder builder(context);
    builder.setInsertionPoint(vf);
    auto loc = vf.getLoc();
    auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, originalFuncInfo.funcName, originalFuncInfo.funcType);
    // Forward argument attributes
    wrapperFunc.setArgAttrsAttr(originalFuncInfo.funcArgAttr);
    auto kernelAttr =
        StringAttr::get(context, hivm_regbaseintrins::kDavinciKernelAttrName);
    wrapperFunc->setAttr(kernelAttr, builder.getUnitAttr());
    auto entryAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
        hacc::HACCToLLVMIRTranslateAttr::ENTRY);
    wrapperFunc->setAttr(entryAttr, builder.getUnitAttr());
    wrapperFunc->setAttr(
        hacc::HACCFuncTypeAttr::name,
        hacc::HACCFuncTypeAttr::get(context, hacc::HACCFuncType::DEVICE));

    // Set the target device name
    ModuleOp m = vf->getParentOfType<ModuleOp>();
    hacc::TargetAttr target =
        m->getAttrOfType<hacc::TargetAttr>(hacc::TargetAttr::name);
    assert(target && "Target device not specified!");

    StringAttr targetAttr =
        StringAttr::get(context, hivm_regbaseintrins::kDavinciTargetAttrName);
    wrapperFunc->setAttr(
        targetAttr, hivm_regbaseintrins::SIMT_TargetAttr::get(
                        context, hacc::getArch(hacc::symbolizeTargetDeviceEnum(
                                     target.getTarget()))));

    Block *entryBlock = wrapperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);
    populateEntry(vf, context, builder, loc, wrapperFunc);
  }

  void convertGridDPXToArgs(LLVM::LLVMFuncOp funcOp, MLIRContext *context) {
    Block *entryBlock = &funcOp.getBody().front();
    OpBuilder builder(context);
    auto loc = funcOp.getLoc();

    auto oldType = funcOp.getFunctionType();
    SmallVector<Type, 4> argTypes(oldType.getParams().begin(),
                                  oldType.getParams().end());
    // TODO: the nctaid.x/y/z might be duplicated if the grid dims are passed in
    // as kernel parameters. Do we need to optimize this?

    // append six i64s (nctaid.x/y/z then ctaid.x/y/z)
    // always, even if present
    argTypes.append(6, builder.getI64Type());
    argTypes.push_back(LLVM::LLVMPointerType::get(context, 6));

    auto newFuncType =
        LLVM::LLVMFunctionType::get(oldType.getReturnType(), argTypes, false);
    funcOp.setType(newFuncType);
    funcOp.setArgAttr(argTypes.size() - 1, hivm::SharedMemoryAttr::name,
                      builder.getUnitAttr());

    // add arguments in the same order we appended them:
    Value DimX = entryBlock->addArgument(builder.getI64Type(), loc);
    Value DimY = entryBlock->addArgument(builder.getI64Type(), loc);
    Value DimZ = entryBlock->addArgument(builder.getI64Type(), loc);
    Value IdxX = entryBlock->addArgument(builder.getI64Type(), loc);
    Value IdxY = entryBlock->addArgument(builder.getI64Type(), loc);
    Value IdxZ = entryBlock->addArgument(builder.getI64Type(), loc);
    Value ub =
        entryBlock->addArgument(LLVM::LLVMPointerType::get(context, 6), loc);

    auto DPXOpToArg = [&](Operation *op) -> Value {
      if (dyn_cast<ascend_dpx::BlockIdxXOp>(op))
        return IdxX;
      if (dyn_cast<ascend_dpx::BlockIdxYOp>(op))
        return IdxY;
      if (dyn_cast<ascend_dpx::BlockIdxZOp>(op))
        return IdxZ;
      if (dyn_cast<ascend_dpx::GridDimXOp>(op))
        return DimX;
      if (dyn_cast<ascend_dpx::GridDimYOp>(op))
        return DimY;
      if (dyn_cast<ascend_dpx::GridDimZOp>(op))
        return DimZ;
      return nullptr;
    };

    funcOp.walk([&](Operation *dpxOp) {
      Value arg = DPXOpToArg(dpxOp);
      if (arg) {
        OpBuilder builder(dpxOp);
        Value newVal = builder.create<LLVM::TruncOp>(dpxOp->getLoc(),
                                                     builder.getI32Type(), arg);
        dpxOp->getResult(0).replaceAllUsesWith(newVal);
        dpxOp->erase();
      }
    });

    funcOp.walk([&](LLVM::AddressOfOp addrOp) {
      if (addrOp.getGlobalName() == "global_smem") {
        addrOp.getResult().replaceAllUsesWith(ub);
      } else {
        for (auto &use :
             llvm::make_early_inc_range(addrOp.getResult().getUses())) {
          use.getOwner()->erase();
        }
      }
      addrOp.erase();
    });
  }

  void pruneUnusedSIMTArgs(LLVM::LLVMFuncOp funcOp) {
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return;

    Block &entryBlock = funcOp.getBody().front();
    unsigned originalVFArgCount = entryBlock.getNumArguments();

    // find unused VF arguments
    SmallVector<int> unusedArgumentInd;
    for (const BlockArgument &bArg : entryBlock.getArguments()) {
      if (bArg.use_empty() &&
          bArg != entryBlock.getArguments().back()) { // do not prune UB!!!
        unusedArgumentInd.push_back(bArg.getArgNumber());
      }
    }
    if (unusedArgumentInd.empty()) {
      return;
    }

    // update call sites to remove operands corresponding to unused args
    auto callSites = funcOp.getSymbolUses(module);
    if (callSites.has_value()) {
      for (auto &symUse : callSites.value()) {
        Operation *userOp = symUse.getUser();
        if (!userOp)
          continue;

        SmallVector<Value> operands(userOp->operand_begin(),
                                    userOp->operand_end());
        SmallVector<Value> newOperands;

        if (auto launch = dyn_cast<hivm_regbaseintrins::LaunchFuncOp>(userOp)) {
          const unsigned launchHead = 3;
          for (unsigned i = 0;
               i < std::min<unsigned>(launchHead, operands.size()); ++i)
            newOperands.push_back(operands[i]);

          for (unsigned opIdx = launchHead; opIdx < operands.size(); ++opIdx) {
            unsigned argIndex = opIdx - launchHead;
            if (argIndex < originalVFArgCount &&
                llvm::is_contained(unusedArgumentInd, (int)argIndex)) {
              continue;
            }
            newOperands.push_back(operands[opIdx]);
          }
          userOp->setOperands(newOperands);
        } else {
          for (unsigned i = 0; i < operands.size(); ++i) {
            if (i < originalVFArgCount &&
                llvm::is_contained(unusedArgumentInd, (int)i)) {
              continue;
            }
            newOperands.push_back(operands[i]);
          }
          userOp->setOperands(newOperands);
        }
      }
    }

    // erase unused VF argument
    entryBlock.eraseArguments([&](BlockArgument bArg) {
      return llvm::is_contained(unusedArgumentInd, (int)bArg.getArgNumber());
    });

    // rebuild VF function type and argument attributes
    auto oldFuncType = funcOp.getFunctionType();
    SmallVector<Type> keptParamTypes;
    SmallVector<Attribute> keptArgAttrs;
    auto oldParamTypes = oldFuncType.getParams();
    for (unsigned i = 0; i < oldParamTypes.size(); ++i) {
      if (!llvm::is_contained(unusedArgumentInd, (int)i)) {
        keptParamTypes.push_back(oldParamTypes[i]);
        if (auto dict = funcOp.getArgAttrDict(i)) {
          keptArgAttrs.push_back(dict);
        } else {
          keptArgAttrs.push_back(DictionaryAttr::get(funcOp.getContext()));
        }
      }
    }
    auto newFuncType = LLVM::LLVMFunctionType::get(oldFuncType.getReturnType(),
                                                   keptParamTypes, false);
    funcOp.setType(newFuncType);
    funcOp.setArgAttrsAttr(ArrayAttr::get(funcOp.getContext(), keptArgAttrs));
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    SymbolTable symbolTable(m);
    auto *context = &getContext();
    context->loadDialect<hacc::HACCDialect>();

    SmallVector<Operation *> toBeRemoved;
    m.walk([&](Operation *nestedOp) {
      if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(nestedOp)) {
        // Triton by default generates nvvm.kernel to indicate a SIMT kernel
        // Functionally works, keeping the name for now.
        if (funcOp->hasAttr("nvvm.kernel")) {
          FuncInfo originalFuncInfo{funcOp.getFunctionType(),
                                    funcOp.getSymNameAttr(),
                                    funcOp.getArgAttrsAttr()};
          std::string newName =
              originalFuncInfo.funcName.str() + utils::simtVFSuffix.str();
          StringAttr newNameAttr = StringAttr::get(context, newName);
          if (failed(symbolTable.rename(funcOp, newNameAttr))) {
            funcOp->emitWarning()
                << "Failed to rename function from "
                << originalFuncInfo.funcName.str() << " to " << newName;
            return;
          }

          // Create an main scalar entry function, with a launch function op to
          // call the nvvm.kernel
          convertGridDPXToArgs(funcOp, context);
          createLaunchFunction(funcOp, context, originalFuncInfo);
          pruneUnusedSIMTArgs(funcOp);
        }
      }
      if (isa<LLVM::GlobalOp>(nestedOp)) {
        toBeRemoved.push_back(nestedOp);
      }
    });
    for (Operation *op : toBeRemoved)
      op->erase();
  }
};
} // namespace

std::unique_ptr<Pass>
bishengir::triton::createAdaptGPUKernelPass(TritonRemapOptions options) {
  return std::make_unique<AdaptGPUKernelPass>(options);
}
