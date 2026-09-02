//===-------- FlattenModule.cpp - ModuleOp flattening Pass ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Twine.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Iterators.h"

namespace mlir {
#define GEN_PASS_DEF_FLATTENMODULE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "hivm-flatten-module"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace {
static constexpr llvm::StringLiteral kCIfacePrefix = "_mlir_ciface_";
static constexpr llvm::StringLiteral kHIVMRegbaseIntrinPrefix =
    "hivm_regbaseintrins";
static constexpr llvm::StringLiteral kSIMTModule = "hacc.simt_module";
/// Flatten all subModules so that the top moduleOp does not have nested modules
/// link functions in pairs with the same function names (private & public)
// assume we only have two-level depth
// e.g.
// module{
//   module @sub_mod0{
//   ...
//   }
//   module @sub_mod1{
//   ...
//   }
// }
struct FlattenModulePass : public impl::FlattenModuleBase<FlattenModulePass> {
public:
  void resolveRenamingConflicts(ModuleOp nested, SymbolTable &topSymTable) {
    SmallVector<Operation *, 8> nestedOps;
    for (Operation &op : nested.getBody()->getOperations()) {
      if (isa<LLVM::LLVMFuncOp>(op))
        nestedOps.push_back(&op);
    }
    for (Operation *incomingOp : nestedOps) {
      StringAttr oldName = SymbolTable::getSymbolName(incomingOp);
      Operation *existingOp = topSymTable.lookup(oldName);
      if (existingOp) {
        bool merge = satisfyFuncMergingCondition(existingOp, incomingOp);
        // resolve conflict by renaming functions in submodule
        if (!merge) {
          LLVM_DEBUG(DBGS() << "Renaming function under submodule\n";);
          StringAttr newName = findUniqueNameInTopMod(topSymTable, oldName);
          SymbolTable::setSymbolName(incomingOp, newName);
          if (failed(SymbolTable::replaceAllSymbolUses(oldName, newName,
                                                       nested))) {
            nested->emitError("Multi-level nested modules not supported");
          }
        }
      }
    }
  }

  void mergeOperation(Operation *incomingOp, Operation *existingOp,
                      SymbolTable &topSymbolTable) {
    StringAttr funcName = SymbolTable::getSymbolName(incomingOp);
    bool topIsDecl = isDeclaration(existingOp);
    bool isCIface = isCWrapper(funcName.strref());

    if (isCIface) {
      incomingOp->erase();
      LLVM_DEBUG(DBGS() << "both refer to the same template function\n");
      return;
    }
    // top module has declaration -- delete it and move its definition out of
    // the submodule
    // top module has definition -- delete its declaration in the submodule
    if (topIsDecl) {
      LLVM_DEBUG(DBGS() << "top is decl, inc is def\n");
      // special case for simt function linking
      if (hivm::util::isSIMTVF(existingOp)) {
        mergeSimtFunc(existingOp, incomingOp, topSymbolTable);
      } else {
        topSymbolTable.erase(existingOp);
        incomingOp->remove();
        topSymbolTable.insert(incomingOp);
      }
    } else {
      LLVM_DEBUG(DBGS() << "top is def, inc is decl\n");
      incomingOp->erase();
    }
  }

  void hoistContent(SymbolTable &topSymTable, ModuleOp &nested) {
    LLVM::LLVMFuncOp simtWrapper = nullptr;
    nested->walk([&](LLVM::LLVMFuncOp funcOp) {
      if (!funcOp->hasAttr(hivm_regbaseintrins::kDavinciKernelAttrName))
        return;
      simtWrapper = funcOp;
    });
    // first link simt function if found
    if (simtWrapper) {
      StringAttr symName = SymbolTable::getSymbolName(simtWrapper);

      Operation *existingOp = topSymTable.lookup(symName);
      mergeOperation(simtWrapper, existingOp, topSymTable);
    }
    // hoist the rest
    nested->walk([&](Operation *incomingOp) {
      if (!isa<LLVM::LLVMFuncOp>(incomingOp))
        return;
      StringAttr symName = SymbolTable::getSymbolName(incomingOp);

      Operation *existingOp = topSymTable.lookup(symName);
      // operation with the same name found in topMod
      if (existingOp) {
        mergeOperation(incomingOp, existingOp, topSymTable);
      } else {
        incomingOp->remove();
        topSymTable.insert(incomingOp);
      }
    });
  }
  void convertSIMTWrapper(LLVM::LLVMFuncOp funcOp) {
    if (funcOp.isDeclaration() || !hivm::util::isSIMTVF(funcOp)) {
      return;
    }
    funcOp.eraseBody();
    funcOp.setLinkage(LLVM::Linkage::External);
  }
  void runOnOperation() override {
    auto topMod = getOperation();
    SymbolTable topSymTable(topMod);
    ModuleOp mainMod = nullptr;
    SmallVector<ModuleOp, 2> subModules;

    for (auto nested : topMod.getOps<ModuleOp>()) {
      if (nested->hasAttr(kSIMTModule))
        subModules.push_back(nested);
      else {
        assert(!mainMod && "only one main module shall exist");
        mainMod = nested;
      }
    };

    flattenMainMod(topMod, mainMod);
    flattenSIMTMods(topMod, subModules);
  }

private:
  bool isCWrapper(StringRef name) { return name.starts_with(kCIfacePrefix); }
  bool isDeclaration(Operation *op) {
    auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(op);
    assert(llvmFunc && "Expect a llvm function");
    return llvmFunc.isExternal();
  }
  bool funcSignatureMatch(Operation *funcA, Operation *funcB) {
    auto fA = dyn_cast<FunctionOpInterface>(funcA);
    auto fB = dyn_cast<FunctionOpInterface>(funcB);
    if (!fA || !fB)
      return false;
    return fA.getFunctionType() == fB.getFunctionType();
  }
  bool satisfyFuncMergingCondition(Operation *existingOp,
                                   Operation *incomingOp) {
    auto funcName = SymbolTable::getSymbolName(incomingOp).strref();
    if (isCWrapper(funcName) || hivm::util::isSIMTVF(existingOp))
      return true;
    bool sameOpType = (existingOp->getName() == incomingOp->getName());
    bool sigMatch = sameOpType && funcSignatureMatch(existingOp, incomingOp);
    bool topIsDecl = isDeclaration(existingOp);
    bool incIsDecl = isDeclaration(incomingOp);
    return sigMatch && ((topIsDecl && !incIsDecl) || (!topIsDecl && incIsDecl));
  }

  StringAttr findUniqueNameInTopMod(SymbolTable &topSymTable,
                                    StringAttr baseName) {
    if (!topSymTable.lookup(baseName))
      return baseName;
    int counter = 0;
    while (true) {
      SmallString<32> nameBuf;
      (Twine(baseName) + "_" + Twine(counter++)).toVector(nameBuf);
      if (!topSymTable.lookup(nameBuf)) {
        return StringAttr::get(baseName.getContext(), nameBuf);
      }
    }
  }

  void mergeSimtFunc(Operation *declOp, Operation *defOp,
                     SymbolTable &topSymbolTable) {
    auto llvmDefFuncOp = dyn_cast<LLVM::LLVMFuncOp>(defOp);
    auto llvmDeclFuncOp = dyn_cast<LLVM::LLVMFuncOp>(declOp);
    if (!llvmDefFuncOp || !llvmDeclFuncOp)
      return;
    LLVM::LLVMFuncOp simtEntry = nullptr;
    hivm_regbaseintrins::LaunchFuncOp launchFunc;
    llvmDefFuncOp->walk([&](hivm_regbaseintrins::LaunchFuncOp launchFuncOp) {
      launchFunc = launchFuncOp;
      auto simtFuncSymAttr = launchFuncOp.getKernelAttr();
      simtEntry = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
          launchFuncOp, simtFuncSymAttr);
    });
    assert(simtEntry);
    // need certain hivm_regbaseintrins attribute to attach metadata
    for (auto attr : defOp->getAttrs()) {
      StringRef attrName = attr.getName().getValue();
      if (!attrName.starts_with(kHIVMRegbaseIntrinPrefix))
        continue;
      simtEntry->setAttr(attr.getName(), attr.getValue());
    }
    // detach the definition because we still need some operations inside it
    SymbolTable simtSymTable(defOp->getParentOfType<ModuleOp>());

    auto callableIface = dyn_cast<CallableOpInterface>(defOp);
    if (!callableIface)
      return;

    Region *simtRegion = callableIface.getCallableRegion();
    Block &entryBlock = llvmDefFuncOp.getBody().front();
    /// Just preserve used args.
    SmallVector<unsigned> usedIndices;
    for (auto &block : *simtRegion) {
      for (auto &op : block) {
        for (Value operand : op.getOperands()) {
          if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
            if (blockArg.getOwner() == &entryBlock) {
              usedIndices.push_back(blockArg.getArgNumber());
            }
          }
        }
      }
    }
    llvm::sort(usedIndices);
    usedIndices.erase(std::unique(usedIndices.begin(), usedIndices.end()),
                      usedIndices.end());

    OpBuilder builder(simtRegion->getContext());
    Block &origEntry = simtRegion->front();

    auto callSites = SymbolTable::getSymbolUses(declOp, topSymbolTable.getOp());

    if (callSites.has_value()) {
      for (auto callSite : callSites.value()) {
        auto call = cast<LLVM::CallOp>(callSite.getUser());

        builder.setInsertionPoint(call);

        IRMapping mapping;
        for (auto idx : usedIndices) {
          mapping.map(origEntry.getArgument(idx), call->getOperand(idx));
        }
        for (Operation &op : origEntry.getOperations()) {
          if (op.hasTrait<OpTrait::IsTerminator>()) {
            continue;
          }
          builder.clone(op, mapping);
        }

        call.erase();
      }
    }

    StringRef simtVfName = SymbolTable::getSymbolName(simtEntry);
    topSymbolTable.erase(declOp);
    simtSymTable.remove(defOp);
    simtEntry->remove();

    if (topSymbolTable.lookup(simtVfName)) {
      topSymbolTable.getOp()->emitError("Simt_VF's name is not unique");
      return;
    }
    topSymbolTable.insert(simtEntry);
    simtEntry->setAttr("noinline", builder.getBoolAttr(false));
    simtEntry->removeAttr("hivm_regbaseintrins.kernel");

    defOp->erase();
  }

  void moveGlobals(ModuleOp curMod, ModuleOp topMod) {
    SymbolTable parentTable(topMod);

    SmallVector<LLVM::GlobalOp, 2> globals;
    for (auto global : curMod.getOps<LLVM::GlobalOp>()) {
      globals.push_back(global);
    }

    for (auto global : globals) {
      auto symName = global.getSymName();
      // keep one copy of globalOp of the same name
      if (parentTable.lookup(symName)) {
        global->erase();
      } else {
        global->moveBefore(topMod.getBody(), topMod.getBody()->begin());
        parentTable.insert(global);
      }
    }
  }

  // The main module goes first so that there should not be any naming
  // conflicts. This converts empty SIMT wrapper to external call, which gets
  // merged later when we flatten simt modules. Copying module-level
  // attributes to ensure correct device binary lowering.
  void flattenMainMod(ModuleOp topMod, ModuleOp mainMod) {
    if (!mainMod)
      return;
    SymbolTable topSymTable(topMod);
    mainMod->walk([&](LLVM::LLVMFuncOp funcOp) { convertSIMTWrapper(funcOp); });
    hoistContent(topSymTable, mainMod);
    // Copy all module-level attributes
    for (auto attr : mainMod->getAttrs()) {
      topMod->setAttr(attr.getName(), attr.getValue());
    }
    // TODO: handle other symbolOps
    moveGlobals(mainMod, topMod);

    if (mainMod.getBody()->empty()) {
      mainMod->erase();
    }
  }

  // For SIMT modules, we perform a function naming check in case that any
  // helper functions share the same name (unlikely if properly handled
  // early). Linking the simt wrapper with the actual simt function (i.e. not
  // the wrapper that calls simt func inside simt module), and inserting
  // simt_vf setup before it is called.
  void flattenSIMTMods(ModuleOp topMod,
                       const SmallVector<ModuleOp, 2> &simtMods) {
    SymbolTable topSymTable(topMod);

    for (auto simtMod : simtMods) {

      // first rename functions that cannot be merged
      resolveRenamingConflicts(simtMod, topSymTable);

      // Move ops up, merging definitions where appropriate
      hoistContent(topSymTable, simtMod);

      // handle other symbolOps
      moveGlobals(simtMod, topMod);

      // cleanup empty module
      if (simtMod.getBody()->empty()) {
        simtMod->erase();
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createFlattenModulePass() {
  return std::make_unique<FlattenModulePass>();
}
