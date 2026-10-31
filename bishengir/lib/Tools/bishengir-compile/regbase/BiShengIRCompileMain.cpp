//===- BiShengIRCompileMain.cpp - RegBase compile orchestration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RegBase (A5) pipeline orchestration with retry/fallback logic.
// Migrated from AscendNPU-IR-Dev BishengIRCompileMain.cpp.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/Utils/AuxBitcodePaths.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/regbase/PassPipeline.h"
#include "bishengir/Tools/bishengir-compile/regbase/Utility.h"
#include "bishengir/Tools/bishengir-compile/regbase/VectorizationFailure.h"
#include "bishengir/Tools/hivmc/BishengAsyncRuntime.h"
#include "bishengir/Tools/hivmc/Config.h"
#include "bishengir/Tools/hivmc/HIVMC.h"
#include "bishengir/Tools/hivmc/HIVMCA5.h"
#include "bishengir/Tools/hivmc/Utility.h"
#include "bishengir/Version/Version.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VersionTuple.h"
#include <functional>
#include <set>

#define DEBUG_TYPE "bishengir-compile-regbase"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

using PipelineBuilder = std::function<void(mlir::PassManager &,
                                           const BiShengIRCompileMainConfig &)>;

enum class CompileFlow {
  Affinity,
  Mixed,
  PureSimt,
  Simd,
};

static bool isShapedElementwise(Operation *op) {
  if (!op->hasTrait<OpTrait::Elementwise>())
    return false;
  auto isShaped = [](mlir::Type type) { return isa<ShapedType>(type); };
  return llvm::any_of(op->getOperandTypes(), isShaped) ||
         llvm::any_of(op->getResultTypes(), isShaped);
}

static bool isDataValue(mlir::Value value) {
  return !isa<IndexType>(value.getType());
}

/// Return true if an external output has a data source other than a reduction
/// or an initializer. Compute operations have already been checked globally,
/// so this traversal only needs to follow structural and control-flow values.
static bool hasNonReductionSource(mlir::Value value,
                                  llvm::DenseSet<mlir::Value> &visiting) {
  if (!isDataValue(value))
    return false;
  if (!visiting.insert(value).second)
    return false;

  auto finish = [&](bool result) {
    visiting.erase(value);
    return result;
  };

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(owner->getParentOp())) {
      unsigned argNumber = blockArg.getArgNumber();
      if (argNumber > 0 && argNumber <= forOp.getInitArgs().size())
        return finish(hasNonReductionSource(forOp.getInitArgs()[argNumber - 1],
                                            visiting));
    }
    return finish(true);
  }

  Operation *producer = value.getDefiningOp();
  if (!producer)
    return finish(true);
  if (isa<linalg::ReduceOp>(producer))
    return finish(false);
  if (isa<linalg::FillOp, tensor::EmptyOp>(producer) ||
      producer->hasTrait<OpTrait::ConstantLike>())
    return finish(false);

  if (auto forOp = dyn_cast<scf::ForOp>(producer)) {
    unsigned resultNumber = cast<OpResult>(value).getResultNumber();
    bool hasInvalidInit =
        hasNonReductionSource(forOp.getInitArgs()[resultNumber], visiting);
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    bool hasInvalidYield =
        hasNonReductionSource(yield.getOperand(resultNumber), visiting);
    return finish(hasInvalidInit || hasInvalidYield);
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(producer)) {
    unsigned resultNumber = cast<OpResult>(value).getResultNumber();
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      if (region->empty())
        continue;
      auto yield = cast<scf::YieldOp>(region->front().getTerminator());
      if (hasNonReductionSource(yield.getOperand(resultNumber), visiting))
        return finish(true);
    }
    return finish(false);
  }

  bool hasDataOperand = false;
  for (mlir::Value operand : producer->getOperands()) {
    if (!isDataValue(operand))
      continue;
    hasDataOperand = true;
    if (hasNonReductionSource(operand, visiting))
      return finish(true);
  }
  return finish(!hasDataOperand);
}

/// Return true if a memref is a function argument or a view derived from one.
/// Such a value is externally visible and a copy into it is an output sink.
static bool isFunctionArgumentBacked(mlir::Value value,
                                     llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return isa<func::FuncOp>(blockArg.getOwner()->getParentOp());

  Operation *producer = value.getDefiningOp();
  if (!producer || producer->getDialect()->getNamespace() !=
                       memref::MemRefDialect::getDialectNamespace())
    return false;
  return llvm::any_of(producer->getOperands(), [&](mlir::Value operand) {
    return isa<MemRefType>(operand.getType()) &&
           isFunctionArgumentBacked(operand, visited);
  });
}

/// Check the module-wide compute-op whitelist. Structural, control-flow,
/// initialization and cast operations are auxiliary; the only payload compute
/// operation allowed outside a reduction region is linalg.reduce itself.
static bool hasOnlyReductionComputeOps(ModuleOp module) {
  bool hasReduction = false;
  Operation *unsupportedComputeOp = nullptr;
  module.walk([&](Operation *op) {
    if (unsupportedComputeOp)
      return WalkResult::interrupt();
    if (isa<linalg::ReduceOp>(op)) {
      hasReduction = true;
      return WalkResult::advance();
    }
    // The scalar combiner is part of the reduction itself, not an independent
    // module-level payload computation.
    if (op->getParentOfType<linalg::ReduceOp>())
      return WalkResult::advance();
    if (isa<linalg::FillOp>(op) || op->hasTrait<OpTrait::ConstantLike>() ||
        isa<CastOpInterface>(op))
      return WalkResult::advance();
    if ((isa<linalg::LinalgOp>(op) || isShapedElementwise(op))) {
      unsupportedComputeOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (unsupportedComputeOp) {
    LDBG("Not reduce-only: module contains a non-reduction compute op: "
         << *unsupportedComputeOp);
    return false;
  }
  return hasReduction;
}

/// Check that no externally visible output bypasses the allowed reductions.
static bool hasOnlyReductionOutputs(ModuleOp module) {
  bool hasInvalidOutput = false;
  module.walk([&](Operation *op) {
    if (hasInvalidOutput)
      return WalkResult::interrupt();

    SmallVector<mlir::Value> outputValues;
    if (auto returnOp = dyn_cast<func::ReturnOp>(op))
      llvm::append_range(outputValues, returnOp.getOperands());
    if (auto materialize =
            dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
      llvm::DenseSet<mlir::Value> visited;
      if (isa<MemRefType>(materialize.getDest().getType()) &&
          isFunctionArgumentBacked(materialize.getDest(), visited))
        outputValues.push_back(materialize.getSource());
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      llvm::DenseSet<mlir::Value> visited;
      if (isFunctionArgumentBacked(copy.getTarget(), visited))
        outputValues.push_back(copy.getSource());
    }

    for (mlir::Value output : outputValues) {
      if (!isDataValue(output))
        continue;
      llvm::DenseSet<mlir::Value> visiting;
      if (hasNonReductionSource(output, visiting)) {
        LDBG("Not reduce-only: external output has a non-reduction data "
             "contributor: "
             << *op);
        hasInvalidOutput = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return !hasInvalidOutput;
}

static bool isReduceOnlyModule(ModuleOp module) {
  return hasOnlyReductionComputeOps(module) && hasOnlyReductionOutputs(module);
}

CompileFlow getCompileFlow(const BiShengIRCompileMainConfig &config) {
  if (config.getEnableTlaMode())
    return CompileFlow::Affinity;
  if (config.getEnableSimdSimtMixCompile())
    return CompileFlow::Mixed;
  if (config.getPureSimt())
    return CompileFlow::PureSimt;
  return CompileFlow::Simd;
}

/// Get the lib directory path (../lib relative to bishengir-compile
/// executable). Returns canonical absolute path without ".." or ".".
static std::string getLibDirFromExecutable(StringRef executablePath) {
  if (executablePath.empty() ||
      (!executablePath.contains('/') && !executablePath.contains('\\')))
    return "";
  llvm::SmallString<256> absPath(executablePath);
  if (llvm::sys::fs::make_absolute(absPath))
    return "";
  llvm::SmallString<256> realPath;
  if (!llvm::sys::fs::real_path(absPath, realPath))
    absPath = realPath;
  llvm::sys::path::remove_filename(absPath);
  llvm::sys::path::append(absPath, "..", "lib");
  llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
  return std::string(absPath.str());
}

/// Add bitcode path attributes to ModuleOp from ../lib/*.bc files.
/// Paths are canonical (no ".." or ".") before being stored in attributes.
static void addBitcodeAttrsToModule(ModuleOp module, StringRef executablePath,
                                    const BiShengIRCompileMainConfig &config) {
  std::string libDir = getLibDirFromExecutable(executablePath);
  MLIRContext *ctx = module->getContext();
  ctx->loadDialect<mlir::hivm::HIVMDialect>();

  auto addIfExists = [&](const char *filename, llvm::StringRef attrName,
                         auto createAttr) {
    llvm::SmallString<256> bcPath(libDir);
    llvm::sys::path::append(bcPath, filename);
    if (!llvm::sys::fs::exists(bcPath))
      return;
    llvm::SmallString<256> canonicalPath;
    if (llvm::sys::fs::real_path(bcPath, canonicalPath))
      return;
    module->setAttr(
        attrName,
        createAttr(ctx, mlir::StringAttr::get(ctx, canonicalPath.str().str())));
  };

  // Template bitcode is installed with an optimization-level suffix
  // (e.g. meta_op.aic.c220.O2.bc); pick the variant with priority: explicit
  // --enable-optimized-metaop > CANN version (>= 9.2.0 uses O2) > default O2.
  std::string optLevel = resolveTemplateBitcodeOptLevel(config);
  auto bitcodeFileName = [&](StringRef base) -> std::string {
    return (base + "." + optLevel + ".bc").str();
  };

  if (mlir::hacc::utils::isAscend950(config.getTarget())) {
    addIfExists(bitcodeFileName("meta_op.aic.c310").c_str(),
                mlir::hivm::AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIC_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.aiv.c310").c_str(),
                mlir::hivm::AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIV_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.mix.aic.c310").c_str(),
                mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.mix.aiv.c310").c_str(),
                mlir::hivm::MIX_AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
                });
  } else if (mlir::hacc::utils::isAscend960(config.getTarget())) {
    addIfExists(bitcodeFileName("meta_op.aic.920r1").c_str(),
                mlir::hivm::AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIC_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.aiv.920r1").c_str(),
                mlir::hivm::AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIV_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.mix.aic.920r1").c_str(),
                mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.mix.aiv.920r1").c_str(),
                mlir::hivm::MIX_AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
                });
  } else {
    addIfExists(bitcodeFileName("meta_op.aic.c220").c_str(),
                mlir::hivm::AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIC_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.aiv.c220").c_str(),
                mlir::hivm::AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIV_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.mix.aic.c220").c_str(),
                mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists(bitcodeFileName("meta_op.mix.aiv.c220").c_str(),
                mlir::hivm::MIX_AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
                });
  }
  addIfExists(bitcodeFileName("host").c_str(),
              mlir::hivm::HOST_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::HOST_BITCODEAttr::get(c, s);
              });
}

//===--------------------------------------------------------------------===//
// Pure-path base aux resolver (A5, target-dependent subcoreTag)
//===--------------------------------------------------------------------===//
//
// These helpers run *before* the first RegBase PassManager (in the for-loop
// below), so they must not read Module Attrs.  They depend solely on `config`
// (CLI), `executablePath`, and the filesystem.

/// Sub-core bitcode tag for the regbase target.  Three branches matching
/// `addBitcodeAttrsToModule` above (c310 / 920r1 / c220).  Dev's original
/// helper only distinguished c310 vs c220; this corrects that gap so the
/// early pure-path calculator stays byte-consistent with the Attr writes for
/// Ascend960 targets.
static StringRef
auxSubcoreTagForTarget(const BiShengIRCompileMainConfig &config) {
  if (mlir::hacc::utils::isAscend950(config.getTarget()))
    return "c310";
  if (mlir::hacc::utils::isAscend960(config.getTarget()))
    return "920r1";
  return "c220";
}

/// Resolve base aux paths via the pure path calculator (no MLIR Attr read).
static std::map<SubCoreTarget, std::string>
resolveAuxBitcodePathsForConfig(const BiShengIRCompileMainConfig &config) {
  return bishengir::toSubCoreTargetPaths(
      bishengir::auxbitcode::resolveAuxBitcodePaths(
          config.getExecutablePath(), auxSubcoreTagForTarget(config),
          resolveTemplateBitcodeOptLevel(config)));
}

/// Per-lane distinct base aux path list for A5.  Base aux is the set of aux
/// files needed by EVERY scenario (PureVV / PureCube / MixCV) on that lane,
/// so it can be preloaded BEFORE the topology is known.  Conditionally-
/// needed aux (mix BC, which only MixCV uses) is deferred to Final Delta,
/// submitted after the topology is known — this prevents a PureVV compile
/// from failing PreloadBase on a corrupt/missing mix BC it never actually
/// consumes.  A5 `compileDeviceKernel` builds `auxPaths` as:
/// appendBishengOptions aux -> extraDeviceBCPaths -> metaOp -> mixAux.
/// The distinct BASE preload set per lane is:
///   AIC: aic + extraDeviceBCPaths + appendBishengOptions aux
///   AIV: aiv + extraDeviceBCPaths + appendBishengOptions aux
/// mixAic/mixAiv are deferred to Final Delta (MixCV-only).
/// Host BC is *not* preloaded (host never enters device worker auxPaths).
struct BaseAuxByLane {
  std::vector<std::string> aic;
  std::vector<std::string> aiv;
};

static BaseAuxByLane
splitBaseAuxByLane(const std::map<SubCoreTarget, std::string> &paths,
                   const BiShengIRCompileMainConfig &config) {
  BaseAuxByLane out;
  auto addIf = [&](SubCoreTarget t, std::vector<std::string> &dst) {
    auto it = paths.find(t);
    if (it != paths.end() && !it->second.empty())
      dst.push_back(it->second);
  };
  // Base aux: only the per-lane metaOp (aic/aiv).  Mix BC (mixAic/mixAiv) is
  // deferred to Final Delta so PureVV/PureCube never preload (and never fail
  // on) a mix BC they do not consume.
  addIf(SubCoreTarget::AIC, out.aic);
  addIf(SubCoreTarget::AIV, out.aiv);
  // Extra device BC paths (--cce-link-aicore-ll-module) shared by both lanes.
  for (const auto &bc : config.getLinkAicoreBitcode()) {
    out.aic.push_back(bc);
    out.aiv.push_back(bc);
  }
  // A5-only: appendBishengOptions aux.  Apply the same parsing rule as
  // `parseAppendBishengOptionsAuxFromTokens` in HIVMCMainA5.cpp (scan tokens
  // for `-cce-link-aicore-ll-module`, take the following token as a path).
  // This runs in the regbase translation unit and does not share tokens with
  // the A5 file, so the scan is duplicated here; the rule must stay in sync.
  // These aux paths apply to both lanes (they appear in auxPaths for both
  // AIC and AIV compiles).
  const std::string appendOpts = config.getAppendBishengOptions();
  if (!appendOpts.empty()) {
    std::vector<std::string> tokens;
    splitString(appendOpts, " ", tokens);
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
      if (tokens[i] == "-cce-link-aicore-ll-module") {
        out.aic.push_back(tokens[i + 1]);
        out.aiv.push_back(tokens[i + 1]);
      }
    }
  }
  // HOST intentionally excluded: not preloaded to device workers.
  return out;
}

using MixedModules = std::pair<ModuleOp, SmallVector<ModuleOp, 2>>;

/// Run a pipeline on a module using the regbase builder and A3's runPipeline
/// utility.
static LogicalResult runPipelineRegBase(ModuleOp mod,
                                        const PipelineBuilder &buildPipeline,
                                        BiShengIRCompileMainConfig &config,
                                        const std::string &pipelineName) {
  // Bind the config parameter into the builder to match A3's runPipeline
  // signature.
  auto boundPipeline =
      std::bind(buildPipeline, std::placeholders::_1, std::cref(config));
  return bishengir::runPipeline(mod, boundPipeline, config, pipelineName);
}

//   Scope::ScopeOp SIMT detection logic needed.
static MixedModules getMixedModules(ModuleOp topMod) {
  MixedModules res;
  res.first = nullptr;
  for (auto subMod : topMod.getOps<ModuleOp>()) {
    if (subMod->hasAttr(hacc::SIMTModuleAttr::name)) {
      res.second.push_back(subMod);
    } else {
      assert(!res.first && "only one main module is allowed");
      res.first = subMod;
    }
  };
  // if no main module, return the top module
  if (!res.first)
    res.first = topMod;

  return res;
}

static bool runModulePipeline(ModuleOp module, const PipelineBuilder &builder,
                              BiShengIRCompileMainConfig &config,
                              StringRef pipelineName) {
  return succeeded(
      runPipelineRegBase(module, builder, config, pipelineName.str()));
}

static bool runModulePipelines(ArrayRef<ModuleOp> modules,
                               const PipelineBuilder &builder,
                               BiShengIRCompileMainConfig &config,
                               StringRef pipelineName) {
  bool success = true;
  for (auto module : modules) {
    // Stop this pipeline on each module after earlier pipeline fails.
    success =
        success && runModulePipeline(module, builder, config, pipelineName);
  }
  return success;
}

static bool runMixedPipelines(ModuleOp mixedModule,
                              BiShengIRCompileMainConfig &config) {
  if (!runModulePipeline(mixedModule, regbase::buildBiShengHIRPipeline, config,
                         "BiShengHIR")) {
    return false;
  }

  auto [mainMod, simtMods] = getMixedModules(mixedModule);
  if (!runModulePipelines(simtMods, regbase::buildBiShengTTIRPipeline, config,
                          "BiShengSIMT")) {
    return false;
  }

  if (!runModulePipeline(mixedModule, regbase::buildBiShengHIRFinishPipeline,
                         config, "BishengHIR")) {
    return false;
  }

  if (!runModulePipeline(mainMod, regbase::buildFinalHIVMPipelines, config,
                         "buildFinalHIVMPipelines")) {
    return false;
  }
  return runModulePipeline(mainMod, regbase::buildBiShengHIRAVEToLLVMPipeline,
                           config, "BiShengSIMD");
}

static bool runAffinityPipelines(ModuleOp affinityModule,
                                 BiShengIRCompileMainConfig &config) {
  return runModulePipeline(affinityModule, regbase::buildAffinityPipeline,
                           config, "AffinityHIR");
}

LogicalResult handleSaveTemps(ModuleOp module,
                              BiShengIRCompileMainConfig &config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.opt.mlir";
  std::string outputFile = config.getOutputFile();
  std::unique_ptr<llvm::ToolOutputFile> inputFileHandler;

  // Handle --save-temps=<directory> option to store module.hivm.opt.mlir
  if (!config.getSaveTemps().empty()) {
    llvm::SmallString<256> saveTempsDir(config.getSaveTemps());
    if (llvm::sys::fs::make_absolute(saveTempsDir)) {
      llvm::errs() << "[ERROR] Failed to get absolute path for save-temps.\n";
      return failure();
    }
    if (!llvm::sys::fs::exists(saveTempsDir))
      if (auto ec = llvm::sys::fs::create_directories(saveTempsDir)) {
        llvm::errs() << "[ERROR] Failed to create save-temps directory: "
                     << saveTempsDir << "\n";
        return failure();
      }
    llvm::sys::path::append(saveTempsDir, inputFile);
    std::string errorMessage;
    inputFileHandler = mlir::openOutputFile(saveTempsDir, &errorMessage);
    if (!inputFileHandler) {
      llvm::errs() << "[ERROR] Failed to open save-temps file: " << errorMessage
                   << "\n";
      return failure();
    }
    inputFileHandler->keep();
  } else {
    inputFileHandler = getTempFile(inputFile, tempDirsStore);
    if (!inputFileHandler) {
      llvm::dbgs() << "[ERROR] Failed to create temporary input file needed to "
                      "run hivmc compile.\n";
      return failure();
    }
  }
  inputFile = inputFileHandler->outputFilename();

  module.print(inputFileHandler->os(),
               mlir::OpPrintingFlags().enableDebugInfo(
                   config.getEnableSanitizer() || config.getEnableDebugInfo()));
  inputFileHandler->os().flush();
  return success();
}

HIVMCMainConfig HIVMCFromBiShengIRConfig(BiShengIRCompileMainConfig &config,
                                        BishengAsyncRuntime *runtime) {
  HIVMCMainConfig hivmcConfig;

  /// TODO: add flag difference logic for hivmc and bishengir flags in
  /// Options.td These flags are set manually due to mismatch in
  /// HIVMCConfigCLOptions constructor and Options.td
  ///
  /// Options.td is used both by bishengir pipeline and hivmc pipeline,
  /// So the flags which are both in bishengir pipeline and hivmc pipeline are
  /// just copied from the BiShengIRCompileMainConfig through dispatch table
  /// (which is generated in CompileOptionsGen.cpp)
  /// P.S : the same problem is also on A3 pipeline, check the translation there

  /// A5-specific
  hivmcConfig.autoVectorizeV2(false);
  hivmcConfig.targetBackend(config.getTarget());
  ///

  hivmcConfig.limitAutoMultiBufferForLocalBuffer(true);
  hivmcConfig.deterministicComputing(true);
  hivmcConfig.simtOptimizationMode(1900101);
  hivmcConfig.enableAutoCVBalance(true);
  hivmcConfig.setUseDPX(true);
  hivmcConfig.onlyRunHIVMPipeline(false);

  hivmcConfig.appendBishengOptions(config.getAppendBishengOptions());
  hivmcConfig.compileTriton(config.getEnableTritonKernelCompile());
  hivmcConfig.compileTritonDialect(config.getEnableTritonIRCompile());
  hivmcConfig.enableSimdSimtMixCompile(config.getEnableSimdSimtMixCompile());
  hivmcConfig.enableSIMTOnly(config.getPureSimt());
  hivmcConfig.enableSanitizer(config.getEnableSanitizer());
  hivmcConfig.enableSIMTFastDiv(config.getEnableSIMTFastDiv());
  hivmcConfig.enableDebugVariables(config.getEnableDebugVariables());
  hivmcConfig.enableDebugInfo(config.getEnableDebugInfo());
  hivmcConfig.saveTemps(config.getSaveTemps());
  hivmcConfig.injectBarrierAllSync(config.getEnableHIVMInjectBarrierAllSync());
  hivmcConfig.setExtraDeviceBCPaths(config.getLinkAicoreBitcode());
  hivmcConfig.setDisableFMA(config.getDisableFMA());
  hivmcConfig.setSaveLinkedIR(config.getSaveLinkedIR());
  hivmcConfig.setNumWarps(config.getNumWarps());
  hivmcConfig.setThreadsPerWarp(config.getThreadsPerWarp());
  hivmcConfig.setSharedDynamicSize(config.getSharedMemDynamicSize());
  hivmcConfig.simtStackLimit(config.getSimtStackLimit());
  hivmcConfig.tritonMetadataOutput(config.getTritonMetadataOutput());
  hivmcConfig.disableDecomposeReduction(config.getDisableDecomposeReduction());
  hivmcConfig.disableReorderInstruction(config.getDisableReorderInstruction());

  if (hivmcConfig.getTritonGridDim().size() > 3) {
    report_fatal_error(
        "Invalid --simt-triton-grid: at most 3 elements allowed x,y,z\n");
  }
  hivmcConfig.setOutputFile(config.getOutputFile());

  StringTmpPath path(hivmcConfig.outputFile());

  // TODO: investigate if this check is redundant
  llvm::cantFail(llvm::errorCodeToError(hivmcCanonicalizePath(path)));
  hivmcConfig.setOutputFile(path.str().str());
  hivmcConfig.setRuntime(runtime);
  return hivmcConfig;
}

} // namespace

LogicalResult
bishengir::regbase::runRegBasePipeline(ModuleOp mod,
                                       BiShengIRCompileMainConfig config,
                                       BishengAsyncRuntime *runtime) {
  if (failed(checkInOutOptionsValidity(config))) {
    return failure();
  }

  if (isReduceOnlyModule(mod)) {
    LDBG("Detected reduce-only data flow; forcing TreeReduceV2 mode=all with "
         "Ascend-style RA");
    config.setEnableTreeReduceV2(true);
    config.setEnableTreeReduceMode(hfusion::TreeReduceMode::All);
    config.setUseAscendStyleRA(true);
  }

  // ---- LoadInit submission (before the first RegBase Pass, no wait) ----
  // submitLoadInit submits the LoadInit command (dlmopen + dlsym +
  // bishengLlcInit) only; it does NOT preload any aux path.  LoadInit is
  // independent of aux paths (the SO path is frozen by the runtime at
  // construction from BISHENG_LLC_LIBRARY env).  Submitted once before the
  // for-loop so all retry attempts share the same SO load.  Submit returns
  // immediately (does not block); the main thread proceeds into the first
  // RegBase Pass so LoadInit overlaps with RegBase lowering.
  // Base aux paths are resolved and PreloadBase is submitted after this;
  // PreloadFinalDelta is submitted once the final topology is known.
  // Compute the early pure-path base aux ONCE before the for-loop so both
  // the PreloadBase submit (below) and the per-attempt SHA cross-check
  // (inside the loop) reference the same resolved paths.  When runtime is
  // null (Process/CPU-Runner) this is left empty and unused.  The
  // reduce-only config tweak above runs first so the pure-path base aux
  // resolution sees the finalized config.
  std::map<SubCoreTarget, std::string> earlyAuxPaths;
  if (runtime) {
    runtime->initTrace();
    runtime->emitEvent("MAIN", "compile_scope", "begin");
    runtime->emitEvent("MAIN", "early_start_submit", "point");
    // mask = Both (conservative; lane usage is inferred later from a pure
    // path calculator).  LoadInit only loads SO + init, no aux preload.
    runtime->submitLoadInit(LaneMask::Both);
    // Resolve base aux paths via the pure path calculator (no MLIR Attr
    // read; `addBitcodeAttrsToModule` has NOT run yet at this point) and
    // submit PreloadBase ONCE before the retry for-loop so all attempts
    //     share the same preload.  Paths are stable across attempts
    //     (executablePath/target/optLevel don't change), and
    //     `preloadedPaths_` in the worker dedups across attempts.
    //     LaneMask::Both is conservative; `compileDeviceKernel` only drains
    //     the actually-used lane ticket, so an unused-lane preload failure
    //     does not kill the Compile.  Base aux unavailability (empty paths)
    //     submits a no-op PreloadBase and does NOT block LoadInit.
    earlyAuxPaths = resolveAuxBitcodePathsForConfig(config);
    auto baseByLane = splitBaseAuxByLane(earlyAuxPaths, config);
    runtime->submitPreloadBase(LaneMask::Both, baseByLane.aic, baseByLane.aiv);
  }

  // Overflow tracking: UB gets a Vector-only multi-buffer fallback; L1/L0C
  // overflow still uses the coarse enable-auto-multi-buffer switch.
  bool hasUboverflow = false;
  bool hasVectorizationPipelineFailure = false;
  MLIRContext *ctx = mod->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<Diagnostic> collectedDiagnostics;
  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID = diagEngine.registerHandler([&](Diagnostic &diag) {
    // VF fusion may cause ub overflow. in this case, it will fallback to allop
    // fused to decrease ub occupation
    // Todo: use Enum to standardize the format of error message printing
    if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
      std::string errMsg;
      llvm::raw_string_ostream errStream(errMsg);
      errStream << diag;
      const std::string &msg = errStream.str();
      if (msg.find("ub overflow") != std::string::npos) {
        hasUboverflow = true;
      }
      if (isVectorizationPipelineFailure(msg))
        hasVectorizationPipelineFailure = true;
    } else if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning) {
      std::string warnMsg;
      llvm::raw_string_ostream warnStream(warnMsg);
      warnStream << diag;
      if (isVectorizationPipelineFailure(warnStream.str()))
        hasVectorizationPipelineFailure = true;
    }
    collectedDiagnostics.emplace_back(std::move(diag));
  });

  bool hirCompileSuccess = false;
  int tryTimes = 2;
  // triton compile has nothing to do with HFusion auto schedule, so we don't
  // need to tune for it.
  //
  // TODO: refactor this ad-hoc retry loop into a dedicated retryPassManager
  // so each fallback policy is composable and explicit. Planned policies:
  //   - OpFusion retry policy: bump tiling-max-counter each attempt, up to 5
  //     retries (the current default branch).
  //   - AutoBlockify retry policy: progressively disable hoisting and then
  //     multi-buffer.
  //   - MultiBuffer retry policy: disable auto-multi-buffer.
  // Once the retryPassManager exists, the tryTimes / nested-if logic below
  // should be replaced by composing those policies.
  if (config.getEnableTuningMode()) {
    tryTimes = 1;
  }
  CompileFlow compileFlow = getCompileFlow(config);

  // Set once this driver has turned the VF stack limit on itself, which the
  // UB overflow fallback below distinguishes from the user having asked for
  // the limit on the command line: only a rescue in progress extends the
  // attempt budget and says so.
  bool vfStackLimitRetried = false;
  // Keeps the UB overflow fallback below once per compilation.
  bool ubOverflowFallbackApplied = false;
  // Retries a kernel whose vector function overflowed the VF stack with the
  // conservative stack slot limit, which keeps that function within the
  // stack. The retry is enabled by default; users can disable it explicitly
  // when they need the pre-retry behavior. Only an overflow is retried:
  // failures in other stages or tools are not helped by a smaller vector
  // function. Decline too when the limit is already in force, where the
  // retry would change nothing.
  //
  // Tuning mode is deliberately not excluded, even though it caps tryTimes
  // at 1 above: that cap is about the plan-memory retries a tuner sweeps
  // itself, and no tuner varies the stack limit -- it is undocumented and
  // unreferenced outside this driver. A tuner therefore sees the same retry
  // behavior as normal compilation unless the option is explicitly disabled.
  auto tryVFStackRetry = [&](int attempt) {
    // The limit being in force already covers having retried once, since the
    // retry is what turns it on and nothing turns it back off.
    if (!config.getEnableVFStackOverflowRetry() ||
        config.getEnableVFStackLimit())
      return false;
    vfStackLimitRetried = true;
    config.setEnableVFStackLimit(true);
    llvm::errs() << "[NOTE] VF stack overflow suspected; retrying with "
                    "enable-vf-stack-limit=true.\n";
    // Budget two more attempts, one per lever still unused against an
    // overflow, applied cumulatively as the UB overflow fallback is too:
    //   the next attempt    the stack limit applied just above;
    //   the one after that  the buffer-count bump at the bottom of the loop,
    //                       which raises maxBufferCnt and so shrinks the
    //                       tiles the vector function works on.
    // Budgeting only one would leave that second lever unused, and a kernel
    // the buffer bump rescues today would start failing instead.
    if (tryTimes < attempt + 3)
      tryTimes = attempt + 3;
    return true;
  };

  // Restore the default diagnostic handler on every exit path, including the
  // early return inside the loop below (eraseHandler is keyed, so a double
  // erase is a no-op).
  struct DiagHandlerGuard {
    ~DiagHandlerGuard() { diagEngine.eraseHandler(handlerID); }
    mlir::DiagnosticEngine &diagEngine;
    mlir::DiagnosticEngine::HandlerID handlerID;
  } diagHandlerGuard{diagEngine, handlerID};

  for (int i = 0; i < tryTimes; i++) {
    LDBG("Attempt number: " << i << " with max buffer count tuning delta: "
                            << config.getHfusionMaxBufferCountTuning());
    ModuleOp hirCompileMode = mod.clone();
    bool success = true;
    hasUboverflow = false;
    hasVectorizationPipelineFailure = false;

    if (compileFlow == CompileFlow::Affinity) {
      success = runAffinityPipelines(hirCompileMode, config);
    } else if (compileFlow == CompileFlow::Mixed) {
      success = runMixedPipelines(hirCompileMode, config);
    } else if (compileFlow == CompileFlow::PureSimt) {
      success =
          runModulePipeline(hirCompileMode, regbase::buildBiShengTTIRPipeline,
                            config, "BiShengSIMT");
    } else {
      // Standard HIR compile path (no SIMD/SIMT split)
      success =
          runModulePipeline(hirCompileMode, regbase::buildBiShengHIRPipeline,
                            config, "BiShengHIR");
      // Stop final HIVM lowering after earlier pipeline fails.
      success = success && runModulePipeline(hirCompileMode,
                                             regbase::buildFinalHIVMPipelines,
                                             config, "buildFinalHIVMPipelines");
    }
    if (!success) {
      if (hasVectorizationPipelineFailure &&
          config.getEnableOptimizeScalarTransfers()) {
        config.setEnableOptimizeScalarTransfers(false);
        tryTimes += 1;

        llvm::errs() << "[NOTE] ";
        if (hasUboverflow && hasVectorizationPipelineFailure)
          llvm::errs() << "UB overflow and vectorization pipeline failure";
        else if (hasUboverflow)
          llvm::errs() << "UB overflow";
        else
          llvm::errs() << "vectorization pipeline failure";
        llvm::errs() << " detected; retrying with "
                        "enable-optimize-scalar-transfers=false.\n";
        collectedDiagnostics.clear();
        continue;
      }
      // The pre-existing UB overflow fallback (all-op fusion plus more
      // auto-schedule buffers); combined with the stack limit when that was
      // already applied as a retry.
      if (hasUboverflow && !ubOverflowFallbackApplied &&
          (vfStackLimitRetried ||
           config.getVfFusionMode() ==
               mlir::analysis::FusionMode::MaxParallel)) {
        ubOverflowFallbackApplied = true;
        if (vfStackLimitRetried) {
          llvm::errs() << "[NOTE] the VF stack retry hit a UB overflow; "
                          "retrying with vf-fusion-mode=all-op and "
                          "hfusion-max-buffer-count-tuning+2.\n";
          tryTimes += 1;
        }
        LDBG("ub overflow detected at attempt "
             << (i + 1) << "/" << tryTimes << ", fallback with all-op mode");
        config.setVfFusionMode(mlir::analysis::FusionMode::AllOp);
        // increase max buffers by 2 in HFusion auto schedule
        config.increaseHfusionMaxBufferCountTuning(2);
        collectedDiagnostics.clear();
        continue;
      }
      LDBG("can't retry at attempt " << (i + 1) << "/" << tryTimes
                                     << ", no fallback");
      break;
    }

    if (compileFlow == CompileFlow::Simd) {
      // Stop SIMD lowering after earlier pipeline fails.
      success = runModulePipeline(hirCompileMode,
                                  regbase::buildBiShengHIRAVEToLLVMPipeline,
                                  config, "BiShengSIMD");
    }

    addBitcodeAttrsToModule(hirCompileMode, config.getExecutablePath(), config);
    // Cross-check the early-resolved paths against the Attr paths just
    // written by `addBitcodeAttrsToModule`.  Mismatch is warning-only
    // (perf diagnostic): handleCompile delta preload will still cover the
    // authoritative Attr paths.  Computed once before the for-loop; each
    // attempt cross-checks against its own cloned hirCompileMode Attrs.
    if (runtime)
      verifyEarlyAuxSha(hirCompileMode, earlyAuxPaths);

    auto savedTemp = handleSaveTemps(hirCompileMode, config);
    if (failed(savedTemp)) {
      return failure();
    }

    auto hivmcConfig = HIVMCFromBiShengIRConfig(config, runtime);
    // Only a VF stack overflow is worth retrying with the stack limit; the
    // error is otherwise unused since every stage reports its own failure on
    // stderr.
    bool vfStackOverflow = false;
    if (success) {
      llvm::Error err = runHIVMCCompileA5(hirCompileMode, hivmcConfig);
      if (!err) {
        hirCompileSuccess = true;
        mod = hirCompileMode.clone();
        break;
      }
      vfStackOverflow = isVFStackOverflowError(err);
      llvm::consumeError(std::move(err));
    }

    if (vfStackOverflow && tryVFStackRetry(i)) {
      continue;
    }

    // increase max buffers by 2 in HFusion auto schedule
    config.increaseHfusionMaxBufferCountTuning(2);
  }

  // Restore the default handler so the re-emitted diagnostics below print
  // instead of being collected again.
  diagEngine.eraseHandler(handlerID);

  if (!hirCompileSuccess) {
    for (auto &diag : llvm::reverse(collectedDiagnostics)) {
      diagEngine.emit(std::move(diag));
    }
    if (runtime)
      runtime->emitEvent("MAIN", "compile_scope", "end", "fail");
    return failure();
  }

  if (config.shouldEnableCPURunner()) {
    auto fileHandle = mlir::openOutputFile(config.getOutputFile());
    assert(fileHandle != nullptr);
    fileHandle->os() << mod << '\n';
    fileHandle->keep();
    if (runtime)
      runtime->emitEvent("MAIN", "compile_scope", "end");
    return success();
  }

  // Runtime drains warmup tickets + joins workers in ~CompileScope at main()
  // scope exit.  Emit the compile_scope end marker here for trace clarity;
  // the runtime object itself outlives this function.
  if (runtime)
    runtime->emitEvent("MAIN", "compile_scope", "end");
  return success();
}
