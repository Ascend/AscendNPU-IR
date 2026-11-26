//===- PassPipeline.cpp - BiShengIR HIVMC pass pipeline -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/hivmc/PassPipelineA5.h"
#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Pipelines/Passes.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Tools/hivmc/HIVMC.h"
#include "bishengir/Transforms/Passes.h"

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
#include "bishengir/Dialect/Torch/Pipelines/Passes.h"
#endif

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitCPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace bishengir {

namespace {
static Location scopeInlinedFiles(Location loc, LLVM::DILocalScopeAttr scope,
                                  bool inlined = false) {
  if (auto call = dyn_cast<CallSiteLoc>(loc))
    return CallSiteLoc::get(
        scopeInlinedFiles(call.getCallee(), scope, true),
        scopeInlinedFiles(call.getCaller(), scope, inlined));
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    if (auto explicitScope =
            dyn_cast_or_null<LLVM::DILocalScopeAttr>(fused.getMetadata()))
      scope = explicitScope;
    SmallVector<Location> children;
    for (Location child : fused.getLocations())
      children.push_back(scopeInlinedFiles(child, scope, inlined));
    return FusedLoc::get(children, fused.getMetadata(), loc.getContext());
  }
  if (auto named = dyn_cast<NameLoc>(loc))
    return NameLoc::get(named.getName(),
                        scopeInlinedFiles(named.getChildLoc(), scope, inlined));
  if (auto opaque = dyn_cast<OpaqueLoc>(loc))
    return OpaqueLoc::get(
        opaque.getUnderlyingLocation(), opaque.getUnderlyingTypeID(),
        scopeInlinedFiles(opaque.getFallbackLocation(), scope, inlined));

  auto source = dyn_cast<FileLineColLoc>(loc);
  if (!inlined || !source)
    return loc;
  auto scopeFile =
      llvm::TypeSwitch<LLVM::DILocalScopeAttr, LLVM::DIFileAttr>(scope)
          .Case<LLVM::DISubprogramAttr, LLVM::DILexicalBlockAttr,
                LLVM::DILexicalBlockFileAttr>(
              [](auto attr) { return attr.getFile(); })
          .Default([](auto) { return LLVM::DIFileAttr(); });
  StringRef path = source.getFilename().getValue();
  auto sourceFile =
      LLVM::DIFileAttr::get(loc.getContext(), llvm::sys::path::filename(path),
                            llvm::sys::path::parent_path(path));
  if (sourceFile == scopeFile)
    return loc;

  // LLVM exports a bare FileLineColLoc using its inherited scope's file.
  auto fileScope = LLVM::DILexicalBlockFileAttr::get(scope, sourceFile, 0);
  return FusedLoc::get({loc}, fileScope, loc.getContext());
}

struct PreserveInlinedSourceFiles
    : public PassWrapper<PreserveInlinedSourceFiles, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreserveInlinedSourceFiles)

  void runOnOperation() override {
    getOperation()->walk([](LLVM::LLVMFuncOp func) {
      auto subprogram =
          func.getLoc()->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>();
      if (!subprogram)
        return;
      auto scope = cast<LLVM::DILocalScopeAttr>(subprogram.getMetadata());
      func.getBody().walk([&](Operation *op) {
        op->setLoc(scopeInlinedFiles(op->getLoc(), scope));
      });
    });
  }
};

struct NormalizeBooleanDebugTypes
    : public PassWrapper<NormalizeBooleanDebugTypes, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeBooleanDebugTypes)

  void runOnOperation() override {
    AttrTypeReplacer replacer;
    replacer.addReplacement([](LLVM::DIBasicTypeAttr type) {
      if (type.getTag() != llvm::dwarf::DW_TAG_base_type ||
          type.getEncoding() != llvm::dwarf::DW_ATE_boolean ||
          type.getSizeInBits() != 1)
        return type;

      // CCEC derives DW_AT_byte_size from whole bytes of the DI type.
      // The source Boolean occupies one byte even when its SSA value is i1.
      return LLVM::DIBasicTypeAttr::get(type.getContext(), type.getTag(),
                                        type.getName(), 8, type.getEncoding());
    });
    replacer.recursivelyReplaceElementsIn(getOperation(), /*replaceAttrs=*/true,
                                          /*replaceLocs=*/true);
  }
};

struct ExpandSIMTUnsignedMultiplyHigh
    : public PassWrapper<ExpandSIMTUnsignedMultiplyHigh,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandSIMTUnsignedMultiplyHigh)

  void runOnOperation() override {
    namespace regbase = mlir::hivm_regbaseintrins;
    constexpr StringLiteral helper = "_mlir_ciface_simt_umulhi_uint32_t";
    auto declaration = getOperation().lookupSymbol<LLVM::LLVMFuncOp>(helper);
    if (!declaration || !declaration.isExternal())
      return;

    // The debug backend can end incoming argument locations around an
    // out-of-line SIMT call. Expand this fixed-width template operation so
    // that its arithmetic can be selected directly in the caller.
    getOperation()->walk([&](LLVM::CallOp call) {
      if (call.getCallee().value_or("") != helper ||
          !call->getAttrOfType<regbase::SIMT_CallableAttr>(
              regbase::kDavinciCallingConvAttrName))
        return;
      auto operands = call.getCalleeOperands();
      if (operands.size() != 2 || !operands[0].getType().isInteger(32) ||
          !operands[1].getType().isInteger(32) || call.getNumResults() != 1 ||
          !call.getResult().getType().isInteger(32))
        return;

      OpBuilder builder(call);
      Location resultLoc = call.getLoc();
      Location loc = resultLoc;
      while (auto named = dyn_cast<NameLoc>(loc))
        loc = named.getChildLoc();
      mlir::Type i64 = builder.getI64Type();
      mlir::Value lhs = builder.create<LLVM::ZExtOp>(loc, i64, operands[0]);
      mlir::Value rhs = builder.create<LLVM::ZExtOp>(loc, i64, operands[1]);
      mlir::Value product = builder.create<LLVM::MulOp>(loc, lhs, rhs);
      mlir::Value bits = builder.create<LLVM::ConstantOp>(loc, i64, 32);
      mlir::Value high = builder.create<LLVM::LShrOp>(loc, product, bits);
      mlir::Value result =
          builder.create<LLVM::TruncOp>(resultLoc, builder.getI32Type(), high);
      call.getResult().replaceAllUsesWith(result);
      call.erase();
    });
  }
};
} // namespace

void buildSIMTPipeline(OpPassManager &pm, const HIVMCMainConfig &config) {
  buildLowerToLLVMPipeline(pm, config);
}

/// Build the pipeline to lower BiShengHIR to LLVM Dialect IR
void buildLowerToLLVMPipeline(OpPassManager &pm,
                              const HIVMCMainConfig &config) {
  ConvertHIVMToLLVMOptions hivm2llvmOptions;
  hivm2llvmOptions.useBarePtrCallConv =
      config.tryBarePtrCallConvForStaticShape();
  pm.addPass(createConvertHIVMToLLVMPass(hivm2llvmOptions));
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (config.shouldCompileTriton() &&
      (!config.shouldEnableSimdSimtMixCompile())) {
    pm.addPass(createTritonGlobalKernelArgsToLLVMPass());
  }
  if (config.shouldEnableSanitizer()) {
    pm.addPass(createAppendSanitizerDataPass());
  }
  bool emitDebugInfo = config.shouldEnableSanitizer() ||
                       config.shouldEnableDebugInfo() ||
                       config.shouldEnableDebugVariables();
  if (emitDebugInfo) {
    // Add lineNo debug info to the IR
    // The info here Corresponds to that in the source file.
    pm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
  }
  if (config.shouldEnableDebugVariables()) {
    pm.addPass(std::make_unique<ExpandSIMTUnsignedMultiplyHigh>());
    pm.addPass(LLVM::createLLVMDILocalVariablePass());
    pm.addPass(std::make_unique<NormalizeBooleanDebugTypes>());
  }

  if (emitDebugInfo)
    pm.addPass(std::make_unique<PreserveInlinedSourceFiles>());

  pm.addPass(createAppendUsePrintDebugDataPass());
  const bool ascendDebugLLVM = StringRef(getenv("ASCEND_DEBUG_LLVM")) == "1";
  if (ascendDebugLLVM || !config.getDebugLLVMActions().empty()) {
    DebugLLVMOptions debugLLVMOptions{config.getDebugLLVMActions(),
                                      config.getDebugLLVMCacheFilePath()};
    pm.addPass(createDebugLLVMPass(debugLLVMOptions));
  }
}

void buildBiShengHIRAVEToLLVMPipeline(OpPassManager &pm,
                                      const HIVMCMainConfig &config) {
  if (config.shouldLowerToLLVM()) {
    buildLowerToLLVMPipeline(pm, config);
  }
}

void buildFinalMixVFCompilePipeline(OpPassManager &pm,
                                    const HIVMCMainConfig &config) {

  pm.addPass(hivm::createFlattenModulePass());

  if (config.shouldCompileTriton()) {
    pm.addPass(createTritonGlobalKernelArgsToLLVMPass());
  }
}

/// We define the HIVMC Compile Pass here not in a tablegen file because
/// there potentially many options that are controlled by cmake options, and
/// it's more flexible to define in cpp.
struct HIVMCCompilePass
    : public PassWrapper<HIVMCCompilePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HIVMCCompilePass)
  HIVMCCompilePass() = default;
  HIVMCCompilePass &operator=(const HIVMCCompilePass &pass) = delete;
  HIVMCCompilePass(const HIVMCCompilePass &pass)
      : PassWrapper<HIVMCCompilePass, OperationPass<ModuleOp>>(pass) {
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
    enableTorchCompile = pass.enableTorchCompile;
#endif
    enableTritonKernelCompile = pass.enableTritonKernelCompile;
    enableAutoBlockifyLoop = pass.enableAutoBlockifyLoop;
    enableHFusionCompile = pass.enableHFusionCompile;
    enableHIVMCompile = pass.enableHIVMCompile;
    enableLIRCompile = pass.enableLIRCompile;
    onlyRunHIVMPipeline = pass.onlyRunHIVMPipeline;
    enableManageHostResources = pass.enableManageHostResources;
    enableStaticBarePtr = pass.enableStaticBarePtr;
    enableBinRelocation = pass.enableBinRelocation;
    enableSymbolAnalysis = pass.enableSymbolAnalysis;
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
    ensureNoImplicitBroadcast = pass.ensureNoImplicitBroadcast;
#endif
#if (!BISHENGIR_PUBLISH)
    enableCpuTraceIntrinsic = pass.enableCpuTraceIntrinsic;
#endif
    enableSanitizer = pass.enableSanitizer;
    enableDebugVariables = pass.enableDebugVariables;
    enableDebugInfo = pass.enableDebugInfo;
    outputFile = pass.outputFile;
    enableAutoMultiBuffer = pass.enableAutoMultiBuffer;
    limitAutoMultiBufferOnlyForLocalBuffer =
        pass.limitAutoMultiBufferOnlyForLocalBuffer;
    limitAutoMultiBufferOfLocalBuffer = pass.limitAutoMultiBufferOfLocalBuffer;
    limitMixAutoMultiBufferBuffer = pass.limitMixAutoMultiBufferBuffer;
    enableCodeMotion = pass.enableCodeMotion;
    enableOpsReorder = pass.enableOpsReorder;
    enableTuningMode = pass.enableTuningMode;
    blockDim = pass.blockDim;
    maxHorizontalFusionSize = pass.maxHorizontalFusionSize;
    enableMultiKernel = pass.enableMultiKernel;
    enableCountBufferDmaOpt = pass.enableCountBufferDmaOpt;
    maxBufferCntTuning = pass.maxBufferCntTuning;
    cubeTilingTuning = pass.cubeTilingTuning;
    enableHIVMInjectBarrierAllSync = pass.enableHIVMInjectBarrierAllSync;
    enableInjectBlockAllSync = pass.enableInjectBlockAllSync;
    enableUnitFlagSync = pass.enableUnitFlagSync;
    enableGlobalWorkspaceReuse = pass.enableGlobalWorkspaceReuse;
    enableAutoStorageAlign = pass.enableAutoStorageAlign;
    enableND2NZOnVector = pass.enableND2NZOnVector;
    enablefusedMultiplyAdd = pass.enablefusedMultiplyAdd;
    simtVFDynamicSize = pass.simtVFDynamicSize;
  }
  StringRef getArgument() const override { return "hivmc"; }
  StringRef getDescription() const override {
    return "Compile BiShengIR module to binary.";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    HIVMCMainConfig config;
    // Use fluent API to set the pass option into config.

    // Feature control options
    config
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
        .compileTorch(enableTorchCompile)
#endif
        .compileTriton(enableTritonKernelCompile)
        .compileHFusion(enableHFusionCompile)
        .compileHIVM(enableHIVMCompile)
        .compileLIR(enableLIRCompile)
        .onlyRunHIVMPipeline(onlyRunHIVMPipeline)
        .manageHostResource(enableManageHostResources)
        .barePtrCallConvForStaticShape(enableStaticBarePtr)
        .relocateBinary(enableBinRelocation)
        .symbolAnalysis(enableSymbolAnalysis)
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
        .noImplicitBroadcast(ensureNoImplicitBroadcast)
#endif
        .multiKernel(enableMultiKernel);

    config
#if (!BISHENGIR_PUBLISH)
        // DFX control options
        .cpuTraceIntrinsic(enableCpuTraceIntrinsic)
#endif
        .enableSanitizer(enableSanitizer)
        .enableDebugVariables(enableDebugVariables)
        .enableDebugInfo(enableDebugInfo);

    // Output setting options
    config.setOutputFile(outputFile);

    // General optimization control options
    config.autoMultiBuffer(enableAutoMultiBuffer)
        .limitAutoMultiBufferForLocalBuffer(
            limitAutoMultiBufferOnlyForLocalBuffer)
        .limitAutoMultiBufferOfLocalBuffer(limitAutoMultiBufferOfLocalBuffer)
        .limitMixAutoMultiBufferBuffer(limitMixAutoMultiBufferBuffer)
        .autoBindSubBlock(enableAutoBindSubBlock)
        .deterministicComputing(enableDeterministicComputing)
        .codeMotion(enableCodeMotion)
        .reorderOps(enableOpsReorder)
        .tuningMode(enableTuningMode)
        .setBlockDim(blockDim);

    // HFusion optimization control options
    config.setMaxHorizontalFusionSize(maxHorizontalFusionSize)
        .setMaxBufferCountTuning(maxBufferCntTuning)
        .optimizeCountBufferForDma(enableCountBufferDmaOpt)
        .setCubeTilingTuningParams(cubeTilingTuning);

    // HIVM optimization control options
    config.injectBarrierAllSync(enableHIVMInjectBarrierAllSync)
        .injectBlockAllSync(enableInjectBlockAllSync)
        .unitFlagSync(enableUnitFlagSync)
        .globalWorkspaceReuse(enableGlobalWorkspaceReuse)
        .autoStorageAlign(enableAutoStorageAlign)
        .enableND2NZOnVector(enableND2NZOnVector)
        .enablefusedMultiplyAdd(enablefusedMultiplyAdd)
        .autoBlockifyLoop(enableAutoBlockifyLoop)
        .setSimtVFDynamicSize(simtVFDynamicSize);
    if (failed(runHIVMCCompile(moduleOp, config)))
      signalPassFailure();
  }

protected:
  // -------------------------------------------------------------------------//
  //                       Feature control options
  // -------------------------------------------------------------------------//
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  Pass::Option<bool> enableTorchCompile{
      *this, "enable-torch-compile",
      llvm::cl::desc("Enable compile from Torch dialect"),
      llvm::cl::init(false)};
#endif
  Pass::Option<bool> enableTritonKernelCompile{
      *this, "enable-triton-kernel-compile",
      llvm::cl::desc("Enable Triton kernel compile"), llvm::cl::init(false)};
  Pass::Option<bool> enableHFusionCompile{
      *this, "enable-hfusion-compile",
      llvm::cl::desc("Enable BiShengHIR HFusion compile"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableHIVMCompile{
      *this, "enable-hivm-compile",
      llvm::cl::desc("Enable BiShengHIR HIVM compile"), llvm::cl::init(true)};
  Pass::Option<bool> enableLIRCompile{
      *this, "enable-lir-compile", llvm::cl::desc("Enable BiShengLIR compile"),
      llvm::cl::init(true)};
  Pass::Option<bool> onlyRunHIVMPipeline{
      *this, "only-run-hivm-pipeline",
      llvm::cl::desc("Only run BiShengHIR HIVM pipeline"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableManageHostResources{
      *this, "enable-manage-host-resources",
      llvm::cl::desc("Enable managing resource for Host functions"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableStaticBarePtr{
      *this, "enable-static-bare-ptr",
      llvm::cl::desc("Enable generating bare ptr calling convention for static "
                     "shaped kernels"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableBinRelocation{
      *this, "enable-bin-relocation",
      llvm::cl::desc("Enable binary relocation"), llvm::cl::init(true)};
  Pass::Option<bool> enableSymbolAnalysis{
      *this, "enable-symbol-analysis", llvm::cl::desc("Enable symbol analysis"),
      llvm::cl::init(false)};
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  Pass::Option<bool> ensureNoImplicitBroadcast{
      *this, "ensure-no-implicit-broadcast",
      llvm::cl::desc("Whether to ensure that there is no implicit broadcast "
                     "semantics.If there is a dynamic to dynamic dim "
                     "broadcast, raise a runtime error."),
      llvm::cl::init(false)};
#endif
  Pass::Option<bool> enableMultiKernel{
      *this, "enable-hfusion-multi-kernel",
      llvm::cl::desc("When disabled, graph must fuse as single kernel; when "
                     "enabled, outline multiple kernels."),
      llvm::cl::init(false)};
  // -------------------------------------------------------------------------//
  //                           DFX control options
  // -------------------------------------------------------------------------//
#if (!BISHENGIR_PUBLISH)
  Pass::Option<bool> enableCpuTraceIntrinsic{
      *this, "enable-cpu-trace-intrinsic",
      llvm::cl::desc("Enable to generate cpu-accepted IR by eliminating HIVM "
                     "special traits"),
      llvm::cl::init(false)};
#endif
  Pass::Option<bool> enableSanitizer{*this, "enable-sanitizer",
                                     llvm::cl::desc("Enable ascend sanitizer"),
                                     llvm::cl::init(false)};

  Pass::Option<bool> enableDebugVariables{*this, "enable-debug-variables",
                                          llvm::cl::desc("Enable ascend debug"),
                                          llvm::cl::init(false)};

  Pass::Option<bool> enableDebugInfo{*this, "enable-debug-info",
                                     llvm::cl::desc("Enable debug info"),
                                     llvm::cl::init(false)};
  // -------------------------------------------------------------------------//
  //                        Output setting options
  // -------------------------------------------------------------------------//
  Pass::Option<std::string> outputFile{
      *this, "o", llvm::cl::desc("Specify output bin name"),
      llvm::cl::init("-")};
  // -------------------------------------------------------------------------//
  //                  General optimization control options
  // -------------------------------------------------------------------------//
  Pass::Option<bool> enableLayoutOptimization{
      *this, "enable-layout-optimization",
      llvm::cl::desc("Enable Layout Optimization"), llvm::cl::init(false)};

  Pass::Option<bool> enableMixedCV{
      *this, "enable-mixed-cv", llvm::cl::desc("Enable mixed CV compilation"),
      llvm::cl::init(false)};

  Pass::Option<bool> enableAutoMultiBuffer{
      *this, "enable-auto-multi-buffer",
      llvm::cl::desc("Enable auto multi buffer"), llvm::cl::init(false)};

  Pass::Option<bool> limitAutoMultiBufferOnlyForLocalBuffer{
      *this, "limit-auto-multi-buffer-only-for-local-buffer",
      llvm::cl::desc("Disable workspace multi-buffer optimization"),
      llvm::cl::init(false)};

  Pass::Option<MultiBufferStrategy> limitAutoMultiBufferOfLocalBuffer{
      *this, "limit-auto-multi-buffer-of-local-buffer",
      llvm::cl::desc("When enable-auto-multi-buffer = true, limit local buffer "
                     "mode"),
      llvm::cl::init(MultiBufferStrategy::CUBE_NO_L0C),
      llvm::cl::values(
          clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit", "No limit"),
          clEnumValN(MultiBufferStrategy::CUBE_NO_L0C, "no-l0c",
                     "Disable l0c multi buffer"))};

  Pass::Option<MultiBufferStrategy> limitMixAutoMultiBufferBuffer{
      *this, "limit-auto-multi-buffer-buffer",
      llvm::cl::desc("When enable-auto-multi-buffer = true, limit it only work"
                     "for ONLY_CUBE, ONLY_VECTOR or NO_LIMIT"),
      llvm::cl::init(MultiBufferStrategy::ONLY_CUBE)};

  Pass::Option<bool> enableAutoBindSubBlock{
      *this, "enable-auto-bind-sub-block",
      llvm::cl::desc("Enable auto bind sub block"), llvm::cl::init(true)};

  Pass::Option<bool> enableDeterministicComputing{
      *this, "enable-deterministic-computing",
      llvm::cl::desc("If enabled, the computation result is deterministic. If "
                     "disabled, we will enable extra optimizations that might "
                     "boost performance, e.g. bind reduce to multiple cores. "
                     "However, the result will be non-deterministic."),
      llvm::cl::init(true)};

  Pass::Option<bool> enableCodeMotion{
      *this, "enable-code-motion",
      llvm::cl::desc("Enable code-motion and subset-hoist (Default = ON)"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableOpsReorder{
      *this, "enable-ops-reorder",
      llvm::cl::desc("Enable ops reorder to opt pipeline (Default = ON)"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableTuningMode{
      *this, "enable-tuning-mode",
      llvm::cl::desc("Enable tuning mode and will not try compile multi times"),
      llvm::cl::init(false)};

  Pass::Option<unsigned> blockDim{*this, "block-dim",
                                  llvm::cl::desc("Number of blocks to use"),
                                  llvm::cl::init(1)};
  // -------------------------------------------------------------------------//
  //                  HFusion optimization control options
  // -------------------------------------------------------------------------//
  Pass::Option<int> maxHorizontalFusionSize{
      *this, "hfusion-max-horizontal-fusion-size",
      llvm::cl::desc(
          "Number of horizontal fusion attempt (Default: unlimited)"),
      llvm::cl::init(-1)};
  Pass::Option<bool> enableCountBufferDmaOpt{
      *this, "enable-hfusion-count-buffer-dma-opt",
      llvm::cl::desc("If enabled, the buffer used by DMA operations will not "
                     "bereused by vector operations"),
      llvm::cl::init(false)};
  Pass::Option<int64_t> maxBufferCntTuning{
      *this, "hfusion-max-buffer-count-tuning",
      llvm::cl::desc("Allow tuning auto-schedule max buffer count"),
      llvm::cl::init(0)};
  Pass::ListOption<int64_t> cubeTilingTuning{
      *this, "hfusion-cube-tiling-tuning",
      llvm::cl::desc("Allow tuning auto-schedule cube block sizes")};
  // -------------------------------------------------------------------------//
  //                  HIVM optimization control options
  // -------------------------------------------------------------------------//
  Pass::Option<bool> enableHIVMInjectBarrierAllSync{
      *this, "enable-hivm-inject-barrier-all-sync",
      llvm::cl::desc("Enable barrier all mode for HIVM inject sync"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableInjectBlockAllSync{
      *this, "enable-hivm-inject-block-all-sync",
      llvm::cl::desc("Enable inject all block sync for HIVM inject block sync"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableUnitFlagSync{
      *this, "enable-hivm-unit-flag-sync",
      llvm::cl::desc(
          "Enable inject sync pass to use unit-flag modes for synchronization"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableGlobalWorkspaceReuse{
      *this, "enable-hivm-global-workspace-reuse",
      llvm::cl::desc("Enable global workspace reuse for plan memory"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableAutoStorageAlign{
      *this, "enable-hivm-auto-storage-align",
      llvm::cl::desc("Enable mark/enable HIVM storage align (Default = ON)"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableND2NZOnVector{
      *this, "enable-hivm-nd2nz-on-vector",
      llvm::cl::desc("Enable nd2nz on vector (Default = OFF)"),
      llvm::cl::init(false)};
  Pass::Option<bool> enablefusedMultiplyAdd{
      *this, "enable-fused-multiply-add",
      llvm::cl::desc("Enable fused multiply add"), llvm::cl::init(false)};
  Pass::Option<bool> enableAutoBlockifyLoop{
      *this, "enable-auto-blockify-loop",
      llvm::cl::desc(
          "Enable auto loop on blocks for all parallel (Default = OFF)"),
      llvm::cl::init(false)};
  Pass::Option<int> simtVFDynamicSize{
      *this, "simt-vf-dynamic-size",
      llvm::cl::desc("Dynamic ub size(KB) for simt VF. Default is 216"),
      llvm::cl::init(216)};
};

} // namespace bishengir

// void bishengir::registerHIVMCCompilePass() {
//   PassRegistration<bishengir::HIVMCCompilePass>();
// }
