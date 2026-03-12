//===- BiShengIRCompileConfig.cpp - BiShengIR Compile Config -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/Utility.h"

#include "llvm/ADT/STLExtras.h" // interleaveComma
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h" // report_fatal_error
#include "llvm/Support/ManagedStatic.h"

using namespace bishengir;
using namespace llvm;
using namespace mlir::triton;

namespace {
static cl::OptionCategory featCtrlCategory("BiShengIR Feature Control Options");
static cl::OptionCategory dfxCtrlCategory("BiShengIR DFX Control Options");
static cl::OptionCategory
    generalOptCategory("BiShengIR General Optimization Options");
static cl::OptionCategory
    hfusionOptCategory("BiShengIR HFusion Optimization Options");
static cl::OptionCategory
    hivmOptCategory("BiShengIR HIVM Optimization Options");
static cl::OptionCategory protonCategory("BiShengIR Proton Options");
static cl::OptionCategory targetCategory("BiShengIR Target Options");
static cl::OptionCategory
    simtOptCategory("BiShengIR SIMT Optimization Options");

/// This class is intended to manage the handling of command line options for
/// creating bishengir-compile config. This is a singleton.
/// Options that are not exposed to the user should not be added here.
struct BiShengIRCompileMainConfigCLOptions : public BiShengIRCompileMainConfig {
  BiShengIRCompileMainConfigCLOptions() {
    // These options are static but all uses ExternalStorage to initialize the
    // members of the parent class. This is unusual but since this class is a
    // singleton it basically attaches command line option to the singleton
    // members.

    // -------------------------------------------------------------------------//
    //                       Feature control options
    // -------------------------------------------------------------------------//

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
    static cl::opt<bool, /*ExternalStorage=*/true> enableTorchCompile(
        "enable-torch-compile", cl::desc("Enable compile from Torch dialect"),
        cl::location(enableTorchCompileFlag), cl::init(false),
        cl::cat(featCtrlCategory));
#endif

#if BISHENGIR_ENABLE_TRITON_COMPILE
    static cl::opt<bool, /*ExternalStorage=*/true> enableTritonIRCompile(
        "enable-triton-ir-compile",
        cl::desc("Enable compile from Triton dialect"),
        cl::location(enableTritonIRCompileFlag), cl::init(false),
        cl::cat(featCtrlCategory));
#endif
    static cl::opt<bool, /*ExternalStorage=*/true> enableLayoutOptimization(
        "enable-layout-optimization", cl::desc("Enable Layout Optimization"),
        cl::location(enableLayoutOptimizationFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> enableMixedCV(
        "enable-mixed-cv", cl::desc("Enable mixed CV compilation"),
        cl::location(enableMixedCVFlag), cl::init(false), cl::Hidden,
        cl::callback([](const bool &) {
          errs() << "[WARNING] --enable-mixed-cv is deprecated.\n";
        }));

    static cl::opt<bool, /*ExternalStorage=*/true> enableTritonKernelCompile(
        "enable-triton-kernel-compile",
        cl::desc("Enable Triton kernel compile (lowered from triton-adaptor)"),
        cl::location(enableTritonKernelCompileFlag), cl::init(false),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableDotScaledCompile(
        "enable-dot-scaled-compile", cl::desc("Enable dot scaled compile"),
        cl::location(enableDotScaledCompileFlag), cl::init(false),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> disableFFTS(
        "disable-ffts", cl::desc("Force disabling FFTS."),
        cl::location(disableFFTSFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> disableHFusionVectorize(
        "disable-hfusion-vectorize",
        cl::desc("Disable hfusion auto vectorize."),
        cl::location(disableHFusionVectorizeFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> enableFullSIMT(
        "pure-simt", cl::desc("Full SIMT compile."),
        cl::location(enableFullSIMTFlag), cl::init(false));

    static cl::opt<int, /*ExternalStorage=*/true> simtVFDynamicSize(
        "simt-vf-dynamic-size",
        cl::desc("Dynamic ub size(KB) for simt VF. Default is 216."),
        cl::location(simtVFDynamicSizeFlag), cl::init(216));

    static cl::list<int64_t> gridDim(
        "simt-triton-grid",
        cl::CommaSeparated, // allow comma-separated like 2,2,2
        cl::desc(
            "Grid dimensions (e.g., -simt-triton-grid=2,2,2). Default is 1."),
        cl::ZeroOrMore);
    gridDimCLPtr = std::addressof(gridDim);

    static cl::opt<bool, /*ExternalStorage=*/true> enableHFusionCompile(
        "enable-hfusion-compile", cl::desc("Enable BiShengHIR HFusion compile"),
        cl::location(enableHFusionCompileFlag), cl::init(false),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableHIVMCompile(
        "enable-hivm-compile", cl::desc("Enable BiShengHIR HIVM compile"),
        cl::location(enableHIVMCompileFlag), cl::init(true),
        cl::cat(featCtrlCategory));

#if (!BISHENGIR_PUBLISH)
    static cl::opt<bool, /*ExternalStorage=*/true> enableLIRCompile(
        "enable-lir-compile", cl::desc("Enable BiShengLIR compile"),
        cl::location(enableLIRCompileFlag), cl::init(true),
        cl::cat(featCtrlCategory));
#endif

    // TODO: remove it after changing -enable-lir-compile to control if compile
    // lowering hivm to binary.
    static cl::opt<bool, /*ExternalStorage=*/true> onlyRunHIVMPipeline(
        "only-run-hivm-pipeline", cl::desc("Only run BiShengHIR HIVM pipeline"),
        cl::location(onlyRunHIVMPipelineFlag), cl::init(false),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableManageHostResources(
        "enable-manage-host-resources",
        cl::desc("Enable managing resource for Host functions"),
        cl::location(enableManageHostResourcesFlag), cl::init(false),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableStaticBarePtr(
        "enable-static-bare-ptr",
        cl::desc("Enable generating bare ptr calling convention for static "
                 "shaped kernels"),
        cl::location(enableStaticBarePtrFlag), cl::init(true),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableBinRelocation(
        "enable-bin-relocation", cl::desc("Enable binary relocation"),
        cl::location(enableBinRelocationFlag), cl::init(true),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableSymbolAnalysis(
        "enable-symbol-analysis", cl::desc("Enable symbol analysis"),
        cl::location(enableSymbolAnalysisFlag), cl::init(false),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableTreeReduce(
        "enable-tree-reduce", cl::desc("Enable tree reduce"),
        cl::location(enableTreeReduceFlag), cl::init(false),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> skipScope(
        "skip-scope", cl::desc("Skip passes like flattenOps when scope exists"),
        cl::location(skipScopeFlag), cl::init(true), cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableAutoVectorizeV2(
        "enable-auto-vectorize-v2", cl::desc("Enable auto vectorize v2"),
        cl::location(enableAutoVectorizeV2Flag), cl::init(true),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableVFFusion(
        "enable-vf-fusion", cl::desc("Enable vf fusion"),
        cl::location(enableVFFusionFlag), cl::init(false),
        cl::cat(featCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableHighPrecision(
        "enable-high-precision",
        cl::desc("Enable high precision calculation for sin/cos in HFusion"),
        cl::location(enableHighPrecisionFlag), cl::init(true),
        cl::cat(featCtrlCategory));

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
    static cl::opt<bool, /*ExternalStorage=*/true> ensureNoImplicitBroadcast(
        "ensure-no-implicit-broadcast",
        cl::desc("Whether to ensure that there is no implicit broadcast "
                 "semantics. If there is a dynamic to dynamic dim "
                 "broadcast, raise a runtime error."),
        cl::location(ensureNoImplicitBroadcastFlag), cl::init(false),
        cl::cat(featCtrlCategory));
#endif

    static cl::opt<bool, /*ExternalStorage=*/true> saveLinkedIR(
        "save-linked-ir",
        cl::desc("Enable saving linked IR before compile to binary"),
        cl::location(this->saveLinkedIR), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> enableMultiKernel(
        "enable-hfusion-multi-kernel",
        cl::desc("When disabled, graph must fuse as single kernel; when "
                 "enabled, outline multiple kernels."),
        cl::location(enableMultiKernelFlag), cl::init(false),
        cl::cat(featCtrlCategory));

#if BISHENGIR_ENABLE_TRITON_COMPILE
    static cl::opt<int32_t, /*ExternalStorage=*/true> numWarps(
        "num-warps", cl::desc("Number of warps"), cl::location(numWarpsFlag),
        cl::init(4), cl::cat(featCtrlCategory));

    static cl::opt<int32_t, /*ExternalStorage=*/true> threadsPerWarp(
        "threads-per-warp", cl::desc("Number of threads per warp"),
        cl::location(threadsPerWarpFlag), cl::init(32),
        cl::cat(featCtrlCategory));

    static cl::opt<int32_t, /*ExternalStorage=*/true> sharedDynamicSize(
        "shared-mem-dynamic-size",
        cl::desc("Dynamic size of shared memory (in bytes)"),
        cl::location(sharedDynamicSizeFlag), cl::init(122880),
        cl::cat(featCtrlCategory));
    static cl::opt<bool, /*ExternalStorage=*/true> enableSimdSimtMixCompile(
        "enable-simd-simt-mix-compile",
        cl::desc("Enable simd-simt mix kernel compile"),
        cl::location(enableSimdSimtMixCompileFlag), cl::init(false),
        cl::cat(hivmOptCategory));
#endif

    // -------------------------------------------------------------------------//
    //                           DFX control options
    // -------------------------------------------------------------------------//

#if (!BISHENGIR_PUBLISH)
    static cl::opt<bool, /*ExternalStorage=*/true> enableCpuTraceIntrinsic(
        "enable-cpu-trace-intrinsic",
        cl::desc("Enable to generate host-accepted IR by eliminating HIVM "
                 "special traits"),
        cl::location(enableCpuTraceIntrinsicFlag), cl::init(false),
        cl::cat(dfxCtrlCategory));
#endif
    static cl::opt<bool, /*ExternalStorage=*/true> enableSanitizer(
        "enable-sanitizer", cl::desc("Enable ascend sanitizer"),
        cl::location(enableSanitizerFlag), cl::init(false),
        cl::cat(dfxCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableDebugInfo(
        "enable-debug-info", cl::desc("Enable debug info"),
        cl::location(enableDebugInfoFlag), cl::init(false),
        cl::cat(dfxCtrlCategory));

    static cl::opt<bool, /*ExternalStorage=*/true>
        enablePrintMemoryAllocatedSize(
            "enable-print-memory-allocated-size",
            cl::desc("Enable print memory allocated size"),
            cl::location(enablePrintMemoryAllocatedSizeFlag), cl::init(false),
            cl::cat(dfxCtrlCategory));

    static cl::opt<std::string, /*ExternalStorage=*/true> injectIrFromFile(
        "inject-ir-from-file",
        cl::desc("Path to IR file for inject-ir pass; when set, matching "
                 "functions are replaced with those from the file for debug"),
        cl::location(injectIrFromFileFlag), cl::init(""),
        cl::cat(dfxCtrlCategory));

    // -------------------------------------------------------------------------//
    //                        Output setting options
    // -------------------------------------------------------------------------//

    static cl::opt<std::string, /*ExternalStorage=*/true> outputFile(
        "o", cl::desc("Specify output bin name"), cl::location(outputFileFlag),
        cl::init("-"));

    // -------------------------------------------------------------------------//
    //                  General optimization control options
    // -------------------------------------------------------------------------//

    static cl::opt<bool, /*ExternalStorage=*/true> enableAutoMultiBuffer(
        "enable-auto-multi-buffer", cl::desc("Enable auto multi buffer"),
        cl::location(enableAutoMultiBufferFlag), cl::init(false),
        cl::cat(generalOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableDropUnitDims(
        "enable-drop-unit-dims", cl::desc("Enable drop-unit-dims pass"),
        cl::location(enableDropUnitDimsFlag), cl::init(true),
        cl::cat(generalOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableFlatten(
        "enable-flatten", cl::desc("Enable flatten pass"),
        cl::location(enableFlattenFlag), cl::init(true),
        cl::cat(generalOptCategory));

    static cl::opt<int, /*ExternalStorage=*/true>
        enableBishengirSimtOptimization(
            "enable-bishengir-simt-optimization",
            cl::desc("enable bishengir simt optimization"),
            cl::location(enableBishengirSimtOptimizationFlag),
            cl::init(900101));

    static cl::opt<std::string> simtStackLimitStr(
        "simt-stack-limit", cl::desc("SIMT stack limit."), cl::value_desc("N"),
        cl::callback([this](const std::string &arg) {
          int32_t v;
          llvm::StringRef s(arg);
          if (s.getAsInteger(0, v))
            report_fatal_error("Invalid value for --simt-stack-limit: " + s);
          this->simtStackLimitFlag = v;
        }));

    static cl::opt<std::string, /*ExternalStorage=*/true> tritonMetadataOutput(
        "triton-metadata-output",
        cl::desc("File to dump triton metadata. -- means stdout"),
        cl::location(tritonMetadataOutputPath), cl::init(""));

    static cl::opt<bool, /*ExternalStorage=*/true> disableDecomposeReduction(
        "disable-decompose-reduction",
        cl::desc("Disable SIMT decompose reduction pass"),
        cl::location(disableDecomposeReductionFlag), cl::init(false),
        cl::cat(simtOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> disableReorderInstruction(
        "disable-reorder-instruction",
        cl::desc("Disable reorder instruction pass"),
        cl::location(disableReorderInstructionFlag), cl::init(false),
        cl::cat(simtOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true>
        limitAutoMultiBufferOnlyForLocalBuffer(
            "limit-auto-multi-buffer-only-for-local-buffer",
            cl::desc("When enable-auto-multi-buffer = true, limit it only "
                     "work for local buffer"),
            cl::location(limitAutoMultiBufferOnlyForLocalBufferFlag),
            cl::init(true), cl::cat(generalOptCategory));

    static cl::opt<MultiBufferStrategy, /*ExternalStorage=*/true>
        limitAutoMultiBufferOfLocalBuffer(
            "limit-auto-multi-buffer-of-local-buffer",
            cl::desc("When enable-auto-multi-buffer = true, limit local buffer "
                     "mode"),
            cl::location(limitAutoMultiBufferOfLocalBufferFlag),
            cl::init(MultiBufferStrategy::CUBE_NO_L0C),
            cl::values(clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit",
                                  "No limit"),
                       clEnumValN(MultiBufferStrategy::CUBE_NO_L0C, "no-l0c",
                                  "Disable l0c multi buffer")),
            cl::cat(generalOptCategory));

    static cl::opt<MultiBufferStrategy, /*ExternalStorage=*/true>
        limitMixAutoMultiBufferBuffer(
            "limit-auto-multi-buffer-buffer",
            cl::desc("When enable-auto-multi-buffer = true, limit it only-only "
                     "only-cube, only-vector Or no limit"),
            cl::location(limitMixAutoMultiBufferBufferFlag),
            cl::init(MultiBufferStrategy::ONLY_CUBE),
            cl::values(clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit",
                                  "No limit"),
                       clEnumValN(MultiBufferStrategy::ONLY_CUBE, "only-cube",
                                  "Limit to only cube"),
                       clEnumValN(MultiBufferStrategy::ONLY_VECTOR,
                                  "only-vector", "Limit to only vector")),
            cl::cat(generalOptCategory));

    static cl::opt<unsigned, /*ExternalStorage=*/true> workspaceMultiBufferNum(
        "set-workspace-multibuffer",
        cl::desc("Override number of multibuffers for workspace, defaults to 1 "
                 "(off)"),
        cl::location(workspaceMultiBufferNumFlag), cl::init(1),
        cl::cat(generalOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableAutoBindSubBlock(
        "enable-auto-bind-sub-block", cl::desc("Enable auto bind sub block"),
        cl::location(enableAutoBindSubBlockFlag), cl::init(true),
        cl::cat(generalOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableDeterministicComputing(
        "enable-deterministic-computing",
        cl::desc("If enabled, the computation result is deterministic. If "
                 "disabled, we will enable extra optimizations that might "
                 "boost performance, e.g. bind reduce to multiple cores. "
                 "However, the result will be non-deterministic."),
        cl::location(enableDeterministicComputingFlag), cl::init(true),
        cl::cat(generalOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableCodeMotion(
        "enable-code-motion", cl::desc("Enable code-motion/subset-hoist"),
        cl::location(enableCodeMotionFlag), cl::init(true),
        cl::cat(generalOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableOpsReorder(
        "enable-ops-reorder", cl::desc("Enable ops reorder to opt pipeline"),
        cl::location(enableOpsReorderFlag), cl::init(true),
        cl::cat(generalOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableTuningMode(
        "enable-tuning-mode",
        cl::desc("Enable tuning mode and will not try compile multi times in "
                 "case of plan memory failure"),
        cl::location(enableTuningModeFlag), cl::init(false),
        cl::cat(generalOptCategory));

    static cl::opt<unsigned, /*ExternalStorage=*/true> blockDim(
        "block-dim", cl::desc("Number of blocks to use"),
        cl::location(blockDimFlag), cl::init(1), cl::cat(generalOptCategory));

    static cl::opt<int32_t, /*ExternalStorage=*/true> enableVfMergeLevel(
        "enable-vf-merge-level",
        cl::desc("Enable vector function merge with level"),
        cl::location(enableVfMergeLevelFlag), cl::init(1));

    // -------------------------------------------------------------------------//
    //                  HFusion optimization control options
    // -------------------------------------------------------------------------//

    static cl::opt<int32_t, /*ExternalStorage=*/true> maxHorizontalFusionSize(
        "hfusion-max-horizontal-fusion-size",
        cl::desc("Number of horizontal fusion attempt (Default: unlimited)"),
        cl::location(maxHorizontalFusionSizeFlag), cl::init(-1),
        cl::cat(hfusionOptCategory));
    static cl::opt<int32_t, /*ExternalStorage=*/true> maxFusedElementwiseOps(
        "hfusion-max-fused-elementwise-ops",
        cl::desc("Maximum number of elementwise ops to fuse in "
                 "PreVectorizationFusion (Default: unlimited)"),
        cl::location(maxFusedElementwiseOpsFlag), cl::init(-1),
        cl::cat(hfusionOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableCountBufferDmaOpt(
        "enable-hfusion-count-buffer-dma-opt",
        cl::desc("If enabled, the buffer used by DMA operations will not "
                 "be reused by Vector operations"),
        cl::location(enableCountBufferDmaOptFlag), cl::init(false),
        cl::cat(hfusionOptCategory));

    static cl::opt<int64_t, /*ExternalStorage=*/true> maxBufferCntTuning(
        "hfusion-max-buffer-count-tuning",
        cl::desc("Max buffer count tuning in HFusion auto schedule"),
        cl::location(maxBufferCntTuningFlag), cl::init(0),
        cl::cat(hfusionOptCategory));

    static cl::list<int64_t> cubeTilingTuning(
        "hfusion-cube-tiling-tuning",
        cl::desc("Cube block size tuning in HFusion auto schedule"),
        cl::CommaSeparated, cl::cat(hfusionOptCategory));
    cubeTilingTuningCLPtr = std::addressof(cubeTilingTuning);

    static cl::opt<bool, /*ExternalStorage=*/true>
        enableHIVMInjectBarrierAllSync(
            "enable-hivm-inject-barrier-all-sync",
            cl::desc("Enable barrier all mode for HIVM inject sync"),
            cl::location(enableHIVMInjectBarrierAllSyncFlag), cl::init(false),
            cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableInjectBlockAllSync(
        "enable-hivm-inject-block-all-sync",
        cl::desc("Enable inject all block sync for HIVM inject block sync"),
        cl::location(enableInjectBlockAllSyncFlag), cl::init(false),
        cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> disableAutoInjectBlockSync(
        "disable-auto-inject-block-sync",
        cl::desc("Disable auto generating sync block wait/set by "
                 "InjectBlockSync pass"),
        cl::location(disableAutoInjectBlockSyncFlag), cl::init(false),
        cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableHIVMGraphSyncSolver(
        "enable-hivm-graph-sync-solver",
        cl::desc("Enable HIVM Graph-Sync-Solver pass to do auto-sync."),
        cl::location(enableHIVMGraphSyncSolverFlag), cl::init(false),
        cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableUnitFlagSync(
        "enable-hivm-unit-flag-sync",
        cl::desc("Enable inject sync pass to use unit-flag modes for "
                 "synchronization"),
        cl::location(enableUnitFlagSyncFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> enableAutoCVBalance(
        "enable-hivm-auto-cv-balance",
        cl::desc("Enable balancing during cv-pipelining"),
        cl::location(enableAutoCVBalanceFlag), cl::init(true),
        cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableGlobalWorkspaceReuse(
        "enable-hivm-global-workspace-reuse",
        cl::desc("Enable global workspace reuse"),
        cl::location(enableGlobalWorkspaceReuseFlag), cl::init(false),
        cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableAutoStorageAlign(
        "enable-hivm-auto-storage-align",
        cl::desc("Enable mark/enable storage align"),
        cl::location(enableAutoStorageAlignFlag), cl::init(true),
        cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableFusedMultiplyAdd(
        "enable-fused-multiply-add", cl::desc("Enable fused multiply add"),
        cl::location(enableFusedMultiplyAddFlag), cl::init(false),
        cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableND2NZOnVector(
        "enable-hivm-nd2nz-on-vector", cl::desc("Enable nd2nz on vector"),
        cl::location(enableND2NZOnVectorFlag), cl::init(false),
        cl::cat(hivmOptCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> enableAutoBlockifyLoop(
        "enable-auto-blockify-loop",
        cl::desc("Enable auto loop on blocks for all parallel"),
        cl::location(enableAutoBlockifyLoopFlag), cl::init(false),
        cl::cat(hivmOptCategory));

    static cl::opt<int, /*ExternalStorage=*/true> maxReductionSplitNum(
        "max-reduction-split", cl::desc("Max split times for reductionLoop."),
        cl::location(maxReductionSplitNumFlag), cl::init(1),
        cl::cat(hivmOptCategory));

    // -------------------------------------------------------------------------//
    //                            proton options
    // -------------------------------------------------------------------------//

    static cl::opt<proton::MetricType, /*ExternalStorage=*/true> clMetricType(
        "proton-metric-type",
        cl::desc("The performance counter metric type we are profiling"),
        cl::location(protonGPUCompileConfig.metricType),
        cl::values(clEnumValN(proton::MetricType::CYCLE, "cycle", "Cycle")),
        cl::init(proton::MetricType::CYCLE), cl::cat(protonCategory));

    static cl::opt<proton::SamplingStrategy, /*ExternalStorage=*/true>
        clSamplingStrategy(
            "proton-sampling-strategy", cl::desc("Profiling sampling strategy"),
            cl::location(protonGPUCompileConfig.samplingStrategy),
            cl::values(clEnumValN(proton::SamplingStrategy::NONE, "none",
                                  "No Sampling"),
                       clEnumValN(proton::SamplingStrategy::SELECTIVE,
                                  "selective", "Selective Sampling")),
            cl::init(proton::SamplingStrategy::NONE), cl::cat(protonCategory));

    static cl::opt<std::string, /*ExternalStorage=*/true> clSamplingOptions(
        "proton-sampling-options", cl::desc("Profiling sampling options"),
        cl::location(protonGPUCompileConfig.samplingOptions), cl::init(""),
        cl::cat(protonCategory));

    static cl::opt<proton::gpu::Granularity, /*ExternalStorage=*/true>
        clGranularity(
            "proton-granularity",
            cl::desc("Profiling granularity: warp, warp_group, or cta"),
            cl::location(protonGPUCompileConfig.granularity),
            cl::values(
                clEnumValN(proton::gpu::Granularity::THREAD, "thread",
                           "Thread"),
                clEnumValN(proton::gpu::Granularity::WARP, "warp", "Warp"),
                clEnumValN(proton::gpu::Granularity::WARP_2, "warp-2",
                           "2 Warps"),
                clEnumValN(proton::gpu::Granularity::WARP_4, "warp-4",
                           "4 Warps"),
                clEnumValN(proton::gpu::Granularity::WARP_8, "warp-8",
                           "8 Warps"),
                clEnumValN(proton::gpu::Granularity::CTA, "cta", "CTA"),
                clEnumValN(proton::gpu::Granularity::WARP_GROUP, "warp-group",
                           "Warp Group"),
                clEnumValN(proton::gpu::Granularity::WARP_GROUP_2,
                           "warp-group-2", "2 Warp Groups"),
                clEnumValN(proton::gpu::Granularity::WARP_GROUP_4,
                           "warp-group-4", "4 Warp Groups"),
                clEnumValN(proton::gpu::Granularity::WARP_GROUP_8,
                           "warp-group-8", "8 Warp Groups")),
            cl::init(proton::gpu::Granularity::WARP), cl::cat(protonCategory));

    static cl::opt<proton::gpu::BufferStrategy, /*ExternalStorage=*/true>
        clBufferStrategy(
            "proton-buffer-strategy",
            cl::desc("Profiler buffer recording strategy (circular or flush)"),
            cl::location(protonGPUCompileConfig.bufferStrategy),
            cl::values(clEnumValN(proton::gpu::BufferStrategy::CIRCULAR,
                                  "circular", "Circular Buffer"),
                       clEnumValN(proton::gpu::BufferStrategy::FLUSH, "flush",
                                  "Flush Buffer")),
            cl::init(proton::gpu::BufferStrategy::CIRCULAR),
            cl::cat(protonCategory));

    static cl::opt<proton::gpu::BufferType, /*ExternalStorage=*/true>
        clBufferType("proton-buffer-type",
                     cl::desc("Internal buffer type (SHARED, GLOBAL) that "
                              "stores the profiling data"),
                     cl::location(protonGPUCompileConfig.bufferType),
                     cl::values(clEnumValN(proton::gpu::BufferType::SHARED,
                                           "shared", "Shared Memory"),
                                clEnumValN(proton::gpu::BufferType::GLOBAL,
                                           "global", "Global Memory")),
                     cl::init(proton::gpu::BufferType::SHARED),
                     cl::cat(protonCategory));

    static cl::opt<int32_t, /*ExternalStorage=*/true> clBufferSize(
        "proton-buffer-size",
        cl::desc("Internal buffer byte size that stores the profiling data. 0 "
                 "means auto-size based on the device's `maxSharedMemSize`"),
        cl::location(protonGPUCompileConfig.bufferSize), cl::init(0),
        cl::cat(protonCategory));

    static cl::opt<int32_t, /*ExternalStorage=*/true> clMaxSharedMemSize(
        "proton-max-shared-mem",
        cl::desc("Maximum available shared memory size per CTA"),
        cl::location(protonGPUCompileConfig.maxSharedMemSize), cl::init(32768),
        cl::cat(protonCategory));

    static cl::opt<int64_t, /*ExternalStorage=*/true> clProfileScratchSize(
        "proton-profile-scratch-size",
        cl::desc("Profiler global scratch memory size per CTA"),
        cl::location(protonGPUCompileConfig.profileScratchSize),
        cl::init(32768), cl::cat(protonCategory));

    static cl::opt<int32_t, /*ExternalStorage=*/true> clProfileScratchAlignment(
        "proton-profile-scratch-alignment",
        cl::desc("Profiler global scratch memory alignment"),
        cl::location(protonGPUCompileConfig.profileScratchAlignment),
        cl::init(128), cl::cat(protonCategory));

    static cl::opt<bool, /*ExternalStorage=*/true> clclockExtension(
        "proton-clk-ext",
        cl::desc("Use long clock if true, otherwise use 32-bit clock"),
        cl::location(protonGPUCompileConfig.clockExtension), cl::init(false),
        cl::cat(protonCategory));

    // -------------------------------------------------------------------------//
    //                            Target options
    // -------------------------------------------------------------------------//

    static cl::opt<mlir::hacc::TargetDevice, /*ExternalStorage=*/true> target(
        "target", cl::desc("Target device name"),
        cl::location(targetBackendFlag),
        cl::init(mlir::hacc::TargetDevice::Ascend910B1),
        cl::values(
#define TO_STRING(x) #x
#define REGISTER_TARGET(TARGET)                                                \
  clEnumValN(mlir::hacc::TargetDevice::TARGET, TO_STRING(TARGET),              \
             TO_STRING(TARGET))
            // Ascend910B series
            REGISTER_TARGET(Ascend910B1), REGISTER_TARGET(Ascend910B2),
            REGISTER_TARGET(Ascend910B3), REGISTER_TARGET(Ascend910B4),
            // Ascend910_93 series
            REGISTER_TARGET(Ascend910_9362), REGISTER_TARGET(Ascend910_9372),
            REGISTER_TARGET(Ascend910_9381), REGISTER_TARGET(Ascend910_9382),
            REGISTER_TARGET(Ascend910_9391), REGISTER_TARGET(Ascend910_9392),
            // Ascend310B series
            REGISTER_TARGET(Ascend310B1), REGISTER_TARGET(Ascend310B2),
            REGISTER_TARGET(Ascend310B3), REGISTER_TARGET(Ascend310B4),
            // Ascend950 series
            REGISTER_TARGET(Ascend910_950z), REGISTER_TARGET(Ascend910_9579),
            REGISTER_TARGET(Ascend910_957b), REGISTER_TARGET(Ascend910_957d),
            REGISTER_TARGET(Ascend910_9581), REGISTER_TARGET(Ascend910_9589),
            REGISTER_TARGET(Ascend910_958a), REGISTER_TARGET(Ascend910_958b),
            REGISTER_TARGET(Ascend910_9599), REGISTER_TARGET(Ascend950PR_950z),
            REGISTER_TARGET(Ascend950PR_9579),
            REGISTER_TARGET(Ascend950PR_957a),
            REGISTER_TARGET(Ascend950PR_957b),
            REGISTER_TARGET(Ascend950PR_957c),
            REGISTER_TARGET(Ascend950PR_957d),
            REGISTER_TARGET(Ascend950PR_9589),
            REGISTER_TARGET(Ascend950PR_958a),
            REGISTER_TARGET(Ascend950PR_958b),
            REGISTER_TARGET(Ascend950PR_958c),
            REGISTER_TARGET(Ascend950PR_958d),
            REGISTER_TARGET(Ascend950PR_9599),
            REGISTER_TARGET(Ascend950PR_959a),
            REGISTER_TARGET(Ascend950PR_959b),
            REGISTER_TARGET(Ascend950DT_950x),
            REGISTER_TARGET(Ascend950DT_950y),
            REGISTER_TARGET(Ascend950DT_9571),
            REGISTER_TARGET(Ascend950DT_9572),
            REGISTER_TARGET(Ascend950DT_9573),
            REGISTER_TARGET(Ascend950DT_9574),
            REGISTER_TARGET(Ascend950DT_9575),
            REGISTER_TARGET(Ascend950DT_9576),
            REGISTER_TARGET(Ascend950DT_9577),
            REGISTER_TARGET(Ascend950DT_9578),
            REGISTER_TARGET(Ascend950DT_9581),
            REGISTER_TARGET(Ascend950DT_9582),
            REGISTER_TARGET(Ascend950DT_9583),
            REGISTER_TARGET(Ascend950DT_9584),
            REGISTER_TARGET(Ascend950DT_9585),
            REGISTER_TARGET(Ascend950DT_9586),
            REGISTER_TARGET(Ascend950DT_9587),
            REGISTER_TARGET(Ascend950DT_9588),
            REGISTER_TARGET(Ascend950DT_9591),
            REGISTER_TARGET(Ascend950DT_9592),
            REGISTER_TARGET(Ascend950DT_9595),
            REGISTER_TARGET(Ascend950DT_9596),
            REGISTER_TARGET(Ascend950DT_95A1),
            REGISTER_TARGET(Ascend950DT_95A2), REGISTER_TARGET(Unknown)
#undef REGISTER_TARGET
#undef TO_STRING
                ),
        cl::cat(targetCategory));

    // -------------------------------------------------------------------------//
    //                            Other options
    // -------------------------------------------------------------------------//

    static cl::opt<bool, /*ExternalStorage=*/true> allowUnregisteredDialects(
        "allow-unregistered-dialect",
        cl::desc("Allow operation with no registered dialects"),
        cl::location(allowUnregisteredDialectsFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> enableDirectHIVMLowering(
        "enable-direct-hivm-lowering", cl::desc("enable-direct-hivm-lowering"),
        cl::location(enableDirectHIVMLoweringFlag), cl::init(false));

    // when enableSanitizer is true, enable printDebugInfoOpt
    auto &opts = cl::getRegisteredOptions();
    if ((enableSanitizer || enableDebugInfo) &&
        (opts.count("mlir-print-debuginfo") != 0)) {
      static_cast<cl::opt<bool> *>(opts["mlir-print-debuginfo"])
          ->setValue(true);
    }

    static cl::opt<std::string, /*ExternalStorage=*/true> appendBishengOptions(
        "append-bisheng-options",
        cl::desc("Append options when calling bisheng"),
        cl::location(appendBishengOptionsFlag), cl::init(""));

    static cl::opt<bool, /*ExternalStorage=*/true> useDPX(
        "use-dpx", cl::desc("Enable SIMT lowering through DPX Dialect."),
        cl::location(useDPXFlag), cl::init(true));
  }

  /// Set the callback to get the tiling tuning.
  void setCubeTilingTuningCallback();
  void setGridDimCallback();

  /// Pointer to static cubeTilingTuning variable in constructor.
  cl::list<int64_t> *cubeTilingTuningCLPtr = nullptr;
  cl::list<int64_t> *gridDimCLPtr = nullptr;
};
} // namespace

ManagedStatic<BiShengIRCompileMainConfigCLOptions> clOptionsConfig;

void BiShengIRCompileMainConfig::registerCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptionsConfig;

  clOptionsConfig->setCubeTilingTuningCallback();
  clOptionsConfig->setGridDimCallback();
}

BiShengIRCompileMainConfig BiShengIRCompileMainConfig::createFromCLOptions() {
  // Enforce <= 3 items
  if (clOptionsConfig->gridDimFlags.size() > 3) {
    report_fatal_error(
        "Invalid --simt-triton-grid: at most 3 elements allowed x,y,z.\n");
  }
  clOptionsConfig->setCubeTilingTuningCallback();
  StringTmpPath path(clOptionsConfig->outputFile());
  llvm::cantFail(llvm::errorCodeToError(canonicalizePath(path)),
                 "failed to canonicalize output file path.");
  clOptionsConfig->setOutputFile(path.str().str());
  return *clOptionsConfig;
}

void BiShengIRCompileMainConfigCLOptions::setCubeTilingTuningCallback() {
  if (this->cubeTilingTuningCLPtr)
    this->cubeTilingTuningCLPtr->setCallback(
        [&](int64_t tiling) { this->cubeTilingTuningFlags.push_back(tiling); });
}

void BiShengIRCompileMainConfigCLOptions::setGridDimCallback() {
  if (this->gridDimCLPtr)
    this->gridDimCLPtr->setCallback(
        [&](int64_t grid) { this->gridDimFlags.push_back(grid); });
}
