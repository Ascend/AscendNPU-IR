#include "bishengir/Conversion/ProtonAscendGPUToLLVM/ProtonAscendGPUToLLVM.h"

#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "bishengir/Conversion/GPUToDPX/GPUOpToDPX.h"
#include "bishengir/Conversion/ProtonAscendGPUToLLVM/PatternProtonAscendGPUOpToLLVM.h"
#include "bishengir/Conversion/ProtonAscendGPUToLLVM/TargetInfo.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/PatternTritonAscendGPUOpToLLVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir::triton::proton::gpu {
#define GEN_PASS_DEF_CONVERTPROTONASCENDGPUTOLLVM
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir::triton::proton::gpu

using namespace mlir;
using namespace mlir::triton;

namespace {

class ProtonLLVMConversionTarget : public ConversionTarget {
public:
  explicit ProtonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::ascend_dpx::AscendDPXDialect>();
    addIllegalDialect<mlir::triton::proton::gpu::ProtonGPUDialect>();
    addIllegalDialect<mlir::triton::proton::ProtonDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertProtonAscendGPUToLLVM
    : public mlir::triton::proton::gpu::impl::ConvertProtonAscendGPUToLLVMBase<
          ConvertProtonAscendGPUToLLVM> {
  explicit ConvertProtonAscendGPUToLLVM() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    if (!triton::util::getPassColumnDigit(mod, "convert-triton-gpu-to-llvm")) {
      return;
    }

    auto tritonTargetInfo =
        mlir::triton::ascend::TargetInfo();
    auto protonTargetInfo =
        mlir::triton::proton::gpu::ASCEND::TargetInfo(tritonTargetInfo);
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(
      context, option, tritonTargetInfo);
    populateTypeConversions(typeConverter, protonTargetInfo);
    mlir::triton::proton::gpu::populateProtonGPUOpPatterns(
        typeConverter, patterns, protonTargetInfo, 1);
    mlir::triton::proton::gpu::ASCEND::populateProtonGPUOpAscendPatterns(
        typeConverter, patterns, protonTargetInfo, 1);
    mlir::arith::populateArithToLLVMConversionPatterns(
      typeConverter, patterns);
    triton::ascend::populateGPUOpToDPXPatterns(typeConverter, patterns,
                                               1);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(
      typeConverter, patterns);
    auto convTarget = ProtonLLVMConversionTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    OpPassManager pm;
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, mod)))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::proton::gpu {

std::unique_ptr<Pass> createConvertProtonAscendGPUToLLVMPass() {
  return std::make_unique<ConvertProtonAscendGPUToLLVM>();
}

} // namespace mlir::troton::proton::gpu
