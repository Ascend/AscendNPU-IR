#ifndef PROTONGPU_TO_LLVM_TARGETINFO_ASCEND_H
#define PROTONGPU_TO_LLVM_TARGETINFO_ASCEND_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/TargetInfo.h"

namespace mlir::triton::proton::gpu::ASCEND {
class TargetInfo : public mlir::triton::proton::gpu::TargetInfoBase {
public:
  explicit TargetInfo(const mlir::triton::ascend::TargetInfo &helper)
      : mlir::triton::proton::gpu::TargetInfoBase(helper) {}

  const mlir::triton::ascend::TargetInfo &getTritonTargetInfo() const override {
    return static_cast<const mlir::triton::ascend::TargetInfo &>(helper);
  }

  Value clock(ConversionPatternRewriter &rewriter, Location loc,
              bool isClock64) const override;

  Value processorId(ConversionPatternRewriter &rewriter,
                    Location loc) const override;

  int getAddressSpace(Attribute addressSpace) const override;

  int getIndexPtrAddrSpace() const override;

  ~TargetInfo() {}
};
} // namespace mlir::triton::proton::gpu::ASCEND

#endif // PROTONGPU_TO_LLVM_TARGETINFO_ASCEND_H
