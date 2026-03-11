#include "bishengir/Conversion/ProtonAscendGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::proton::gpu::ASCEND {

Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {
  auto Type = isClock64 ? i64_ty : i32_ty;
  if (isClock64)
    return rewriter.create<mlir::ascend_dpx::Clock64Op>(loc, Type);
  else
    return rewriter.create<mlir::ascend_dpx::Clock32Op>(loc, Type);
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  return rewriter.create<mlir::ascend_dpx::CoreIdOp>(loc, i32_ty);
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = (int)ascend_dpx::AscendDPXAddressSpace::SHARED_MEM;
  } else if (mlir::isa<proton::gpu::GlobalMemorySpaceAttr>(addressSpace)) {
    spaceId = (int)ascend_dpx::AscendDPXAddressSpace::GLOBAL_MEM;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace, "
                             "and GlobalMemorySpace for now");
  }
  return spaceId;
}

int TargetInfo::getIndexPtrAddrSpace() const {
  return 1;
}

} // namespace mlir::triton::proton::gpu::ASCEND
