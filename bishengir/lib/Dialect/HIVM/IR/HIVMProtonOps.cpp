#include "bishengir/Dialect/HIVM/IR/HIVMProtonOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;
using namespace mlir::hivm;

llvm::LogicalResult ProtonScopeBoundaryOp::verify() {
  int64_t regionId = getRegionIdAttr().getInt();
  if (regionId < 0 || regionId > static_cast<int64_t>(proton::kMaxRegionId)) {
    return emitOpError("region_id must fit in 12 bits");
  }
  return success();
}
