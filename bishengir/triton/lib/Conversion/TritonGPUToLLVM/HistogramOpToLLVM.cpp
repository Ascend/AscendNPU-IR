#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

static void atomicAdd(Value ptr, Value val, Location loc,
                      ConversionPatternRewriter &rewriter) {
#if BSPUB_DAVINCI_BISHENGIR
  rewriter.create<ascend_dpx::AtomicAddOp>(loc, val.getType(), ptr, val);
#else
  rewriter.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, ptr, val,
                                     LLVM::AtomicOrdering::monotonic, "cta");
#endif
}

static Value castHistogramIndexToI32(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     Value value) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto intTy = cast<IntegerType>(value.getType());
  unsigned width = intTy.getWidth();
  if (width < 32)
    return b.zext(i32_ty, value);
  if (width > 32)
    return b.trunc(i32_ty, value);
  return value;
}

static Value intConstantLike(Location loc, ConversionPatternRewriter &rewriter,
                             Value value, int64_t constant) {
  Type type = value.getType();
  return rewriter.create<LLVM::ConstantOp>(
      loc, type, rewriter.getIntegerAttr(type, constant));
}

static Value getHistogramInRangePredicate(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          Value value, int numBins) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto intTy = cast<IntegerType>(value.getType());
  unsigned width = intTy.getWidth();
  if (intTy.isUnsigned()) {
    if (width < 32) {
      Value extended = b.zext(i32_ty, value);
      return b.icmp_ult(extended, b.i32_val(numBins));
    }
    return b.icmp_ult(value,
                      intConstantLike(loc, rewriter, value, numBins));
  }

  if (width < 32) {
    Value extended = b.sext(i32_ty, value);
    Value nonNegative = b.icmp_sge(extended, b.i32_val(0));
    Value lessThanNumBins = b.icmp_slt(extended, b.i32_val(numBins));
    return b.and_(nonNegative, lessThanNumBins);
  }
  Value nonNegative =
      b.icmp_sge(value, intConstantLike(loc, rewriter, value, 0));
  Value lessThanNumBins =
      b.icmp_slt(value, intConstantLike(loc, rewriter, value, numBins));
  return b.and_(nonNegative, lessThanNumBins);
}

static SmallVector<Value>
computeHistogram(Location loc, ConversionPatternRewriter &rewriter,
                 Value baseSharedMemPtr, const SmallVector<Value> &srcValues,
                 const SmallVector<Value> &maskValues, int numBins,
                 int scratchNumBins, int numThreadPerWarp,
                 const SmallVector<Value> &indices, Value threadId,
                 int numWarps) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> histogramValues;
  // Initialize the shared memory with zeros.
  int64_t numElementPerThread =
      ceil<int64_t>(scratchNumBins, numThreadPerWarp * numWarps);
  for (int i = 0; i < numElementPerThread; ++i) {
    Value offset =
        b.add(threadId, b.i32_val((i * numWarps * numThreadPerWarp)));
    offset = b.urem(offset, b.i32_val(scratchNumBins));
    Value sharedMemPtr =
        b.gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, offset);
    b.store(b.i32_val(0), sharedMemPtr);
  }
  b.barrier();

  // Apply atomic add to update the histogram in shared memory.  Values outside
  // the real output bin range are mapped to bin 0 with an increment of 0, so the
  // GEP is always in bounds while invalid inputs do not affect the result.
  for (size_t i = 0; i < srcValues.size(); ++i) {
    Value value = srcValues[i];
    Value updatePred =
        getHistogramInRangePredicate(loc, rewriter, value, numBins);
    if (!maskValues.empty())
      updatePred = b.and_(updatePred, maskValues[i]);

    Value index = castHistogramIndexToI32(loc, rewriter, value);
    index = b.select(updatePred, index, b.i32_val(0));
    Value increment = b.select(updatePred, b.i32_val(1), b.i32_val(0));
    Value sharedMemPtr =
        b.gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, index);
    atomicAdd(sharedMemPtr, increment, loc, rewriter);
  }

  b.barrier();
  // load the histogram to register with the right layout.
  for (Value index : indices) {
    Value sharedMemPtr =
        b.gep(baseSharedMemPtr.getType(), i32_ty, baseSharedMemPtr, index);
    Value val = b.load(i32_ty, sharedMemPtr);
    histogramValues.push_back(val);
  }
  return histogramValues;
}

namespace {
struct HistogramOpConversion
    : public ConvertOpToLLVMPattern<triton::HistogramOp> {
public:
  using ConvertOpToLLVMPattern<triton::HistogramOp>::ConvertOpToLLVMPattern;

  explicit HistogramOpConversion(LLVMTypeConverter &typeConverter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    SmallVector<Value> srcValues = unpackLLElements(loc, input, rewriter);

    Value llMask = adaptor.getMask();
    SmallVector<Value> maskValues;
    if (llMask)
      maskValues = unpackLLElements(loc, llMask, rewriter);

    int numBins = op.getType().getDimSize(0);
    auto mod = op->getParentOfType<ModuleOp>();
    int numThreadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    assert(numThreadsPerWarp == 32 ||
           numThreadsPerWarp == 64 &&
               "Only supports 32 or 64 threads per warp");
    int numWarps = triton::gpu::lookupNumWarps(op);
    // Pad out the bins so that we have at least one bin per thread within a
    // warp.
    int scratchNumBins = std::max(numBins, numThreadsPerWarp);
    Value threadId = getThreadId(rewriter, loc);
    auto srcType = op.getSrc().getType();

    // Use atomic adds to update the histogram in shared memory.
    Value baseSharedMemPtr =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto dstType = op.getType();
    Attribute dstEncoding = dstType.getEncoding();
    auto indices = emitIndices(op.getLoc(), rewriter, targetInfo, dstEncoding,
                               dstType, true);
    SmallVector<Value> innerDimIndices;
    for (size_t i = 0; i < indices.size(); ++i)
      innerDimIndices.push_back(indices[i][0]);
    SmallVector<Value> histogramValue = computeHistogram(
        loc, rewriter, baseSharedMemPtr, srcValues, maskValues, numBins,
        scratchNumBins, numThreadsPerWarp, innerDimIndices, threadId, numWarps);

    // Depending on the layout, some threads may have duplicate data. We can
    // account for this by calculating a "replication factor" and dividing the
    // results by it to avoid overcounting.
    auto replicationFactor = numWarps * numThreadsPerWarp;
    auto threadsPerWarp = getThreadsPerWarp(srcType);
    auto warpsPerCTA =
        getWarpsPerCTA(srcType.getEncoding(), srcType.getShape());
    replicationFactor /= std::accumulate(
        threadsPerWarp.begin(), threadsPerWarp.end(), 1, std::multiplies<>());
    replicationFactor /= std::accumulate(warpsPerCTA.begin(), warpsPerCTA.end(),
                                         1, std::multiplies<>());

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (size_t i = 0; i < histogramValue.size(); ++i) {
      histogramValue[i] =
          b.sdiv(histogramValue[i], b.i32_val(replicationFactor));
    }

    Value results = packLLElements(loc, typeConverter, histogramValue, rewriter,
                                   op.getType());
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::populateHistogramOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<HistogramOpConversion>(typeConverter, targetInfo, benefit);
}
