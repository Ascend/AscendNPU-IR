#include "AscendReduceScanCommon.h"

#include "bishengir/Conversion/TritonAscendGPUToLLVM/TargetInfo.h"
#include "bishengir/Analysis/AscendUtility.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/PatternTritonAscendGPUOpToLLVM.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ascend;

using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DistributedEncodingTrait;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getThreadOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct AscendReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  AscendReduceOpConversion(LLVMTypeConverter &typeConverter,
                           const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AscendReduceOpHelper helper(op);
    assert(helper.isReduceWithinCTA() &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();
    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    Operation *BaseOp = op.getOperation();
    bool ReplaceButterflyReduction = (util::getPassColumnDigit(BaseOp, "reduce-op") != 0);
    bool isVariadic = (op.getNumResults() > 1);
    // Variadic outputs are not supported.
    if (isVariadic)
      ReplaceButterflyReduction = false;
    LDBG("Transforming ReduceOps in TritonAscendGPUToLLVM");
    LDBG(" * Butterfly replacement is " << (ReplaceButterflyReduction ? "ON" : "OFF"));
    LDBG(" * is variadic: " << (isVariadic ? "TRUE" : "FALSE"));
    auto srcShape = helper.getSrcShape();
    if (srcShape.size() == 1) {
      ReplaceButterflyReduction = false;
      LDBG(" * Butterfly replacement is disabled for 1D shapes");
    }
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Then reduce across threads within a warp. We can use shared memory
    // if this is the last stage as we can do some warp packing. For the moment
    // keep accs.size() to 1, can be extended to handle larger cases.
    bool ReplaceBFlyWithinWarps =
        ReplaceButterflyReduction && helper.isSharedMemoryReductionPreferred();
    auto mod = op->getParentOfType<ModuleOp>();
    LDBG(" * accs.size() = " << accs.size());
    LDBG(" * sizeIntraWarps = " << helper.getIntraWarpSizeWithUniqueData());
    LDBG(" * interleave = " << helper.getThreadOffsetOnReductionAxis());
    LDBG(" * numLanes = " << triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    LDBG(" * numWarps = " << triton::gpu::lookupNumWarps(op));
    LDBG(" * shared mem preferred = " << helper.isSharedMemoryReductionPreferred());
    LDBG(" * Butterfly replacement within warps is "
        << (ReplaceBFlyWithinWarps ? "ON" : "OFF"));
    reduceWithinWarps(helper, accs, rewriter, ReplaceBFlyWithinWarps);

    if (helper.isWarpSynchronous()) {
      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
      LDBG(" * Warp-synchronous reduction does not require further butterfly replacement");
      return success();
    }

    // Compute a shared memory base per operand.
    auto smemShape = helper.getScratchRepShape();
    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter,
		                  ReplaceButterflyReduction);

    sync(rewriter, loc, op);

    // The second round of shuffle reduction
    //   now the problem size: sizeInterWarps, s1, s2, .. , sn
    //   where sizeInterWarps is 2^m
    //
    // Each thread needs to process:
    //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
    accumulatePartialReductions(helper, smemBases, rewriter, ReplaceButterflyReduction);

    // We could avoid this barrier in some of the layouts, however this is not
    // the general case.
    // TODO: optimize the barrier in case the layouts are accepted.
    sync(rewriter, loc, op);

    // set output values
    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter,
		               ReplaceButterflyReduction);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size()) {
      acc.resize(results.size());
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.barrier();
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      AscendReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);

    // Thread X might hold the same input value in two registers.  Get the
    // indices in `offsets` that hold unique values, and only accumulate over
    // those.
    llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
    for (size_t i = 0; i < offsets.size(); ++i) {
      uniqueOffsets.insert({offsets[i], i});
    }
    auto *combineOp = &op.getCombineOp();
    auto srcIndices = emitIndices(op.getLoc(), rewriter, targetInfo,
                                  helper.getSrcLayout(), operandType, true);

    // reduce within threads
    for (const auto &[_, i] : uniqueOffsets) {
      SmallVector<unsigned> key = offsets[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(op.getLoc(), rewriter, *combineOp, accs[key], srcValues[i]);
      if (isFirst)
        indices[key] = srcIndices[i];
    }
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave,
                  Value pred = {}) const {
    auto success = targetInfo.warpReduce(rewriter, loc, acc, op,
                                         numLaneToReduce, interleave);
    if (success)
      return;
    for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
      SmallVector<Value> shfl(acc.size());
      for (unsigned i = 0; i < acc.size(); ++i) {
        shfl[i] = targetInfo.shuffleXor(rewriter, loc, acc[i], N * interleave);
      }
      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl, pred);
    }
  }

  void loadVectorAndAccumulateSingleAcc(AscendReduceOpHelper &helper, SmallVector<Value> &smemBases,
		    Value WarpReadOffset, SmallVector<Value> &acc, Value threadId,
                    ConversionPatternRewriter &rewriter, Location loc, unsigned sizeInterWarps,
		    unsigned stride, Value &threadIsNeeded) const {
    triton::ReduceOp op = helper.getOperation();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    std::unordered_map<unsigned, SmallVector<Value>> items;
    LDBG(" * using stride-based accumulation");
    auto *combineOp = &op.getCombineOp();
    for (unsigned VIdx = 0; VIdx < sizeInterWarps; VIdx++) {
      Value readOffset = b.add(threadId, b.i32_val(stride * VIdx));
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
        items[VIdx].push_back(targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                               threadIsNeeded));
      }
      if (VIdx == 0) {
        for (unsigned AccIdx = 0; AccIdx < op.getNumOperands(); ++AccIdx) {
          acc[AccIdx] = items[0][AccIdx];
        }
      } else {
        // Now accumulate.
        accumulate(op.getLoc(), rewriter, *combineOp, acc, items[VIdx]);
      }
    }
  }

  // Reduce across threads within each warp.
  void
  reduceWithinWarps(AscendReduceOpHelper &helper,
                    std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                    ConversionPatternRewriter &rewriter,
                    bool ReplaceButterflyReduction) const {
    triton::ReduceOp op = helper.getOperation();
    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned threadOffsetOnReductionAxis =
        helper.getThreadOffsetOnReductionAxis();
    // If profitable, use shared memory.
    if (ReplaceButterflyReduction) {
      LDBG(" * Butterfly replacement within warps applied");
      auto smemShape = helper.getScratchRepShape();
      unsigned axis = op.getAxis();
      SmallVector<Value> smemBases =
          getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);
      auto mod = op->getParentOfType<ModuleOp>();
      int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      int numWarps = triton::gpu::lookupNumWarps(op);
      int interleave = (int)helper.getThreadOffsetOnReductionAxis();
      int numThreads = numLanes * numWarps;
      int workingThreads = numThreads / static_cast<int>(sizeIntraWarps);
      LDBG(" * workingthreads = " << workingThreads);
      LDBG(" * numLanes = " << numLanes);
      LDBG(" * numWarps = " << numWarps);
      int accOffset = workingThreads * (int)sizeIntraWarps;
      Location loc = op.getLoc();
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      Value threadId = getThreadId(rewriter, loc);
      Value newLane = b.udiv(threadId, b.i32_val(sizeIntraWarps*interleave));
      Value newIndex = b.urem(threadId, b.i32_val(sizeIntraWarps*interleave));
      if (interleave > 1) {
        newLane = b.add(b.mul(newLane, b.i32_val(interleave)), b.urem(threadId, b.i32_val(interleave)));
        newIndex = b.udiv(newIndex, b.i32_val(interleave));
      }
      auto *combineOp = &op.getCombineOp();
      LDBG(" * axis = " << axis);
      LDBG(" * smemShape fields = " << (product<unsigned>(smemShape)));
      unsigned neededTIDs = product<unsigned>(smemShape) / sizeIntraWarps;
      LDBG(" * neededTIDs = " << neededTIDs);
      
      // Now store data in the shared memory so that we can repack warps. Here is how
      // we do this:
      // 1. We compute the number of working threads, each of which will sum up sizeIntraWarps
      //    number of values. So workingThreads * sizeIntraWarps is the amount of memory to use
      //    for each acc (and there can be multiple per thread).
      // 2. We define newLane to determine the thread that is going to reduce subsequent
      //    sizeIntraWarps elements.
      // 3. We then define newIndex which will be the multiplier to offset storing of elements
      //    each thread reduces, where the multiplicand is the number of workingThreads.
      // 4. For each subsequent acc, we offset the entire address computation by
      //    workingThreads * sizeIntraWarps.
      unsigned AccIdx = 0;
      for (auto &it : accs) {
        const SmallVector<unsigned> &key = it.first;
        SmallVector<Value> &acc = accs[key];
        // Now store data in memory, such that each working thread is accessing data
        Value dataOffset = b.add(newLane, b.mul(newIndex, b.i32_val(workingThreads)));
        Value writeOffset = b.add(dataOffset, b.i32_val(accOffset * AccIdx));
        Value ValTrue = b.true_val();
        for (size_t i = 0; i < op.getNumOperands(); i++) {
          auto elemTy = getElementType(op, i);
          Value writePtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
          targetInfo.storeShared(rewriter, loc, writePtr, acc[i], /*writeIsNeeded*/ ValTrue);
        }
        AccIdx++;
      }

      // Now put a barrier to ensure all data is in shared memory.
      sync(rewriter, loc, op);

      // Now we put in an if condition to only work on threads that are packed into working warps
      SmallVector<Type> resultTypes;
      auto threadIsNeeded = b.icmp_slt(threadId, b.i32_val(neededTIDs));
      auto IfStmt = rewriter.create<scf::IfOp>(loc, resultTypes, threadIsNeeded, true);
      rewriter.setInsertionPointToStart(&IfStmt.getThenRegion().front());
      SmallVector<Value> resultsThen;
      AccIdx = 0;
      for (size_t idx = 0; idx < accs.size(); ++idx) {
        // Point the rewriter to the body of the 'then' block and generate the reduction.
        Value ValTrue = b.true_val();
        SmallVector<Value> local_acc(op.getNumOperands());
        for (unsigned VIdx = 0; VIdx < sizeIntraWarps; VIdx++) {
          SmallVector<Value> items;
          Value readOffset =
              b.add(threadId, b.i32_val(VIdx * workingThreads + AccIdx * accOffset));
          for (unsigned i = 0; i < op.getNumOperands(); ++i) {
            auto elemTy = getElementType(op, i);
            Value readPtr =
                b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
            items.push_back(targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                                  ValTrue));
          }
          if (VIdx == 0) {
            for (unsigned OpIdx = 0; OpIdx < op.getNumOperands(); ++OpIdx) {
              local_acc[OpIdx] = items[OpIdx];
            }
          } else {
            // Now accumulate.
            accumulate(op.getLoc(), rewriter, *combineOp, local_acc, items);
          }
        }
        // Store accumulator result into shared memory at the threadId spot.
        for (size_t i = 0; i < op.getNumOperands(); i++) {
          auto elemTy = getElementType(op, i);
          Value ValTrue = b.true_val();
          Value writeOffset =
              b.add(threadId, b.i32_val(AccIdx * accOffset));
          Value resultPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
          targetInfo.storeShared(rewriter, loc, resultPtr, local_acc[i], ValTrue);
        }
        // Go to the next acc.
        AccIdx++;
      }
      // Now terminate the IfStmt and restore insertion point.
      rewriter.setInsertionPointAfter(IfStmt);

      // Now synchronize with a barrier after the computation is done.
      sync(rewriter, loc, op);

      // Now read all data to each thread. Threads now read data from their lane
      // (newLane) that was computing the reduction for the entire thread set.
      AccIdx = 0;
      for (auto &it : accs) {
        const SmallVector<unsigned> &key = it.first;
        SmallVector<Value> &acc = accs[key];
        for (size_t i = 0; i < op.getNumOperands(); i++) {
          auto elemTy = getElementType(op, i);
          Value ValTrue = b.true_val();
          Value readOffset =
              b.add(newLane, b.i32_val(AccIdx * accOffset));
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy, ValTrue);
        }
        AccIdx++;
      }
    } else {
      LDBG(" * Butterfly replacement not profitable within warps");
      for (auto &it : accs) {
        const SmallVector<unsigned> &key = it.first;
        SmallVector<Value> &acc = accs[key];
        warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                   threadOffsetOnReductionAxis);
      }
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(AscendReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (unsigned j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }

  void storeWarpReduceToSharedMemory(
      AscendReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter,
      bool &ReplaceButterflyReduction) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcLayout =
        mlir::cast<DistributedEncodingTrait>(helper.getSrcLayout());
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchRepShape();

    // Lezcano: We should move all the shared memory logic to use LLs natively
    auto srcShape = helper.getSrcShape();
    auto kLane = rewriter.getStringAttr("lane");
    auto [multiDimLaneId, isRepresentativeLane] =
        delinearize(rewriter, loc, srcLayout, srcShape, kLane, laneId);
    auto kWarp = rewriter.getStringAttr("warp");
    auto [multiDimWarpId, isRepresentativeWarp] =
        delinearize(rewriter, loc, srcLayout, srcShape, kWarp, warpId);

    Value laneIdAxis = multiDimLaneId[axis];
    Value laneZero = b.icmp_eq(laneIdAxis, b.i32_val(0));
    Value write =
        b.and_(b.and_(isRepresentativeLane, isRepresentativeWarp), laneZero);

    Value warpIdAxis = multiDimWarpId[axis];

    // Check if the number of threads required to do the job would be reduced
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned numLanes = static_cast<unsigned>(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    int numWarps = triton::gpu::lookupNumWarps(op);
    int numThreads = static_cast<int>(numLanes) * numWarps;
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    unsigned threadsNeeded = elems / sizeInterWarps;
    LDBG(" * MEM write -> threads needed = " << threadsNeeded);
    LDBG(" * MEM write -> threads available = " << numThreads);

    // The warp packing only makes sense if we use fewer threads. Now we can still
    // do more work per thread if needed, but that will be a future extension.
    if (numThreads < static_cast<int>(threadsNeeded) * 2)
      ReplaceButterflyReduction = false;

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    LDBG(" * MEM write -> axis = " << axis);
    LDBG(" * MEM write -> smemOrder.size() = " << smemOrder.size());
    if (ReplaceButterflyReduction) {
      // For next stage we want the warp threads taking adjacent elements from memory, but
      // each thread takes an element that it itself will be reducing, so not related to
      // other threads. We do this to optimize memory access pattern. So, we need a stride for
      // data storage. We can easily do this by swapping smemOrder, assuming smemOrder has
      // more than 1 element. One possible improvement is to add padding to the memory
      // reorganization, which is more than just reshaping the tensor. That would speed
      // things up even for some of the more odd shapes.
      unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
      if ((smemOrder.size() > 1) && (elemsPerThread <= numLanes)) {
        auto tmp = smemOrder[smemOrder.size() - 1];
        smemOrder[smemOrder.size() - 1] = smemOrder[0];
        smemOrder[0] = tmp;
      } else {
        // If there was a reason to abandon reduction, then flag that
        // so at the read stage we don't retrieve incorrect results.
        ReplaceButterflyReduction = false;
      }
    }
    LDBG(" * MEM write -> accs.size() = " << accs.size());
    for (auto &it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;
      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        targetInfo.storeShared(rewriter, loc, writePtr, acc[i], write);
      }
    }
  }

  // Load the reduction of each warp and accumulate them to a final value and
  // store back to shared memory.
  void accumulatePartialReductions(AscendReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   ConversionPatternRewriter &rewriter,
				   bool ReplaceButterflyReduction) const {
    triton::ReduceOp op = helper.getOperation();
    auto smemShape = helper.getScratchRepShape();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto mod = op->getParentOfType<ModuleOp>();
    unsigned numLanes = static_cast<unsigned>(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    int numWarps = triton::gpu::lookupNumWarps(op);
    int numThreads = static_cast<int>(numLanes) * numWarps;
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = b.i32_val(numLanes);
    Value laneId = b.urem(threadId, warpSize);

    Value zero = b.i32_val(0);
    Value ValTrue = b.true_val();
    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    if (!ReplaceButterflyReduction) {
      LDBG(" * Using original implementation.");
      LDBG(" * elems = " << elems);
      LDBG(" * sizeInterWarps = " << sizeInterWarps);
      LDBG(" * axis = " << op.getAxis());
      LDBG(" * numLanes = " << triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
      LDBG(" * numWarps = " << triton::gpu::lookupNumWarps(op));
      Value threadIsNeeded = b.icmp_slt(threadId, b.i32_val(elems));
      Value readOffset = threadId;
      for (unsigned round = 0; round < elemsPerThread; ++round) {
        SmallVector<Value> acc(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          auto elemTy = getElementType(op, i);
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy,
                                         threadIsNeeded);
        }
        warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */,
                   threadIsNeeded);
        // only the first thread in each sizeInterWarps is writing
        Value writeOffset = readOffset;
        SmallVector<Value> writePtrs(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          auto elemTy = getElementType(op, i);
          writePtrs[i] =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
        }

        Value laneIdModSizeInterWarps = b.urem(laneId, b.i32_val(sizeInterWarps));
        Value laneIdModSizeInterWarpsIsZero =
            b.icmp_eq(laneIdModSizeInterWarps, zero);
        Value pred = b.and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          targetInfo.storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
        }
        if (round != elemsPerThread - 1) {
          readOffset = b.add(readOffset, b.i32_val(numThreads));
        }
      }
    } else {
      LDBG(" * Using shared-memory warp packing optimization");
      // Now since we are reducing using only 1 thread and only one thread per
      // sizeInterWarps is writing, then technically we only need
      // elems / sizeInterWarps threads to do the job. So lets pack those threads
      // into minimum number of warps and use them to do the reduction together.
      // This should allow us to make all other warps no-ops for the purpose of this
      // step.
      auto axis = op.getAxis();
      auto reductionAxisSize = smemShape[axis];
      unsigned stride = product<unsigned>(smemShape) / reductionAxisSize;
      int maxThreadsNeeded = static_cast<int>(elems / sizeInterWarps);
      // Create an if statement to contain the reduction, to allow non-participants
      // to skip through this step.
      SmallVector<Value> acc(op.getNumOperands());
      LDBG(" * sizeInterWarps = " << sizeInterWarps);
      LDBG(" * stride = " << stride);
      LDBG(" * axis = " << axis);
      LDBG(" * redAxisSize = " << reductionAxisSize);
      LDBG(" * numLanes = " << triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
      LDBG(" * numWarps = " << triton::gpu::lookupNumWarps(op));
      LDBG(" * maxThreadsNeeded = " << maxThreadsNeeded);
      LDBG(" * elements = " << elems);
      Value readOffset = threadId;
      SmallVector<Type> resultTypes;
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        resultTypes.push_back(getElementType(op, i));
      }
      auto threadIsNeeded = b.icmp_slt(threadId, b.i32_val(maxThreadsNeeded));
      auto IfStmt = rewriter.create<scf::IfOp>(loc, resultTypes, threadIsNeeded, true);
      rewriter.setInsertionPointToStart(&IfStmt.getThenRegion().front());

      // Point the rewriter to the body of the 'then' block and generate the reduction.
      loadVectorAndAccumulateSingleAcc(helper, smemBases, readOffset, acc, threadId,
                                       rewriter, loc, sizeInterWarps, stride, ValTrue);
      SmallVector<Value> resultsThen;
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        resultsThen.push_back(acc[i]);
      }
      rewriter.create<scf::YieldOp>(loc, resultsThen);
      // Return results in the else region.
      rewriter.setInsertionPointToStart(&IfStmt.getElseRegion().front());

      SmallVector<Value> resultsElse;
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        resultsElse.push_back(rewriter.create<LLVM::UndefOp>(loc, resultTypes[i]));
      }
      rewriter.create<scf::YieldOp>(loc, resultsElse);

      // Now terminate the IfStmt and restore insertion point.
      rewriter.setInsertionPointAfter(IfStmt);

      // Because of how we are accessing memory and how we are writing it back to shared memory,
      // there is a race condition, since we are writing in the following store to a location
      // another warp could be reading from. Thus, put a barrier.
      sync(rewriter, loc, op);

      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] =
            b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
      }
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        targetInfo.storeShared(rewriter, loc, writePtrs[i], IfStmt.getResult(i), threadIsNeeded);
      }
    }
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(AscendReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter,
				  bool ReplaceButterflyReduction) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    if (ReplaceButterflyReduction) {
      // Change read order to match how we organized memory in the first place.
      auto tmp = smemOrder[0];
      smemOrder[0] = smemOrder[smemOrder.size() - 1];
      smemOrder[smemOrder.size() - 1] = tmp;
    }
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        // nd-tensor where n >= 1
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, targetInfo,
                                         resultLayout, resultTy, true);
        auto resultShape = resultTy.getShape();
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), b.i32_val(0));
          for (size_t resultIdx = 0, resultDim = resultShape.size();
               resultIdx < resultDim; ++resultIdx) {
            auto smemIdx = resultIdx < op.getAxis() ? resultIdx : resultIdx + 1;
            if (resultShape[resultIdx] > smemShape[smemIdx]) {
              // When srcShape smaller than src sizePerThread, only srcShape
              // elements is accumulated in smem. Modulo smemShape effectively
              // replicates srcShape elements to src sizePerThread.
              readIdx[smemIdx] =
                  b.urem(readIdx[smemIdx], b.i32_val(smemShape[smemIdx]));
            }
          }
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], readOffset);
          resultVals[j] = b.load(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = b.load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }
};
} // namespace

void mlir::triton::ascend::populateAscendReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<AscendReduceOpConversion>(typeConverter, targetInfo, benefit);
}
