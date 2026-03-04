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

    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Then reduce across threads within a warp. We can use shared memory
    // if this is the last stage as we can do some warp packing. For the moment
    // keep accs.size() to 1, can be extended to handle larger cases.
    bool ReplaceBFlyWithinWarps =
	ReplaceButterflyReduction && helper.isSharedMemoryReductionPreferred() &&
	(accs.size() == 1);
    reduceWithinWarps(helper, accs, rewriter, ReplaceBFlyWithinWarps);

    if (helper.isWarpSynchronous()) {
      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
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
    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);

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
    if (stride == 1) {
      auto *combineOp = &op.getCombineOp();
      for (unsigned VIdx = 0; VIdx < sizeInterWarps; VIdx++) {
        Value bankOffset = b.urem(b.add(threadId, b.i32_val(VIdx)), b.i32_val(sizeInterWarps));
        Value readOffset = b.add(WarpReadOffset, bankOffset);
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
    } else {
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

    if (ReplaceButterflyReduction && (accs.size() == 1)) {
      auto smemShape = helper.getScratchRepShape();
      SmallVector<Value> smemBases =
          getSmemBases(op, product<unsigned>(smemShape), rewriter, targetInfo);
      auto mod = op->getParentOfType<ModuleOp>();
      int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      int numWarps = triton::gpu::lookupNumWarps(op);
      int numThreads = numLanes * numWarps;
      int workingThreads = numThreads / static_cast<int>(sizeIntraWarps);
      
      // accs.size() is expected to be 1 here, since we have not provisined memory for
      // more than this. If a case requires this and accs is > 1, then we need to make
      // sure we have enough memory for this reduction.
      for (auto it : accs) {
        const SmallVector<unsigned> &key = it.first;
        SmallVector<Value> &acc = accs[key];
      
        // Now store data in memory, such that each working thread is accessing data
        // with a stride equal to the number of working threads.  Data is coming in
        // sequentially, so each batch of consecutive sizeIntraWarps needs to be reduced.
        Location loc = op.getLoc();
        auto b = TritonLLVMOpBuilder(loc, rewriter);
        Value threadId = getThreadId(rewriter, loc);
        Value newLane = b.udiv(threadId, b.i32_val(sizeIntraWarps));
        Value newIndex = b.urem(threadId, b.i32_val(sizeIntraWarps));
        Value writeOffset = b.add(newLane, b.mul(newIndex, b.i32_val(workingThreads))); 
        Value ValTrue = b.true_val();
        for (size_t i = 0; i < op.getNumOperands(); i++) {
          auto elemTy = getElementType(op, i);
          Value writePtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], writeOffset);
          targetInfo.storeShared(rewriter, loc, writePtr, acc[i], ValTrue);
        }
        sync(rewriter, loc, op);

        // Create an if statement to contain the reduction, to allow non-participants
        // to skip through this step.
        Value readOffset = threadId;
        SmallVector<Type> resultTypes;
        for (size_t i = 0; i < op.getNumOperands(); i++) {
          resultTypes.push_back(getElementType(op, i));
        }
        auto threadIsNeeded = b.icmp_slt(threadId, b.i32_val(workingThreads));
        auto IfStmt = rewriter.create<scf::IfOp>(loc, resultTypes, threadIsNeeded, true);
        rewriter.setInsertionPointToStart(&IfStmt.getThenRegion().front());

        // Point the rewriter to the body of the 'then' block and generate the reduction.
        SmallVector<Value> local_acc(op.getNumOperands());
        loadVectorAndAccumulateSingleAcc(helper, smemBases, readOffset, local_acc, threadId,
                                       rewriter, loc, sizeIntraWarps, workingThreads, ValTrue);
        SmallVector<Value> resultsThen;
        for (size_t i = 0; i < op.getNumOperands(); i++) {
          resultsThen.push_back(local_acc[i]);
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

        // We are writing to memory we just used, so no hazards here. We can write before the barrier
        // and then read from corresponding locations.
        for (size_t i = 0; i < op.getNumOperands(); i++) {
          auto elemTy = getElementType(op, i);
          Value resultPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], threadId);
          targetInfo.storeShared(rewriter, loc, resultPtr, IfStmt.getResult(i), threadIsNeeded);
        }

        sync(rewriter, loc, op);

        // Now read all data to each thread
        for (size_t i = 0; i < op.getNumOperands(); i++) {
          auto elemTy = getElementType(op, i);
          Value readPtr =
              b.gep(smemBases[i].getType(), elemTy, smemBases[i], newLane);
          acc[i] = targetInfo.loadShared(rewriter, loc, readPtr, elemTy, ValTrue);
        }
      }
    } else {
      for (auto it : accs) {
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
      bool ReplaceButterflyReduction) const {
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

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    if (ReplaceButterflyReduction) {
      // For next stage we want the warp threads taking adjacent elements from memory, but
      // each thread takes an element that it itself will be reducing, so not related to
      // other threads. We do this to optimize memory access pattern. So, we need a stride for
      // data storage. We can easily do this by swapping smemOrder, assuming smemOrder has
      // more than 1 element. One possible improvement is to add padding to the memory
      // reorganization, which is more than just reshaping the tensor. That would speed
      // things up even for some of the more odd shapes.
      auto mod = op->getParentOfType<ModuleOp>();
      unsigned numLanes = static_cast<unsigned>(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
      int numWarps = triton::gpu::lookupNumWarps(op);
      int numThreads = static_cast<int>(numLanes) * numWarps;
      unsigned elems = product<unsigned>(smemShape);
      unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
      if ((smemOrder.size() > 1) && (elemsPerThread <= numLanes)) {
        auto tmp = smemOrder[smemOrder.size() - 1];
        smemOrder[smemOrder.size() - 1] = smemOrder[0];
        smemOrder[0] = tmp;
      }
    }
    for (auto it : accs) {
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
    if ((!ReplaceButterflyReduction) || (elemsPerThread > numLanes)) {
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
      Value readOffset = b.mul(threadId, b.i32_val(sizeInterWarps));
      SmallVector<Type> resultTypes;
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        resultTypes.push_back(getElementType(op, i));
      }
      auto threadIsNeeded = b.icmp_slt(threadId, b.i32_val(maxThreadsNeeded));
      auto IfStmt = rewriter.create<scf::IfOp>(loc, resultTypes, threadIsNeeded, true);
      rewriter.setInsertionPointToStart(&IfStmt.getThenRegion().front());

      // Point the rewriter to the body of the 'then' block and generate the reduction.
      SmallVector<Value> acc(op.getNumOperands());
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
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
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
