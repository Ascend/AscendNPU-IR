#include "bishengir/Conversion/TritonAscendGPUToLLVM/PatternTritonAscendGPUOpToLLVM.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/TargetInfo.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

bool mlir::triton::ascend::isSupportedPrintRuntimeType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 1:
    case 8:
    case 16:
    case 32:
    case 64:
      return true;
    default:
      return false;
    }
  }
  return isa<Float16Type, BFloat16Type, Float32Type>(type);
}

namespace {

using namespace mlir;
using namespace mlir::hivm;

static bool isSupportedAssertRuntimeType(Type type) {
  if (auto memdescType = dyn_cast<triton::gpu::MemDescType>(type))
    return memdescType.getElementType().isInteger(8);
  return type.isInteger(1);
}

static bool isSupportedPrintRuntimeOperand(Type type, bool isBool) {
  if (auto memdescType = dyn_cast<triton::gpu::MemDescType>(type)) {
    Type elementType = memdescType.getElementType();
    if (isBool)
      return elementType.isInteger(8);
    return !elementType.isInteger(1) &&
           triton::ascend::isSupportedPrintRuntimeType(elementType);
  }
  return triton::ascend::isSupportedPrintRuntimeType(type);
}

// Returns failure (after emitting a diagnostic) for unsupported operands; the
// caller must abort the rewrite rather than emit a call to a non-existent
// symbol.
FailureOr<std::string>
getOpLibraryCallName(Operation *op, Value arg, bool isSigned,
                     const std::string &baseName = "print",
                     bool isBool = false) {
  Location loc = op->getLoc();
  std::string callName = "_mlir_ciface_" + baseName;
  auto argTy = arg.getType();
  if (baseName == "assert" && !isSupportedAssertRuntimeType(argTy)) {
    op->emitError("device assert runtime does not support condition type ")
        << argTy;
    return failure();
  }
  if (isa<triton::PointerType>(argTy))
    return callName + "_scalar_uint64_t_gm_simt";

  if (baseName == "print" && !isSupportedPrintRuntimeOperand(argTy, isBool)) {
    op->emitError("device print runtime does not support operand type ")
        << argTy;
    return failure();
  }

  hivm::TypeFn casting =
      isSigned ? hivm::TypeFn::cast_signed : hivm::TypeFn::cast_unsigned;
  if (auto argBufTy = dyn_cast_or_null<triton::gpu::MemDescType>(argTy)) {
    int rank = argBufTy.getRank();
    int maxOpRank = 8; // StaticMaxRankTrait
    if (rank > maxOpRank) {
      op->emitError("DebugOp requires rank <= maxOpRank");
      return failure();
    }
    std::string libCallDim = std::to_string(rank) + "d";
    std::string dataTypeStr =
        isBool ? "bool"
               : hivm::util::getTypeName(loc, argBufTy.getElementType(),
                                         casting);
    if (dataTypeStr == "UNKNOWN") {
      op->emitError("device print does not support tensor element type ")
          << argBufTy.getElementType();
      return failure();
    }
    callName += "_" + libCallDim + "_" + dataTypeStr;
    Attribute addressSpace = argBufTy.getMemorySpace();

    if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
      callName = callName + "_" + "ubuf";
    } else {
      op->emitError("print-to-libcall currently only supports UB");
      return failure();
    }
  } else {
    std::string dataTypeStr = hivm::util::getTypeName(loc, argTy, casting);
    if (dataTypeStr == "UNKNOWN") {
      op->emitError("device print does not support operand type ")
          << argTy;
      return failure();
    }
    callName = callName + "_scalar_" + dataTypeStr;
    callName += "_gm";
  }
  return callName + "_simt";
}

static FailureOr<LLVM::CallOp>
createLibCall(ConversionPatternRewriter &rewriter, ModuleOp mod, Operation *op,
              const std::string &libCallName,
              const SmallVector<Value> &inputOperands) {
  Location loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  FlatSymbolRefAttr fnNameAttr = SymbolRefAttr::get(ctx, libCallName);
  // Create external declaration if not already present
  // The actual implementation comes from the template library
  // (meta_op.aiv.xnnn.bc)
  if (!mod.lookupSymbol(fnNameAttr.getAttr())) {
    auto libFnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        llvm::to_vector(ValueRange(inputOperands).getTypes()));
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(mod.getBody(), std::prev(mod.getBody()->end()));

    LLVM::LLVMFuncOp funcOp = rewriter.create<LLVM::LLVMFuncOp>(
        mlir::FileLineColLoc::get(ctx, "internal", 0, 0), fnNameAttr.getValue(),
        libFnType);

    auto haccAlwaysInlineAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
        hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE);
    funcOp->setAttr(haccAlwaysInlineAttr, rewriter.getUnitAttr());
    auto llvmEmitCAttr = LLVM::LLVMDialect::getEmitCWrapperAttrName();
    funcOp->setAttr(llvmEmitCAttr, rewriter.getUnitAttr());
    funcOp->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                    rewriter.getStringAttr("private"));
    funcOp->setAttr(
        mlir::hivm::TFuncCoreTypeAttr::name,
        hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIC_OR_AIV));
    funcOp->setAttr(
        "hivm_regbaseintrins.cconv",
        rewriter.getStringAttr("hivm_regbaseintrins.simt_callable"));
  }

  return rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fnNameAttr.getValue(),
                                       inputOperands);
}

MemRefDescriptor convertTTGMemDescToMemRefWithSMEM(
    Location loc, Value llvmStruct, triton::gpu::MemDescType memdescTy,
    const TypeConverter *typeConverter, ConversionPatternRewriter &rewriter) {

  auto shape = memdescTy.getShape();
  int rank = memdescTy.getRank();
  Type elemTy = memdescTy.getElementType();

  // NOTE: ConvertDebugOpToAscendDPX always feeds us a freshly-created
  // local_alloc with vec=perPhase=maxPhase=1 (no swizzle, no slicing), so
  // getShmemOffset() returns zero here. If you start consuming swizzled
  // allocations, revisit the descriptor below -- the row-major strides we
  // synthesize below assume offset == 0.
  // ------------------------------------------------------------
  // 1. Build SharedMemoryObject
  // ------------------------------------------------------------
  auto smemObj = mlir::LLVM::getSharedMemoryObjectFromStruct(loc, llvmStruct,
                                                             elemTy, rewriter);

  // ------------------------------------------------------------
  // 2. Base pointer
  // ------------------------------------------------------------
  Value basePtr = smemObj.getBase();

  // Launder basePtr from addrspace 3 (Triton shared memory) to addrspace
  // 6 (Ascend UB) via ptrtoint+inttoptr. AscendDPXToHIVMRegbaseIntrins's
  // remap walk reliably relabels standalone ptr<3> SSA values but fails
  // to propagate ptr<3>->ptr<6> through the descriptor struct's field
  // types we build below. On A5 (M300/C310) ptr<3> and ptr<6> alias the
  // same physical memory; the int round-trip preserves the address.
  Type i64Ty = rewriter.getI64Type();
  Type ptrTy6 =
      LLVM::LLVMPointerType::get(rewriter.getContext(), /*addressSpace=*/6);
  Value basePtrAsInt = rewriter.create<LLVM::PtrToIntOp>(loc, i64Ty, basePtr);
  basePtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy6, basePtrAsInt);

  // ------------------------------------------------------------
  // 3. Compute swizzle-aware offset
  // ------------------------------------------------------------
  // are we sure there is no swizzle, no slicing no padding?
  // if so we can replace below with offset = 0
  Value offset = smemObj.getShmemOffset(loc, rewriter, memdescTy);
  // The descriptor crosses the _mlir_ciface_ ABI to the print runtime, whose
  // memref_t (Template/include/Utils.h) uses int64_t offset/sizes/strides and
  // is shared with the SIMD path. On-core SIMT indexing stays i32 (the pass
  // type converter), but these descriptor fields must be i64, so widen the
  // (i32) shmem offset here.
  if (offset.getType() != i64Ty)
    offset = rewriter.create<LLVM::ZExtOp>(loc, i64Ty, offset);

  // ------------------------------------------------------------
  // 4. Create descriptor
  // ------------------------------------------------------------
  // can we get rid of memrefTy
  auto memrefTy = MemRefType::get(
      shape, elemTy,
      /* layout */
      StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                             SmallVector<int64_t>(rank, ShapedType::kDynamic)),
      rewriter.getI32IntegerAttr(6));
  // Build the descriptor through a 64-bit-index converter so its offset/sizes/
  // strides match the runtime memref_t (see offset note above). The pass's own
  // typeConverter has overrideIndexBitwidth(32) and must not be used here.
  (void)typeConverter;
  LowerToLLVMOptions abiOpts(rewriter.getContext());
  abiOpts.overrideIndexBitwidth(64);
  LLVMTypeConverter abiConverter(rewriter.getContext(), abiOpts);
  MemRefDescriptor desc = MemRefDescriptor::undef(
      rewriter, loc, abiConverter.convertType(memrefTy));

  desc.setAllocatedPtr(rewriter, loc, basePtr);
  desc.setAlignedPtr(rewriter, loc, basePtr);
  desc.setOffset(rewriter, loc, offset);

  // ------------------------------------------------------------
  // 5. Sizes
  // ------------------------------------------------------------
  SmallVector<Value> sizes(rank);

  for (int i = 0; i < rank; ++i) {
    sizes[i] = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(shape[i]));

    desc.setSize(rewriter, loc, i, sizes[i]);
  }

  // ------------------------------------------------------------
  // 6. Row-major strides (safe after blocked layout)
  // ------------------------------------------------------------
  SmallVector<Value> strides(rank);

  if (rank > 0) {
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(1));
    strides[rank - 1] = one;

    for (int i = rank - 2; i >= 0; --i) {
      strides[i] =
          rewriter.create<LLVM::MulOp>(loc, strides[i + 1], sizes[i + 1]);
    }

    for (int i = 0; i < rank; ++i) {
      desc.setStride(rewriter, loc, i, strides[i]);
    }
  }

  return desc;
}

static Value createGlobalStringInAS(Location loc, OpBuilder &builder,
                                    StringRef name, StringRef value,
                                    LLVM::Linkage linkage, unsigned addrSpace) {
  auto module = builder.getInsertionBlock()
                    ->getParentOp()->getParentOfType<ModuleOp>();
  MLIRContext *ctx = builder.getContext();
  auto arrayTy =
      LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), value.size());

  LLVM::GlobalOp global;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    global = builder.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/true, linkage, name,
        builder.getStringAttr(value), /*alignment=*/0, addrSpace);
  }

  // AddressOfOp infers ptr<addrSpace> from the global, so it can't disagree.
  Value addr = builder.create<LLVM::AddressOfOp>(loc, global);
  auto ptrTy = cast<LLVM::LLVMPointerType>(addr.getType());
  return builder.create<LLVM::GEPOp>(loc, ptrTy, arrayTy, addr,
                                     ArrayRef<LLVM::GEPArg>{0, 0});
}

// Builds an scf::If whose then-block runs only on logical thread (0,0,0), and
// moves the insertion point into that then-block so the caller can emit
// the guarded body. The caller replaces the original op with the result.
static scf::IfOp createThread0GuardIf(ConversionPatternRewriter &rewriter,
                                      Location loc) {
  Value tidX = getThreadId(rewriter, loc);
  Value tidY = rewriter.create<ascend_dpx::ThreadIdYOp>(
      loc, rewriter.getIntegerType(32));
  Value tidZ = rewriter.create<ascend_dpx::ThreadIdZOp>(
      loc, rewriter.getIntegerType(32));
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  Value cmpX =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tidX, zero);
  Value cmpY =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tidY, zero);
  Value cmpZ =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tidZ, zero);
  Value isThread0 = rewriter.create<arith::AndIOp>(
      loc, rewriter.create<arith::AndIOp>(loc, cmpX, cmpY), cmpZ);

  auto ifOp = rewriter.create<scf::IfOp>(loc, isThread0, /*withElse=*/false);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  return ifOp;
}

// The input print op contains:
//  - a "prefix" (string) specified by the user, and
//  - one or more "operands" (tensors).
struct PrintOpConversion : public ConvertOpToLLVMPattern<ascend_dpx::PrintOp> {
  explicit PrintOpConversion(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<ascend_dpx::PrintOp>(typeConverter,
                                                          benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ascend_dpx::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    // Convert the prefix attribute to two arguments of the called func
    static int prefixNumber = 0;
    StringAttr prefixAttr = op.getPrefixAttr();
    auto prefixStrName = "_debug_prefix_" + std::to_string(prefixNumber++);
    auto prefixValue = createGlobalStringInAS(
      op.getLoc(), rewriter, prefixStrName, prefixAttr.strref(),
      LLVM::Linkage::Private, (unsigned)ascend_dpx::AscendDPXAddressSpace::GLOBAL_MEM);

    auto prefixLenType = IntegerType::get(ctx, 64);
    auto prefixLenValue = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), prefixLenType,
        rewriter.getI64IntegerAttr(prefixAttr.size()));

    // only thread0 prints
    auto ifOp = createThread0GuardIf(rewriter, loc);

    ValueRange args = op.getArgs();
    ValueRange llvmArgs = adaptor.getArgs();
    ArrayRef<int32_t> isSignedAttrs = op.getIsSigned();
    ArrayRef<int32_t> isBoolAttrs = op.getIsBool();
    for (size_t i = 0, e = args.size(); i < e; ++i) {
      Value origVal = args[i];
      Value llvmStruct = llvmArgs[i];
      bool isSigned = isSignedAttrs[i] != 0;
      bool isBool = isBoolAttrs[i] != 0;

      SmallVector<Value> funcOperands;
      // do we want to print prefix for every operand?
      funcOperands.push_back(prefixValue);
      funcOperands.push_back(prefixLenValue);

      // Add DebugOp's (non-attr) arguments
      if (auto memdescTy =
              dyn_cast<triton::gpu::MemDescType>(origVal.getType())) {
        MemRefDescriptor desc = convertTTGMemDescToMemRefWithSMEM(
            loc, llvmStruct, memdescTy, getTypeConverter(), rewriter);
        Value descValue = desc;

        auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
        Value one = rewriter.create<LLVM::ConstantOp>(
            loc, getTypeConverter()->convertType(rewriter.getIndexType()),
            rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
        Value allocated = rewriter.create<LLVM::AllocaOp>(
            loc, ptrTy, descValue.getType(), one, /*alignment=*/0);
        rewriter.create<LLVM::StoreOp>(loc, descValue, allocated);

        funcOperands.push_back(allocated);
      } else {
        if (isa<triton::PointerType>(origVal.getType())) {
          if (!isa<LLVM::LLVMPointerType>(llvmStruct.getType())) {
            op.emitError("expected converted pointer operand, got ")
                << llvmStruct.getType();
            return failure();
          }
          auto i64Ty = IntegerType::get(ctx, 64);
          funcOperands.push_back(
              rewriter.create<LLVM::PtrToIntOp>(loc, i64Ty, llvmStruct));
        } else {
          funcOperands.push_back(llvmStruct);
        }
      }

      // dispatch to different lib calls for assert/print
      FailureOr<std::string> libCallName = getOpLibraryCallName(
          op, origVal, isSigned, /*baseName=*/"print", isBool);
      if (failed(libCallName))
        return failure();

      // Convert the hex attr to an argument of the print lib call
      bool hexBool = op.getHex() || isa<triton::PointerType>(origVal.getType());
      Value hexInt = rewriter.create<arith::ConstantOp>(
          op->getLoc(), rewriter.getI8IntegerAttr(hexBool));
      funcOperands.push_back(hexInt);

      // create the library call and modify the corresponding funcOp's
      // TFuncCoreType
      auto moduleOp = op->template getParentOfType<ModuleOp>();
      if (failed(createLibCall(rewriter, moduleOp, op, *libCallName,
                               funcOperands)))
        return failure();
    }

    rewriter.replaceOp(op, ifOp);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

struct AssertOpConversion
    : public ConvertOpToLLVMPattern<ascend_dpx::AssertOp> {
  explicit AssertOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<ascend_dpx::AssertOp>(typeConverter,
                                                           benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ascend_dpx::AssertOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();

    static int msgNumber = 0;
    StringAttr msgAttr = op.getMsgAttr();
    auto msgStrName = "_debug_msg_" + std::to_string(msgNumber++);
    auto msgValue = createGlobalStringInAS(
        loc, rewriter, msgStrName, msgAttr.strref(),
        LLVM::Linkage::Private,
        (unsigned)ascend_dpx::AscendDPXAddressSpace::GLOBAL_MEM);

    auto msgLenType = IntegerType::get(ctx, 64);
    auto msgLenValue = rewriter.create<LLVM::ConstantOp>(
        loc, msgLenType, rewriter.getI64IntegerAttr(msgAttr.size()));

    auto ifOp = createThread0GuardIf(rewriter, loc);

    Value cond = op.getCond();
    Value llvmCond = adaptor.getCond();

    SmallVector<Value> funcOperands;
    funcOperands.push_back(msgValue);
    funcOperands.push_back(msgLenValue);

    if (auto memdescTy =
            dyn_cast<triton::gpu::MemDescType>(cond.getType())) {
      MemRefDescriptor desc = convertTTGMemDescToMemRefWithSMEM(
          loc, llvmCond, memdescTy, getTypeConverter(), rewriter);
      Value descValue = desc;

      auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, getTypeConverter()->convertType(rewriter.getIndexType()),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
      Value allocated = rewriter.create<LLVM::AllocaOp>(
          loc, ptrTy, descValue.getType(), one, /*alignment=*/0);
      rewriter.create<LLVM::StoreOp>(loc, descValue, allocated);

      funcOperands.push_back(allocated);
    } else {
      funcOperands.push_back(llvmCond);
    }

    FailureOr<std::string> libCallName =
        getOpLibraryCallName(op, cond, true, "assert");
    if (failed(libCallName))
      return failure();

    auto moduleOp = op->template getParentOfType<ModuleOp>();
    if (failed(createLibCall(rewriter, moduleOp, op, *libCallName,
                              funcOperands)))
      return failure();

    rewriter.replaceOp(op, ifOp);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::ascend::populateDebugOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<PrintOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<AssertOpConversion>(typeConverter, targetInfo, benefit);
}
