//===- CreateHostMain.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements wrapping logic for the single host function
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/ExecutionEngine/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"

#define DEBUG_TYPE "execution-engine-create-host-main"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_EXECUTIONENGINEHOSTMAINCREATOR
#include "bishengir/ExecutionEngine/Passes.h.inc"
} // namespace mlir

namespace {

using namespace mlir;
using namespace std::placeholders;

struct CreateHostMainPass
    : public impl::ExecutionEngineHostMainCreatorBase<CreateHostMainPass> {
  using Base::Base;
  using ShapedValue = TypedValue<ShapedType>;

  static std::string getFuncNameWithType(const char *funcName, Type type) {
    auto typeStr = llvm::to_string(type);
    TypeSwitch<Type>(type)
        .Case([&typeStr](IndexType type) {
          typeStr.front() = llvm::toUpper(typeStr.front());
        })
        .Default([&typeStr](Type type) {
          llvm::transform(typeStr, typeStr.begin(), llvm::toUpper);
        });
    return funcName + typeStr;
  }

  inline static auto getPrintDataFuncName =
      std::bind(getFuncNameWithType, "printData", _1);
  inline static auto getGetDataFuncName =
      std::bind(getFuncNameWithType, "getData", _1);
  inline static StringRef getFileHandleFuncName = "getFileHandle";
  inline static StringRef closeFileHandleFuncName = "closeFileHandle";

  struct ShapedMetadata {
    ShapedType type;
    SmallVector<Value> dynSizes;

    ShapedMetadata() = default;
    explicit ShapedMetadata(ShapedType type) : type(type) {}
  };

  ValueRange createLibCall(const Location loc, IRRewriter &rewriter,
                           ArrayRef<Type> results, StringRef libFuncName,
                           ArrayRef<Value> operands = {},
                           ArrayRef<NamedAttribute> attrs = {}) {
    auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        getOperation(), rewriter.getStringAttr(libFuncName));
    if (!funcOp) {
      IRRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(getOperation().getBody());
      const auto funcType = rewriter.getFunctionType(
          llvm::map_to_vector(operands,
                              [](Value value) { return value.getType(); }),
          results);
      funcOp = rewriter.create<func::FuncOp>(loc, libFuncName, funcType, attrs);
      funcOp.setPrivate();
    }
    return rewriter.create<func::CallOp>(loc, funcOp, operands).getResults();
  }

  template <typename ResultType, typename = std::enable_if_t<!llvm::is_one_of<
                                     ResultType, void, ValueRange>::value>>
  TypedValue<ResultType>
  createLibCall(const Location loc, IRRewriter &rewriter, ResultType resultType,
                StringRef libFuncName, ArrayRef<Value> operands = {},
                ArrayRef<NamedAttribute> attrs = {}) {
    auto results = createLibCall(loc, rewriter, ArrayRef<Type>(resultType),
                                 libFuncName, operands, attrs);
    return cast<TypedValue<ResultType>>(results.front());
  }

  void createLibCall(const Location loc, IRRewriter &rewriter,
                     StringRef libFuncName, ArrayRef<Value> operands = {},
                     ArrayRef<NamedAttribute> attrs = {}) {
    (void)createLibCall(loc, rewriter, {}, libFuncName, operands, attrs);
  }

  SmallVector<ShapedMetadata> getInputMetadata(IRRewriter &rewriter,
                                               func::FuncOp wrapperFunc,
                                               func::FuncOp kernelFunc) {
    const auto loc = kernelFunc.getLoc();
    SmallVector<ShapedMetadata> metadata;
    for (auto type : kernelFunc.getFunctionType().getInputs())
      metadata.push_back(
          TypeSwitch<Type, ShapedMetadata>(type)
              .Case([&](ShapedType type) {
                ShapedMetadata data(type);
                for (int64_t i = 0; i < type.getRank(); i++)
                  if (type.isDynamicDim(i)) {
                    wrapperFunc.insertArgument(wrapperFunc.getNumArguments(),
                                               rewriter.getIndexType(), {},
                                               loc);
                    data.dynSizes.push_back(wrapperFunc.args_end()[-1]);
                  }
                return data;
              })
              .Default([&](Type type) -> ShapedMetadata {
                llvm::report_fatal_error(("Error: parameter type \"" +
                                          llvm::to_string(type) +
                                          "\" is not currently supported!\n" +
                                          llvm::to_string(kernelFunc))
                                             .c_str());
                return {};
              }));

    return metadata;
  }

  void printTensors(IRRewriter &rewriter,
                    const SmallVector<ShapedValue> &memrefs,
                    func::FuncOp wrapperFunc,
                    std::size_t argumentInsertionIndex) {
    if (memrefs.empty())
      return;
    const auto loc = wrapperFunc.getLoc();
    wrapperFunc.insertArgument(argumentInsertionIndex,
                               rewriter.getType<LLVM::LLVMPointerType>(), {},
                               loc);
    auto fileHandle = createLibCall(
        loc, rewriter, rewriter.getType<LLVM::LLVMPointerType>(),
        getFileHandleFuncName, wrapperFunc.getArgument(argumentInsertionIndex));
    for (auto memref : memrefs)
      createLibCall(
          loc, rewriter,
          getPrintDataFuncName(memref.getType().getElementType()),
          {fileHandle, memref},
          rewriter.getNamedAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                                rewriter.getUnitAttr()));
    createLibCall(loc, rewriter, closeFileHandleFuncName, fileHandle);
  }

  static bool isRankedShapedType(ShapedType type) {
    return isa<RankedTensorType, MemRefType>(type);
  }

  static bool isUnrankedShapedType(ShapedType type) {
    return isa<UnrankedTensorType, UnrankedMemRefType>(type);
  }

  static bool isAnyTensorType(ShapedType type) { return isa<TensorType>(type); }

  static bool isAnyMemRefType(ShapedType type) {
    return isa<BaseMemRefType>(type);
  }

  static ShapedType getUnrankedVersion(ShapedType type) {
    if (isUnrankedShapedType(type))
      return type;
    if (isAnyTensorType(type))
      return UnrankedTensorType::get(type.getElementType());
    return UnrankedMemRefType::get(type.getElementType(), {});
  }

  static ShapedType getUnrankedMemRefType(ShapedValue value) {
    return UnrankedMemRefType::get(value.getType().getElementType(), {});
  }

  static ShapedType getRankedVersion(ShapedType type, ArrayRef<int64_t> sizes) {
    return type.clone(sizes);
  }

  ShapedValue adaptToType(IRRewriter &rewriter, ShapedValue shapedValue,
                          ShapedType targetType) {
    ShapedType sourceType = shapedValue.getType();
    Location loc = shapedValue.getLoc();
    Value result = shapedValue;

    // Handle ranked unranked conversions
    bool toUnranked =
        isRankedShapedType(sourceType) && isUnrankedShapedType(targetType);
    bool toRanked =
        isUnrankedShapedType(sourceType) && isRankedShapedType(targetType);
    if (toUnranked || toRanked) {
      ShapedType castType =
          toUnranked ? getUnrankedVersion(sourceType)
                     : getRankedVersion(sourceType, targetType.getShape());

      if (isAnyTensorType(sourceType)) {
        result = rewriter.create<tensor::CastOp>(loc, castType, result);
      } else {
        result = rewriter.create<memref::CastOp>(loc, castType, result);
      }
    }

    // Handle tensor memref conversions
    bool toMemRef = isAnyTensorType(sourceType) && isAnyMemRefType(targetType);
    bool toTensor = isAnyMemRefType(sourceType) && isAnyTensorType(targetType);

    if (toMemRef) {
      result = rewriter.create<bufferization::ToMemrefOp>(loc, targetType,
                                                          result, nullptr);
    } else if (toTensor) {
      result = rewriter.create<bufferization::ToTensorOp>(
          loc, targetType, result, rewriter.getUnitAttr(),
          rewriter.getUnitAttr());
    }

    return cast<ShapedValue>(result);
  }

  SmallVector<ShapedValue>
  adaptToTypes(IRRewriter &rewriter,
               const SmallVector<ShapedValue> &&shapedValues,
               const SmallVector<ShapedType> &shapedTypes) {
    return llvm::map_to_vector(
        llvm::zip_equal(shapedValues, shapedTypes), [&](auto &&shapedData) {
          auto [shapedValue, shapedType] = shapedData;
          LDBG("Converting from '" << shapedValue << "' to '" << shapedType
                                   << "'");
          return adaptToType(rewriter, shapedValue, shapedType);
        });
  }

  SmallVector<ShapedValue>
  getMemrefs(const Location loc, IRRewriter &rewriter,
             const SmallVector<ShapedMetadata> &&inputMetadata) {
    auto allocedMemrefs = llvm::map_to_vector(
        inputMetadata, [&rewriter, loc](const ShapedMetadata &info) {
          return cast<ShapedValue>(
              rewriter
                  .create<memref::AllocOp>(
                      loc,
                      MemRefType::get(info.type.getShape(),
                                      info.type.getElementType()),
                      info.dynSizes)
                  .getResult());
        });
    auto memrefUnrankedTypes = llvm::map_to_vector(
        inputMetadata, [](const ShapedMetadata &info) -> ShapedType {
          return UnrankedMemRefType::get(info.type.getElementType(), {});
        });
    auto operands =
        adaptToTypes(rewriter, std::move(allocedMemrefs), memrefUnrankedTypes);
    for (auto operand : operands)
      createLibCall(
          loc, rewriter, getGetDataFuncName(operand.getType().getElementType()),
          operand,
          rewriter.getNamedAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                                rewriter.getUnitAttr()));
    return operands;
  }

  hacc::HACCFunction getKernelFunc() {
    auto moduleOp = cast<ModuleOp>(getOperation());
    if (auto *mainFunc = SymbolTable::lookupSymbolIn(moduleOp, wrapperName);
        mainFunc) {
      mainFunc->emitError(StringRef("\"") + wrapperName +
                          "\" function already exists.");
      return nullptr;
    }

    hacc::HACCFunction kernelFunc = nullptr;
    const auto status = moduleOp->walk([&kernelFunc](hacc::HACCFunction op) {
      if (!op.isHost() || op.getHostFuncType() != hacc::HostFuncType::kEntry)
        return WalkResult::advance();

      if (kernelFunc) {
        op.emitError("found multiple host functions.");
        return WalkResult::interrupt();
      }

      kernelFunc = op;
      return WalkResult::advance();
    });
    if (status.wasInterrupted())
      return nullptr;

    if (!kernelFunc) {
      moduleOp.emitError("didn't find host functions.");
      return nullptr;
    }

    return kernelFunc;
  }

  SmallVector<ShapedValue> getResults(func::CallOp kernelCall) {
    SmallVector<ShapedValue> results;

    auto kernelFunc = cast<func::FuncOp>(
        SymbolTable::lookupSymbolIn(getOperation(), kernelCall.getCallee()));
    for (auto paramIdx : llvm::seq(kernelFunc.getNumArguments()))
      if (const auto argTypeAttr =
              kernelFunc.getArgAttrOfType<hacc::KernelArgTypeAttr>(
                  paramIdx, hacc::KernelArgTypeAttr::name);
          argTypeAttr &&
          (argTypeAttr.getArgType() == hacc::KernelArgType::kOutput ||
           argTypeAttr.getArgType() == hacc::KernelArgType::kInputAndOutput))
        results.push_back(cast<ShapedValue>(kernelCall.getOperand(paramIdx)));

    if (kernelCall->getNumResults() > 0)
      results.append(
          llvm::map_to_vector(kernelCall.getResults(), [](Value val) {
            return cast<ShapedValue>(val);
          }));

    return results;
  }

  void runOnOperation() override {
    auto kernelFunc =
        dyn_cast_or_null<func::FuncOp>(getKernelFunc().getOperation());
    if (!kernelFunc) {
      signalPassFailure();
      return;
    }

    const auto loc = kernelFunc.getLoc();
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPointAfter(kernelFunc);

    // Wrapping procedure:
    auto wrapperFunc = rewriter.create<func::FuncOp>(
        loc, wrapperName, rewriter.getFunctionType({}, {}));
    rewriter.setInsertionPointToStart(wrapperFunc.addEntryBlock());

    // Step 1: Parse kernel arguments and create placeholders for unknowns
    auto memrefMetadata = getInputMetadata(rewriter, wrapperFunc, kernelFunc);
    // Step 2: Create Memrefs to pass to the kernel
    auto memrefs = getMemrefs(loc, rewriter, std::move(memrefMetadata));

    // Step 3: Print the randomly generated data to a file with a placeholder
    // path
    auto unrankedMemRefTypes =
        llvm::map_to_vector(memrefs, getUnrankedMemRefType);
    auto operands =
        adaptToTypes(rewriter, std::move(memrefs), unrankedMemRefTypes);
    printTensors(rewriter, operands, wrapperFunc, 0);

    // Step 4: Call the kernel and get the results
    auto kernelOperands = adaptToTypes(
        rewriter, std::move(operands),
        llvm::map_to_vector(
            kernelFunc.getFunctionType().getInputs(),
            static_cast<ShapedType (*)(const Type &)>(cast<ShapedType>)));
    auto kernelCall = rewriter.create<func::CallOp>(
        loc, kernelFunc,
        SmallVector<Value>(kernelOperands.begin(), kernelOperands.end()));

    // Step 5: Print the results to a file with a placeholder path
    auto results = getResults(kernelCall);
    unrankedMemRefTypes = llvm::map_to_vector(results, getUnrankedMemRefType);
    operands = adaptToTypes(rewriter, std::move(results), unrankedMemRefTypes);
    printTensors(rewriter, operands, wrapperFunc, 1);

    rewriter.create<func::ReturnOp>(loc);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::execution_engine::createCreateHostMainPass(
    const ExecutionEngineHostMainCreatorOptions &options) {
  return std::make_unique<CreateHostMainPass>(options);
}
