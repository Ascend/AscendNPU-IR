//===- DebugMemory.cpp - ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a LLVM IR dialect debugging.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/ADT/SmallVector.h"

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"

#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "debug-memory"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace bishengir {
#define GEN_PASS_DEF_DEBUGMEMORY
#include "bishengir/Transforms/Passes.h.inc"
} // namespace bishengir

using namespace mlir;

#define PASS_NAME "debug-memory"
//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

// #if defined (_DEBUG)  || defined (DEBUG)
// #define DEBUG_MEMORY_ENABLE_LOGGING 1
// #endif
// #define DEBUG_MEMORY_ENABLE_LOGGING 1

#define DEBUG_MEMORY_ENABLE_LOGGING 1

#if defined(DEBUG_MEMORY_ENABLE_LOGGING)
#define DEBUG_LLVM_LOG_DEBUG_COUT 1
class DebugMemoryLoger {
  bool initialized = false;
  bool dumpToCacheFile = false;
  std::unique_ptr<llvm::raw_fd_ostream> fileStream;

  DebugMemoryLoger() = default;
  ~DebugMemoryLoger() = default;
  DebugMemoryLoger(DebugMemoryLoger&) = delete;
  DebugMemoryLoger(DebugMemoryLoger&&) = delete;
  DebugMemoryLoger& operator=(DebugMemoryLoger&) = delete;
  DebugMemoryLoger& operator=(DebugMemoryLoger&&) = delete;
public:
  void init(const std::string &cacheFilePath) {
    if (!initialized && !cacheFilePath.empty()) {
      std::error_code EC;
      fileStream = std::make_unique<llvm::raw_fd_ostream>(cacheFilePath, EC, llvm::sys::fs::OF_Append);
      if (EC) {
          llvm::errs() << "[DebugMemoryLoger] Error opening file: " << EC.message() << "\n";
      } else {
        dumpToCacheFile = true;
      }
      initialized = true;
    }
  }

  template <typename... Args>
  void logInfo(Args... args) {
    if (dumpToCacheFile) {
      ( (*fileStream << args), ... );
    }
    else {
      ( (llvm::outs() << args), ... );
    }
  }

  template <typename... Args>
  void dbgInfo(Args... args) {
#if defined(DEBUG_LLVM_LOG_DEBUG_COUT)
    ( (std::cout << args), ... );
#else
    if (dumpToCacheFile) {
      ( (*fileStream << args), ... );
    }
    else {
      ( (llvm::dbgs() << args), ... );
    }
#endif
  }

  template <typename... Args>
  void logError(Args... args) {
    ( (llvm::errs() << args), ... );
  }

  void dumpFunction(
    ModuleOp moduleOp,
    const std::string& functionName) {
    auto functionOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(functionName);
    if (dumpToCacheFile) {
      functionOp.print(*fileStream);
      (*fileStream) << "\n";
    } else {
      functionOp.dump();
      llvm::outs() << "\n";
    }
  }

  void dumpOp(Operation *op) {
    if (dumpToCacheFile) {
      op->print(*fileStream);
      (*fileStream) << "\n";
    } else {
      op->dump();
      llvm::outs() << "\n";
    }
  }

  static DebugMemoryLoger& get(){
    static DebugMemoryLoger logger;
    return logger;
  }
};

#if defined(DEBUG_LLVM_LOG_DEBUG_COUT)
#define DEBUG_LLVM_LOG_DEBUG(...) \
  DebugMemoryLoger::get().dbgInfo("[DEBUG ASCEND]" __VA_OPT__(,' ',) __VA_ARGS__); \
  std::cout << std::endl;
#else
#define DEBUG_LLVM_LOG_DEBUG(...) \
  DebugMemoryLoger::get().dbgInfo("[DEBUG ASCEND]" __VA_OPT__(,' ',) __VA_ARGS__); \
  llvm::dbgs() << "\n";
#endif

#if defined(DEBUG_LLVM_LOG_DEBUG_COUT)
#define DEBUG_LLVM_LOG_DEBUG_EXEC(msg, func) \
  std::cout << "[DEBUG ASCEND] " << __func__  << ": " << msg << std::flush; \
  func; \
  std::cout << std::endl;
#else
#define DEBUG_LLVM_LOG_DEBUG_EXEC(msg, func) \
  llvm::dbgs() << "[DEBUG ASCEND] " << __func__  << ": " << msg << "\n"; \
  func; \
  llvm::dbgs() << "\n";
#endif

#define DEBUG_LLVM_LOG_INFO(...) DebugMemoryLoger::get().logInfo("[INFO ASCEND]" __VA_OPT__(,' ',) __VA_ARGS__)
#define DEBUG_LLVM_LOG_ERROR(...) DebugMemoryLoger::get().logError("[ERROR ASCEND]" __VA_OPT__(,' ',) __VA_ARGS__)
#define DEBUG_LLVM_LOG_FUNC(moduleOp, functionName) DebugMemoryLoger::get().dumpFunction(moduleOp, functionName)
#define DEBUG_LLVM_LOG_OP(op) DebugMemoryLoger::get().dumpOp(op)

#else

#define DEBUG_LLVM_LOG_DEBUG(...)
#define DEBUG_LLVM_LOG_DEBUG_EXEC(msg, func)

#define DEBUG_LLVM_LOG_INFO(...)
#define DEBUG_LLVM_LOG_ERROR(...)
#define DEBUG_LLVM_LOG_FUNC(moduleOp, functionName)
#define DEBUG_LLVM_LOG_OP(op)

#endif

enum class ActionType {
  UNKNOWN,
  UBPROBE
};

ActionType convertToActionT(const llvm::StringRef& act) {
  if (act.lower() == "ubprobe") {
    return ActionType::UBPROBE;
  }
  return ActionType::UNKNOWN;
}

class DebugActionHandler {
public:
  using uptr = std::unique_ptr<DebugActionHandler>;

  virtual LogicalResult handleAction(ModuleOp, DictionaryAttr) = 0;

  virtual ~DebugActionHandler() = default;

  static uptr getHandler(ActionType actT);

protected:
  static Operation* findOperation(
    ModuleOp moduleOp,
    const std::string& functionName,
    const std::string& instructionName,
    const size_t argumentIndex) {
    Operation* foundOperation = nullptr;

    moduleOp.walk([&](func::FuncOp functionOp) {
      const auto functionSymName = functionOp.getSymName();
      if (functionSymName.str() == functionName) {
        if (functionOp->getNumRegions() != 1) {
          DEBUG_LLVM_LOG_ERROR(functionSymName + " function regions amount is bigger then 1\n");
          return WalkResult::interrupt();
        }

        Block& entryBlock = functionOp.getRegion().front();
        llvm::ArrayRef<BlockArgument> args = entryBlock.getArguments();

        if (args.size() <= argumentIndex) {
          DEBUG_LLVM_LOG_ERROR("argument index ", argumentIndex, " is not correct (", args.size(), ")\n");
          return WalkResult::interrupt();
        }

        BlockArgument arg = args[argumentIndex];

        std::list<Operation*> operations;
        for (OpOperand& operand : arg.getUses()) {
          operations.push_back(operand.getOwner());
        }

        while (!operations.empty()) {
          auto it = operations.begin();
          Operation* operation = *it;

          DEBUG_LLVM_LOG_DEBUG_EXEC("findOperation:", operation->dump());

          operations.remove(*it);

          if (operation->getName().getStringRef().str() == instructionName) {
            foundOperation = operation;
            return WalkResult::interrupt();
          }

          auto results = operation->getResults();
          for (Value value : results) {
            for (OpOperand& operand : value.getUses()) {
              operations.push_back(operand.getOwner());
            }
          }
        }
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    DEBUG_LLVM_LOG_DEBUG("done");
    return foundOperation;
  }

  static std::vector<Operation*> findOperations(ModuleOp& moduleOp, const std::string& operationName) {
    std::vector<Operation*> foundOperations;
    moduleOp->walk([&](Operation* op) {
      if (op->getName().getStringRef().str() == operationName) {
        foundOperations.push_back(op);
      }
      return WalkResult::advance();
    });
    return foundOperations;
  }

  static std::vector<Operation*> findOperations(Operation* op, const std::vector<std::string>& operationNames) {
    std::vector<Operation*> foundOperations;
    op->walk([&](Operation* op) {
      for (const std::string& operationName : operationNames) {
        if (op->getName().getStringRef().str() == operationName) {
            foundOperations.push_back(op);
          }
      }
      return WalkResult::advance();
    });
    return foundOperations;
  }

  static std::vector<func::FuncOp> findEntryFunction(Operation* op) {
    std::vector<func::FuncOp> foundOperations;

    op->walk([&](func::FuncOp func) {
      auto argAttrs = func.getArgAttrs();
      if (argAttrs) {
        foundOperations.push_back(func);
      }
      return WalkResult::advance();
    });

    return foundOperations;
  }
};

const static std::string maskedStore = "ave.hir.masked_store";

const std::vector<std::string> tensorChangeOperations = {
  "ave.hir.vmul",
  "ave.hir.vadd",
  "ave.hir.vintlv"
};

class Config {
public:
  static std::string getenv(const std::string& name) {
    const auto value = ::getenv(name.c_str());
    return value == nullptr ? "" : std::string(value);
  }

  static size_t getenvInt(const std::string& name, const size_t defaultValue) {
    const auto value = ::getenv(name.c_str());
    return ((value == nullptr) || (value[0] == '\0')) ? defaultValue : std::atoi(value);
  }

  static Config getEnv() {
    return Config(
      std::string(getenv("ASCEND_DEBUG_PRINT")),
      getenvInt("ASCEND_DEBUG_PROB_OFFSET", 0),
      getenvInt("ASCEND_DEBUG_PROB_ALLOC_SIZE", 0),
      StringRef(getenv("TRITON_HIVMC_DB_VF_DISABLED")) == "1",
      StringRef(getenv("TRITON_HIVMC_DB_VF_CALCULATE_OFFSET")) == "1",
      StringRef(getenv("TRITON_HIVMC_DB_VF_STORE_DISABLED")) == "1",
      StringRef(getenv("TRITON_HIVMC_DB_VF_SUBVIEW_DISABLED")) == "1",
      StringRef(getenv("TRITON_HIVMC_DB_VF_PGE_DISABLED")) == "1",
      StringRef(getenv("TRITON_HIVMC_DB_CAST_DISABLED")) == "1",
      StringRef(getenv("TRITON_HIVMC_DB_STORE_DISABLED")) == "1",
      StringRef(getenv("TRITON_HIVMC_DB_STORE_REPLACE_OUTPUT_DISABLED")) == "1",
      StringRef(getenv("TRITON_HIVMC_DB_VF_INSTRUCTIONS_DISABLED")).str());
  }

  std::string str() const {
    std::stringstream ss;
    ss << "\tdebugPrint=" << debugPrint << std::endl;
    ss << "\tprobOffset=" << probOffset << std::endl;
    ss << "\tprobAllocSize=" << probAllocSize << std::endl;
    ss << "\tvfDisabled=" << vfDisabled << std::endl;
    ss << "\tvfCalculateOffset=" << vfCalculateOffset << std::endl;
    ss << "\tvfStoreDisabled=" << vfStoreDisabled << std::endl;
    ss << "\tvfSubviewDisabled=" << vfSubviewDisabled << std::endl;
    ss << "\tvfPgeDisabled=" << vfPgeDisabled << std::endl;
    ss << "\tcastDisabled=" << castDisabled << std::endl;
    ss << "\tstoreDisabled=" << storeDisabled << std::endl;
    ss << "\treplaceOutputDisabled=" << replaceOutputDisabled << std::endl;
    ss << "\tvfInstructionsDisabled=" << vfInstructionsDisabled << std::endl;
    return ss.str();
  }

  Config(
    const std::string& debugPrint,
    const size_t probOffset,
    const size_t probAllocSize,
    const bool vfDisabled,
    const bool vfCalculateOffset,
    const bool vfStoreDisabled,
    const bool vfSubviewDisabled,
    const bool vfPgeDisabled,
    const bool castDisabled,
    const bool storeDisabled,
    const bool replaceOutputDisabled,
    const std::string& vfInstructionsDisabled) :
      debugPrint(debugPrint),
      probOffset(probOffset),
      probAllocSize(probAllocSize),
      vfDisabled(vfDisabled),
      vfCalculateOffset(vfCalculateOffset),
      vfStoreDisabled(vfStoreDisabled),
      vfSubviewDisabled(vfSubviewDisabled),
      vfPgeDisabled(vfPgeDisabled),
      castDisabled(castDisabled),
      storeDisabled(storeDisabled),
      replaceOutputDisabled(replaceOutputDisabled),
      vfInstructionsDisabled(vfInstructionsDisabled) {}

  const std::string debugPrint;
  const size_t probOffset;
  const size_t probAllocSize;

  const bool vfDisabled;
  const bool vfCalculateOffset;
  const bool vfStoreDisabled;
  const bool vfSubviewDisabled;
  const bool vfPgeDisabled;
  const bool castDisabled;
  const bool storeDisabled;
  const bool replaceOutputDisabled;
  const std::string vfInstructionsDisabled;
};

class OpBuilderContextItem {
public:
  OpBuilderContextItem(
    const std::string name,
    const Type type,
    const mlir::Attribute attributeValue) :
      name(name),
      type(type),
      attributeValue(attributeValue) {}

  OpBuilderContextItem(
    const std::string name,
    const Type type,
    const mlir::Value value1,
    const mlir::Value value2) :
      name(name),
      type(type),
      value1(value1),
      value2(value2) {}

  bool operator==(const OpBuilderContextItem& other) const {
    return
      (name == other.name) &&
      (type == other.type) &&
      (attributeValue == other.attributeValue) &&
      (value1 == other.value1) &&
      (value2 == other.value2);
  }

  const std::string name;
  const Type type;
  const mlir::Attribute attributeValue;
  const mlir::Value value1;
  const mlir::Value value2;
};

struct ContextItemHash {
  size_t operator()(const OpBuilderContextItem& item) const {
    std::hash<std::string> h;
    const size_t typeHash = DenseMapInfo<Type>::getHashValue(item.type);
    const size_t attributeValueHash = item.attributeValue ?
      DenseMapInfo<Attribute>::getHashValue(item.attributeValue) : 0;
    const size_t value1Hash = item.value1 ?
      DenseMapInfo<Value>::getHashValue(item.value1) : 0;
    const size_t value2Hash = item.value2 ?
      DenseMapInfo<Value>::getHashValue(item.value2) : 0;
    return h(item.name) + typeHash + attributeValueHash + value1Hash * 2 + value2Hash * 4;
  }
};

class OpBuilderContext {
public:
  template <typename T>
  T get(mlir::Type type, mlir::Attribute value) const {
    const auto name = llvm::getTypeName<T>().str();
    OpBuilderContextItem item(name, type, value);
    auto it = items.find(item);
    if (it == items.end()) {
      return T();
    }
    return dyn_cast<T>(it->second);
  }

  template <typename T>
  T get(mlir::Type type, mlir::Value value1, mlir::Value value2) const {
    const auto name = llvm::getTypeName<T>().str();
    OpBuilderContextItem item(name, type, value1, value2);
    auto it = items.find(item);
    if (it == items.end()) {
      return T();
    }
    return dyn_cast<T>(it->second);
  }

  template <typename T>
  void add(mlir::Type type, mlir::Attribute value, T& instance) {
    const auto name = llvm::getTypeName<T>().str();
    OpBuilderContextItem item(name, type, value);
    items.emplace(item, instance);
  }

  template <typename T>
  void add(mlir::Type type, mlir::Value value1, mlir::Value value2, T& instance) {
    const auto name = llvm::getTypeName<T>().str();
    OpBuilderContextItem item(name, type, value1, value2);
    items.emplace(item, instance);
  }

private:
  std::unordered_map<OpBuilderContextItem, Operation*, ContextItemHash> items;
};

class DebugMemoryContext {
public:
  arith::ConstantOp createConstant(
      OpBuilder& functionBuilder,
      Location functionLocation,
      const Type type,
      const TypedAttr value) {
    if (!type) {
      DEBUG_LLVM_LOG_ERROR("type is empty");
      return nullptr;
    }

    if (!value) {
      DEBUG_LLVM_LOG_ERROR("attribute is empty");
      return nullptr;
    }

    arith::ConstantOp constantOp = context.get<arith::ConstantOp>(type, value);
    if (!constantOp) {
      constantOp = functionBuilder.create<arith::ConstantOp>(
        functionLocation,
        type,
        value);
      context.add<arith::ConstantOp>(type, value, constantOp);
    }
    return constantOp;
  }

  hivmave::VFPgeOp createVFPgeOp(
      OpBuilder& functionBuilder,
      Location functionLocation,
      const VectorType outVecType,
      const mlir::hivmave::PgePattern pgePattern) {
    const auto patternAttr = mlir::hivmave::PgePatternAttr::get(
      functionBuilder.getContext(),
      pgePattern);

    hivmave::VFPgeOp pgeOp = context.get<hivmave::VFPgeOp>(outVecType, patternAttr);
    if (!pgeOp) {
      auto maskType = VectorType::get(
        SmallVector<int64_t> {outVecType.getNumElements()},
        functionBuilder.getI1Type());

      pgeOp = functionBuilder.create<hivmave::VFPgeOp>(
        functionLocation,
        maskType,
        patternAttr);

      context.add<hivmave::VFPgeOp>(outVecType, patternAttr, pgeOp);
    }
    return pgeOp;
  }

  hivmave::VFBroadcastScalarMaskOp createVFBroadcastScalarMaskOp(
      OpBuilder& functionBuilder,
      Location functionLocation,
      const VectorType outVecType,
      const Value constant,
      const Value mask) {
    auto broadcastOp = context.get<hivmave::VFBroadcastScalarMaskOp>(
      outVecType,
      constant,
      mask);
    if (!broadcastOp) {
      broadcastOp = functionBuilder.create<hivmave::VFBroadcastScalarMaskOp>(
        functionLocation,
        outVecType,
        constant,
        mask);

      context.add<hivmave::VFBroadcastScalarMaskOp>(
        outVecType,
        constant,
        mask,
        broadcastOp);
    }
    return broadcastOp;
  }

private:
  OpBuilderContext context;
};

class HeaderBuilder {
public:
  enum class OperationId {
    OPERATION_VMULL = 1,
    OPERATION_VADD,
    OPERATION_UNKNOWN
  };

  enum class TypeId {
    TYPE_F16 = 1,
    TYPE_F32,
    TYPE_UNKNOWN
  };

  HeaderBuilder(
      const DebugMemoryContext& context,
      const OpBuilder& functionBuilder,
      const Location& functionLocation) :
      context(context),
      functionBuilder(functionBuilder),
      functionLocation(functionLocation) {}

  class HeaderDefinition {
  public:
    HeaderDefinition(
      const size_t size,
      const arith::ConstantOp dataSizeConstantOp,
      const hivmave::VFBroadcastScalarMaskOp dataSizeBroadcastOp,
      const hivmave::VFBroadcastScalarMaskOp typeIdBroadcastOp,
      const arith::ConstantOp instructionIdConstantOp,
      const arith::ConstantOp offsetConstantOp,
      const hivmave::VFBroadcastScalarMaskOp instructionIdBroadcastOp) :
        size(size),
        dataSizeConstantOp(dataSizeConstantOp),
        dataSizeBroadcastOp(dataSizeBroadcastOp),
        typeIdBroadcastOp(typeIdBroadcastOp),
        instructionIdConstantOp(instructionIdConstantOp),
        offsetConstantOp(offsetConstantOp),
        instructionIdBroadcastOp(instructionIdBroadcastOp) {
    }

    const size_t size;
    const arith::ConstantOp dataSizeConstantOp;
    hivmave::VFBroadcastScalarMaskOp dataSizeBroadcastOp;
    hivmave::VFBroadcastScalarMaskOp typeIdBroadcastOp;
    arith::ConstantOp instructionIdConstantOp;
    arith::ConstantOp offsetConstantOp;
    hivmave::VFBroadcastScalarMaskOp instructionIdBroadcastOp;
  };

  class Header {
  public:
    Header() : headerSize(0), dataSize(0) {}

    Header(
      const memref::SubViewOp subViewOp,
      const arith::AddIOp addIOp,
      const Value loopIterArg,
      const size_t headerSize,
      const size_t dataSize) :
        subViewOp(subViewOp),
        addIOp(addIOp),
        loopIterArg(loopIterArg),
        headerSize(headerSize),
        dataSize(dataSize) {
    }

    explicit operator bool() const {
      return (subViewOp && addIOp && loopIterArg && (headerSize != 0) && (dataSize != 0));
    }

    memref::SubViewOp subViewOp;
    arith::AddIOp addIOp;
    const Value loopIterArg;
    const size_t headerSize;
    const size_t dataSize;
  };

  Header createHeader(
      OpBuilder& storeBuilder,
      Location& storeLocation,
      Operation& operation,
      const Value functionProbArg,
      const Value loopIterArg,
      arith::AddIOp offsetIncrementAddOp,
      memref::SubViewOp intermediateSubView,
      OperandRange& indices) {
    ::llvm::ArrayRef<int64_t> staticSizes = intermediateSubView.getStaticSizes();
    if (staticSizes.empty()) {
      DEBUG_LLVM_LOG_ERROR("staticSizes is empty");
      return Header();
    }

    if (staticSizes.size() != 1) {
      DEBUG_LLVM_LOG_ERROR("staticSizes: unexpected size: " + std::to_string(staticSizes.size()));
      return Header();
    }

    const auto dataSize = staticSizes[0];
    const MemRefType resultType = intermediateSubView.getType();
    const Type type = resultType.getElementType();

    auto headerDefinition = createHeaderDefinition(operation, dataSize, type);

    const hivmave::StoreDist storePattern = hivmave::StoreDist::NORM_B32;

    auto resultTypes = operation.getResultTypes();
    const Type outType = resultTypes.front();
    const VectorType outVecType = cast<VectorType>(outType);

    IRMapping map;
    map.map(intermediateSubView->getOperand(0), functionProbArg);
    map.map(intermediateSubView->getOperand(1), loopIterArg);
    auto subView = storeBuilder.clone(*intermediateSubView, map);
    memref::SubViewOp subViewOp = dyn_cast<memref::SubViewOp>(subView);

    {
      const auto pgeOp = context.createVFPgeOp(
        storeBuilder,
        storeLocation,
        outVecType,
        mlir::hivmave::PgePattern::ALL);

      storeBuilder.create<hivmave::VFMaskedStoreOp>(
        storeLocation,
        storePattern,
        subViewOp,
        indices,
        pgeOp->getResult(0),
        headerDefinition.instructionIdBroadcastOp);
    }

    {
      const auto pgeOp = context.createVFPgeOp(
        storeBuilder,
        storeLocation,
        outVecType,
        mlir::hivmave::PgePattern::VL2);

      storeBuilder.create<hivmave::VFMaskedStoreOp>(
        storeLocation,
        storePattern,
        subViewOp,
        indices,
        pgeOp->getResult(0),
        headerDefinition.typeIdBroadcastOp);
    }

    {
      const auto pgeOp = context.createVFPgeOp(
        storeBuilder,
        storeLocation,
        outVecType,
        mlir::hivmave::PgePattern::VL1);

      storeBuilder.create<hivmave::VFMaskedStoreOp>(
        storeLocation,
        storePattern,
        subViewOp,
        indices,
        pgeOp->getResult(0),
        headerDefinition.dataSizeBroadcastOp);
    }

    Operation* offsetIncrementAdd = offsetIncrementAddOp.getOperation();
    IRMapping addMap;
    addMap.map(offsetIncrementAddOp->getOperand(0), loopIterArg);
    addMap.map(offsetIncrementAddOp->getOperand(1), headerDefinition.offsetConstantOp);
    Operation* add = storeBuilder.clone(*offsetIncrementAdd, addMap);

    arith::AddIOp addOp = dyn_cast<arith::AddIOp>(add);
    const Value newLoopIterArg = addOp.getResult();

    return Header(subViewOp, addOp, newLoopIterArg, headerDefinition.size, dataSize);
  }

private:
  static OperationId getOperationId(const Operation& op) {
    if (isa<hivmave::VFMulOp>(op)) {
      return OperationId::OPERATION_VMULL;
    } else if (isa<hivmave::VFAddOp>(op)) {
      return OperationId::OPERATION_VADD;
    } else {
      return OperationId::OPERATION_UNKNOWN;
    }
  }

  static TypeId getTypeId(const Type type, OpBuilder& builder) {
    if (builder.getF16Type() == type) {
      return TypeId::TYPE_F16;
    } else if (builder.getF32Type() == type) {
      return TypeId::TYPE_F32;
    } else {
      return TypeId::TYPE_UNKNOWN;
    }
  }

  mlir::TypedAttr getAttribute(const Type type, const size_t value) {
    if (functionBuilder.getF16Type() == type) {
      return functionBuilder.getF16FloatAttr(value);
    } else if (functionBuilder.getF32Type() == type) {
      return functionBuilder.getF32FloatAttr(value);
    } else if (functionBuilder.getIndexType() == type) {
      return functionBuilder.getIndexAttr(value);
    } else {
      return nullptr;
    }
  }

  HeaderDefinition createHeaderDefinition(
      Operation& operation,
      const size_t dataSize,
      const Type type) {
    auto dataSizeConstantOp = context.createConstant(
      functionBuilder,
      functionLocation,
      type,
      getAttribute(type, dataSize));

    const size_t operationId = static_cast<size_t>(getOperationId(operation));
    auto instructionIdConstantOp = context.createConstant(
      functionBuilder,
      functionLocation,
      type,
      getAttribute(type, operationId));

    const size_t typeId = static_cast<size_t>(getTypeId(type, functionBuilder));
    auto typeIdConstantOp = context.createConstant(
      functionBuilder,
      functionLocation,
      type,
      getAttribute(type, typeId));

    const size_t headerSize = 8;
    const Type indexType = functionBuilder.getIndexType();
    auto offsetConstantOp = context.createConstant(
      functionBuilder,
      functionLocation,
      indexType,
      getAttribute(indexType, headerSize));

    auto resultTypes = operation.getResultTypes();
    const Type outType = resultTypes.front();
    const VectorType outVecType = cast<VectorType>(outType);

    hivmave::VFBroadcastScalarMaskOp dataSizeBroadcastOp;
    {
      const Value pgeOp = context.createVFPgeOp(
        functionBuilder,
        functionLocation,
        outVecType,
        mlir::hivmave::PgePattern::ALL);

      dataSizeBroadcastOp = context.createVFBroadcastScalarMaskOp(
        functionBuilder,
        functionLocation,
        outVecType,
        dataSizeConstantOp,
        pgeOp);
    }

    hivmave::VFBroadcastScalarMaskOp typeIdBroadcastOp;
    {
      const Value pgeOp = context.createVFPgeOp(
        functionBuilder,
        functionLocation,
        outVecType,
        mlir::hivmave::PgePattern::ALL);

      typeIdBroadcastOp = context.createVFBroadcastScalarMaskOp(
        functionBuilder,
        functionLocation,
        outVecType,
        typeIdConstantOp,
        pgeOp);
    }

    hivmave::VFBroadcastScalarMaskOp instructionIdBroadcastOp;
    {
      const Value pgeOp = context.createVFPgeOp(
        functionBuilder,
        functionLocation,
        outVecType,
        mlir::hivmave::PgePattern::ALL);

      instructionIdBroadcastOp = context.createVFBroadcastScalarMaskOp(
        functionBuilder,
        functionLocation,
        outVecType,
        instructionIdConstantOp,
        pgeOp);
    }

    return HeaderDefinition(
      headerSize,
      dataSizeConstantOp,
      dataSizeBroadcastOp,
      typeIdBroadcastOp,
      instructionIdConstantOp,
      offsetConstantOp,
      instructionIdBroadcastOp);
  }

  DebugMemoryContext context;
  OpBuilder functionBuilder;
  Location functionLocation;
};

class UbProbeActionHandler : public DebugActionHandler {
public:
  BlockArgument addArgument(
    mlir::func::FuncOp funcOp,
    mlir::Type newArgType
  ) {
    mlir::FunctionType funcType = funcOp.getFunctionType();

    Block& block = funcOp.getRegion().front();
    const llvm::ArrayRef<BlockArgument> args = block.getArguments();

    llvm::SmallVector<mlir::Type> newInputs(
      funcType.getInputs().begin(),
      funcType.getInputs().end());
    newInputs.push_back(newArgType);

    mlir::FunctionType newFuncType = mlir::FunctionType::get(
        funcOp.getContext(),
        newInputs,
        funcType.getResults());

    funcOp.setFunctionType(newFuncType);

    const auto index = args.size();
    BlockArgument blockArg = block.insertArgument(index, newArgType, funcOp.getLoc());
    return blockArg;
  }

  mlir::func::CallOp addArgument(ModuleOp& moduleOp, mlir::func::CallOp& callOp, mlir::Value& arg) {
    DEBUG_LLVM_LOG_DEBUG_EXEC("prev function call:", callOp->dump());

    OpBuilder builder(callOp->getBlock(), std::next(callOp->getIterator()));

    const StringRef calleeName = callOp.getCallee();

    auto funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(calleeName);
    mlir::FunctionType funcType = funcOp.getFunctionType();
    llvm::SmallVector<mlir::Type> newInputTypes(
      funcType.getInputs().begin(),
      funcType.getInputs().end());
    newInputTypes.push_back(arg.getType());

    auto operands = callOp.getOperands();

    llvm::SmallVector<mlir::Value> newOperands;
    newOperands.reserve(operands.size() + 1);
    for (auto operation : operands) {
      newOperands.push_back(operation);
    }
    newOperands.push_back(arg);

    llvm::SmallVector<mlir::Type> outputTypes;
    auto newCallOp = builder.create<mlir::func::CallOp>(
      callOp->getLoc(),
      calleeName,
      outputTypes,
      newOperands
    );
    newCallOp->setAttr("hivm.vector_function", builder.getUnitAttr());
    newCallOp->setAttr("no_inline", builder.getUnitAttr());

    callOp.replaceAllUsesWith(newCallOp);
    callOp->erase();

    DEBUG_LLVM_LOG_DEBUG_EXEC("new function call with prob was added:", newCallOp.dump());
    return newCallOp;
  }

  LogicalResult handleFunctionCall(
        ModuleOp moduleOp,
        mlir::func::CallOp callOp,
        mlir::Value entryOutputArg,
        mlir::Value& castedEntryProbArg,
        const Config& config) {
    if (config.castDisabled) {
      auto operands = callOp->getOperands();
      auto resultArg = operands[operands.size() - 1];

      DEBUG_LLVM_LOG_DEBUG_EXEC("PointerCastOp was disabled (castDisabled=" + std::to_string(config.castDisabled) + "):", resultArg.dump());
      castedEntryProbArg = resultArg;
    } else {
      OpBuilder builder(callOp->getBlock(), std::prev(callOp->getIterator()));
      mlir::Type type = builder.getI64Type();

      const int64_t value = config.probOffset;
      mlir::IntegerAttr attribute = builder.getI64IntegerAttr(value);
      auto ubOffset = builder.create<arith::ConstantOp>(
        callOp->getLoc(),
        type,
        attribute);

      DEBUG_LLVM_LOG_DEBUG_EXEC("offset:", ubOffset.dump());

      MemRefType vecType = cast<MemRefType>(entryOutputArg.getType());
      Type elementType = vecType.getElementType();

      StridedLayoutAttr layout = {};
      mlir::MLIRContext* context = elementType.getContext();

      auto memorySpace = hivm::AddressSpaceAttr::get(context, hivm::AddressSpace::UB);
      const SmallVector<int64_t> shape({static_cast<int64_t>(config.probAllocSize)});
      MemRefType memrefType = MemRefType::get(shape, elementType, layout, memorySpace);

      castedEntryProbArg = builder.create<hivm::PointerCastOp>(
        ubOffset->getLoc(),
        memrefType,
        ubOffset);

      DEBUG_LLVM_LOG_DEBUG_EXEC("PointerCastOp was created", castedEntryProbArg.dump());
    }

    addArgument(moduleOp, callOp, castedEntryProbArg);
    return LogicalResult::success();
  }

  scf::ForOp getForOp(memref::SubViewOp intermediateSubviewOp) {
    scf::ForOp forOp;

    DEBUG_LLVM_LOG_DEBUG("intermediate subview dynamicOffsets:");
    llvm::SmallVector<mlir::OpFoldResult> mixedOffsets = intermediateSubviewOp.getMixedOffsets();
    for (const auto& offset : mixedOffsets) {
      if (offset.is<Value>()) {
        const Value offsetValue = dyn_cast<Value>(offset);
        auto blockArg = dyn_cast<mlir::BlockArgument>(offsetValue);
        if (blockArg != nullptr) {
          mlir::Block* parentBlock = blockArg.getOwner();
          mlir::Operation* ownerOp = parentBlock->getParentOp();
          forOp = dyn_cast<mlir::scf::ForOp>(ownerOp);
          if (forOp) {
            return forOp;
          }
        }
      }
    }
    return forOp;
  }

  mlir::Operation* updateForOpIterArgs(
      scf::ForOp forOp,
      Value extraInitArg,
      arith::AddIOp& offsetIncrement,
      Value& loopIterArg,
      scf::YieldOp& newYieldOp) {
    OpBuilder rewriter(forOp);

    Location loc = forOp.getLoc();
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();

    SmallVector<Value> newInitArgs(forOp.getInitArgs());
    newInitArgs.push_back(extraInitArg);

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
      loc, lowerBound, upperBound, step, newInitArgs,
      [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange newBodyArgs) {
        DEBUG_LLVM_LOG_DEBUG("newBodyArgs (" + std::to_string(newBodyArgs.size()) + ")");
        DEBUG_LLVM_LOG_DEBUG_EXEC("iv:", iv.dump());

        Block* oldBody = forOp.getBody();
        Block* newBody = b.getInsertionBlock();

        Value iterArg = newBodyArgs.front();
        loopIterArg = iterArg;

        const Value step = forOp.getStep();
        offsetIncrement = b.create<arith::AddIOp>(bodyLoc, iterArg, step);
        DEBUG_LLVM_LOG_DEBUG_EXEC("intermediate offsetIncrement:", offsetIncrement->dump());

        auto yieldOp = cast<scf::YieldOp>(oldBody->getTerminator());
        SmallVector<Value> newYieldOperands(yieldOp.getOperands());
        newYieldOperands.push_back(offsetIncrement);

        newYieldOp = b.create<scf::YieldOp>(bodyLoc, newYieldOperands);

        Operation* prevOperation = nullptr;
        b.setInsertionPointToStart(newBody);

        MutableArrayRef<BlockArgument> oldBodyArgs = oldBody->getArguments();
        assert(oldBodyArgs.size() == newBodyArgs.size() && "unexpected body arguments");

        std::vector<std::pair<Value, Value>> results;
        for (auto& op : llvm::make_early_inc_range(*oldBody)) {
          if (isa<scf::YieldOp>(op)) {
            continue;
          }

          DEBUG_LLVM_LOG_DEBUG_EXEC("op movement: original:", op.dump());

          mlir::OperandRange operands = op.getOperands();
          DEBUG_LLVM_LOG_DEBUG("op operands:");

          IRMapping map;

          for (auto operand : operands) {
            for (auto result : results) {
              if (result.first == operand) {
                map.map(result.first, result.second);
                DEBUG_LLVM_LOG_DEBUG_EXEC("operand is reassigned: original:", result.first.dump());
                DEBUG_LLVM_LOG_DEBUG_EXEC("operand is reassigned: new:", result.second.dump());
              }
            }
          }

          {
            if (prevOperation != nullptr) {
              b.setInsertionPointAfter(prevOperation);
            }

            map.map(oldBodyArgs[0], iv);
            Operation* clonedOp = b.clone(op, map);

            const auto numberResults = op.getNumResults();
            assert(numberResults == clonedOp->getNumResults() && "results numbers are different");

            for (size_t number = 0; number < numberResults; ++number) {
              Value v1 = op.getResult(number);
              Value v2 = clonedOp->getResult(number);
              results.push_back(std::pair<mlir::Value, mlir::Value>(v1, v2));
            }

            DEBUG_LLVM_LOG_DEBUG_EXEC("op movement: cloned:", clonedOp->dump());
            prevOperation = clonedOp;
          }
        }
      });
    return newForOp.getOperation();
  }

  LogicalResult getIntermediateSubviewOp(
        Operation* operation,
        hivmave::VFLoadOp& loadOp,
        memref::SubViewOp& intermediateSubviewOp) {

    LogicalResult result = LogicalResult::success();

    Operation* load = operation->getOperand(0).getDefiningOp();
    if (load == nullptr) {
      return result;
    }

    const auto tmpLoadOp = dyn_cast<hivmave::VFLoadOp>(load);
    if (!tmpLoadOp) {
      return result;
    }
    loadOp = tmpLoadOp;

    Operation* intermediateSubview = load->getOperand(0).getDefiningOp();
    if (intermediateSubview == nullptr) {
      return result;
    }

    const auto tmpIntermediateSubviewOp = dyn_cast<memref::SubViewOp>(intermediateSubview);
    if (!tmpIntermediateSubviewOp) {
      return result;
    }
    intermediateSubviewOp = tmpIntermediateSubviewOp;

    return result;
  }

  static LogicalResult getIntValue(const Value value, int64_t& intValue) {
    Operation* constant = value.getDefiningOp();
    arith::ConstantOp constantOp = dyn_cast<arith::ConstantOp>(constant);
    if (!constantOp) {
      DEBUG_LLVM_LOG_ERROR("Operation is not constant");
      return LogicalResult::failure();
    }

    const Attribute attr = constantOp.getValue();
    const IntegerAttr intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr) {
      DEBUG_LLVM_LOG_ERROR("Unexpected constant value type");
      return LogicalResult::failure();
    }

    intValue = intAttr.getInt();
    return LogicalResult::success();
  }

  LogicalResult handleFunctionOp(
        DebugMemoryContext& context,
        ModuleOp moduleOp,
        func::FuncOp functionOp,
        mlir::Value entryOutputArg,
        mlir::Value& castedEntryProbArg,
        const Config& config,
        size_t& storeOffset) {

    LogicalResult result = LogicalResult::success();

    arith::AddIOp offsetIncrementAddOp;
    arith::AddIOp offsetIncrementAddOpOriginal;
    Value loopIterArg;
    scf::YieldOp newYieldOp;
    scf::ForOp newForOp;

    {
      const std::vector<Operation*> operations = DebugActionHandler::findOperations(functionOp, tensorChangeOperations);
      if (operations.size() > 0) {
        hivmave::VFLoadOp loadOp;
        memref::SubViewOp intermediateSubviewOp;

        result = getIntermediateSubviewOp(operations[0], loadOp, intermediateSubviewOp);
        if (!result.succeeded()) {
          return result;
        }

        if ((!loadOp) && (!intermediateSubviewOp)) {
          DEBUG_LLVM_LOG_ERROR("load and subview operations were not found");
          return LogicalResult::failure();
        }

        if (intermediateSubviewOp) {
          DEBUG_LLVM_LOG_DEBUG_EXEC("intermediate SubviewOp:", intermediateSubviewOp.dump());

          scf::ForOp forOp = getForOp(intermediateSubviewOp);
          if (forOp) {
            DEBUG_LLVM_LOG_DEBUG_EXEC("intermediate ForOp:", forOp.dump());

            OpBuilder prevForOpBuilder(forOp->getBlock(), std::prev(forOp->getIterator()));

            const int64_t initialOffsetValue = 0;
            arith::ConstantOp initialOffsetConstant = prevForOpBuilder.create<arith::ConstantOp>(
              forOp->getLoc(),
              prevForOpBuilder.getIndexType(),
              prevForOpBuilder.getIndexAttr(initialOffsetValue));
            DEBUG_LLVM_LOG_DEBUG_EXEC("initialOffsetConstant:", initialOffsetConstant.dump());

            auto newFor = updateForOpIterArgs(
              forOp,
              initialOffsetConstant,
              offsetIncrementAddOp,
              loopIterArg,
              newYieldOp);

            newForOp = dyn_cast<scf::ForOp>(newFor);
            offsetIncrementAddOpOriginal = offsetIncrementAddOp;
            forOp.erase();
          } else {
            DEBUG_LLVM_LOG_DEBUG("intermediate ForOp was not found");
          }
        } else {
          DEBUG_LLVM_LOG_DEBUG("intermediate SubviewOp was not found");
        }
      }
    }

    const std::vector<Operation*> operations = DebugActionHandler::findOperations(functionOp, tensorChangeOperations);
    if (operations.empty()) {
      DEBUG_LLVM_LOG_DEBUG("change operations were not found for function '" + functionOp->getName().getStringRef().str() + "'");
      return LogicalResult::success();
    }

    DEBUG_LLVM_LOG_DEBUG_EXEC(
      "change operations were found (" + std::to_string(operations.size()) + ") function before:\n",
      functionOp->dump());

    DEBUG_LLVM_LOG_DEBUG_EXEC("module with updated function 1:\n", moduleOp->dump());

    const auto outlinedFunctionName = functionOp.getSymName().str();
    {
      moduleOp.walk([&](mlir::func::CallOp callOp) {
        const StringRef calleeName = callOp.getCallee();
        if (calleeName.str() == outlinedFunctionName) {
          DEBUG_LLVM_LOG_DEBUG_EXEC("function call was found: " + outlinedFunctionName + ":", callOp.dump());

          result = handleFunctionCall(moduleOp, callOp, entryOutputArg, castedEntryProbArg, config);
          return WalkResult::interrupt();
        }

        DEBUG_LLVM_LOG_DEBUG_EXEC("function ignored:", callOp.dump());
        return WalkResult::advance();
      });

      if (!result.succeeded()) {
        return result;
      }
    }

    BlockArgument functionProbArg;
    {
      functionProbArg = addArgument(functionOp, castedEntryProbArg.getType());
      if (!functionProbArg) {
        return LogicalResult::failure();
      }

      DEBUG_LLVM_LOG_DEBUG_EXEC("prob arg was added in function declaration:\n", functionOp->dump());
    }

    DEBUG_LLVM_LOG_DEBUG_EXEC("module with updated function 2:\n", moduleOp->dump());

    Operation* intermediateSubview = nullptr;
    Operation* intermediateLoad = nullptr;
    hivmave::VFLoadOp loadOp;
    memref::SubViewOp intermediateSubviewOp;
    Operation* existingMaskedStoreOp = nullptr;

    OpBuilder functionBuilder(functionOp);
    Block& entryBlock = functionOp.getRegion().front();
    if (entryBlock.empty()) {
        DEBUG_LLVM_LOG_ERROR("function region is empty");
        return LogicalResult::failure();
    }

    functionBuilder.setInsertionPointAfter(&entryBlock.front());
    Location functionLocation = entryBlock.front().getLoc();
    HeaderBuilder headerBuilder(context, functionBuilder, functionLocation);

    for (Operation* operation : operations) {
      if ((intermediateSubview == nullptr) && (intermediateLoad == nullptr)) {
        result = getIntermediateSubviewOp(operations[0], loadOp, intermediateSubviewOp);
        if (!result.succeeded()) {
          return result;
        }

        if ((!loadOp) && (!intermediateSubviewOp)) {
          DEBUG_LLVM_LOG_ERROR("load and subview operations were not found");
          return LogicalResult::failure();
        }

        if (loadOp) {
          intermediateLoad = loadOp.getOperation();
          DEBUG_LLVM_LOG_DEBUG_EXEC(
            "loadOp (" + std::to_string(loadOp->getNumOperands()) + "):",
            loadOp->dump());
        }

        if (intermediateSubviewOp) {
          intermediateSubview = intermediateSubviewOp.getOperation();
          DEBUG_LLVM_LOG_DEBUG_EXEC(
            "intermediateSubviewOp (" + std::to_string(intermediateSubviewOp->getNumOperands()) + "):",
            intermediateSubviewOp->dump());
        }
      }

      const std::string operationName = operation->getName().getStringRef().str();
      if (config.vfInstructionsDisabled.find(operationName) != std::string::npos) {
        DEBUG_LLVM_LOG_DEBUG("operation '" + operationName + "' is disabled");
        continue;
      }

      DEBUG_LLVM_LOG_DEBUG_EXEC("change operation '" + operationName + "':", operation->dump());

      if (config.vfPgeDisabled) {
        DEBUG_LLVM_LOG_DEBUG("PgeOp operation was disabled");
      } else {
        OpBuilder builder(operation->getBlock(), std::next(operation->getIterator()));
        auto location = operation->getLoc();

        auto resultTypes = operation->getResultTypes();
        Type outType = resultTypes.front();
        VectorType outVecType = cast<VectorType>(outType);

        auto pgeOp = context.createVFPgeOp(builder, location, outVecType, mlir::hivmave::PgePattern::ALL);
        auto operationResult = operation->getResult(0);
        auto pgeOpResult = pgeOp->getResult(0);

        if (existingMaskedStoreOp == nullptr) {
          std::vector<Operation*> existingMaskedStores = DebugActionHandler::findOperations(
            functionOp,
            { maskedStore });
          if (existingMaskedStores.size() != 1) {
            DEBUG_LLVM_LOG_DEBUG("unexpected existing masked stores: " + std::to_string(existingMaskedStores.size()));
            return LogicalResult::failure();
          }
          existingMaskedStoreOp = existingMaskedStores[0];
        }

        hivmave::VFMaskedStoreOp existingMaskedStore = dyn_cast<hivmave::VFMaskedStoreOp>(existingMaskedStoreOp);

        Value base;
        if (config.vfSubviewDisabled) {
          DEBUG_LLVM_LOG_DEBUG("view operation is disabled");
        } else {
          if (intermediateSubview != nullptr) {
            {
              auto indices = existingMaskedStore.getIndices();
              auto opLocation = operation->getLoc();

              OpBuilder headerOpBuilder(operation->getBlock(), std::next(operation->getIterator()));
              headerOpBuilder.setInsertionPointAfter(operation);

              const auto header = headerBuilder.createHeader(
                headerOpBuilder,
                opLocation,
                *operation,
                functionProbArg,
                loopIterArg,
                offsetIncrementAddOp,
                intermediateSubviewOp,
                indices);
              if (!header) {
                DEBUG_LLVM_LOG_ERROR("header was not created");
                return LogicalResult::failure();
              }

              storeOffset += header.headerSize + header.dataSize;
              loopIterArg = header.loopIterArg;
            }

            IRMapping map;
            map.map(intermediateSubview->getOperand(0), functionProbArg);
            map.map(intermediateSubview->getOperand(1), loopIterArg);
            Operation* probSubViewOp = builder.clone(*intermediateSubview, map);
            DEBUG_LLVM_LOG_DEBUG_EXEC("probSubViewOp:", probSubViewOp->dump());

            base = probSubViewOp->getResult(0);

            Operation* offsetIncrementAdd = offsetIncrementAddOp.getOperation();
            IRMapping addMap;
            addMap.map(offsetIncrementAddOp->getOperand(0), loopIterArg);
            Operation* addOp = builder.clone(*offsetIncrementAdd, addMap);

            DEBUG_LLVM_LOG_DEBUG_EXEC("addOp:", addOp->dump());

            loopIterArg = addOp->getResult(0);
            offsetIncrementAdd = dyn_cast<arith::AddIOp>(addOp);
          } else if (intermediateLoad != nullptr) {
            base = functionProbArg;
          } else {
            DEBUG_LLVM_LOG_ERROR("can not create store operation");
            return LogicalResult::failure();
          }
          DEBUG_LLVM_LOG_DEBUG_EXEC("base:", base.dump());

          if (config.vfStoreDisabled) {
            DEBUG_LLVM_LOG_DEBUG("store operation was disabled");
          } else {
            auto indices = existingMaskedStore.getIndices();
            auto mask = pgeOpResult;

            hivmave::StoreDist storePattern = hivmave::StoreDist::NORM_B32;
            builder.create<hivmave::VFMaskedStoreOp>(
              location,
              storePattern,
              base,
              indices,
              mask,
              operationResult);
          }
        }
      }
    }

    if (newYieldOp) {
      OpBuilder builder(newYieldOp->getBlock(), std::next(newYieldOp->getIterator()));

      SmallVector<Value> operands;
      operands.push_back(loopIterArg);
      builder.create<scf::YieldOp>(newYieldOp.getLoc(), operands);
      newYieldOp.erase();
    }

    if (!operations.empty()) {
      DEBUG_LLVM_LOG_DEBUG_EXEC("function after insertion:\n", functionOp->dump());
    }

    if (offsetIncrementAddOpOriginal) {
      offsetIncrementAddOpOriginal.erase();
    }

    if (newForOp != nullptr) {
      int64_t lowerBound;
      result = getIntValue(newForOp.getLowerBound(), lowerBound);
      if (!result.succeeded()) {
        DEBUG_LLVM_LOG_ERROR("lowerBound was not received");
        return result;
      }
      if (lowerBound < 0) {
        DEBUG_LLVM_LOG_ERROR("lowerBound has to be zero or positive: " + std::to_string(lowerBound));
        return LogicalResult::failure();
      }

      int64_t upperBound;
      result = getIntValue(newForOp.getUpperBound(), upperBound);
      if (!result.succeeded()) {
        DEBUG_LLVM_LOG_ERROR("upperBound was not received");
        return result;
      }
      if (upperBound < 0) {
        DEBUG_LLVM_LOG_ERROR("upperBound has to be zero or positive: " + std::to_string(upperBound));
        return LogicalResult::failure();
      }

      if (lowerBound >= upperBound) {
        DEBUG_LLVM_LOG_ERROR(std::string("not expected loop parameters: ") +
          "lowerBound(" + std::to_string(lowerBound) + "), " +
          "upperBound (" + std::to_string(upperBound) + ")");
        return LogicalResult::failure();
      }

      int64_t step;
      result = getIntValue(newForOp.getStep(), step);
      if (!result.succeeded()) {
        DEBUG_LLVM_LOG_ERROR("step was not received");
        return result;
      }
      if (step <= 0) {
        DEBUG_LLVM_LOG_ERROR("step has to be positive: " + std::to_string(step));
        return LogicalResult::failure();
      }

      if (((upperBound - lowerBound) % step) != 0) {
        DEBUG_LLVM_LOG_ERROR(std::string("not expected loop parameters: ") +
          "lowerBound(" + std::to_string(lowerBound) + "), " +
          "upperBound (" + std::to_string(upperBound) + "), " +
          "step (" + std::to_string(step) + ")");
        return LogicalResult::failure();
      }

      storeOffset = storeOffset * static_cast<size_t>((upperBound - lowerBound) / step);
    }
    return result;
  }

  LogicalResult handleFunctionOps(
        DebugMemoryContext& context,
        ModuleOp moduleOp,
        func::FuncOp entryFunction,
        mlir::Value entryOutputArg,
        mlir::Value& castedEntryProbArg,
        const Config& config,
        size_t& storeOffset) {

    LogicalResult result = LogicalResult::success();
    moduleOp.walk([&](func::FuncOp functionOp) {
      if (entryFunction == functionOp) {
        return WalkResult::advance();
      }

      const auto outlinedFunctionName = functionOp.getSymName().str();
      DEBUG_LLVM_LOG_DEBUG("outlinedFunctionName: " + outlinedFunctionName);

      size_t functionStoreOffset = 0;
      result = handleFunctionOp(
        context,
        moduleOp,
        functionOp,
        entryOutputArg,
        castedEntryProbArg,
        config,
        functionStoreOffset);
      if (!result.succeeded()) {
        return WalkResult::interrupt();
      }

      storeOffset += functionStoreOffset;

      return WalkResult::advance();
    });

    return result;
  }

  LogicalResult handleAction(ModuleOp moduleOp, DictionaryAttr actionArgs) override {
    DEBUG_LLVM_LOG_DEBUG("was started");

    const auto config = Config::getEnv();
    DEBUG_LLVM_LOG_DEBUG(config.str());

    if (config.debugPrint == "") {
      DEBUG_LLVM_LOG_DEBUG("Ascend debugging is disabled, use ASCEND_DEBUG_PRINT=ALL to enable");
    }

    if (config.debugPrint != "ALL") {
      DEBUG_LLVM_LOG_ERROR("ASCEND_DEBUG_PRINT supports only 'ALL' value, but passed value: '" + config.debugPrint + "'\n");
    }

    if (config.probOffset == 0) {
      DEBUG_LLVM_LOG_ERROR("ASCEND_DEBUG_PROB_OFFSET is not set\n");
      return LogicalResult::failure();
    }

    if (config.probAllocSize == 0) {
      DEBUG_LLVM_LOG_ERROR("ASCEND_DEBUG_PROB_ALLOC_SIZE is not set\n");
      return LogicalResult::failure();
    }

    const std::vector<func::FuncOp> entryFunctions = DebugActionHandler::findEntryFunction(moduleOp);
    if (entryFunctions.size() != 1) {
      DEBUG_LLVM_LOG_DEBUG("entry function was not found");
      return LogicalResult::failure();
    }

    func::FuncOp entryFunction = entryFunctions[0];
    DEBUG_LLVM_LOG_DEBUG_EXEC("entry function was found: " + entryFunction.getName().str() + "\n", entryFunction->dump());

    mlir::BlockArgument entryOutputArg;
    mlir::BlockArgument entryProbArg;

    Block& entryBlock = entryFunction.getRegion().front();
    llvm::ArrayRef<BlockArgument> args = entryBlock.getArguments();
    for (BlockArgument tmpArg : args) {
      mlir::Type type = tmpArg.getType();
      if (isa<MemRefType>(type)) {
        entryOutputArg = entryProbArg;
        entryProbArg = tmpArg;
      }
    }

    if (!entryOutputArg) {
      DEBUG_LLVM_LOG_ERROR("entry output arg was not found");
      return LogicalResult::failure();
    }
    DEBUG_LLVM_LOG_DEBUG_EXEC(
      "entry output arg (" + std::to_string(entryOutputArg.getArgNumber()) << ")",
      entryOutputArg.dump());

    if (!entryProbArg) {
      DEBUG_LLVM_LOG_ERROR("entry prob arg was not found");
      return LogicalResult::failure();
    }
    DEBUG_LLVM_LOG_DEBUG_EXEC("entry prob arg (" + std::to_string(entryProbArg.getArgNumber()) + ")", entryProbArg.dump());

    DebugMemoryContext context;
    size_t storeOffset = 0;
    mlir::Value castedEntryProbArg;
    if (!config.vfDisabled) {
      const auto result = handleFunctionOps(
        context,
        moduleOp,
        entryFunction,
        entryOutputArg,
        castedEntryProbArg,
        config,
        storeOffset);
      if (!result.succeeded()) {
        return LogicalResult::failure();
      }
    } else {
      DEBUG_LLVM_LOG_DEBUG("function update was skipped)");
    }

    if (!castedEntryProbArg) {
      DEBUG_LLVM_LOG_DEBUG("module doesn't have handled operations");
      return LogicalResult::success();
    }

    if (!config.storeDisabled) {
      DEBUG_LLVM_LOG_DEBUG_EXEC("store:\n", moduleOp.dump());

      const std::string entryFunctionName = entryFunction.getName().str();
      auto entryProbStore = findOperation(moduleOp, entryFunctionName, "hivm.hir.store", entryProbArg.getArgNumber());
      if (!entryProbStore) {
        DEBUG_LLVM_LOG_ERROR("DebugActionHandler::findOperation: entryProbStore was not found in '", entryFunctionName, "' function for argument '", entryProbArg, "')\n");
        return LogicalResult::failure();
      }

      DEBUG_LLVM_LOG_DEBUG_EXEC("entryProbStore:", entryProbStore->dump());

      auto operands = entryProbStore->getOperands();
      DEBUG_LLVM_LOG_DEBUG("operands:" + std::to_string(operands.size()));

      auto originalOutputOperand = operands[0];
      DEBUG_LLVM_LOG_DEBUG_EXEC("operands[0]:", originalOutputOperand.dump());

      Value probValue;

      Operation* sharedOperation = originalOutputOperand.getDefiningOp();
      memref::SubViewOp sharedSubviewOp = dyn_cast<memref::SubViewOp>(sharedOperation);
      if (sharedSubviewOp) {
        DEBUG_LLVM_LOG_DEBUG_EXEC("sharedSubview:", sharedSubviewOp->dump());
        Operation* entryProbSubViewOp;
        {
          OpBuilder builder(sharedSubviewOp->getBlock(), std::next(sharedSubviewOp->getIterator()));
          IRMapping map;
          if (!config.replaceOutputDisabled) {
            map.map(sharedSubviewOp->getOperand(0), castedEntryProbArg);
          }
          entryProbSubViewOp = builder.clone(*sharedSubviewOp.getOperation(), map);

          DEBUG_LLVM_LOG_DEBUG_EXEC(
            "entryProbSubViewOp (replaceOutputDisabled=" + std::to_string(config.replaceOutputDisabled) + "):",
            entryProbSubViewOp->dump());
        }

        probValue = entryProbSubViewOp->getResult(0);
      } else {
        DEBUG_LLVM_LOG_DEBUG_EXEC("sharedOperation:", sharedOperation->dump());
        DEBUG_LLVM_LOG_DEBUG("sharedOperation->getName():", sharedOperation->getName().getStringRef().str());
        if (isa<hivm::PointerCastOp>(sharedOperation)) {
          DEBUG_LLVM_LOG_DEBUG_EXEC("castedEntryProbArg: ", castedEntryProbArg.dump());
          probValue = castedEntryProbArg;
        } else {
          DEBUG_LLVM_LOG_DEBUG_EXEC("Unknow shared operation type", sharedOperation->dump());
          DEBUG_LLVM_LOG_ERROR("Unknow shared operation type");
          return LogicalResult::failure();
        }
      }
      DEBUG_LLVM_LOG_DEBUG_EXEC("probValue:", probValue.dump());

      OpBuilder cloneBuilder(entryProbStore);
      Operation* clonedInputProbSubView = nullptr;
      {
        auto entryProbSubView = probValue.getDefiningOp();

        auto storeOffsetConstant = context.createConstant(
          cloneBuilder,
          sharedOperation->getLoc(),
          cloneBuilder.getIndexType(),
          cloneBuilder.getIndexAttr(storeOffset));

        IRMapping map;
        map.map(entryProbSubView->getOperand(1), storeOffsetConstant);
        clonedInputProbSubView = cloneBuilder.clone(*entryProbSubView, map);

        auto source = entryProbSubView->getResult(0);
        auto target = clonedInputProbSubView->getResult(0);
        source.replaceAllUsesWith(target);
        entryProbSubView->erase();
      }

      Operation* clonedOuputProbSubView = nullptr;
      {
        auto storeOffsetConstant = context.createConstant(
          cloneBuilder,
          sharedOperation->getLoc(),
          cloneBuilder.getIndexType(),
          cloneBuilder.getIndexAttr(storeOffset));
        Operation* ouputProbSubView = entryProbStore->getOperand(1).getDefiningOp();

        IRMapping map;
        map.map(ouputProbSubView->getOperand(1), storeOffsetConstant);
        clonedOuputProbSubView = cloneBuilder.clone(*ouputProbSubView, map);

        auto source = ouputProbSubView->getResult(0);
        auto target = clonedOuputProbSubView->getResult(0);
        source.replaceAllUsesWith(target);
        ouputProbSubView->erase();
      }


      IRMapping map;
      map.map(entryProbStore->getOperand(0), clonedInputProbSubView->getResult(0));
      map.map(entryProbStore->getOperand(1), clonedOuputProbSubView->getResult(0));
      cloneBuilder.clone(*entryProbStore, map);

      entryProbStore->erase();
    }

    return LogicalResult::success();
  }
};

DebugActionHandler::uptr DebugActionHandler::getHandler(ActionType actT) {
    switch(actT) {
      case ActionType::UBPROBE:
        return std::make_unique<UbProbeActionHandler>();
      default:
        llvm_unreachable("Unsupported LLVM debug action kind");
    }
}

struct DebugMemoryPass : public bishengir::impl::DebugMemoryBase<DebugMemoryPass> {

  using Base = bishengir::impl::DebugMemoryBase<DebugMemoryPass>;

  explicit DebugMemoryPass(const bishengir::DebugMemoryOptions &options) : Base(options) {
#if defined(DEBUG_MEMORY_ENABLE_LOGGING)
    if (cacheFilePath.getValue().size() > 0) {
      DebugMemoryLoger::get().init(cacheFilePath.getValue());
    }
#endif
  }

  void runOnOperation() override {
    ModuleOp moduleOp = isa<ModuleOp>(getOperation())
                        ? cast<ModuleOp>(getOperation())
                        : getOperation()->getParentOfType<ModuleOp>();
    DictionaryAttr actionArgs;
    const std::string actionType = "ubprobe";
    DEBUG_LLVM_LOG_INFO("Action type: '", actionType, "' was started\n");

    auto handler = DebugActionHandler::getHandler(convertToActionT(actionType));
    LogicalResult actionStatus = handler->handleAction(moduleOp, actionArgs);

    if (actionStatus.failed()) {
      DEBUG_LLVM_LOG_ERROR("Action type: '", actionType, "' failed\n");
    } else {
      DEBUG_LLVM_LOG_INFO("Action type: '", actionType, "' was completed\n");
    }
  }
};
} // namespace

std::unique_ptr<Pass> bishengir::createDebugMemoryPass(const bishengir::DebugMemoryOptions &options) {
  return std::make_unique<DebugMemoryPass>(options);
}
