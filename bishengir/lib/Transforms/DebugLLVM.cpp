//===- DebugLLVM.cpp - ---------------------------------------------------===//
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

#include <list>
#include <iostream>
#include <unordered_set>
#include <optional>
#include <functional>

#include "bishengir/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "debug-llvm"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace bishengir {
#define GEN_PASS_DEF_DEBUGLLVM
#include "bishengir/Transforms/Passes.h.inc"
} // namespace bishengir

using namespace mlir;

#define PASS_NAME "debug-llvm"
//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

#define DEBUG_MEMORY_ENABLE_LOGGING 1

#if defined(DEBUG_MEMORY_ENABLE_LOGGING)
#define DEBUG_LLVM_LOG_COUT 1
class DebugLLVMLoger {
  bool initialized = false;
  bool dumpToCacheFile = false;
  std::unique_ptr<llvm::raw_fd_ostream> fileStream;

  DebugLLVMLoger() = default;
  ~DebugLLVMLoger() = default;
  DebugLLVMLoger(DebugLLVMLoger&) = delete;
  DebugLLVMLoger(DebugLLVMLoger&&) = delete;
  DebugLLVMLoger& operator=(DebugLLVMLoger&) = delete;
  DebugLLVMLoger& operator=(DebugLLVMLoger&&) = delete;
public:
  void init(const std::string &cacheFilePath) {
    if (!initialized && !cacheFilePath.empty()) {
      std::error_code EC;
      fileStream = std::make_unique<llvm::raw_fd_ostream>(cacheFilePath, EC, llvm::sys::fs::OF_Append);
      if (EC) {
          llvm::errs() << "[DebugLLVMLoger] Error opening file: " << EC.message() << "\n";
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
  void logError(Args... args) {
    ( (llvm::errs() << args), ... );
  }

  void dump(
    ModuleOp moduleOp,
    const std::string& functionName) {
    auto functionOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(functionName);
    if (!functionOp) {
      return;
    }

    if (dumpToCacheFile) {
      functionOp.print(*fileStream);
      (*fileStream) << "\n";
    } else {
      functionOp.dump();
      llvm::outs() << "\n";
    }
  }

  void dump(Operation *op) {
    if (dumpToCacheFile) {
      op->print(*fileStream);
      (*fileStream) << "\n";
    } else {
      op->dump();
      llvm::outs() << "\n";
    }
  }

  static DebugLLVMLoger& get(){
    static DebugLLVMLoger logger;
    return logger;
  }
};

#if defined(DEBUG_LLVM_LOG_COUT)
#define DEBUG_LLVM_LOG_DEBUG(...) \
  DebugLLVMLoger::get().logInfo("[DEBUG ASCEND]" __VA_OPT__(,' ',) __VA_ARGS__); \
  std::cout << std::endl;
#else
#define DEBUG_LLVM_LOG_DEBUG(...) \
  DebugLLVMLoger::get().logInfo("[DEBUG ASCEND]" __VA_OPT__(,' ',) __VA_ARGS__); \
  llvm::dbgs() << "\n";
#endif

#if defined(DEBUG_LLVM_LOG_COUT)
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

#if defined(DEBUG_LLVM_LOG_COUT)
#define DEBUG_LLVM_LOG_ERROR_EXEC(msg, func) \
  std::cout << "[ERROR ASCEND] " << __func__  << ": " << msg << std::flush; \
  func; \
  std::cout << std::endl;
#else
#define DEBUG_LLVM_LOG_ERROR_EXEC(msg, func) \
  llvm::errs() << "[ERROR ASCEND] " << __func__  << ": " << msg << "\n"; \
  func; \
  llvm::errs() << "\n";
#endif

#define DEBUG_LLVM_LOG_INFO(...) DebugLLVMLoger::get().logInfo("[INFO ASCEND]" __VA_OPT__(,' ',) __VA_ARGS__)
#define DEBUG_LLVM_LOG_ERROR(...) DebugLLVMLoger::get().logError("[ERROR ASCEND]" __VA_OPT__(,' ',) __VA_ARGS__)
#define DEBUG_LLVM_LOG_FUNC(moduleOp, functionName) DebugLLVMLoger::get().dump(moduleOp, functionName)
#define DEBUG_LLVM_LOG_OP(op) DebugLLVMLoger::get().dump(op)

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
  SSA,
  REMOVE,
  MOVE,
  TYPE_CHANGE
};

class Config {
public:
  static Config getEnv() {
    return Config(
      std::string(getEnvString("ASCEND_DEBUG_LLVM")) == "1",
      std::string(getEnvString("ASCEND_DEBUG_LLVM_ACTION")),
      std::string(getEnvString("ASCEND_DEBUG_LLVM_FUNCTION")),
      getEnvInt("ASCEND_DEBUG_LLVM_INPUT", -1),
      getEnvInt("ASCEND_DEBUG_LLVM_SOURCE_INPUT", -1),
      getEnvInt("ASCEND_DEBUG_LLVM_TARGET_INPUT", -1),
      StringRef(getEnvString("ASCEND_DEBUG_LLVM_INSTRUCTION")).str(),
      StringRef(getEnvString("ASCEND_DEBUG_LLVM_SOURCE_INSTRUCTION")).str(),
      StringRef(getEnvString("ASCEND_DEBUG_LLVM_TARGET_INSTRUCTION")).str(),
      StringRef(getEnvString("ASCEND_DEBUG_LLVM_TYPE")).str());
  }

  std::string str() const {
    std::stringstream ss;
    ss << "\tdebugLLVM=" << debugLLVM << std::endl;
    ss << "\taction=" << action << std::endl;
    ss << "\tfunction=" << function << std::endl;
    ss << "\tinput=" << input << std::endl;
    ss << "\tsourceInput=" << sourceInput << std::endl;
    ss << "\ttargetInput=" << targetInput << std::endl;
    ss << "\tinstruction=" << instruction << std::endl;
    ss << "\tsourceInstruction=" << sourceInstruction << std::endl;
    ss << "\ttargetInstruction=" << targetInstruction << std::endl;
    ss << "\ttype=" << type << std::endl;
    return ss.str();
  }

  DictionaryAttr toDictionaryArg(mlir::MLIRContext& context) const {
    mlir::OpBuilder builder(&context);
    std::vector<NamedAttribute> attributes;

    attributes.push_back(mlir::NamedAttribute(
      builder.getStringAttr("function"),
      builder.getStringAttr(function)));

    const auto intType = builder.getI32Type();

    pushIfNotConsists(attributes, mlir::NamedAttribute(
      builder.getStringAttr("action"),
      builder.getStringAttr(action)));

    pushIfNotConsists(attributes, mlir::NamedAttribute(
      builder.getStringAttr("input"),
      builder.getIntegerAttr(intType, input)));

    pushIfNotConsists(attributes, mlir::NamedAttribute(
      builder.getStringAttr("source_input"),
      builder.getIntegerAttr(intType, sourceInput)));

    pushIfNotConsists(attributes, mlir::NamedAttribute(
      builder.getStringAttr("target_input"),
      builder.getIntegerAttr(intType, targetInput)));

    pushIfNotConsists(attributes, mlir::NamedAttribute(
      builder.getStringAttr("instruction"),
      builder.getStringAttr(instruction)));

    pushIfNotConsists(attributes, mlir::NamedAttribute(
      builder.getStringAttr("source_instruction"),
      builder.getStringAttr(sourceInstruction)));

    pushIfNotConsists(attributes, mlir::NamedAttribute(
      builder.getStringAttr("target_instruction"),
      builder.getStringAttr(targetInstruction)));

    pushIfNotConsists(attributes, mlir::NamedAttribute(
      builder.getStringAttr("type"),
      builder.getStringAttr(type)));

    return DictionaryAttr::get(&context, attributes);
  }

  Config(
    const bool debugLLVM,
    const std::string& action,
    const std::string& function,
    const int input,
    const int sourceInput,
    const int targetInput,
    const std::string& instruction,
    const std::string& sourceInstruction,
    const std::string& targetInstruction,
    const std::string& type) :
      debugLLVM(debugLLVM),
      action(action),
      function(function),
      input(input),
      sourceInput(sourceInput),
      targetInput(targetInput),
      instruction(instruction),
      sourceInstruction(sourceInstruction),
      targetInstruction(targetInstruction),
      type(type) {}

  const bool debugLLVM;
  const std::string action;
  const std::string function;
  const int input;
  const int sourceInput;
  const int targetInput;
  const std::string instruction;
  const std::string sourceInstruction;
  const std::string targetInstruction;
  const std::string type;

private:
  static void pushIfNotConsists(
    std::vector<NamedAttribute>& attributes,
    const NamedAttribute& attr) {
    for (NamedAttribute attribure : attributes) {
      if (attribure.getName() == attr.getName()) {
        return;
      }
    }
    attributes.push_back(attr);
  }

  static size_t getEnvInt(const std::string& name, const size_t defaultValue) {
    const std::string value = getEnvString(name);
    return value.empty() ? defaultValue : std::atoi(value.c_str());
  }

  static std::string getEnvString(const std::string& name) {
    const auto value = getenv(name.c_str());
    return value == nullptr ? "" : std::string(value);
  }
};

ActionType convertToActionT(const llvm::StringRef act) {
  if (act.lower() == "ssa") {
    return ActionType::SSA;
  }
  if (act.lower() == "remove") {
    return ActionType::REMOVE;
  }
  if (act.lower() == "move") {
    return ActionType::MOVE;
  }
  if (act.lower() == "type_change") {
    return ActionType::TYPE_CHANGE;
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
  static LLVM::LLVMFuncOp findFunction(
        ModuleOp moduleOp,
        const std::string& functionName) {
    LLVM::LLVMFuncOp resultFunctionOp;

    moduleOp.walk([&](LLVM::LLVMFuncOp functionOp) {
      if (functionOp.getSymName().starts_with(functionName)) {
        resultFunctionOp = functionOp;
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return resultFunctionOp;
  }

  static Operation* findOperation(
    ModuleOp moduleOp,
    const std::string& functionName,
    const std::string& instructionName,
    const size_t argumentIndex) {
    Operation* foundOperation = nullptr;

    moduleOp.walk([&](LLVM::LLVMFuncOp functionOp) {
      const auto functionSymName = functionOp.getSymName();

      if (functionSymName.starts_with(functionName)) {
        Block& entryBlock = functionOp.getRegion().front();
        llvm::ArrayRef<BlockArgument> args = entryBlock.getArguments();

        if (args.size() <= argumentIndex) {
          DEBUG_LLVM_LOG_ERROR("DebugActionHandler::findOperation: argument index ", argumentIndex, " is not correct\n");
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

    return foundOperation;
  }

  static Operation* findOperation(
    ModuleOp moduleOp,
    const std::string& functionName,
    const std::string& instructionName) {
    Operation* foundOperation = nullptr;

    moduleOp.walk([&](LLVM::LLVMFuncOp functionOp) {
      const auto functionSymName = functionOp.getSymName();
      if (functionSymName.starts_with(functionName)) {
        functionOp.walk([&](Operation* operation) {
          if (operation->getName().getStringRef().str() == instructionName) {
            foundOperation = operation;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return foundOperation;
  }
};

class SSAActionHandler : public DebugActionHandler {
public:
  LogicalResult handleAction(ModuleOp moduleOp, DictionaryAttr actionArgs) override {
    const int sourceInput = actionArgs.getAs<IntegerAttr>("source_input").getInt();
    const int targetInput = actionArgs.getAs<IntegerAttr>("target_input").getInt();
    const std::string sourceInstruction = actionArgs.getAs<StringAttr>("source_instruction").str();
    const std::string targetInstruction = actionArgs.getAs<StringAttr>("target_instruction").str();
    const std::string function = actionArgs.getAs<StringAttr>("function").str();

    Operation* sourceOp = findOperation(moduleOp, function, sourceInstruction, sourceInput);
    if (sourceOp == nullptr) {
      DEBUG_LLVM_LOG_ERROR("Source operation '", sourceInstruction, "' was not found in function '", function, "'\n");
      return LogicalResult::failure();
    }

    auto sourceResults = sourceOp->getResults();
    if (sourceResults.size() != 1) {
      DEBUG_LLVM_LOG_ERROR("Source operation should has one result\n");
      return LogicalResult::failure();
    }
    OpResult source = sourceResults[0];

    Operation* targetOp = findOperation(moduleOp, function, targetInstruction, targetInput);
    if (targetOp == nullptr) {
      DEBUG_LLVM_LOG_ERROR("Target operation '", targetInstruction, "' was not found in function '", function, "'\n");
      return LogicalResult::failure();
    }

    auto targetResults = targetOp->getResults();
    if (targetResults.size() != 1) {
      DEBUG_LLVM_LOG_ERROR("Target operation should has one result\n");
      return LogicalResult::failure();
    }
    OpResult target = targetResults[0];

    if (source == target) {
      DEBUG_LLVM_LOG_ERROR("Source and target ssa operands are the same\n");
      return LogicalResult::failure();
    }

    DEBUG_LLVM_LOG_INFO("Using of the result of operation:\n");
    DEBUG_LLVM_LOG_OP(sourceOp);
    DEBUG_LLVM_LOG_INFO("Was replaced with the result of operation:\n");
    DEBUG_LLVM_LOG_OP(targetOp);
    source.replaceAllUsesWith(target);
    return LogicalResult::success();
  };
};

class RemoveActionHandler : public DebugActionHandler {
public:
  LogicalResult handleAction(ModuleOp moduleOp, DictionaryAttr actionArgs) override {
    const int input = actionArgs.getAs<IntegerAttr>("input").getInt();
    const std::string instruction = actionArgs.getAs<StringAttr>("instruction").str();
    const std::string function = actionArgs.getAs<StringAttr>("function").str();

    Operation* op = findOperation(moduleOp, function, instruction, input);
    if (op == nullptr) {
      DEBUG_LLVM_LOG_INFO("RemoveActionHandler::handleAction: operation '", instruction, "' in function '", function, "' was not found\n");
      return LogicalResult::failure();
    }

    DEBUG_LLVM_LOG_INFO("RemoveActionHandler::handleAction: operation '", op->getName(), "' was found in function '", function, "'\n");
    if (op->getNumResults() != 0) {
      DEBUG_LLVM_LOG_ERROR("RemoveActionHandler::handleAction: operation '", op->getName(), "' has results and can not be removed\n");
      return LogicalResult::failure();
    }

    DEBUG_LLVM_LOG_INFO("Operation '", op->getName(), "' is removed in function '", function, "'\n");

    op->erase();
    return LogicalResult::success();
  };
};

class MoveActionHandler : public DebugActionHandler {
public:
  LogicalResult handleAction(ModuleOp moduleOp, DictionaryAttr actionArgs) override {
    const int sourceInput = actionArgs.getAs<IntegerAttr>("source_input").getInt();
    const int targetInput = actionArgs.getAs<IntegerAttr>("target_input").getInt();
    const std::string sourceInstruction = actionArgs.getAs<StringAttr>("source_instruction").str();
    const std::string targetInstruction = actionArgs.getAs<StringAttr>("target_instruction").str();
    const std::string function = actionArgs.getAs<StringAttr>("function").str();

    LLVM::LLVMFuncOp funcOp = findFunction(moduleOp, function);
    mlir::Region& bodyRegion = funcOp.getBody();
    if (bodyRegion.empty()) {
      DEBUG_LLVM_LOG_ERROR("Function '", function, "' body is empty");
      return LogicalResult::failure();
    }

    mlir::Block& entryBlock = bodyRegion.front();

    Operation* sourceOp = findOperation(moduleOp, function, sourceInstruction, sourceInput);
    if (sourceOp == nullptr) {
      DEBUG_LLVM_LOG_ERROR("Source operation '", sourceInstruction, "' was not found in function '", function, "'\n");
      return LogicalResult::failure();
    }

    auto sourceResults = sourceOp->getResults();
    if (sourceResults.size() != 1) {
      DEBUG_LLVM_LOG_ERROR("Source operation '", sourceInstruction, "' should has one result\n");
      return LogicalResult::failure();
    }
    OpResult source = sourceResults[0];

    Operation* targetOp = findOperation(moduleOp, function, targetInstruction, targetInput);
    if (targetOp == nullptr) {
      DEBUG_LLVM_LOG_ERROR("Target operation '", targetInstruction, "' was not found\n");
      return LogicalResult::failure();
    }

    OpBuilder builder(&entryBlock, std::prev(targetOp->getIterator()));
    Operation* clonedSourceOp = builder.clone(*sourceOp);

    auto clonedResults = clonedSourceOp->getResults();
    OpResult clonedResult = clonedResults[0];
    source.replaceAllUsesWith(clonedResult);
    DEBUG_LLVM_LOG_OP(clonedSourceOp);

    sourceOp->erase();
    return LogicalResult::success();
  };
};

class TypeChangeActionHandler : public DebugActionHandler {
public:
  LogicalResult handleAction(ModuleOp moduleOp, DictionaryAttr actionArgs) override {
    llvm::dbgs() << "TypeChangeActionHandler.handleAction\n";
    llvm::dbgs() << "moduleOp.dump: begin\n";
    moduleOp.dump();
    llvm::dbgs() << "moduleOp.dump: end\n";

    const int input = actionArgs.getAs<IntegerAttr>("input").getInt();
    const std::string instruction = actionArgs.getAs<StringAttr>("instruction").str();
    const std::string typeString = actionArgs.getAs<StringAttr>("type").str();

    mlir::MLIRContext* context = moduleOp.getContext();
    const Type type = getType(context, typeString);

    const std::string function = actionArgs.getAs<StringAttr>("function").str();
    LLVM::LLVMFuncOp funcOp = findFunction(moduleOp, function);
    if (!funcOp) {
      DEBUG_LLVM_LOG_ERROR("Function '", function, "' was not found");
      return LogicalResult::failure();
    }

    mlir::Region& bodyRegion = funcOp.getBody();
    if (bodyRegion.empty()) {
      DEBUG_LLVM_LOG_ERROR("Function '", function, "' body is empty");
      return LogicalResult::failure();
    }

    Operation* op = (input == -1) ?
      op = findOperation(moduleOp, function, instruction) :
      op = findOperation(moduleOp, function, instruction, input);
    if (op == nullptr) {
      DEBUG_LLVM_LOG_ERROR("Operation '", instruction, "' was not found in function '", function, "'\n");
      return LogicalResult::failure();
    }

    return changeType(context, type, op);
  }

private:
  class Context {
  public:
    template <typename T>
    T get(mlir::Type type, size_t value) {
        const auto name = llvm::getTypeName<T>().str();
        auto it = items.find(name);
        if (it == items.end()) {
          return T();
        }

        auto& map = it->second;
        auto it2 = map.find(type);
        if (it2 == map.end()) {
          return T();
        }

        auto it3 = it2->second.find(value);
        if (it3 == it2->second.end()) {
          return T();
        }

        return dyn_cast<T>(it3->second);
    }

    template <typename T>
    void add(mlir::Type type, size_t value, T& instance) {
      const auto name = llvm::getTypeName<T>().str();
      auto it = items.find(name);

      if (it == items.end()) {
        auto map = llvm::DenseMap<mlir::Type, std::unordered_map<size_t, Operation*>>();
        items.emplace(name, map);
      }

      it = items.find(name);
      auto& map = it->second;

      auto it2 = map.find(type);
      if (it2 == map.end()) {
        auto map2 = std::unordered_map<size_t, Operation*>();
        map.insert({type, map2});
      }

      it2 = map.find(type);
      it2->second.emplace(value, instance);
    }

  private:
    std::unordered_map<
      std::string,
      llvm::DenseMap<mlir::Type, std::unordered_map<size_t, Operation*>>> items;
  };

  struct VcvtffMask {
    size_t value;
    size_t bitWidth;
  };

  static bool isSupported(Operation* op, SmallVector<Value>& operands) {
    const auto name = op->getName().getStringRef().str();
    if (
      (name == "hivm_regbaseintrins.intr.hivm.vadd.s.x") ||
      (name == "hivm_regbaseintrins.intr.hivm.vmul.s.x")
      ) {
      SmallVector<Value> allOperands = op->getOperands();
      operands.emplace_back(allOperands[0]);
      operands.emplace_back(allOperands[1]);
      return true;
    }

    return false;
  }

  static Type getType(mlir::MLIRContext* context, const std::string& type) {
    OpBuilder builder(context);
    if (type == "i16") {
      return builder.getI16Type();
    } else if (type == "i32") {
      return builder.getI32Type();
    } else if (type == "f16") {
      return builder.getF16Type();
    } else if (type == "f32") {
      return builder.getF32Type();
    } else {
      const std::string message = "Unsupported type '" + type + "'";
      DEBUG_LLVM_LOG_ERROR(message);
      llvm_unreachable(message.c_str());
    }
  }

  LLVM::ConstantOp createConstant(
      OpBuilder& builder,
      mlir::Location loc,
      const mlir::Type type,
      const size_t value) {
    auto constant = executionContext.get<LLVM::ConstantOp>(type, value);
    if (!constant) {
      constant = builder.create<LLVM::ConstantOp>(loc, type, builder.getI32IntegerAttr(value));
      executionContext.add<LLVM::ConstantOp>(type, value, constant);
    }
    return constant;
  }

  hivm_regbaseintrins::PgeB32 createPgeB32(
      OpBuilder& builder,
      mlir::Location loc,
      const VcvtffMask& mask) {
    Type type = VectorType::get(SmallVector<int64_t>{hivm::util::PREDICATE_BITS}, builder.getI1Type());
    DEBUG_LLVM_LOG_DEBUG_EXEC("hivm_regbaseintrins::PgeB32: dstType: ", type.dump());
    auto pgeB32 = executionContext.get<hivm_regbaseintrins::PgeB32>(type, 0);
    if (!pgeB32) {
      auto pattern = createConstant(builder, loc, builder.getI32Type(), 8);
      auto zero = createConstant(builder, loc, builder.getI32Type(), 0);

      pgeB32 = builder.create<hivm_regbaseintrins::PgeB32>(loc, type, pattern, zero);
      pgeB32->setAttr(
        mlir::utils::maskBitWidth,
        builder.getIntegerAttr(builder.getIntegerType(mask.bitWidth), mask.bitWidth));
      executionContext.add<hivm_regbaseintrins::PgeB32>(type, 0, pgeB32);
    }
    return pgeB32;
  }

  Operation* createVcvtff(
      OpBuilder& builder,
      mlir::Location loc,
      const Type& sourceType,
      const Type& targetType,
      const VcvtffMask& mask,
      Value input) {
    auto pgeB32 = createPgeB32(builder, loc, mask);
    Value maskValue = pgeB32->getResult(0);

    auto zero = createConstant(builder, loc, builder.getI32Type(), 0);

    Operation* convertOp = nullptr;

    const auto sourceVectorType = dyn_cast<VectorType>(sourceType);
    const auto source = sourceVectorType.getElementType();

    const auto targetVectorType = dyn_cast<VectorType>(targetType);
    const auto target = targetVectorType.getElementType();

    if ((source == builder.getF32Type()) && (target == builder.getF16Type())) {
      auto op = builder.create<hivm_regbaseintrins::VcvtffF322F16InstrOp>(
        loc,
        targetType,
        input,
        maskValue,
        zero,
        zero,
        zero);
      convertOp = op.getOperation();
    } else if ((source == builder.getF16Type()) && (target == builder.getF32Type())) {
      auto op = builder.create<hivm_regbaseintrins::VcvtffF162F32InstrOp>(
        loc,
        targetType,
        input,
        maskValue,
        zero);
      convertOp = op.getOperation();
    } else {
      DEBUG_LLVM_LOG_ERROR("Source type '", source, "' and target type '", target, "' are not supported");
      llvm_unreachable("Types are not supported");
    }

    return convertOp;
  }

  Operation* createOperation(
      OpBuilder& builder,
      mlir::Location loc,
      const Type& sourceType,
      const Type& targetType,
      Operation* sourceOp,
      const SmallVector<Value>& newOperands) {
    const auto sourceVectorType = dyn_cast<VectorType>(sourceType);
    const auto source = sourceVectorType.getElementType();

    const auto targetVectorType = dyn_cast<VectorType>(targetType);
    const auto target = targetVectorType.getElementType();

    if (dyn_cast<hivm_regbaseintrins::VmulSXInstrOp>(sourceOp) != nullptr) {
      if (((source == builder.getF32Type()) && (target == builder.getF16Type())) ||
         ((source == builder.getF16Type()) && (target == builder.getF32Type()))) {
        return builder.create<hivm_regbaseintrins::VmulSXInstrOp>(
          loc,
          newOperands[0].getType(), // targetType,
          newOperands[0], newOperands[1],
          sourceOp->getOperand(2));
      }

      DEBUG_LLVM_LOG_ERROR("Unknow source '", source, "' type and target '", target, "' type");
      llvm_unreachable("Unknow types");
    } else if (dyn_cast<hivm_regbaseintrins::VaddSXInstrOp>(sourceOp) != nullptr) {
      if (((source == builder.getF32Type()) && (target == builder.getF16Type())) ||
        ((source == builder.getF16Type()) && (target == builder.getF32Type()))) {
        return builder.create<hivm_regbaseintrins::VaddSXInstrOp>(
          loc,
          newOperands[0].getType(), // targetType,
          newOperands[0], newOperands[1],
          sourceOp->getOperand(2));
      }

      DEBUG_LLVM_LOG_ERROR("Unknow source '", source, "' type and target '", target, "' type");
      llvm_unreachable("Unknow types");
    } else {
      DEBUG_LLVM_LOG_ERROR_EXEC("Unknow source operation: ", sourceOp->dump());
      llvm_unreachable("Unknow source operation");
    }
  }

  Type createType(
      Builder& builder,
      const Type operandType,
      const Type targetType,
      VcvtffMask& mask) {
    const auto vectorType = dyn_cast<VectorType>(operandType);
    if (vectorType) {
      const auto sourceType = vectorType.getElementType();
      const auto sourceShape = vectorType.getShape();

      ArrayRef<int64_t> targetShape;
      if ((sourceType == builder.getF32Type()) && (targetType == builder.getF16Type())) {
        targetShape = ArrayRef<int64_t>{ sourceShape[0] * 2 };
        mask.value = 8;
        mask.bitWidth = 32;
      } else if ((sourceType == builder.getF16Type()) && (targetType == builder.getF32Type())) {
        targetShape = ArrayRef<int64_t>{ sourceShape[0] / 2 };
        mask.value = 8;
        mask.bitWidth = 32;
      } else {
        DEBUG_LLVM_LOG_ERROR("Source type is not supported: ", sourceType);
        llvm_unreachable("Source type is not supported");
      }

      return VectorType::get(targetShape, targetType);
    }

    DEBUG_LLVM_LOG_DEBUG_EXEC("Unsupported operand type", operandType.dump());
    llvm_unreachable("Unsupported operand type");
  }

  LogicalResult changeType(
      mlir::MLIRContext* context,
      const Type type,
      Operation* op) {
    SmallVector<Value> operands;
    if (!isSupported(op, operands)) {
      DEBUG_LLVM_LOG_ERROR("operation is not supported: ");
      op->dump();
      return LogicalResult::success();
    }

    SmallVector<Value> newOperands;
    Type sourceType;
    Type targetType;
    VcvtffMask mask;

    OpBuilder builder(context);
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();

    for (Value operand : operands) {
      if (type != operand.getType()) {
        if (sourceType) {
          if (sourceType != operand.getType()) {
            DEBUG_LLVM_LOG_ERROR_EXEC("operand type is different ", operand.getType().dump());
            return LogicalResult::failure();
          }
        } else {
          sourceType = operand.getType();
        }

        Operation* ownerOperation = operand.getDefiningOp();
        if (ownerOperation == nullptr) {
          DEBUG_LLVM_LOG_DEBUG("ownerOperation is nullable");
          continue;
        }

        targetType = createType(builder, sourceType, type, mask);
        auto convertOp = createVcvtff(
          builder,
          loc,
          sourceType,
          targetType,
          mask,
          operand);
        newOperands.emplace_back(convertOp->getResult(0));

        continue;
      }
    }

    auto newOp = createOperation(
      builder,
      loc,
      sourceType,
      targetType,
      op,
      newOperands);

    Operation* convertBackOp = createVcvtff(
      builder,
      loc,
      targetType,
      sourceType,
      mask,
      newOp->getResult(0));

    auto source = op->getResult(0);
    auto target = convertBackOp->getResult(0);
    source.replaceAllUsesWith(target);

    op->erase();
    return LogicalResult::success();
  }

  Context executionContext;
};

DebugActionHandler::uptr DebugActionHandler::getHandler(const ActionType actT) {
    switch(actT) {
      case ActionType::SSA:
        return std::make_unique<SSAActionHandler>();
      case ActionType::REMOVE:
        return std::make_unique<RemoveActionHandler>();
      case ActionType::MOVE:
        return std::make_unique<MoveActionHandler>();
      case ActionType::TYPE_CHANGE:
        return std::make_unique<TypeChangeActionHandler>();
      default:
        llvm_unreachable("Unsupported LLVM debug action kind");
    }
}

struct DebugLLVMPass : public bishengir::impl::DebugLLVMBase<DebugLLVMPass> {

  using Base = bishengir::impl::DebugLLVMBase<DebugLLVMPass>;

  explicit DebugLLVMPass(const bishengir::DebugLLVMOptions &options) : Base(options) {
    if (!cacheFilePath.getValue().empty()) { // TODO: check cache option
      DebugLLVMLoger::get().init(cacheFilePath.getValue());
    }
  }

  void runOnOperation() override {
    const llvm::SmallVector<DictionaryAttr> actions = parseActionsList();

    ModuleOp moduleOp = cast<ModuleOp>(getOperation());
    for (const DictionaryAttr &actionArgs : actions) {
      DEBUG_LLVM_LOG_INFO("LLVM function before transformation\n");
      DEBUG_LLVM_LOG_FUNC(moduleOp, actionArgs.getAs<StringAttr>("function").str());

      const std::string actionType = actionArgs.getAs<StringAttr>("action").str();
      auto handler = DebugActionHandler::getHandler(convertToActionT(actionType));
      LogicalResult actionStatus = handler->handleAction(moduleOp, actionArgs);

      if (actionStatus.failed()) {
        DEBUG_LLVM_LOG_ERROR("Action type: '", actionType, "' failed\n");
      } else {
        DEBUG_LLVM_LOG_INFO("Action type: '", actionType, "' was completed\n");
      }

      DEBUG_LLVM_LOG_INFO("LLVM function after transformation\n");
      DEBUG_LLVM_LOG_FUNC(moduleOp, actionArgs.getAs<StringAttr>("function").str());
    }
  }

private:
  llvm::SmallVector<DictionaryAttr> parseActionsList() {
    llvm::SmallVector<DictionaryAttr> actions;

    const auto config = Config::getEnv();
    if (config.debugLLVM) {
      mlir::MLIRContext& context = getContext();
      actions.push_back(config.toDictionaryArg(context));
    } else {
      llvm::StringRef debugLlvmString(actionsList.getValue());
      llvm::SmallVector<llvm::StringRef> debugLlvmStringSplit;
      debugLlvmString.split(debugLlvmStringSplit, ';');

      for (const auto &debugAct : debugLlvmStringSplit) {
        auto actionAtrr = parseAttribute(debugAct.str(), getOperation()->getContext());
        auto parsedDict = dyn_cast<DictionaryAttr>(actionAtrr);
        actions.push_back(parsedDict);
      }
    }

    return actions;
  }
};

} // namespace

std::unique_ptr<Pass> bishengir::createDebugLLVMPass(const bishengir::DebugLLVMOptions &options) {
  return std::make_unique<DebugLLVMPass>(options);
}
