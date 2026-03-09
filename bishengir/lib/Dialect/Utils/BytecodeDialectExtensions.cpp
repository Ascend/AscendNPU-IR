//===- BytecodeDialectExtensions.cpp - Bytecode dialect extensions -------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Utils/BytecodeDialectExtensions.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace {

// TODO: release multiple versions of ascendnpu ir, then
// use the ascendnpu-ir-version to control the upgrade logic.
// TritonAscend version is used temporarily.
enum class TritonAscendVersion : uint64_t {
  V3_2_0 = 320,
  V3_3_0 = 330,
  V3_4_0 = 340,
  V3_5_0 = 350,
};

struct BufferizationDialectVersion : public DialectVersion {
  explicit BufferizationDialectVersion(uint64_t version) : version(version) {}
  uint64_t version;
};

struct BufferizationBytecodeDialectInterface : public BytecodeDialectInterface {
  explicit BufferizationBytecodeDialectInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

private:
  static LogicalResult downgradeToBufferOp(Operation *topLevelOp) {
    MLIRContext *ctx = topLevelOp->getContext();
    ctx->allowUnregisteredDialects();

    SmallVector<Operation *> toBufferOps;
    topLevelOp->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "bufferization.to_buffer")
        toBufferOps.push_back(op);
    });

    OpBuilder builder(ctx);
    for (Operation *op : toBufferOps) {
      builder.setInsertionPoint(op);

      Value tensorInput = op->getOperand(0);
      Type memrefType = op->getResult(0).getType();
      auto readOnlyAttr = op->getAttrOfType<UnitAttr>("read_only");
      auto toMemrefOp = builder.create<bufferization::ToMemrefOp>(
          op->getLoc(), memrefType, tensorInput, readOnlyAttr);
      op->replaceAllUsesWith(toMemrefOp);
      op->erase();
    }

    return success();
  }

public:
  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const override {
    uint64_t version;
    if (failed(reader.readVarInt(version)))
      return nullptr;
    return std::make_unique<BufferizationDialectVersion>(version);
  }

  LogicalResult upgradeFromVersion(Operation *topLevelOp,
                                   const DialectVersion &version) const override {
    if (!topLevelOp)
      return success();

    const auto &bufferizationVersion =
        static_cast<const BufferizationDialectVersion &>(version);
    // TODO: Implement ascendnpu ir upgrade logic.
    // Currently we impl downgrade logic to support TA that has higher versions.
    if (bufferizationVersion.version >=
        static_cast<uint64_t>(TritonAscendVersion::V3_4_0))
      return downgradeToBufferOp(topLevelOp);

    return success();
  }
};

struct FuncDialectVersion : public DialectVersion {
  explicit FuncDialectVersion(uint64_t version) : version(version) {}
  uint64_t version;
};

struct FuncBytecodeDialectInterface : public BytecodeDialectInterface {
  explicit FuncBytecodeDialectInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const override {
    uint64_t version;
    if (failed(reader.readVarInt(version)))
      return nullptr;
    return std::make_unique<FuncDialectVersion>(version);
  }
};

} // namespace

void bishengir::registerBytecodeDialectExtensions(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *,
                            bufferization::BufferizationDialect *dialect) {
    dialect->addInterfaces<BufferizationBytecodeDialectInterface>();
  });
  registry.addExtension(+[](MLIRContext *, func::FuncDialect *dialect) {
    dialect->addInterfaces<FuncBytecodeDialectInterface>();
  });
}
