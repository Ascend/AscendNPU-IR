//===- OpPipeInterface.td - Ops with Pipe Information -----*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
//
// This is the definition file for the OpPipeInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_OPPIPEINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_OPPIPEINTERFACE_TD

include "mlir/IR/OpBase.td"

def OpPipeInterface : OpInterface<"OpPipeInterface"> {
  let description = [{
   HIVM Op has three kinds of op:
     1. op with single pipe.
     2. macro op with input pipe and output pipe.
     3. op with no pipe attribute.
  }];

  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/"judge if it is op with single pipe.",
      /*retTy=*/"bool",
      /*methodName=*/"isSinglePipeOp",
      /*args=*/(ins),
      /*methodBody=*/[{
        return ConcreteOp::template hasTrait<OpTrait::SinglePipeOpTrait>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"judge if it is compound macro op.",
      /*retTy=*/"bool",
      /*methodName=*/"isMacroOp",
      /*args=*/(ins),
      /*methodBody=*/[{
        return ConcreteOp::template hasTrait<OpTrait::MacroOpTrait>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"get op pipe if it is op with single pipe.",
      /*retTy=*/"mlir::hivm::PIPE",
      /*methodName=*/"getPipe",
      /*args=*/(ins),
      /*methodBody=*/[{
        if constexpr(ConcreteOp::template hasTrait<OpTrait::SinglePipeOpTrait>()) {
// TODO : release different version of ascendnpu ir and remove the macro
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
          return $_op.getPipe();
#else
          ConcreteOp op = $_op;
          return op.getPipe();
#endif
        } else {
          return mlir::hivm::PIPE::PIPE_UNASSIGNED;
        }
      }]
    >,
    InterfaceMethod<
      /*desc=*/"get input pipe if it is macro op.",
      /*retTy=*/"mlir::hivm::PIPE",
      /*methodName=*/"getInPipe",
      /*args=*/(ins),
      /*methodBody=*/[{
        if constexpr(ConcreteOp::template hasTrait<OpTrait::MacroOpTrait>()) {
// TODO : release different version of ascendnpu ir and remove the macro
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
          return $_op.getInPipe();
#else
          ConcreteOp op = $_op;
          return op.getInPipe();
#endif
        } else {
          return mlir::hivm::PIPE::PIPE_UNASSIGNED;
        }
      }]
    >,
    InterfaceMethod<
      /*desc=*/"get output pipe if it is macro op.",
      /*retTy=*/"mlir::hivm::PIPE",
      /*methodName=*/"getOutPipe",
      /*args=*/(ins),
      /*methodBody=*/[{
        if constexpr(ConcreteOp::template hasTrait<OpTrait::MacroOpTrait>()) {
// TODO : release different version of ascendnpu ir and remove the macro
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
          return $_op.getOutPipe();
#else
          ConcreteOp op = $_op;
          return op.getOutPipe();
#endif
        } else {
          return mlir::hivm::PIPE::PIPE_UNASSIGNED;
        }
      }]
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_OPPIPEINTERFACE_TD
