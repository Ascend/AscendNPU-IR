//===- AnnotationOps.td - Annotation op definitions -----------*-tablegen-*-==//
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

#ifndef BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONOPS_TD
#define BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONOPS_TD

include "bishengir/Dialect/Annotation/IR/AnnotationBase.td"
include "bishengir/Dialect/Annotation/IR/AnnotationAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MarkOp : Op<Annotation_Dialect, "mark", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface, ["getEffects"]>]> {
  let summary = "Annotate an IR value with key-value attribute pairs";
  let description = [{
    The `annotation.mark` operation can be used to annotated an IR value
    with a set of key-value attributes pairs.

    The value can be both static (i.e. an attribute) or dynamic
    (a.k.a. an IR value).

    The `effects` attribute is used to specify the memory effect that the `annotation.mark` op
    has on the marked value. By default, it has a memory write effect.

    Example:

    ```mlir
    annotation.mark %target keys = ["key"] values = [%val]
    annotation.mark %target {key : val}
    ```
  }];
  let arguments = (ins AnyType:$src,
                       DefaultValuedAttr<StrArrayAttr, 
                       "llvm::ArrayRef<StringRef>{stringifyEffectMode(EffectMode::Write)}">:$effects,
                       Variadic<AnyType>:$values,
                       OptionalAttr<StrArrayAttr>:$keys);
  let assemblyFormat = [{
    $src attr-dict
    (`keys` `=` $keys^)?
    (`values` `=` `[`$values^`:`type($values) `]`)?
    `:`type($src)
  }];
  let builders = [
    OpBuilder<(ins "Value": $src)>,
    OpBuilder<(ins "Value":$src, "ValueRange":$values, "ArrayAttr":$key)>
  ];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let extraClassDeclaration = [{
    /// Return whether the `annotation.mark` op is annotated
    /// by a key-value pair with the target `key`.
    bool isAnnotatedBy(StringRef key);

    bool isAnnotatedByStaticAttr(StringRef key);
    bool isAnnotatedByDynamicAttr(StringRef key);

    /// Return key-value pair's value with the target `key`.
    OpFoldResult getMixedAttrValue(StringRef key);

    Attribute getStaticAttrValue(StringRef key);
    Value getDynamicAttrValue(StringRef key);

    // Return attribute number except attribute `effects`
    int64_t getAttrNum();

    // Return whether attribute number is empty except attribute `effects`
    bool isAttrEmpty();
  }];
}

#endif // BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONOPS_TD
