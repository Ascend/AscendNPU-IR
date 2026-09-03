//===- SymbolOps.td - Symbol dialect operation definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for Symbol dialect operations.
// Symbol dialect operations mainly deals with dynamic shapes representation.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SYMBOL_IR_SYMBOLOPS_TD
#define BISHENGIR_DIALECT_SYMBOL_IR_SYMBOLOPS_TD

include "bishengir/Dialect/Symbol/IR/SymbolBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Symbol Dialect operations.
//===----------------------------------------------------------------------===//

def SymbolicIntOp : Symbol_Op<"symbolic_int", [
  Pure,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Represents a symbolic integer with a constrained range";
  let description = [{
    The "symbolic_int" operation represents a symbolic integer value by name
    (via a symbol attribute) and returns it as an `index`-typed value.
    It includes range constraints through `min_val` and `max_val` attributes
    that specify the inclusive bounds for the symbolic value.

    This operation is typically used to represent dynamic dimensions or
    other symbolic integer values that have known constraints.

    Example:
    ```
    %0 = symbol.symbolic_int @s0 {min_val = 5, max_val = 10} : index
    %1 = symbol.symbolic_int @s1 {min_val = 2, max_val = 20} : index
    %2 = symbol.symbolic_int @s2 [%0, %1], affine_map<()[s1, s2] -> (s1 * s2)> {min_val = 2, max_val = 20} : index
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$symbol_name,
    I64Attr:$min_val,
    I64Attr:$max_val,
    Variadic<Index>:$int_symbols,
    OptionalAttr<Builtin_AffineMapAttr>:$int_expressions
  );
  let results = (outs Index:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$result,
                   "FlatSymbolRefAttr":$symbol_name,
                   "int64_t":$min_val,
                   "int64_t":$max_val)>,
    OpBuilder<(ins "FlatSymbolRefAttr":$symbol_name)>,
    OpBuilder<(ins "FlatSymbolRefAttr":$symbol_name,
                   "ValueRange":$int_symbols,
                   "AffineMapAttr":$int_expressions)>
  ];
}

def BindSymbolicShapeOp : Symbol_Op<"bind_symbolic_shape"> {
  let summary = "Binds shape expressions to tensors using an affine map indexed by shape symbols";
  let description = [{
    The `symbol.bind_symbolic_shape` operation binds shape expressions
    useful to compute the dynamic dimensions of a tensor. It takes a
    variadic of SSA symbols that map 1:1 to the local symbols declared
    in the affine map. The affine map contains a list of affine shape
    expressions for each dim where the terminals are from the declared
    symbols.

    Example:
    ```
    symbol.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>
    symbol.bind_symbolic_shape %out0, [%0, %1, %2], affine_map<()[s0, s1, s2] -> (s0, s1 * 2 + s2, 3)> : tensor<?x?x3xf32>
    ```
  }];
  let arguments = (ins
    AnyShaped:$operand,
    Variadic<Index>:$shape_symbols,
    Builtin_AffineMapAttr:$shape_expressions
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // BISHENGIR_DIALECT_SYMBOL_IR_SYMBOLOPS_TD
