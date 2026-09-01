#!/usr/bin/env bash
#
# Minimal generator for the bishengir pass/dialect .md files.
#
# Runs mlir-tblgen directly on the .td sources — no ninja, no -D defines, no
# depfiles. Output is byte-identical to `ninja bishengir-doc` (verified).
#
# Paths are derived from this script's location; override with env vars:
#   SRC_DIR   project root        (default: parent of docs/scripts/)
#   BUILD_DIR  build output dir   (default: $SRC_DIR/build)
#   OUT_DIR    where .md go       (default: $BUILD_DIR/bishengir-pass-dialect-md)
#
# Usage:
#   docs/scripts/gen_bishengir_pass_dialect_md.sh   # generate all 19 .md into $OUT_DIR
#
# This script only generates .md files. Copying them into the docs source tree
# is the caller's job (see docs/Makefile target gen-bishengir-pass-dialect-md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build}"
OUT_DIR="${OUT_DIR:-$BUILD_DIR/bishengir-pass-dialect-md}"
TBLGEN="$BUILD_DIR/bin/mlir-tblgen"

# Only three shared -I are needed (verified minimal): mlir source .td,
# bishengir source .td, and generated bishengir .td (.td.inc / yamlgen.td).
SHARED_I=(
  "-I$SRC_DIR/third-party/llvm-project/mlir/include"
  "-I$SRC_DIR/bishengir/include"
  "-I$BUILD_DIR/tools/bishengir/bishengir/include"
)

# table: gen_flag | dialect(- = none) | td_relpath | out_filename (snake_case)
TABLE=(
  "-gen-dialect-doc|annotation|bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationOps.td|annotation_dialect.md"
  "-gen-pass-doc|-|bishengir/include/bishengir/Dialect/Annotation/Transforms/Passes.td|annotation_passes.md"
  # HACC: use HACCAttrs.td (not HACCBase.td) — Base only declares the dialect
  # shell; attrs/enums live in HACCAttrs.td, which the old committed doc used.
  "-gen-dialect-doc|hacc|bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td|hacc_dialect.md"
  "-gen-pass-doc|-|bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td|hacc_passes.md"
  "-gen-dialect-doc|hfusion|bishengir/include/bishengir/Dialect/HFusion/IR/HFusionDoc.td|hfusion_dialect.md"
  "-gen-op-doc|-|bishengir/include/bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.td|hfusion_transform_ops.md"
  "-gen-pass-doc|-|bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td|hfusion_passes.md"
  "-gen-dialect-doc|hivm|bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDoc.td|hivm_dialect.md"
  "-gen-op-doc|-|bishengir/include/bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.td|hivm_transform_ops.md"
  "-gen-pass-doc|-|bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td|hivm_passes.md"
  "-gen-pass-doc|-|bishengir/include/bishengir/Dialect/LLVMIR/Transforms/Passes.td|bishengir_llvm_passes.md"
  "-gen-dialect-doc|mathExt|bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.td|math_ext_dialect.md"
  "-gen-dialect-doc|memref_ext|bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtOps.td|memref_ext_dialect.md"
  "-gen-op-doc|-|bishengir/include/bishengir/Dialect/SCF/TransformOps/SCFTransformOps.td|bishengir_scf_loop_transform_ops.md"
  "-gen-dialect-doc|scope|bishengir/include/bishengir/Dialect/Scope/IR/ScopeOps.td|scope_dialect.md"
  "-gen-pass-doc|-|bishengir/include/bishengir/Dialect/Scope/Transforms/Passes.td|scope_passes.md"
  "-gen-dialect-doc|symbol|bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.td|symbol_dialect.md"
  "-gen-pass-doc|-|bishengir/include/bishengir/Dialect/Symbol/Transforms/Passes.td|symbol_passes.md"
  "-gen-pass-doc|-|bishengir/include/bishengir/ExecutionEngine/Passes.td|execution_engine_passes.md"
)

mkdir -p "$OUT_DIR"

echo "[bishengir-pass-dialect-md] generating ${#TABLE[@]} .md files into $OUT_DIR ..."
for row in "${TABLE[@]}"; do
  IFS='|' read -r flag dialect td out <<< "$row"
  args=("$flag" "-allow-hugo-specific-features" "-I$(dirname "$SRC_DIR/$td")" "${SHARED_I[@]}")
  [[ "$dialect" != "-" ]] && args+=("-dialect" "$dialect")
  "$TBLGEN" "${args[@]}" -o "$OUT_DIR/$out" "$SRC_DIR/$td"
done
echo "[bishengir-pass-dialect-md] done. Output in $OUT_DIR"
