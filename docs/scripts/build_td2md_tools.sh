#!/usr/bin/env bash
#
# Independently build the two tools the bishengir doc generator depends on:
#   1. mlir-tblgen                     (vanilla upstream MLIR tablegen, ~234 cmds)
#   2. bishengir-hfusion-ods-yaml-gen (HFusion ODS yaml -> .td, 1 .cpp + MLIRIR)
# Plus the one generated .td it produces (HFusionNamedStructuredOps.yamlgen.td),
# so the resulting build dir is a drop-in BUILD_DIR for the doc generator.
#
# Uses a SEPARATE minimal build dir (default build/td2md-tools) — does NOT touch the
# main build/. Configure is one-time; afterwards only the 3 targets are built
# (~600 commands, ~1/10 of a full build).
#
# Paths derived from script location; override with env vars:
#   ROOT       project root      (default: parent of docs/scripts/)
#   BUILD_DIR  build output dir  (default: $ROOT/build/td2md-tools)
#   CC/CXX     compilers         (default: clang/clang++)
#   JOBS       parallel jobs     (default: nproc)
#
# Usage:
#   docs/scripts/build_td2md_tools.sh                 # configure (if needed) + build
#   BUILD_DIR=$PWD/build bash docs/scripts/build_td2md_tools.sh   # reuse an existing build dir
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/td2md-tools}"
LLVM_SRC="$ROOT/third-party/llvm-project/llvm"
JOBS="${JOBS:-$(nproc)}"

# Targets: the 2 tool binaries + the yamlgen.td they need for HFusion docs.
TARGETS=(
  mlir-tblgen
  bishengir-hfusion-ods-yaml-gen
  BiShengIRHFusionNamedStructuredOpsYamlIncGen   # produces HFusionNamedStructuredOps.yamlgen.td
)

echo "[td2md-tools] project root: $ROOT"
echo "[td2md-tools] build dir:    $BUILD_DIR"

# --- 1. Configure (one-time) -------------------------------------------------
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
  if [ ! -d "$LLVM_SRC" ]; then
    echo "[td2md-tools] ERROR: $LLVM_SRC not found — fetch the llvm-project submodule first"
    echo "           (cd \"$ROOT\" && git submodule update --init --depth 1 third-party/llvm-project)"
    exit 1
  fi
  echo "[td2md-tools] configuring (one-time) — minimal: mlir + bishengir, triton/tests/examples off ..."
  cmake -S "$LLVM_SRC" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC:-clang}" -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_CCACHE_BUILD=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=bishengir \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$ROOT" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
    -DBISHENGIR_BUILD_EXAMPLES=OFF \
    -DBISHENGIR_ENABLE_TRITON_COMPILE=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_BSPUB_DAVINCI_BISHENGIR=ON -DBSPUB_DAVINCI_BISHENGIR=ON \
    -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON
else
  echo "[td2md-tools] build dir already configured, skipping configure."
fi

# --- 2. Build only the needed targets ----------------------------------------
echo "[td2md-tools] building: ${TARGETS[*]}"
ninja -C "$BUILD_DIR" -j "$JOBS" "${TARGETS[@]}"

# --- 3. Report ---------------------------------------------------------------
YAMLGEN_TD="$BUILD_DIR/tools/bishengir/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yamlgen.td"
echo
echo "[td2md-tools] done. Artifacts:"
echo "  tool:  $BUILD_DIR/bin/mlir-tblgen"
echo "  tool:  $BUILD_DIR/bin/bishengir-hfusion-ods-yaml-gen"
[ -f "$YAMLGEN_TD" ] && echo "  gen:   $YAMLGEN_TD"
echo
echo "Point the doc generator at this build dir:"
echo "  BUILD_DIR=\"$BUILD_DIR\" bash docs/scripts/gen_bishengir_pass_dialect_md.sh"
