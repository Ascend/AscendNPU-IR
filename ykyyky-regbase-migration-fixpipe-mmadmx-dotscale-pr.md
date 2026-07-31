## Migration scope
- Author: Yuan Keyu (ykyyky)
- Source branch: `dev/regbase` (merge commits; tips also on `feature/regbase` lineage)
- Source date range: June 29 through July 15, 2026, inclusive for the checklist window; **this PR also includes two later author PRs explicitly requested for migration** (July 21 / July 24)
- Target branch: `master` (base: `merged/master`)
- Source commits / merges, in order:
  - `ac6741dd7` `!1453` merge fix-fixpipe-si8-cast into dev/regbase
    - tips: `2beb4e12c` Fix the fixpipe config when dst is i8; `b639d2b50` Avoid fusing i2i truncate casting into fixpipe
  - `5d19eab9a` `!1577` merge fix-avoid-cross-scope-fixpipe-inlining into dev/regbase
    - tip: `dc4a360d4` Avoid inline fixpipe fuse cross-scope stores
  - `70de8f7c0` `!1747` merge feature-mmadmx-insert-convert-layout into dev/regbase *(outside window; requested)*
    - tip: `f6e038b97` Support mmadmxL1 in InsertConvertLayout and fuse scale convert_layout
  - `90c5b3698` `!1900` merge fix-dot-scale-template into dev/regbase *(outside window; requested)*
    - tip: `dc3416fe0` Fix stride calculating in load_scale

## Summary

Migrate four regbase (A5) fixes/features onto the A3/A5 merged `master`: prevent incorrect fixpipe i2i saturation fusion (already present), refuse fixpipe fusion into VECTOR-scope stores, bring `mmadmxL1` into the pre-bufferization convert_layout + `load_scale` fusion path, and fix L1 loop3 stride for padded scale subviews in the RegBase `load_scale` template.

## Background

On `dev/regbase`, these PRs land as merge commits. Master already had the InlineFixpipe truncate-saturate guard and RegBase i8 PRE_QUANT bit from `!1453`, but was missing cross-scope fixpipe guarding, mmadmx InsertConvertLayout / load_scale combine, and the load_scale `n_tile_ceil` stride fix.

Problematic patterns (minimal):

1. **Cross-scope fixpipe** — cube `fixpipe` result stored inside `scope.scope { hivm.tcore_type = VECTOR }`; fusion would place cube fixpipe in the vector scope.
2. **mmadmx without InsertConvertLayout** — `hivm.hir.mmadmxL1` left ND until post-bufferization InferHIVMDataLayout; no pre-bufferization `convert_layout` / `load_scale` fusion.
3. **load_scale stride** — loop3 dst stride used logical `n/2` instead of physical L1 `strides[0]/strides[1]`, breaking fractal subviews into padded allocs.

## Changes

- **`ac6741dd7` / fix-fixpipe-si8-cast**: **Already on master** — `isVcastInlinableIntoFixpipe` (gated by `isRegBasedArch`) in `InlineFixpipe.cpp`, lit in `inline-fixpipe-portable.mlir`, and i8 PRE_QUANT bit 46 in `bishengir/lib/Template/lib/RegBase/Cube/compat/Fixpipe/Fixpipe.cpp`. No further code change.
- **`5d19eab9a` / cross-scope fixpipe**: `InlineFixpipe.cpp` — `isInsideVectorScope`; refuse inline when `isRegBasedArch && isInsideVectorScope(curOp)`. Test appended to `inline-fixpipe-portable.mlir`.
- **`70de8f7c0` / mmadmx convert_layout** (manual port of `f6e038b97`, not later dn2nz redesign):
  - Layout attrs/interface: `isScaleFractalLayout()`, `FractalOperandLayouts::{scaleA,scaleB}`, `MmadMxL1Op::getOperandsTargetFractalLayout`, `getScaleBlockSizes` → `[16,2]`.
  - `InsertConvertLayoutAroundMmadMxL1` gated by `hacc::utils::isRegBasedArch`; MmadL1 pattern kept separate.
  - CombineOptimized: fuse scale `load + convert_layout` → `load_scale`; ToTranspose / shape utils / Propagate / InferHIVM scale index + non-transposed-only fold.
  - Lit: `insert-convert-layout.mlir`, `combine-optimized-convert-layout.mlir`, `convert-layout-to-transpose.mlir`.
- **`90c5b3698` / load_scale stride**: RegBase-only `loadMXScale.cpp` — `n_tile_ceil = l1->strides[0] / l1->strides[1]` for loop3 dst stride.

## Expected IR difference

**Cross-scope fixpipe (Ascend950):** store inside VECTOR scope is not fused into fixpipe outs.

**mmadmx InsertConvertLayout (Ascend950):** before — ND operands into `mmadmxL1`; after — `convert_layout` on A/B/ScaleA/ScaleB/C to fractal / `SCALEA_zZ` / `SCALEB_nN` (`fractalSizes = [16, 2]` for scales), result converted back to ND; combine may fold scale path to `hivm.hir.load_scale`.

**load_scale template:** padded L1 fractal subviews use parent row pitch for loop3 stride (behavioral; no MLIR lit).

## Conflict resolution
- Files with conflicts: none (manual port; no cherry-pick apply).
- Resolution and rationale:
  - `!1453` template change maps to **RegBase** `compat/Fixpipe/Fixpipe.cpp` on master (shared `lib/Fixpipe/Fixpipe.cpp` is A3/membase and lacks `set_pre_quant_scale` / quant_scale args). Already present — no edit.
  - `MmadMxL1Op::getOperandsTargetFractalLayout` implemented in `HIVMMacroOps.cpp` next to file-local `getScaleBlockSizes` (master’s MmadL1 fractal lives in `GetOperandsLayout.cpp`).
  - InsertConvertLayout kept a **separate** regbase-gated MmadMx pattern instead of fully replacing MmadL1 with `LocalMatmulLikeOpInterface`, to avoid A3 behavior drift.
- Shared/Membase behavior impact:
  - MmadL1 InsertConvertLayout path preserved.
  - Cross-scope guard and mmadmx insert gated by `isRegBasedArch`.
  - `getScaleBlockSizes` / InferHIVM scale indexing only affect MX scale paths.
  - `loadMXScale.cpp` is under `Template/lib/RegBase/` only.

## High-impact review
- Pipeline files reviewed: no HFusion/HIVM pipeline file changes required for these PRs.
- PlanMemory/Normalize/lowering files reviewed: N/A for these commits; InferHIVMDataLayout scale fold updated only.
- Expected behavior confirmed: lit PASS for inline-fixpipe-portable, insert-convert-layout, combine-optimized-convert-layout, convert-layout-to-transpose.
- Remaining risks:
  - Later regbase work (`60c8e56b8`) replaces `load_scale` fusion with `i8→f16 reinterpret_cast + dn2nz`; **not** migrated here (out of requested commit set / design-doc target state).
  - Transposed scale `load_scale` fold is rejected (`llvm_unreachable`), matching `f6e038b97`.
  - `PropagateConvertLayout` now treats any `LocalMatmulLikeOpInterface` user as a stop; low A3 risk (mmadmx is regbase-centric).

## Remaining issues / assumptions
- `ac6741dd7` duplicate SHA in the request list treated as one merge.
- Commits `70de8f7c0` / `90c5b3698` are after the formal June 29–July 15 window; included by explicit request.
- No corresponding-location failures: all source files have master counterparts; Fixpipe i8 fix already lives under RegBase compat rather than shared Fixpipe.

## Validation
- CI/gate: not required for this migration round
- Optional local checks performed:
  - `ninja bishengir-opt` (incremental) succeeded during port
  - `llvm-lit` PASS:
    - `Dialect/HIVM/inline-fixpipe-portable.mlir`
    - `Dialect/HIVM/insert-convert-layout.mlir`
    - `Dialect/HIVM/combine-optimized-convert-layout.mlir`
    - `Dialect/HIVM/convert-layout-to-transpose.mlir`
