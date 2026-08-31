# AscendNPU-IR Agent Guidelines

## LLVM/MLIR Coding Rules

- Follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
  and the conventions used by the surrounding LLVM/MLIR code.
- Format C and C++ changes with the repository-root `.clang-format`. Before a
  commit is created, run the following command on every changed C/C++ source and
  header file:

  ```bash
  clang-format -i -style=file <changed-files>
  ```

- Format only files touched by the change. Do not reformat unrelated code.
- Do not hand-edit generated files. Change the corresponding TableGen source,
  template, or generator instead.
- Keep changes focused and consistent with nearby code. Avoid unrelated cleanup
  or refactoring in the same change.

## Commit Messages

Use the following commit-message format:

```text
[AscendNPU IR][<Module>] <type>: <short description>
[AscendNPU IR][<Module>][<Submodule>] <type>: <short description>

Motivation: <why this change is needed and what problem it solves>
Design: <the general approach and any important algorithm or design choice>
Risks: <potential risks or side effects; write "None" when there are none>

Assisted-by: AI
```

Rules:

- Use one of these commit types: `feat`, `fix`, `doc`, `refactor`, or `chore`.
- Use one or two module levels. For a dialect-related change, use the dialect as
  the first level, for example `[HFusion]` or `[HIVM]`. When the change targets
  a specific pass or component, add it as the second level, for example
  `[HIVM][PlanMemory]`.
- Use the closest owning module for cross-cutting changes. Omit the second level
  when there is no useful, more specific scope. Other module or submodule names
  include `AVE`, `VFFusion`, `SIMT`, `CVPipeline`, and `AutoBlockify`.
- Keep the title concise (about 50 characters when practical) and use imperative
  mood.
- Separate the title and body with a blank line. Wrap body text at about 72
  characters per line.
- Delete all placeholders before committing. Keep `Motivation`, `Design`, and
  `Risks` in the body; use `None` when there is no known risk.
- If AI assisted with the implementation, documentation, review, or commit
  message, the commit message must contain the exact trailer `Assisted-by: AI`.
  Do not include the agent name or model version. Omit the trailer only when the
  change is completely human-written.

## LLVM/MLIR Submodule Changes

`third-party/llvm-project` is a separate upstream-derived submodule. Keep
AscendNPU-IR-specific behavior isolated from community LLVM/MLIR behavior.

- Guard AscendNPU-IR-specific LLVM/MLIR changes with the common
  `BSPUB_DAVINCI_BISHENGIR` feature macro.

- Do not expose AscendNPU-IR-specific APIs or semantics to a community build
  unless the change is intended and suitable for upstream LLVM/MLIR.
- If a community LLVM/MLIR test fails, first determine whether it is an actual
  regression. Fix regressions instead of disabling their tests.
- Only when the failure is caused by an intentional, macro-isolated BiShengIR
  semantic difference may the community test be disabled for the BiShengIR
  configuration. Add this directive to the original test:

  ```text
  // UNSUPPORTED: bspub_davinci_bishengir
  ```

- When disabling a community test this way, copy the test into the matching
  location under `bishengir/test/`, adapt it to the BiShengIR behavior, and keep
  the relevant assertions. For example, an upstream test at
  `mlir/test/Dialect/Tensor/example.mlir` should normally be mirrored as
  `bishengir/test/Dialect/Tensor/example.mlir`.
- Do not weaken, remove, or mark a test unsupported merely to make a test suite
  pass. The copied test must exercise the intended BiShengIR behavior.
- Validate submodule changes with the relevant focused tests, followed by both
  suites before submission:

  ```bash
  ninja -C build check-mlir
  ninja -C build check-bishengir
  ```
