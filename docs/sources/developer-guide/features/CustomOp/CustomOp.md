# CustomOp

## Overview

AscendNPU-IR already supports a rich operator set for upstream models. However, in certain scenarios, there are needs to define their own operators to perform custom computations:

- Supported operaters' combination couldn't fulfill desired computations.
- Vendor wants the custom operator to be private.
- Combining multiple operators could not reach optimal performance.

Custom operator allows users to freely use the interfaces provided by AscendNPU-IR to provide their own operators that compiles with other operators.

---

## Interface

Generic interface for custom op as following:
- name : unique op name.

         Note : there are names reserved for builtins, usually starts with "__builtin".
                Compiler will link these builtins to self-contained template library,
                which comes together within bishengir-compile.

                For normal names/cases, user needs to specify implementation location/compilation commands (TODO),
                and all ther necessary informations.

         Available builtin names :
            "__builtin_gather_load"

- inputs : input parameters.
- outputs : output results, designated "init" operands, which act as initial values for the results
            of the operation or the init locations to which the results of the op will be written.

In order to adapt to future enhancements quickly and dynamically, custom op relies on attributes
to retreive necessary information, required informations are:
- CoreType : which core type to execute on, refer to TCoreTypeAttr.
- Pipe     : which pipe to execute on, refer to PipeAttr.
- VFMode   : which mode to run on vector units, refer to VFModeAttr.
             this attribute is ignored when core type is cube.

             Note : for builtins, user could specify these informations or not,
                    compiler will help to check the correctness and canonicalize.

TODO:
- Impl : user provided implementation and linking process.
- Multi Pipe : custom op wants to use multiple pipes, which is a MacroOp in HIVM's context.

---

## Lowering Process

```
┌─────────────────────────────────────────────────────────────────┐
│                          CustomOp                               │
│    hivm.hir.custom "name" { attrs... } ins(..) outs(...)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  HIVMToStandard                                                 │
│  ───────────────────────────────────────────────────────────────│
│  • Builtins                                                     │
│    -> call to builtins libraries                                │
│  • User provided implementations (WIP) ->                       │
|    -> call to user provided function name                       |
|      -> bishengir-compile link with user provided link commands |
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            BiSheng Compiler compiles to objects
```
---

## Capability & Limitation

### ✅ Capabilities

| Feature                         | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| **CoreType**                    | Custom op execution core.                                    |
| **Pipe**                        | Custom op execution pipe.                                    |
| **VFMode**                      | Custom op running mode on vector core, SIMT/SIMD/MIX.        |
| **Buitlins**                    | Set of builtins (name reserved).                             |

### ⚠️ Limitations

| Limitation                   | Description                                               | Status                                                                 |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **User implementations**     | Custom op lowered to user provided implementations.       | Work in progress.
| **Passes interactions**        Transformation passes that adapt to custom op.            | NA, work in progress.

---

## MLIR Example

### Builtin

```mlir
%0 = hivm.hir.custom
       "__builtin_gather_load"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>

```

### Custom

```mlir
%0 = hivm.hir.custom
      { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD> }
      "my_custom_op"
      ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
          : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
      outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```
