# Version Compatibility

This document describes the CANN software stack and hardware driver environment on which AscendNPU IR depends.

To ensure compilation and runtime stability, configure the environment strictly according to the compatibility relationship provided in this document.

## CANN Compatibility

The following version combinations are verified recommended configurations and should be used preferentially:

| AscendNPU IR Version | Gitcode Branch | CANN Version Dependency | Hardware Support |
| --- | --- | --- | --- |
| `v2.0.0` | `release/v2.0.x` | CANN 9.2.0 | <ul><li>Ascend 950PR&950DT</li><li>Atlas A3 products</li><li>Atlas A2 products</li></ul> |
| `v1.2.0` | `release/v1.2.x` | CANN 9.1.0 | <ul><li>Ascend 950PR&950DT (branch `feature/regbase`)</li><li>Atlas A3 products</li><li>Atlas A2 products</li></ul> |
| `v1.1.0` | `release/v1.1.x` | CANN 9.0.0 | <ul><li>Atlas A3 products</li><li>Atlas A2 products</li></ul> |
| `v1.0.0` | `release/v1.0.0` | CANN 8.5.0 | <ul><li>Atlas A3 products</li><li>Atlas A2 products</li></ul> |

**Important Notes**:

If you want to keep the current CANN version without upgrading or replacing it while still using the new features of AscendNPU IR, you can extract the AscendNPU IR component from the new CANN toolkit package and use it to replace the NPUIR version in your current CANN package. The steps are as follows:

```bash
# In this example, replace the AscendNPU IR in a CANN 8.5.0 installation with the version from the CANN 9.0.0 package.
# 1. Prepare variables: the new CANN package path (<arch> should be filled in according to the actual architecture, such as x86_64), the temporary extraction directory, and the current CANN installation path.
NEW_CANN_PKG="PATH-TO/Ascend-cann_9.0.0_linux-<arch>.run"
TMP_PATH="tmp_pkg_files"
OLD_CANN_PATH="${CANN_850_PATH}/Ascend/cann-8.5.0"

# 2. Extract the new CANN toolkit package (--noexec means extract only, without installation).
bash ${NEW_CANN_PKG} --noexec --extract=cann900

# 3. Extract the ascendnpu-ir subpackage from the extraction directory.
bash cann900/run_package/ascendnpu-ir_*.run --noexec --extract=$TMP_PATH

# 4. Replace the NPUIR in the current CANN package with the extracted new bishengir, and clean up temporary files.
cp -r $TMP_PATH/bishengir/* ${OLD_CANN_PATH}/tools/bishengir/
rm -rf $TMP_PATH

# 5. Generally, the above replacement is sufficient. If problems persist, replace bisheng-compiler in the same way.
bash cann900/run_package/cann-bisheng-compiler_*.run --noexec --extract=$TMP_PATH
BiShengCompilerPath="${TMP_PATH}/bisheng_compiler" # For CANN 9.1.0 and later, the path is "tools/bisheng_compiler/".
cp -r $BiShengCompilerPath/* ${OLD_CANN_PATH}/tools/bisheng_compiler/
rm -rf $TMP_PATH
```

## Python Compatibility

AscendNPU IR also includes Python wheel packages, which can be installed via `pip`:

```bash
pip install ascendnpu-ir
```

After installation, you can directly call the compilation APIs in Python. The following is an example:

```python
import ascendnpuir

# MLIR string to be compiled
mlir_str = """
module {
  func.func @add(%arg0: memref<16xi16, #hivm.address_space<gm>>, %arg1: memref<16xi16, #hivm.address_space<gm>>, %arg2: memref<16xi16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xi16, #hivm.address_space<ub>>, memref<16xi16, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xi16, #hivm.address_space<gm>>)
    return
  }
}
"""
output_path = "example.o"
options = [
    "-enable-hivm-compile=true",
]

# Call the compilation API to generate the object file.
res = ascendnpuir.compile(
    mlir_str,
    output_path=output_path,
    option=options,
)
print(f"Compiled to: {output_path}")
```

The supported Python version range is as follows:

| AscendNPU IR Version | Python Version Support |
| --- | --- |
| `v1.0.0` | `>=3.9, <=3.12` |
| `v1.1.0` | `>=3.10, <=3.13` |
