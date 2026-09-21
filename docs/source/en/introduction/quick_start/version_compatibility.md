# Version Compatibility

This document describes the CANN software stack and hardware driver environment on which AscendNPU IR depends.

To ensure compilation and runtime stability, configure the environment strictly according to the compatibility relationship provided in this document.

## CANN Compatibility

The following version combinations are verified recommended configurations and should be used preferentially:

| AscendNPU IR Version | Gitcode Branch | CANN Version Dependency | Hardware Support |
| --- | --- | --- | --- |
| `v1.2.0` | `release/v1.2.x` | CANN 9.1.0 | <ul><li>Ascend 950PR/Ascend 950DT(branch `feature/regbase`)</li><li>Atlas A3 training products/Atlas A3 inference products</li><li>Atlas A2 training products/Atlas A2 inference products</li></ul> |
| `v1.1.0` | `release/v1.1.x` | CANN 9.0.0 | <ul><li>Ascend 950PR/Ascend 950DT(branch `feature_a5`)</li><li>Atlas A3 training products/Atlas A3 inference products</li><li>Atlas A2 training products/Atlas A2 inference products</li></ul> |
| `v1.0.0` | `release/v1.0.0` | CANN 8.5.0 | <ul><li>Atlas A3 training products/Atlas A3 inference products</li><li>Atlas A2 training products/Atlas A2 inference products</li></ul> |

**Important Notes**:

If you want to keep the current CANN version without upgrading or replacing it while still using the new features of AscendNPU IR, you can extract the AscendNPU IR component from the new CANN toolkit package and use it to replace the NPUIR version in your current CANN package. The steps are as follows:

```bash
# This example replaces the NPUIR in a CANN 8.5.0 installation with the version from the CANN 9.0.0 package.
# 1. Set variables: path to the new CANN package (fill in <arch> with the actual architecture, e.g. x86_64), a temporary extraction directory, and the current CANN installation path.
NEW_CANN_PKG="PATH-TO/Ascend-cann-toolkit_9.0.0_linux-<arch>.run"
TMP_PATH="tmp_pkg_files"
OLD_CANN_PATH="${CANN_850_PATH}/Ascend/cann-8.5.0"

# 2. Extract the new CANN toolkit package (--noexec means extract only, without installing).
bash ${NEW_CANN_PKG} --noexec --extract=cann900

# 3. Extract the ascendnpu-ir sub-package from the extracted directory.
bash cann900/run_package/ascendnpu-ir_*.run --noexec --extract=$TMP_PATH

# 4. Replace the NPUIR in the current CANN package with the extracted new bishengir, then clean up the temporary files.
cp -r $TMP_PATH/bishengir/* ${OLD_CANN_PATH}/tools/bishengir/
rm -rf $TMP_PATH

# 5. Generally, the replacement above is sufficient. If issues persist, replace bisheng-compiler in the same way.
bash cann900/run_package/cann-bisheng-compiler_*.run --noexec --extract=$TMP_PATH
BiShengCompilerPath="${TMP_PATH}/bisheng_compiler" # For CANN 9.1.0 and later, the path is "tools/bisheng_compiler/"
cp -r $BiShengCompilerPath/* ${OLD_CANN_PATH}/tools/bisheng_compiler/
rm -rf $TMP_PATH
```

## Python Compatibility

AscendNPU IR also includes Python wheel packages, which can be installed via `pip`:

```bash
pip install ascendnpu-ir
```

The supported Python version range is as follows:

| AscendNPU IR Version | Python Version Support |
| --- | --- |
| `v1.0.0` | `>=3.9, <=3.12` |
| `v1.1.0` | `>=3.10, <=3.13` |
