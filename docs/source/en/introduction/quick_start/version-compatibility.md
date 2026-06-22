# Version Compatibility

This document describes the CANN software stack and hardware driver environment dependencies for AscendNPU-IR.

To ensure compilation and runtime stability, please strictly follow the version compatibility relationships provided in this document for environment configuration.

## CANN Compatibility

The following version combinations are verified recommended configurations and should be prioritized:

| AscendNPU-IR Version | Gitcode Branch | Required CANN Version | Hardware Support |
| --- | --- | --- | --- |
| v1.0.0 | release/v1.0.0 | CANN 8.5.0 | Ascend A2/A3 |
| v1.1.0 | release/v1.1.0 | CANN 9.0.0 | Ascend A2/A3;<br> 950 Series (branch feature_a5) |

**Important Note:**
If you do not wish to upgrade your CANN version but want to use the new features of AscendNPU IR, you can try the following approach:

```bash
NEW_CANN_PKG="PATH-TO/Ascend-cann-toolkit_9.0.0_linux-aarch64.run"
TMP_PATH="tmp_pkg_files"
OLD_CANN_PATH="${CANN_850_PATH}/Ascend/cann-8.5.0"

bash ${NEW_CANN_PKG} --noexec --extract=cann900
bash cann900/run_package/ascendnpu-ir_*.run --noexec --extract=$TMP_PATH
cp -r $TMP_PATH/bishengir/* ${OLD_CANN_PATH}/tools/bishengir/
rm -rf $TMP_PATH

# Generally, the above steps should work. If there are still issues, try the following fallback method:
bash cann900/run_package/cann-bisheng-compiler_*.run --noexec --extract=$TMP_PATH
BiShengCompilerPath="${TMP_PATH}/bisheng_compiler" # For CANN 9.1.0+, path is "tools/bisheng_compiler/"
cp -r $BiShengCompilerPath/* ${OLD_CANN_PATH}/tools/bisheng_compiler/
rm -rf $TMP_PATH
```

## Python Compatibility

AscendNPU-IR also includes Python wheel packages that can be installed via pip:

```bash
pip install ascendnpu-ir
```

Supported Python versions are as follows:

| AscendNPU-IR Version | Python Version Support |
| --- | --- |
| v1.0.0 | >=3.9, <=3.12 |
| v1.1.0 | >=3.10, <=3.13 |
