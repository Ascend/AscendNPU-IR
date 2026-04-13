# Build and Installation

This document describes dependency installation, build methods (source and binary), and running tests for AscendNPU IR.

## Install Dependencies

### Build Dependencies

#### Compiler and Toolchain Requirements

Minimum requirements:

- CMake >= 3.28
- Ninja >= 1.12.0

Recommended:

- Clang >= 10
- LLD >= 10 (LLVM LLD significantly speeds up builds)

#### Source Preparation

1. Clone the repository (then enter the repo directory, typically `ascendnpu-ir`).

    ```bash
    git clone https://gitcode.com/Ascend/ascendnpu-ir.git
    cd ascendnpu-ir
    ```

2. Initialize and update submodules.

    The project depends on LLVM, Torch-MLIR, and other third-party libraries; submodules must be fetched and updated to the specified commit IDs.

    ```bash
    # Recursively init and update all submodules
    git submodule update --init --recursive
    ```

### Runtime Dependencies

#### CANN Package Installation

End-to-end runs depend on the CANN environment.

1. Download CANN: obtain the toolkit package and the ops package for your hardware from the [Ascend CANN download page](https://www.hiascend.com/cann/download).

2. Install CANN:

    ```bash
    # Example: x86 A3, {version} is CANN version, e.g. 9.0.0
    chmod +x Ascend-cann_{version}_linux-x86_64.run
    chmod +x Ascend-cann-A3-ops_{version}_linux-x86_64.run
    ./Ascend-cann_{version}_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
    ./Ascend-cann-A3-ops_{version}_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
    ```

3. Set environment variables:

    ```bash
    # For version 8.5.0 and earlier, use ${PATH-TO-CANN}/ascend-toolkit/set_env.sh
    source ${PATH-TO-CANN}/cann/set_env.sh
    ```

## Build Commands

### Source Build

#### Using the Build Script (Recommended)

Run `./build-tools/build.sh` from the repo root to configure, build, and install. The script handles CMake configuration, Ninja build, and installation.

**First build** (requires submodule init and patch application):

```bash
# From the repo root
./build-tools/build.sh -o ./build --build-type Release --apply-patches
```

**Subsequent builds** (when build directory already exists):

```bash
./build-tools/build.sh -o ./build --build-type Release
```

**Rebuild** (clean build directory and reconfigure):

```bash
./build-tools/build.sh -o ./build --build-type Release -r
```

#### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--build PATH` | Build output directory | `./build` |
| `--build-type TYPE` | Build type | `Release` |
| `--apply-patches` | Apply patches to third-party submodules; **required on first build** | Off |
| `-r`, `--rebuild` | Clean build directory and reconfigure | Off |
| `-j`, `--jobs N` | Parallel build threads | 3/4 of CPU cores |
| `--install-prefix PATH` | Install path | `BUILD_DIR/install` |
| `--c-compiler PATH` | C compiler path | `clang` |
| `--cxx-compiler PATH` | C++ compiler path | `clang++` |
| `--llvm-source-dir DIR` | LLVM source directory | `third-party/llvm-project` |
| `--build-test` | Build and run tests | Off |
| `--build-bishengir-doc` | Build BiShengIR documentation | Off |
| `-t`, `--build-bishengir-template` | Build BiShengIR template; **requires CANN 9.0.0 to enable this option, required for E2E cases** | Off |
| `--bisheng-compiler PATH` | Path to bisheng compiler; **required when building template** | None |
| `--build-torch-mlir` | Also build torch-mlir | Off |
| `--python-binding` | Enable MLIR python-binding | Off |
| `--enable-cpu-runner` | Enable CPU runner | Off |
| `--disable-ccache` | Disable ccache | On (if installed) |
| `--enable-assertion` | Enable assertions | Off |
| `--fast-build` | Skip installation step | Off |
| `--add-cmake-options OPTIONS` | Append CMake options | None |

#### Common Examples

```bash
# Debug build with tests
./build-tools/build.sh -o ./build --build-type Debug --apply-patches --build-test

# Specify compiler and thread count
./build-tools/build.sh -o ./build --c-compiler /usr/bin/clang-15 --cxx-compiler /usr/bin/clang++-15 -j 256

# Fast build (skip install)
./build-tools/build.sh -o ./build --fast-build

# Rebuild and build template (E2E cases require the template library)
./build-tools/build.sh -r -o ./build --fast-build -t --bisheng-compiler=/usr/Ascend/cann/bin
```

#### Manual Build (Advanced)

For full manual control, follow these steps.

**Prerequisites**: submodule init (`git submodule update --init --recursive`) and patch application (`source build-tools/apply_patches.sh`).

```bash
# From the repo root
mkdir -p build
cd build

# LLVM source path: third-party/llvm-project/llvm
export LLVM_SOURCE_DIR="$(realpath ../third-party/llvm-project)"
cmake ${LLVM_SOURCE_DIR}/llvm -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$(realpath ..)" \
    -DBSPUB_DAVINCI_BISHENGIR=ON \
    # [-DCMAKE_INSTALL_PREFIX="${PWD}/install"] \
    # [-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON] \
    # [-DLLVM_ENABLE_ASSERTIONS=ON] \
    # [-DMLIR_ENABLE_BINDINGS_PYTHON=ON] \
    # [-DLLVM_TARGETS_TO_BUILD="host;Native"] \
    # [-DBISHENGIR_PUBLISH=OFF] \
    # [-DBISHENGIR_BUILD_TEMPLATE=ON -DBISHENG_COMPILER_PATH=/path/to/bisheng-compiler] \
    # [-Dother-options=value]

ninja -j32
```

**Note**: `[]` indicates optional options. To use an option, remove the leading `#` and `[]`, and add the argument to the command.

| Optional Option | Description |
|-----------------|-------------|
| `-DCMAKE_INSTALL_PREFIX="${PWD}/install"` | Install path |
| `-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON` | Required when LLVM >= 21 |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | Enable assertions (common for Debug) |
| `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` | Enable MLIR python-binding |
| `-DLLVM_TARGETS_TO_BUILD="host;Native"` | Enable CPU runner |
| `-DBISHENGIR_PUBLISH=OFF` | Disable unpublished features |
| `-DBISHENGIR_BUILD_TEMPLATE=ON -DBISHENG_COMPILER_PATH=...` | Build BiShengIR template |

### Binary Installation

AscendNPU IR binaries are installed with the CANN toolkit; see [CANN Package Installation](#cann-package-installation) above.

## Running Tests

### Build Test Target

```bash
# From the build directory
cmake --build . --target "check-bishengir"
```

This runs `check-mlir` and `check-bishengir`, executing the BiShengIR test suite via llvm-lit.

#### Typical Output

**When tests pass**:

```text
-- Testing: 388 tests, 8 workers --
...

Testing Time: 45.23s

Total Discovered Tests: 388
  Unsupported: 89  (22.94%)
  Passed     : 299 (77.06)
```

**When tests fail**:

```text
-- Testing: 388 tests, 8 workers --
PASS: bishengir :: bishengir-compile/commandline.mlir (1 of 388)
...
FAIL: bishengir :: test/failing-case.mlir (42 of 388)
******************** TEST 'FAIL: bishengir :: test/failing-case1.mlir' FAILED ********************
...（failure details）...

********************
FAIL: bishengir :: test/failing-case.mlir (256 of 388)
******************** TEST 'FAIL: bishengir :: test/failing-case2.mlir' FAILED ********************
...（failure details）...

********************
********************
Failed Tests (2):
  bishengir :: test/failing-case1.mlir
  bishengir :: test/failing-case2.mlir

Testing Time: 38.12s

Total Discovered Tests: 388
  Unsupported:  86 (22.16%)
  Passed     : 300 (77.32%)
  Failed     :   2 (0.52%)
  ...
```

#### Test Pass Criteria

**Pass**: Exit code 0 and `Failed` is 0. The following count as pass:

- **PASS**: Test passed
- **UNSUPPORTED**: Not supported in current environment (e.g. `UNSUPPORTED: bishengir_published`)
- **XFAIL**: Expected to fail and did fail

**Fail**: Non-zero exit code or `Failed` > 0. The following count as fail:

- **FAIL**: Test failed
- **XPASS**: Expected to fail but passed
- **UNRESOLVED**: Result could not be determined
- **TIMEOUT**: Timed out

### Run Tests with LLVM LIT

```bash
# From the build directory
./bin/llvm-lit ../bishengir/test
```

You can specify a test path, e.g. `./bin/llvm-lit ../bishengir/test/bishengir-compile/commandline.mlir`.

## FAQ

**Q:** When building with `build-tools/build.sh`, I get "ninja: error: loading 'build.ninja': No such file or directory". What should I do?
**A:** Add the `-r` option to rerun CMake and regenerate `build.ninja`.

**Q:** Build fails with "Too many open files". What should I do?
**A:** The number of open files exceeded the system limit. Raise the per-process open-file limit, e.g. `ulimit -n 65535`.

**Q:** Build fails with:

```bash
 The CMAKE_CXX_COMPILER:

  clang++

 is not a full path and was not found in the PATH.
```

What should I do?
**A:** The C++ compiler was not specified or the compiler binary is invalid. First try specifying the compiler with `--cxx-compiler=${CXX-COMPILER-PATH}`. If the error persists after specifying the compiler, reinstall or use another version (e.g. the recommended clang++-15).
