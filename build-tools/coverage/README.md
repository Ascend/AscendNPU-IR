# Coverage Runner

`build-tools/coverage/run.py` runs coverage-enabled tests and generates `lcov` / `genhtml` reports.

## Requirements

- Build the project with `./build-tools/build.sh --coverage`.
- Install `lcov`, `genhtml`, and `llvm-cov`.
- On Ubuntu, `genhtml` is installed together with the `lcov` package.
- The runner uses `llvm-cov` from `PATH`. LLVM 18 is recommended; other versions will print a warning.
- `lit` mode requires `<build-dir>/bin/llvm-lit`.
- `gtest --build-target ...` requires `ninja`.

Install the report tools on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y lcov llvm-18
```

Check the environment before running coverage:

```bash
which lcov
which genhtml
which llvm-cov
llvm-cov --version
```

If `/usr/lib/llvm-18` does not exist, install the LLVM 18 package first:

```bash
sudo apt-get install -y llvm-18
```

If LLVM 18 is installed but `which llvm-cov` still prints nothing, add the LLVM 18 tool directory to `PATH`:

```bash
export PATH=/usr/lib/llvm-18/bin:$PATH
which llvm-cov
```

## Build

```bash
rm -rf ./build-cov && mkdir build-cov
./build-tools/build.sh --coverage \
  --build-triton \
  --disable-cann \
  --enable-lld \
  --disable-mlir-werror \
  --disable-bishengir-werror \
  --disable-werror \
  --enable-assertion \
  --build-type="Release" \
  --build=./build-cov
```

## Run Lit Coverage

Run all `bishengir/test` cases with 4 workers:

```bash
python3 build-tools/coverage/run.py lit \
  --build-dir ./build-cov \
  --lit-test-root ./bishengir/test \
  --jobs 4 \
  --out-dir ./build-cov/coverage-bishengir-full
```

Run one lit case:

```bash
python3 build-tools/coverage/run.py lit \
  --build-dir ./build-cov \
  --lit-test-root ./bishengir/test \
  --test bishengir/test/bishengir-opt/commandline.mlir \
  --jobs 1 \
  --out-dir ./build-cov/coverage-one-lit
```

The runner deletes old `.gcda` files under `--build-dir` before each run. If lit fails after generating `.gcda` files, the runner still generates the coverage report and then returns non-zero.

## Run GTest Coverage

Run one gtest binary:

```bash
python3 build-tools/coverage/run.py gtest \
  --build-dir ./build-cov \
  --binary tools/bishengir/unittests/Dialect/Utils/BiShengIRDialectUtilsTests \
  --out-dir ./build-cov/coverage-gtest
```

Build the gtest target first and run one test filter:

```bash
python3 build-tools/coverage/run.py gtest \
  --build-dir ./build-cov \
  --build-target BiShengIRDialectUtilsTests \
  --binary tools/bishengir/unittests/Dialect/Utils/BiShengIRDialectUtilsTests \
  --gtest-filter 'SelectRoundModeTest.*' \
  --out-dir ./build-cov/coverage-gtest
```

## Outputs

Each run writes these files under `--out-dir`:

- `coverage.info`: raw lcov capture result
- `coverage.filtered.info`: result after extracting `*/bishengir/*`
- `coverage.cleaned.info`: result after removing paths listed in `build-tools/coverage/blacklist.py`
- `html/index.html`: final HTML report

Open the HTML entry directly, for example:

```bash
xdg-open ./build-cov/coverage-bishengir-full/html/index.html
```

## Useful Options

- `--out-dir`: output directory. Default is `<build-dir>/coverage`.
- `--jobs`: lit worker count and gtest pre-build parallelism.
- `--test`: lit test file or directory. Can be passed multiple times.
