# Coverage remove patterns
# Each entry is passed to `lcov --remove` as a shell-style path pattern.
BLACKLIST = [
    "*/build*/*",
    "*/third-party/llvm-project/*",
    "*/third-party/triton/*",
]
