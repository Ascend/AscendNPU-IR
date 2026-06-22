#!/bin/bash
# This script is used to apply or clean-up patches to submodules.
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# All script-level variables are defined here (invoke with "$@" after functions are defined).
init_variables() {
  GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
  if [ -z "$GIT_ROOT" ]; then
    echo "Error: We're not in the AscendNPU IR repo" >&2
    exit 1
  fi

  THIRD_PARTY_FOLDER="$GIT_ROOT/third-party"
  readonly PATCH_FOLDER="$GIT_ROOT/build-tools/patches"
  OPTION="${1:-}"

  # will be inside third-party folder with basename as the submodule folder name
  readonly SUBMODULES=(
    "torch-mlir"
    "llvm-project"
  )

  readonly SUBMODULE_INCLUDES=(
    ""
    "--include=mlir/* --include=llvm/*"
  )

  export PATCH_CMD
}

usage_and_exit() {
  echo "Usage:"
  echo "  apply_patches.sh [--clean-up]"
  echo "     --clean-up  Reset the applied patches. This will also reset changes you made to the submodule."
  exit 1
}

apply_patches() {
  local i
  local folder_name

  echo "Applying patches to submodules"
  cd "$THIRD_PARTY_FOLDER" || exit 1
  for (( i=0; i<${#SUBMODULES[@]}; i++ )); do
    folder_name=${SUBMODULES[i]}
    cd "$THIRD_PARTY_FOLDER/$folder_name" || exit 1
    echo "Applying patches for $folder_name"
    for patch_file in "$PATCH_FOLDER/$folder_name"/*.patch; do
      ${PATCH_CMD:-patch} -p1 --merge < "$patch_file"
    done
  done
  echo "Finished applying patches"
  cd "$GIT_ROOT" || exit 1
}

clean_up_patches() {
  local i
  local folder_name

  for (( i=0; i<${#SUBMODULES[@]}; i++ )); do
    folder_name=${SUBMODULES[i]}
    echo "Resetting" "${folder_name}" "submodule patches."
    echo "[Warning] This will also reset changes you made to the submodule!"
    cd "$THIRD_PARTY_FOLDER/$folder_name" || exit 1
    git reset --hard HEAD
    git clean -df
  done
}

init_variables "$@"

if [[ -n "$OPTION" ]]; then
  if [[ "$OPTION" == "--clean-up" ]]; then
    clean_up_patches
    exit 0
  else
    echo "Unknown flag: $OPTION"
    echo
    usage_and_exit
  fi
fi

clean_up_patches
apply_patches
