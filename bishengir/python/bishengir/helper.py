# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

class BiShengIRHelper:
    print_debug = False

    @staticmethod
    def change_print_debug(print_debug=False):
        BiShengIRHelper.print_debug = print_debug

    @staticmethod
    def print(*args, **kwargs):
        """Wrapper function to print to stdout and optionally to stderr."""
        print(*args, file=sys.stdout, flush=True, **kwargs)
        if BiShengIRHelper.print_debug:
            print(*args, file=sys.stderr, flush=True, **kwargs)

    @staticmethod
    def print_attrs(op):
        """Print all attributes of an operation."""
        BiShengIRHelper.print("\n\n\n------- Printing attributes ---------")
        BiShengIRHelper.print(f"Attributes of {op.name}\n")
        if op.attributes:
            tmp = list(op.attributes)
            BiShengIRHelper.print("Entire Attributes: ", tmp)
            for i, att in enumerate(tmp):
                BiShengIRHelper.print(f' --> Attr {i}: {att.name} | {att.attr} - ')
        else:
            BiShengIRHelper.print(f"Warning: No attributes")

        BiShengIRHelper.print("\n\n\n----------------")
