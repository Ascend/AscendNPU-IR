from ._hfusion_ops_gen import *
from ._hfusion_ops_gen import _Dialect
from ..ir import *
from ._ods_common import (
    get_default_loc_context as _get_default_loc_context,
    _cext as _ods_cext,
    get_op_result_or_op_results as _get_op_result_or_op_results,
)
from typing import Any, List, Union
from bishengir.dialects import arith, func, builtin, hfusion, linalg, math
from bishengir.helper import *

class HACCHelper:
    @staticmethod
    def add_function_kind_attribute(func_op, kind):
        allowed_kinds = {'HOST', 'DEVICE'}
        if kind not in allowed_kinds:
            raise ValueError(f"Invalid function kind: {kind}")
        attr = Attribute.parse(f'#hacc.function_kind<{kind}>')
        func_op.attributes["hacc.function_kind"] = attr

# Usage example:
# HACCHelper.add_function_kind_attribute(host_elemwise_func, "HOST")