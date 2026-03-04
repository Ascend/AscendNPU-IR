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

def create_hfusion_attr(attr_type, value):
    return Attribute.parse(f"#hfusion.{attr_type}<{value}>")

@_ods_cext.register_operation(_Dialect, replace=True)
class ElemwiseUnaryOp(ElemwiseUnaryOp):
    @property
    def type(self):
        return self.results[0].type

    @classmethod
    def create(cls, result_type, input_tensor, output_tensor, fun, cast=None, loc=None, ip=None):
        op = cls(
            result_tensors=[result_type],
            inputs=[input_tensor],
            outputs=[output_tensor],
            fun=create_hfusion_attr("unary_fn", fun),
            cast=cast,
            loc=loc,
            ip=ip
        )
        cls.add_region(op, result_type.element_type, fun)
        return op

    @classmethod
    def add_region(cls, op, element_type, fun):
        block = Block.create_at_start(op.operation.regions[0], [element_type, element_type])
        
        with InsertionPoint(block):
            arg0, arg1 = block.arguments
            result = cls.create_unary_op(fun, arg0)
            linalg.YieldOp([result])

    @staticmethod
    def create_unary_op(fun, arg):
        if fun == "sqrt":
            return math.SqrtOp(arg)
        elif fun == "rsqrt":
            return math.RsqrtOp(arg)
        elif fun == "rec":
            cst = arith.ConstantOp(arg.type, 1.0)
            return arith.DivFOp(cst, arg)
        elif fun == "relu":
            cst = arith.ConstantOp(arg.type, 0.0)
            return arith.MaximumFOp(cst, arg)
        elif fun == "vnot":
            cst = arith.ConstantOp(arg.type, 0)
            return arith.XOrIOp(cst, arg)
        else:
            raise ValueError(f"Unsupported unary function: {fun}")

@_ods_cext.register_operation(_Dialect, replace=True)
class ElemwiseBinaryOp(ElemwiseBinaryOp):
    @property
    def type(self):
        return self.results[0].type

    @classmethod
    def create(cls, result_type, input_tensor1, input_tensor2, output_tensor, fun, cast=None, loc=None, ip=None):
        op = cls(
            result_tensors=[result_type],
            inputs=[input_tensor1, input_tensor2],
            outputs=[output_tensor],
            fun=create_hfusion_attr("binary_fn", fun),
            cast=cast,
            loc=loc,
            ip=ip
        )
        cls.add_region(op, result_type.element_type, fun)
        return op

    @classmethod
    def add_region(cls, op, element_type, fun):
        block = Block.create_at_start(op.operation.regions[0], [element_type, element_type, element_type])
        
        with InsertionPoint(block):
            arg0, arg1, arg2 = block.arguments
            result = cls.create_binary_op(fun, arg0, arg1)
            linalg.YieldOp([result])

    @staticmethod
    def create_binary_op(fun, arg0, arg1):
        if fun == "vand":
            return arith.AndIOp(arg0, arg1)
        elif fun == "vor":
            return arith.OrIOp(arg0, arg1)
        else:
            raise ValueError(f"Unsupported binary function: {fun}")