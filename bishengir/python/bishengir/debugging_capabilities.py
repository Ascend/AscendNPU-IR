# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
 
import torch
 
 
class IntermediateTensor:
    @staticmethod
    def __get_instruction(id: int):
        match id:
            case 1:
                return "vmul"
            case 2:
                return "vadd"
            case _:
                return "UNKNOWN"
        pass
 
    @staticmethod
    def __get_type(id: int):
        match id:
            case 1:
                return "f16"
            case 2:
                return "f32"
            case _:
                return "UNKNOWN"
        pass
 
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
 
    def __str__(self):
        result: str = "{\n"
 
        shape = self.tensor.shape
        result += f"shape={shape},\n"
 
        numel = shape.numel()
        i = 0
        while (i + 8) < numel:
            size = int(self.tensor[i])
            if (size == 0):
                break
            typeId = self.tensor[i + 1]
            instructionId = self.tensor[i + 2]
            i += 8
 
            result += f"instruction={IntermediateTensor.__get_instruction(instructionId)}\n"
            result += f"data (size={size}, type={IntermediateTensor.__get_type(typeId)})=" + "["
 
            index = 0
            for j in range(i, i + size):
                value = self.tensor[j]
                result = result + f"{value},"
                index += 1
            result += "]\n\n"
 
            i += size
        result = result + "}"
        return result


# How to use
# size = (64 * 8,)
# size2 = (64 * 8 * 16,)

# x = torch.full(size, 2.0, device=DEVICE)
# y = torch.full(size, 3.0, device=DEVICE)
# z = torch.full(size, 4.0, device=DEVICE)
# # UB probe tensor result
# prob = torch.full(size2, 0.0, device=DEVICE)

# output_torch = x * y + z
# # kernel execution
# output_triton = mul_add(x, y, z, prob)
# torch.npu.synchronize()

# reader = IntermediateReader(prob)
# print(f"prob={reader}")
