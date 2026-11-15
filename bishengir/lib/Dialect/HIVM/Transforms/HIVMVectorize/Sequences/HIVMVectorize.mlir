//===- HIVMVectorize.mlir - HIVM vectorization entry ----------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

transform.named_sequence @hivm_vectorize(
    %root:      !transform.any_op,
    %options:   !transform.any_param  {transform.readonly}
) {
  %ticket = transform.hivm.transaction @hivm_vectorize_attempt(%root, %options)
      : (!transform.any_op, !transform.any_param) -> !transform.any_param
  %result = transform.hivm.transaction.await %ticket
      : (!transform.any_param) -> !transform.any_param
  %newRoot, %selection = transform.hivm.transaction.select
      @hivm_vectorize_cost(%root, %result)
      : (!transform.any_op, !transform.any_param)
        -> (!transform.any_op, !transform.any_param)

  %vfs = transform.structured.match ops{["func.func"]}
      attributes {hivm.vector_function} in %newRoot
      : (!transform.any_op) -> !transform.any_op
  %annotated = transform.hivm.annotate_vf %vfs
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}

transform.named_sequence @hivm_vectorize_attempt(
    %root:      !transform.any_op,
    %options:   !transform.any_param  {transform.readonly}
) -> !transform.any_param {
  %groups = transform.hivm.partition %root, %options
      {models = ["SameBlock", "UnseparatedByBarrier"]}
      : (!transform.any_op, !transform.any_param) -> !transform.any_param
  %plan = transform.hivm.agglomerate %root, %options, %groups
      {selector = "preserve_groups", cost = ["uniform_cost"]}
      : (!transform.any_op, !transform.any_param, !transform.any_param)
        -> !transform.any_param
  %selected = transform.hivm.extract_groups %plan
      : (!transform.any_param) -> !transform.any_param
  %newRoot, %funcs = transform.hivm.outline %root, %selected
      : (!transform.any_op, !transform.any_param)
        -> (!transform.any_op, !transform.any_op)
  %annotated = transform.hivm.annotate_vf %funcs
      : (!transform.any_op) -> !transform.any_op
  %lifted = transform.hivm.lift_to_linalg %annotated
      : (!transform.any_op) -> !transform.any_op
  // Scalarize reductions that have no native vector lowering before the
  // vectorizer runs. The lifter marks those linalg.reduce ops explicitly.
  %unrolled = transform.apply_registered_pass "hfusion-generic-unroller"
      to %lifted {options = "only-scalarize-tagged=true"}
      : (!transform.any_op) -> !transform.any_op
  %vectorized = transform.hivm.vectorize %unrolled {vector_length = 256}
      : (!transform.any_op) -> !transform.any_op
  %cleaned = transform.hivm.cleanup %vectorized
      : (!transform.any_op) -> !transform.any_op
  %cost = transform.hivm.check_result %cleaned
      : (!transform.any_op) -> !transform.any_param
  transform.yield %cost : !transform.any_param
}

transform.named_sequence @hivm_vectorize_cost(
    %root:      !transform.any_op     {transform.readonly},
    %result:    !transform.any_param  {transform.readonly}
) -> !transform.any_param {
  transform.yield %result : !transform.any_param
}
