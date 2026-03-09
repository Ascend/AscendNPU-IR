# Contributing Guide

We welcome developers to try and contribute to this project. Before contributing, please read the [ascend-community](https://gitcode.com/ascend/community) code of conduct, complete [CLA signing](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1), and understand the contribution workflow for the source repository.

## ✍️ Signing the Ascend Community Contributor License Agreement (CLA)

You must sign the Ascend Community Contributor License Agreement (CLA) before contributing:

- **Individual contributors**: Choose "Sign Individual CLA".
- **Corporations**: Choose "Sign Entity CLA".
- **Corporate employees**: Choose "Entity Contributor Registration"; after signing you will receive an email with subject "Signing CLA on project of xx". Contact the Corporation Managers mentioned in the email for approval.

## Developer contributions

Common contribution scenarios include:

- **Bug fixes**

  If you find a bug and want to fix it, please open an Issue to report and track it.

  You can create a [Bug Report](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Bug-Report|%E7%BC%BA%E9%99%B7%E5%8F%8D%E9%A6%88&template=.gitcode%252FISSUE_TEMPLATE%252Fbug-report.yml) Issue to describe the bug, then comment "/assign" or "/assign @yourself" to assign the Issue to yourself.

- **Code contributions**

  You can create a [Requirement | Feature request](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=%E6%96%B0%E9%9C%80%E6%B1%82&template=.gitcode%252FISSUE_TEMPLATE%252Ffeature.yml) Issue to describe a new sample operator and provide your design. Then comment "/assign" or "/assign @yourself" to assign the Issue to yourself.

  For other Issues you browse:
  - If you intend to work on an Issue, leave a comment stating that you will take it.
  - If the Issue has been open for a long time, do a quick check before implementing.
  - If you resolve an Issue you reported, briefly summarize the outcome before closing it.

- **Questions**

  If you have questions about using this project or other topics, please open an Issue for discussion.

  You can follow the [Question](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Question|%E9%97%AE%E9%A2%98%E5%92%A8%E8%AF%A2&template=.gitcode%252FISSUE_TEMPLATE%252Fquestion.yml) template when creating your Issue.

- **Helping with others’ Issues**

  If you have a solution for someone else’s Issue, please share it in the comments to help the community.

  If the Issue requires code changes, you can comment "/assign" or "/assign @yourself" to assign the Issue to yourself and follow up with a fix.

## Development guidelines

### Code style

This repository follows the LLVM community coding standards and style. See the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html). You can use the following tools for style checks:

- [clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy)
- [CppLint](https://github.com/cpplint/cpplint)
- [CppCheck](http://cppcheck.sourceforge.net/)
- [CMakeLint](https://github.com/cmake-lint/cmake-lint)

### Submitting a PR

- Open an Issue on [GitCode](https://gitcode.com/Ascend/AscendNPU-IR) to propose your idea.
- If the change involves non-trivial design, submit a design document as well.
- After the Issue and design (if any) are agreed upon, fork the repo, implement, and open a Pull Request.
- The PR will be merged, rejected, or closed based on the discussion.
- Merging requires 2 Reviewers to comment `/lgtm` (Looks Good To Me) and 1 Approver to comment `/approve`. The PR author cannot merge their own PR.

### Fork–Pull workflow

Before submitting code to AscendNPU-IR, fork the project to your own repository. Develop in your fork and submit a Pull Request to merge your changes into this project.

### Self-testing your changes

After making code changes, build and run tests in the **build directory** to verify:

```bash
ninja check-bishengir
```

### Pushing and CI

After your changes pass local tests, push your commits to your remote fork.

### Creating a Pull Request to the upstream repo

After pushing to your fork, create a Pull Request from your branch to the AscendNPU-IR `master` branch. After creating the PR, you can comment `compile` in the PR to trigger the CI pipeline. We recommend merging your PR into upstream `master` as soon as it is approved to reduce merge conflicts.

### Dealing with CI failures

Common CI failure types and how to address them:

- **Build failure**: Check the CI log for the cause, fix the issue, and ensure the project builds locally before pushing again.
- **Static analysis failure**: Locate and fix the issues reported by the static checker.
- **CI Pipeline / test failure**: Identify the failing tests from the CI output, fix them, and re-trigger CI.

## Notes

- Avoid including changes unrelated to the PR in your commits.
- Keep commit history clear and logical (squash or rebase when appropriate).
- Rebase your branch onto the latest upstream `master` before opening a PR.
- For bug-fix PRs, reference all related Issues and PRs in the PR description.
