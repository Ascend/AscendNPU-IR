# Contribution Guide

This project welcomes developers to try it out and contribute. Before participating in community contributions, see [ascend-community](https://gitcode.com/ascend/community) to learn about the code of conduct, complete the [CLA signing](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1), and learn about the contribution process of the source repository.

## ✍️ How to Sign the Ascend Community Contributor License Agreement (CLA)

Before contributing to the community, you must sign the Ascend Community Contributor License Agreement (CLA):

- **Individual Contributor**: Select "Sign Individual CLA", fill in your name and email address, and then submit the application.
- **Enterprise**: Select "Sign Corporate CLA", fill in the relevant enterprise information, and submit the application. The enterprise will then receive the corporate CLA signing document from the community. Follow the prompts in the email to complete the subsequent signing.
- **Enterprise Employee**: Select "Corporate Contributor Registration". After signing, you will receive an email with the subject "Signing CLA on project of xx". Contact the Corporation Managers mentioned in the email for approval.

## Developer Contributions

Developer contribution scenarios mainly include:

- Bug Fixes

  If you find a bug in this project and want to fix it, you can create an Issue to report and track it.

  You can create a [Bug-Report](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Bug-Report|%E7%BC%BA%E9%99%B7%E5%8F%8D%E9%A6%88&template=.gitcode%252FISSUE_TEMPLATE%252F400-bug-report.yml) Issue to describe the bug, and then enter "/assign" or "/assign @yourself" in the comment box to claim this Issue and start working on it.

- Code Contributions

  You can create a [Requirement](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=%E6%96%B0%E9%9C%80%E6%B1%82&template=.gitcode%252FISSUE_TEMPLATE%252F500-feature-request.yml) Issue to describe a new sample operator and provide a design proposal, and then enter "/assign" or "/assign @yourself" in the comment box to claim this Issue and start working on it.

  When browsing other Issues, follow these requirements:
    - If you plan to resolve a corresponding issue, leave a comment under the issue stating that you will handle it.
    - If the issue has been open for a long time, perform a pre-check before resolving it.
    - For issues you submit and fix yourself, briefly describe the resolution before closing them.

- Consulation

  If you have questions while using this project, you can create an Issue to provide feedback and seek consultation.

  You can create a [Question](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Question|%E9%97%AE%E9%A2%98%E5%92%A8%E8%AF%A2&template=.gitcode%252FISSUE_TEMPLATE%252F300-usage.yml) type Issue to raise your questions.

- Help resolve others' issues

  If you have a feasible solution to a problem encountered by other community developers, you are welcome to comment on the Issue to discuss and help resolve the problem and pain point, jointly improving usability.

  If the corresponding Issue requires code changes, you can enter "/assign" or "/assign @yourself" in the Issue comment box to claim this Issue and follow up to help resolve it.

## Development Suggestions

### Code Style

This repository follows the coding conventions and programming style commonly used in the LLVM community. See [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html). The following tools can be used for code style checking:

- [Clang-Tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy)
- [CppLint](https://github.com/cpplint/cpplint)
- [Cppcheck](http://cppcheck.sourceforge.net/)
- [CMakeLint](https://github.com/cmake-lint/cmake-lint)

### Submitting a PR

- Propose an idea and create an issue on [GitCode](https://gitcode.com/Ascend/AscendNPU-IR).
- If a new feature involves substantial design details, submit a design proposal as well.
- After reaching a consensus through issue discussion and design proposal review, proceed with Fork development and submit a PR.
- After thorough discussion, the PR will be merged, rejected, or closed based on the discussion outcome.
- Merging a PR requires comments of `/lgtm` (Looks Good To Me) from 2 Reviewers and `/approve` from 1 Approver. The PR submitter cannot perform the merge operation.

### Fork-Pull Development Mode

Before submitting code to the AscendNPU IR project, first fork the project to your personal repository. Subsequently, develop in the forked repository and merge changes into this project through a pull request.

### Self-Testing Code Changes

After completing the code changes, compile and run tests in the **build directory** to verify the functionality:

```bash
ninja check-bishengir
```

### Code Push Verification

After the code is updated and passes the tests, push the commit to your personal remote repository.

### Creating a Pull Request to the Main Repository

After the code is pushed to the remote repository, create a Pull Request between the new branch and the master branch of AscendNPU IR. After the merge request is created, commenting `compile` in the PR triggers the CI build pipeline. It is recommended to merge the PR into the upstream master as soon as possible to reduce the risk of merge conflicts.

### Gate Exception Handling

Code gate exceptions mainly fall into the following categories. Follow the CI prompts to check and fix them one by one.

- **Compilation failure**: Follow the prompts to identify the cause of the compilation failure, fix it, and recompile.
- **Static check failure**: Follow the prompts to locate and fix the static check issues in the code.
- **CI pipeline failure**: Follow the prompts to locate and fix the failed test cases, and then retrigger CI.

## Notes

- Avoid introducing changes unrelated to this modification in the PR.
- Keep the commit history concise and well-organized (squash/rebase can be used as appropriate).
- Before creating a PR, rebase the local branch onto the latest master of the upstream repository.
- For bug-fix PRs, link all related issues and PRs in the description.
