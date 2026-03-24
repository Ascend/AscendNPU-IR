# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请参见 [ascend-community](https://gitcode.com/ascend/community) 了解行为准则，完成 [CLA 协议签署](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1)，并了解源码仓的贡献流程。

## ✍️ 如何签署 Ascend 社区贡献者许可协议（CLA）

在参与社区贡献前，您需要签署 Ascend 社区贡献者许可协议（CLA）：

- **个人贡献者**：请选择「签署个人 CLA」，填写好姓名和邮箱后提交申请即可。
- **企业**：请选择「签署法人 CLA」，填写企业相关信息，提交申请后，企业将接收到社区发出的签署企业 CLA 的文件，请按照邮件中的提示完成后续签署。
- **企业员工**：请选择「法人贡献者登记」；签署后会收到主题为 “Signing CLA on project of xx” 的邮件，请联系邮件中的 Corporation Managers 进行审批。

## 开发者贡献

开发者贡献场景主要包括：

- Bug修复

  如果您在本项目中发现了某些Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

  您可以新建 [Bug-Report|缺陷反馈](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Bug-Report|%E7%BC%BA%E9%99%B7%E5%8F%8D%E9%A6%88&template=.gitcode%252FISSUE_TEMPLATE%252Fbug-report.yml) 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

- 贡献代码

  您可以新建 [Requirement|需求建议](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=%E6%96%B0%E9%9C%80%E6%B1%82&template=.gitcode%252FISSUE_TEMPLATE%252Ffeature.yml) 类Issue对新的样例算子予以说明，并提供您的设计方案，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

  对于您浏览的其他 Issue：
  - 若该问题正是您要解决的，请在问题下评论说明您将负责处理。
  - 若问题已打开较久，解决前请先做预检查。
  - 若您解决了自己报告的问题，关闭前请简要说明处理结果。

- 问题咨询

  如果您在本项目中的使用或者其他内容存在疑问，欢迎您新建Issue进行反馈和咨询。

  您可以按照新建 [Question|问题咨询](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Question|%E9%97%AE%E9%A2%98%E5%92%A8%E8%AF%A2&template=.gitcode%252FISSUE_TEMPLATE%252Fquestion.yml) 类 Issue 提出自己的疑问。

- 帮助解决他人 Issue

  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应 Issue 需要进行代码修改，您可以在 Issue 评论框中输入 “/assign” 或 “/assign @yourself”，将该 Issue 分配给您，跟踪协助解决问题。

## 开发建议

### 代码风格

本代码仓采用 LLVM 社区通用的代码规范与编程风格，参见 [LLVM 编码规范](https://llvm.org/docs/CodingStandards.html)。可使用以下工具进行代码风格检查：

- [Clang-Tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy)
- [CppLint](https://github.com/cpplint/cpplint)
- [Cppcheck](http://cppcheck.sourceforge.net/)
- [CMakeLint](https://github.com/cmake-lint/cmake-lint)

### 提交 PR

- 在 [GitCode](https://gitcode.com/Ascend/AscendNPU-IR) 上提出您的想法作为 Issue。
- 若新功能涉及较多设计细节，请同时提交设计方案。
- 在问题讨论与设计方案审查达成共识后，再进行 Fork 开发并提交 PR。
- PR 经充分讨论后，将根据讨论结果进行合并、拒绝或关闭。
- PR 合入需 2 位 Reviewer 评论 `/lgtm`（Looks Good To Me）及 1 位 Approver 评论 `/approve`；PR 提交者本人无法执行合入操作。

### Fork-Pull 开发模式

在向 AscendNPU-IR 项目提交代码前，请先将项目 Fork 到您自己的仓库。后续在您 Fork 的仓库中进行开发，并通过 Pull Request 将变更合并到本项目。

### 代码更改自测

完成代码更改后，需在**构建目录**下编译并运行测试以验证功能：

```bash
ninja check-bishengir
```

### 代码推送验证

代码更新与测试通过后，将您的 commit 推送到您的远程仓库。

### 向主仓创建拉取请求

将代码推送到您的远程仓库后，在您的新分支与 AscendNPU-IR 的 master 分支之间新建 Pull Request。创建合并请求后，在 PR 中评论 `compile` 可触发 CI 构建流水线。建议尽快将 PR 合并到上游 master，以降低合并冲突风险。

### 门禁异常处理

代码门禁异常主要包括以下几类，请根据 CI 提示信息逐项排查并修复。

- **编译失败**：根据提示检查编译失败原因，修复后重新编译。
- **静态检查失败**：根据提示定位并修复代码中的静态检查问题。
- **CI 流水线未通过**：根据提示定位未通过的测试用例并修复，然后重新触发 CI。

## 注意事项

- 避免在 PR 中引入与本次修改无关的变更。
- 保持提交历史简洁、有序（可适当使用 squash/rebase）。
- 创建 PR 前，请将您的分支 rebase 到上游仓库最新 master。
- 若为错误修复类 PR，请在描述中关联所有相关 Issue 与 PR。
