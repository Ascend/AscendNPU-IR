# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请参见[ascend-community](https://gitcode.com/ascend/community)了解行为准则，完成[CLA协议签署](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1)，并了解源码仓的贡献流程。

## ✍️如何签署Ascend社区贡献者许可协议（CLA）

在参与社区贡献前，需签署Ascend社区贡献者许可协议（CLA）：

- **个人贡献者**：请选择「签署个人CLA」，填写好姓名和邮箱后提交申请即可。
- **企业**：请选择「签署法人CLA」，填写企业相关信息，提交申请后，企业将接收到社区发出的签署企业CLA的文件，请按照邮件中的提示完成后续签署。
- **企业员工**：请选择「法人贡献者登记」；签署后会收到主题为 “Signing CLA on project of xx” 的邮件，请联系邮件中的Corporation Managers进行审批。

## 开发者贡献

开发者贡献场景主要包括：

- Bug修复

  在本项目中发现某些Bug，希望对其进行修复，可新建Issue反馈与跟踪。

  可新建[Bug-Report|缺陷反馈](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Bug-Report|%E7%BC%BA%E9%99%B7%E5%8F%8D%E9%A6%88&template=.gitcode%252FISSUE_TEMPLATE%252Fbug-report.yml)类Issue描述Bug，然后在评论框中输入“/assign”或“/assign @yourself”，认领该Issue开展处理。

- 贡献代码

  可新建[Requirement|需求建议](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=%E6%96%B0%E9%9C%80%E6%B1%82&template=.gitcode%252FISSUE_TEMPLATE%252Ffeature.yml)类Issue对新的样例算子予以说明，并提供设计方案，然后在评论框中输入“/assign”或“/assign @yourself”，认领该Issue开展处理。

  浏览其他Issue时遵循以下要求：
    - 计划解决对应问题，请在问题下评论说明负责处理。
    - 若问题已打开较久，解决前请先做预检查。
    - 自行提交并修复的问题，关闭前请简要说明处理结果。

- 问题咨询

  使用本项目过程中存在疑问，可新建Issue进行反馈和咨询。

  可通过新建[Question|问题咨询](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Question|%E9%97%AE%E9%A2%98%E5%92%A8%E8%AF%A2&template=.gitcode%252FISSUE_TEMPLATE%252Fquestion.yml)类Issue提出疑问。

- 帮助解决他人Issue

  若有可行方案解决社区其他开发者遇到的问题，欢迎在Issue中发表评论交流，协助解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，可在Issue评论框中输入 “/assign” 或 “/assign @yourself”，认领该Issue，跟踪协助解决问题。

## 开发建议

### 代码风格

本代码仓采用LLVM社区通用的代码规范与编程风格，参见[LLVM编码规范](https://llvm.org/docs/CodingStandards.html)。可使用以下工具进行代码风格检查：

- [Clang-Tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy)
- [CppLint](https://github.com/cpplint/cpplint)
- [Cppcheck](http://cppcheck.sourceforge.net/)
- [CMakeLint](https://github.com/cmake-lint/cmake-lint)

### 提交PR

- 在[GitCode](https://gitcode.com/Ascend/AscendNPU-IR)上提出想法创建Issue。
- 若新功能涉及较多设计细节，请同时提交设计方案。
- 在问题讨论与设计方案审查达成共识后，再进行Fork开发并提交PR。
- PR经充分讨论后，将根据讨论结果进行合并、拒绝或关闭。
- PR合入需2位Reviewer评论`/lgtm`（Looks Good To Me）及1位Approver评论`/approve`；PR提交者本人无法执行合入操作。

### Fork-Pull开发模式

在向AscendNPU IR项目提交代码前，请先将项目Fork至个人仓库。后续在Fork仓库中进行开发，并通过Pull Request将变更合并到本项目。

### 代码更改自测

完成代码更改后，需在**构建目录**下编译并运行测试以验证功能：

```bash
ninja check-bishengir
```

### 代码推送验证

代码更新与测试通过后，将commit推送到个人远程仓库。

### 向主仓创建拉取请求

将代码推送到远程仓库后，在新分支与AscendNPU IR的master分支之间新建Pull Request。创建合并请求后，在PR中评论`compile`可触发CI构建流水线。建议尽快将PR合并到上游master，以降低合并冲突风险。

### 门禁异常处理

代码门禁异常主要包括以下几类，请根据CI提示信息逐项排查并修复。

- **编译失败**：根据提示检查编译失败原因，修复后重新编译。
- **静态检查失败**：根据提示定位并修复代码中的静态检查问题。
- **CI流水线未通过**：根据提示定位未通过的测试用例并修复，然后重新触发CI。

## 注意事项

- 避免在PR中引入与本次修改无关的变更。
- 保持提交历史简洁、有序（可适当使用squash/rebase）。
- 创建PR前，请将本地分支rebase到上游仓库最新master。
- 若为错误修复类PR，请在描述中关联所有相关Issue与PR。
