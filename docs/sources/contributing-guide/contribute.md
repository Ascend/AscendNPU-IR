# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[ascend-community](https://gitcode.com/ascend/community)了解行为准则，进行[CLA协议签署](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1)，了解源码仓的贡献流程。

### ✍️ 如何签署Ascend社区贡献者许可协议CLA
在参与社区贡献前，您需要签署Ascend社区贡献者许可协议（CLA）

 - **个人贡献者**：请选择“签署个人CLA”
 - **企业**：请选择“签署法人CLA”
 - **企业员工**：请选择“法人贡献者登记”，签署后会收到主题是Signing CLA on project of xx的邮件，请联系邮件内容里的Corporation Managers进行审批


### 开发者贡献
开发者贡献场景主要包括：

- Bug修复

  如果您在本项目中发现了某些Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

  您可以新建 [Bug-Report|缺陷反馈](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Bug-Report|%E7%BC%BA%E9%99%B7%E5%8F%8D%E9%A6%88&template=.gitcode%252FISSUE_TEMPLATE%252Fbug-report.yml) 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

- 贡献代码

  您可以新建 [Requirement|需求建议](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=%E6%96%B0%E9%9C%80%E6%B1%82&template=.gitcode%252FISSUE_TEMPLATE%252Ffeature.yml) 类Issue对新的样例算子予以说明，并提供您的设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

  对于您在浏览的其他issue
  - 如果您发现这个问题正是您要解决的，请对该问题发表评论，告诉其他人您将负责这个问题。
  - 如果问题已经打开一段时间，请您在解决该问题前进行预检查。
  - 如果您解决了自己报告的问题，在关闭该问题前还需要让其他人知道。
  
- 问题咨询

  如果您在本项目中的使用或者其他内容存在疑问，欢迎您新建Issue进行反馈和咨询。

  您可以按照新建 [Question|问题咨询](https://gitcode.com/Ascend/AscendNPU-IR/issues/create?type=template&title=Question|%E9%97%AE%E9%A2%98%E5%92%A8%E8%AF%A2&template=.gitcode%252FISSUE_TEMPLATE%252Fquestion.yml) 类Issue提出自己的疑问。

  
- 帮助解决他人Issue

  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。

### 开发建议

#### 代码风格

本代码仓采用LLVM社区通用的代码规范与编程风格。[LLVM 编码规范](https://llvm.org/docs/CodingStandards.html))。可以使用以下工具和规范检查自己的代码风格。
* [clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy)
* [CppLint](https://github.com/cpplint/cpplint)
* [CppCheck](http://cppcheck.sourceforge.net/)
* [CMakeLint](https://github.com/cmake-lint/cmake-lint)

#### 提交PR

- 在[GitCode](https://gitcode.com/Ascend/AscendNPU-IR)上提出您的想法作为问题。
- 如果要开发的新功能需要大量设计细节，您还应提交设计方案。
- 在问题讨论和设计方案审查达成共识后，再进行fork开发并提交PR。
- 在PR被充分讨论后，将根据讨论结果对PR进行合并、拒绝或放弃。
- PR的合入需要2个Reviewer评论`/lgtm`（Looks Good To Me）,以及1个Approver评论`/approve` 。请注意PR提交者无法进行合入操作。

#### Fork-Pull开发模式

在您向AscendNPU-IR项目提交自己的代码之前，请确保已经将项目Fork到您自己的存储库。后续您将在自己Fork的项目上进行开发，并通过Pull Request的方式合并到本项目。

#### 代码更改自测

完成代码更改后，需要编译后验证功能是否正确。
```
ninja check-bishengir
```

#### 代码推送验证

代码更新&测试完成后，推送您的commit到您的远程仓库。

#### 向主仓创建拉取请求

代码推送至您的远程仓库后，您需要在您的新分支和与AscendNPU IR的主分支之间新建Pull Request。完成新建合并请求后，评论`compile`后，`CI`将自动设置为您构建流水线测试。您的Pull Request请尽快合并到上游master分支，以降低合并风险。

#### 门禁异常处理

代码门禁异常主要包含以下几种情况，请根据相关提示信息解决门禁异常问题。

- 编译失败

  请根据提示信息，检查编译失败的原因，解决后重新编译即可 。

- 静态检查失败

  请根据提示信息，查找出代码中的异常信息并解决。

- CI流水线未通过

  请根据提示信息，查找出CI流水线未通过的测试用例并检查原因，解决后重新运行CI流水线。

### 注意事项：

-   应避免任何不相关的更改。
-   确保您的提交历史是简洁有序的。
-   创建PR前请rebase上游仓库最新代码。
-   对于错误修复 PR，请确保链接所有相关Issue 和 PR。