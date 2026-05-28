请修复 Issue #$ARGUMENTS，严格按照以下流程操作：

1. 先用 `gh issue view $ARGUMENTS` 查看 Issue 的完整描述
2. 分析问题根因，确定需要修改的文件
3. 实施修复，确保不引入新的问题
4. 运行相关测试（如果有的话），确认修复有效
5. 用 `git add` 和 `git commit` 提交修改
   - commit 信息格式：fix(模块): 一句话描述修复内容 (#Issue编号)
6. 最后汇报：修改了哪些文件、修复思路是什么