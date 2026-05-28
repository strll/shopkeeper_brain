请帮我提交当前的所有代码修改：

1. 先执行 `git status` 和 `git diff --stat` 查看改了什么
2. 分析所有修改内容，理解这次改动的意图
3. 生成一条符合 Conventional Commits 规范的提交信息：
   - feat: 新功能
   - fix: 修复 bug
   - docs: 文档变更
   - refactor: 重构
   - chore: 杂项
   - 格式示例：feat(auth): 添加邮箱格式校验
4. 执行 `git add -A` 然后 `git commit -m "生成的信息"`
5. 告诉我提交成功的 commit hash