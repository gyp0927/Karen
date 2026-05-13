# 开发规范

## 代码修改后流程

每次修改代码后必须执行：

1. **运行测试** — `python test_all.py`
2. **测试通过 → 提交并推送**：
   - `git add -A`
   - `git commit -m "描述"`
   - `git push karen fix-root-code:main`
3. **向用户汇报** — 报告测试结果和推送状态
4. **测试失败 → 先修复再推**

## 例外情况

- 用户明确说"先不推"或"不要推送"时跳过推送
- 仅修改文档（README、注释）时可跳过测试，但仍需推送

## 推送目标

- Remote: `karen`
- 本地分支: `fix-root-code`
- 远程分支: `main`
