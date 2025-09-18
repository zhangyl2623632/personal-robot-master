# 版本管理指南

本指南详细介绍了个人智能问答机器人项目的版本管理规范和使用方法，帮助开发人员正确更新和维护项目版本。

## 版本管理规范

项目采用**语义化版本号（Semantic Versioning）**管理，格式为 `Major.Minor.Patch`：

- **Major（主版本号）**：当进行重大功能变更或架构调整时，增加主版本号
- **Minor（次版本号）**：当新增功能但保持向下兼容时，增加次版本号
- **Patch（补丁版本号）**：当修复bug但不影响现有功能时，增加补丁版本号

## 版本号的位置

当前项目版本号存储在以下位置：

1. **README.md**：在文件开头以 `版本：vX.Y.Z` 格式标注
2. **系统状态API**：通过 `/api/status` 和 `/api/version` 端点可以获取当前版本
3. **Web界面**：在系统状态面板中显示当前版本信息

## 版本管理工具

项目提供了以下工具来管理版本号：

### 1. 命令行工具

使用 `update_version.py` 脚本可以手动更新版本号：

```bash
# 语法
python update_version.py [major|minor|patch]

# 示例：增加补丁版本号（v1.0.0 -> v1.0.1）
python update_version.py patch

# 示例：增加次版本号（v1.0.1 -> v1.1.0）
python update_version.py minor

# 示例：增加主版本号（v1.1.0 -> v2.0.0）
python update_version.py major
```

### 2. Web API

项目提供了以下API端点来管理版本：

- **获取版本信息**：`GET /api/version`
  - 返回当前系统版本号
  
- **获取完整系统状态（包含版本）**：`GET /api/status`
  - 返回包含版本信息的完整系统状态
  
- **更新版本号（仅供开发使用）**：`POST /api/update_version`
  - 参数：`{"part": "major|minor|patch"}`
  - 在实际生产环境中，应添加适当的身份验证

## 版本更新流程

当你完成以下类型的更改时，应更新版本号：

1. **添加新功能**：增加次版本号（Minor）
   ```bash
   python update_version.py minor
   ```
   
2. **修复bug**：增加补丁版本号（Patch）
   ```bash
   python update_version.py patch
   ```
   
3. **进行重大变更**：增加主版本号（Major）
   ```bash
   python update_version.py major
   ```

## 更新日志维护

每次版本更新后，应在README.md文件的"更新日志"章节中记录主要变更内容。更新日志应包含：

- 版本号和发布日期
- 新增功能的详细说明
- 修改的功能点及变更内容
- 修复的bug
- 兼容性说明（如有）

## 注意事项

1. 版本号更新后，README.md中的版本信息会自动更新
2. 在提交代码前，请确保版本号与变更内容匹配
3. 重要版本更新应在团队内部进行充分沟通
4. 生产环境的版本更新应遵循严格的测试流程
5. 版本号变更应作为单独的提交，便于追踪

## 未来改进方向

1. 实现git钩子，在代码提交时自动更新版本号
2. 集成CI/CD流程，实现版本发布的自动化
3. 添加版本历史记录数据库，跟踪所有版本变更
4. 实现版本回滚功能

---

*最后更新时间: 2024-09-15*