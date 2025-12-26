# GitHub 项目更新指南

## 当前状态
- 远程仓库：`git@github.com:kingdoja/MyNewProject.git`
- 当前分支：`main`
- 状态：有一些文件删除和新文件需要提交

## 安全更新步骤（不会物理删除文件）

### 方法一：标准更新流程（推荐）

#### 1. 查看当前更改
```bash
git status
```

#### 2. 添加所有更改（包括删除和新文件）
```bash
# 添加所有更改，包括删除的文件
git add -A
```

或者分别添加：
```bash
# 添加新文件和修改的文件
git add .

# 添加删除的文件
git add -u
```

#### 3. 查看将要提交的更改（确认没有误删重要文件）
```bash
git status
git diff --cached --stat
```

#### 4. 提交更改
```bash
git commit -m "更新项目：删除旧文件，添加新功能"
```

#### 5. 推送到 GitHub
```bash
git push origin main
```

### 方法二：如果只想更新特定文件

#### 1. 只添加新文件，不删除旧文件
```bash
# 只添加新文件和修改的文件，不删除
git add .
```

#### 2. 提交并推送
```bash
git commit -m "添加新功能"
git push origin main
```

### 重要说明

#### ⚠️ 关于文件删除
- `git add -A` 或 `git add -u` 会**标记**文件删除，但**不会物理删除**本地文件
- 删除的文件仍然存在于你的本地文件系统中
- 只有执行 `git rm <file>` 才会从工作区删除文件
- 如果担心，可以先备份重要文件

#### ✅ 安全措施

1. **备份重要文件**（可选但推荐）
```bash
# 创建备份目录
mkdir -p ~/backup_$(date +%Y%m%d)
# 备份重要目录（根据你的需要调整）
cp -r A_sclie2inference ~/backup_$(date +%Y%m%d)/ 2>/dev/null || true
```

2. **查看将要删除的文件**
```bash
# 查看所有标记为删除的文件
git status | grep "^ D"
```

3. **如果误删了文件，可以恢复**
```bash
# 恢复单个文件
git checkout HEAD -- <file_path>

# 恢复所有删除的文件
git checkout HEAD -- .
```

#### 📋 当前需要处理的更改

根据当前状态，有以下更改：
- **删除的文件**：`A_sclie2inference/DataSlice2Inference_main1/` 和 `DataSlice2Inference_main111/` 目录下的文件
- **新文件**：`A_sclie2inference/DataSlice2Inference_main_npu/` 目录

### 快速更新命令（一键执行）

如果你想一次性完成所有更新：

```bash
# 1. 查看更改
git status

# 2. 添加所有更改
git add -A

# 3. 提交
git commit -m "更新项目：清理旧文件，添加NPU支持"

# 4. 推送
git push origin main
```

### 如果遇到问题

#### 问题1：推送被拒绝
```bash
# 先拉取远程更改
git pull origin main --rebase

# 解决冲突后再次推送
git push origin main
```

#### 问题2：大文件推送失败
如果遇到大文件问题，可以：
1. 确保 `.gitignore` 已正确配置（已配置）
2. 使用 Git LFS（如果需要跟踪大文件）
3. 从历史中移除大文件（需要时再处理）

#### 问题3：想撤销更改
```bash
# 撤销暂存区的更改（不删除文件）
git reset HEAD

# 恢复所有文件到上次提交的状态
git checkout -- .
```

## 总结

- ✅ Git 操作**不会物理删除**你的本地文件
- ✅ 删除操作只是从 Git 跟踪中移除，文件仍在本地
- ✅ 可以随时通过 `git checkout` 恢复文件
- ✅ 建议先查看更改再提交
- ✅ 重要文件建议先备份

