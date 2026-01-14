# GitHub 上传指南

## 前置准备

### 1. 检查 Git 配置
```bash
# 检查是否已配置用户信息
git config --global user.name
git config --global user.email

# 如果未配置，请设置（替换为你的信息）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. 检查当前 Git 状态
```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main
git status
```

## 上传步骤

### 步骤 1: 清理和准备

```bash
# 确保在项目根目录
cd /home/ubuntu/lsn/project_new/RT-DETR-main

# 查看哪些文件会被忽略（确认大文件不在其中）
git status --ignored

# 查看将要提交的文件
git status
```

### 步骤 2: 添加文件到暂存区

```bash
# 添加所有文件（.gitignore 会自动排除大文件）
git add .

# 或者选择性添加（推荐，更安全）
git add .gitignore
git add *.md
git add *.py
git add *.txt
git add *.yml
git add benchmark/
git add DataSlice2Inference/
git add rtdetr_pytorch/
git add rtdetrv2_pytorch/
git add rtdetr_paddle/
git add rtdetrv2_paddle/
git add .github/
# 添加其他需要的目录，但不要添加数据目录
```

### 步骤 3: 提交更改

```bash
# 提交更改（使用有意义的提交信息）
git commit -m "Initial commit: RT-DETR project with auto processing tools"

# 或者更详细的提交信息
git commit -m "feat: Add RT-DETR project with auto processing pipeline

- Add auto_process_monitor.py for automated WSI processing
- Add batch inference tools
- Add global visualization tools
- Update .gitignore to exclude large data files"
```

### 步骤 4: 在 GitHub 上创建仓库

1. 登录 GitHub (https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - Repository name: `RT-DETR-main` (或你喜欢的名字)
   - Description: 项目描述
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为本地已有）
4. 点击 "Create repository"

### 步骤 5: 添加远程仓库并推送

```bash
# 添加远程仓库（替换 YOUR_USERNAME 和 REPO_NAME）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 或者使用 SSH（如果已配置 SSH key）
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# 查看远程仓库配置
git remote -v

# 推送代码到 GitHub（首次推送）
git push -u origin master

# 如果默认分支是 main 而不是 master
# git branch -M main
# git push -u origin main
```

### 步骤 6: 验证上传

1. 在浏览器中访问你的 GitHub 仓库
2. 确认所有文件都已上传
3. 确认大文件（DataWSI/, DataPatches/ 等）没有上传

## 常见问题

### 问题 1: 推送时要求输入用户名密码
```bash
# 使用 Personal Access Token（推荐）
# 1. GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
# 2. 生成新 token，勾选 repo 权限
# 3. 推送时使用 token 作为密码

# 或者配置 SSH key（更安全）
# 参考：https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

### 问题 2: 想排除已跟踪的大文件
```bash
# 如果之前已经提交了大文件，需要从 Git 历史中移除
git rm -r --cached DataWSI/
git rm -r --cached DataPatches/
git commit -m "Remove large data files from tracking"
git push
```

### 问题 3: 分支名称不匹配
```bash
# 如果远程仓库使用 main 分支
git branch -M main
git push -u origin main
```

### 问题 4: 查看文件大小
```bash
# 检查仓库大小
du -sh .git

# 查看最大的文件
git ls-files | xargs du -h | sort -rh | head -20
```

### 问题 5: pack exceeds maximum allowed size (2.00 GiB)
如果推送时遇到此错误，说明 Git 历史中包含大文件。需要从历史中移除：

```bash
# 方法 1: 使用 git-filter-repo（推荐，需要先安装）
# Ubuntu/Debian: sudo apt install git-filter-repo

# 移除特定目录/文件
git filter-repo --path 大文件路径/ --invert-paths --force

# 移除多个路径
git filter-repo --path 路径1/ --path 路径2/ --invert-paths --force

# 清理并压缩仓库
git gc --aggressive --prune=now

# 检查仓库大小（应该 < 2GB）
du -sh .git

# 强制推送到远程（历史已重写）
git push origin master --force
# 或
git push origin master --force-with-lease  # 更安全的方式

# 方法 2: 如果历史不重要，可以重新初始化
# 警告：这会丢失所有 Git 历史！
rm -rf .git
git init
git add .
git commit -m "Initial commit"
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git push -u origin master --force
```

## 后续更新

```bash
# 日常更新代码到 GitHub
git add .
git commit -m "项目更新：清理旧文件，添加NPU支持"
git push
```

## 注意事项

1. ✅ **已排除的大文件目录**：
   - DataWSI/ (全图文件)
   - DataPatches/ (切片数据)
   - DataPatchesKeep/ (保留的切片)
   - DataPatchesTrash/ (废弃的切片)
   - DataPatchesInference/ (推理结果)
   - DataSlice2Inference/models/*.pt (模型文件)
   - *.tar.gz, *.zip (压缩包)

2. ✅ **已包含的代码文件**：
   - 所有 Python 脚本
   - README 文件
   - 配置文件
   - 工具脚本

3. ⚠️ **如果需要在 GitHub 上分享模型**：
   - 使用 GitHub Releases 上传大文件
   - 或使用 Git LFS (Large File Storage)
   - 或提供下载链接

## 使用 Git LFS 上传大文件（可选）

如果需要上传模型文件等大文件：

```bash
# 安装 Git LFS
# Ubuntu/Debian:
sudo apt install git-lfs

# 初始化 Git LFS
git lfs install

# 跟踪大文件类型
git lfs track "*.pt"
git lfs track "*.onnx"
git lfs track "*.pth"

# 提交 .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"

# 然后正常添加和提交
git add DataSlice2Inference/models/
git commit -m "Add model files via Git LFS"
git push
```

