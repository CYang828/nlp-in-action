# Docker 环境安装和使用指南

## 🚀 快速开始

### 先决条件

在运行此项目之前，请确保您已安装以下软件：

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 安装 Docker Compose

#### 在 Ubuntu/Debian 上：
```bash
# 更新软件包索引
sudo apt-get update

# 安装 Docker Compose
sudo apt-get install docker-compose-plugin

# 验证安装
docker compose version
```

#### 在 macOS 上：
```bash
# 如果您安装了 Docker Desktop，则已包含 Docker Compose
# 否则，通过 Homebrew 安装
brew install docker-compose

# 验证安装
docker-compose --version
```

#### 在 Windows 上：
Windows 版 Docker Desktop 中已包含 Docker Compose。

### 使用 Docker Compose 运行项目

此项目包含预配置的 JupyterLab 环境，已安装所有必要的依赖项：

```bash
# 克隆仓库
git clone https://github.com/CYang828/nlp-in-action.git
cd nlp-in-action

# 导航到 docker-compose 目录
cd jupyterlab-docker-compose

# 构建并启动 JupyterLab 容器
docker compose up --build

# 或在后台模式运行
docker compose up -d --build
```

容器运行后，在浏览器中打开：
```
http://localhost:8888
```

您将可以访问完整的 JupyterLab 环境，其中预装了所有课程材料和依赖项。

### 停止环境

```bash
# 停止运行中的容器
docker compose down

# 停止并删除卷（如果您想重置所有内容）
docker compose down -v
```

## 故障排除

如果遇到任何问题，请参考项目主页的 FAQ 文档：https://rarelimiting.feishu.cn/wiki/EQTRwntmnidkpSkK9pOcLzOcnze
