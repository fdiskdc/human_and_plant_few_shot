# RNA 可视化Web应用

本项目包含一个前端应用和一个后端API服务。

## 启动指南

### 1. 启动后端服务

后端是一个Flask服务器，负责处理数据分析请求。

```bash
# 进入后端目录
cd backend

# (建议) 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 安装依赖
pip install -r requirements.txt

# 启动服务器
python server.py
```
服务器将在 `http://localhost:5000` 上运行。

### 2. 启动前端应用

前端是一个React应用，负责用户界面和数据展示。

```bash
# 打开一个新的终端，进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```
应用将在 `http://localhost:5173` (或另一个可用端口) 上运行，并会自动在浏览器中打开。
