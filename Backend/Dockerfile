FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 注意这里：指明从 Backend 文件夹复制 requirements.txt
COPY Backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目（包括 News, Backend, Models 等）
COPY . .

# 暴露端口
EXPOSE 8010

# 启动命令：注意路径
# 因为 WORKDIR 是 /app，而代码在 /app/Backend/main.py
# 且我们需要引用 News 模块，所以启动路径要写对
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "8010"]