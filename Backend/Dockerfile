FROM python:3.11-slim

WORKDIR /app

# 安装 TensorFlow 运行所需的系统库
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件（注意路径：假设你在 Backend 目录下 build）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码
COPY . .

# 暴露你的 8010 端口
EXPOSE 8010

# 启动命令：确保模块路径正确。
# 我们使用 Backend.main:app 是因为我们在根目录的上一级启动
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]