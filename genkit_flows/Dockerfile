FROM node:20-slim

WORKDIR /app

# 重点：不要写 genkit_flows/ 前缀
# 我们在部署时把 Context Directory 设为 genkit_flows
COPY package*.json ./

# 安装所有依赖，包括开发依赖（tsc 编译器）
RUN npm install

# 复制当前目录下所有内容
COPY . .

# 执行编译
RUN npm run build

ENV NODE_ENV=production
ENV FLOW_SERVER_PORT=4001
EXPOSE 4001

# 启动命令
CMD ["node", "dist/index.js"]