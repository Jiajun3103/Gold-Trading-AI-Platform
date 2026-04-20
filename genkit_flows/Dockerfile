FROM node:20-slim

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run build

ENV NODE_ENV=production
ENV FLOW_SERVER_PORT=4001
EXPOSE 4001

CMD ["node", "dist/index.js"]