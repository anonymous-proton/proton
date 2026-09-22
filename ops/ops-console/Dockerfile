FROM node:20-alpine AS build
WORKDIR /app

COPY package.json /app/package.json
RUN npm install

COPY . /app
RUN npm run build

FROM alpine:3.20
WORKDIR /app
COPY --from=build /app/dist /app/dist

CMD ["sh", "-c", "rm -rf /dist/* && cp -r /app/dist/* /dist/ && tail -f /dev/null"]
