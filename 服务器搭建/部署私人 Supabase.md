# 部署私人 Supabase

## 官方文档（docker 部署）
https://supabase.com/docs/guides/self-hosting/docker

## 部署要求
- 硬件：4g 以上服务器
- 软件：docker /git

## 部署过程
### 1. 下载源码
从 github 下载源码
```zsh
git clone --depth 1 https://github.com/supabase/supabase
```
下载需要外网，如果服务器无法下载，可以先在本机下载再上传到服务器（本机也需要设置代理）
```zsh
git config --global https.proxy https://127.0.0.1:7890
```

### 2. 进入 docker 文件夹
```zsh
cd supabase/docker
```

### 3. 配置 .env 
1. 创建 .env 文件
	```zsh
	cp .env.example .env
	```
2. 修改 .env
	需要修改的内容包括：
	- `POSTGRES_PASSWORD`
		数据库密码
	- `JWT_SECRET`
		JWT 密钥，可以从官网生成
	- `ANON_KEY`
		从官网生成
	- `SERVICE_ROLE_KEY`
		从官网生成
	- `DASHBOARD_USERNAME` 和 `DASHBOARD_PASSWORD
		图形化界面的用户名和密码
	- `SMTP_*`
		邮箱服务相关配置
	```env
	############
	# Secrets
	# YOU MUST CHANGE THESE BEFORE GOING INTO PRODUCTION
	############
	
	## postgres 数据密码
	POSTGRES_PASSWORD=postgres
	## JWT 密钥（在官网生成）
	JWT_SECRET=eZCpAvifs4j0Mxpqm9L5vUGrlYOjlyJE8yIZDiK9
	# ANON_KEY（在官网生成）
		ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogImFub24iLAogICJpc3MiOiAic3VwYWJhc2UiLAogICJpYXQiOiAxNzM0MDE5MjAwLAogICJleHAiOiAxODkxNzg1NjAwCn0.MUGONbUju_NCJQKGhBeak8ASbPuNVPqLQ8HFi_B9H4Y
		# SERVICE_ROLE_KEY（在官网生成）
		SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogInNlcnZpY2Vfcm9sZSIsCiAgImlzcyI6ICJzdXBhYmFzZSIsCiAgImlhdCI6IDE3MzQwMTkyMDAsCiAgImV4cCI6IDE4OTE3ODU2MDAKfQ.mb6o2xUe-6_RaMIcWlxoullo_yfndv5LU-1BkLhmclw
	## 图形化界面的用户和密码
	DASHBOARD_USERNAME=root
	DASHBOARD_PASSWORD=xh20000507
	
	############
	# Database - You can change these to any PostgreSQL database that has logical replication enabled.
	############
	
	POSTGRES_HOST=db
	POSTGRES_DB=postgres
	POSTGRES_PORT=5432
	# default user is postgres
	
	############
	# Supavisor -- Database pooler
	############
	POOLER_PROXY_PORT_TRANSACTION=6543
	POOLER_DEFAULT_POOL_SIZE=20
	POOLER_MAX_CLIENT_CONN=100
	POOLER_TENANT_ID=your-tenant-id
	
	
	############
	# API Proxy - Configuration for the Kong Reverse proxy.
	############
	
	KONG_HTTP_PORT=8000
	KONG_HTTPS_PORT=8443
	
	
	############
	# API - Configuration for PostgREST.
	############
	
	PGRST_DB_SCHEMAS=public,storage,graphql_public
	
	
	############
	# Auth - Configuration for the GoTrue authentication server.
	############
	
	## General
	SITE_URL=http://localhost:3000
	ADDITIONAL_REDIRECT_URLS=
	JWT_EXPIRY=3600
	DISABLE_SIGNUP=false
	API_EXTERNAL_URL=http://localhost:8000
	
	## Mailer Config
	MAILER_URLPATHS_CONFIRMATION="/auth/v1/verify"
	MAILER_URLPATHS_INVITE="/auth/v1/verify"
	MAILER_URLPATHS_RECOVERY="/auth/v1/verify"
	MAILER_URLPATHS_EMAIL_CHANGE="/auth/v1/verify"
	
	## Email auth
	ENABLE_EMAIL_SIGNUP=true
	ENABLE_EMAIL_AUTOCONFIRM=false
	SMTP_ADMIN_EMAIL=admin@example.com
	SMTP_HOST=supabase-mail
	SMTP_PORT=2500
	SMTP_USER=fake_mail_user
	SMTP_PASS=fake_mail_password
	SMTP_SENDER_NAME=fake_sender
	ENABLE_ANONYMOUS_USERS=false
	
	## Phone auth
	ENABLE_PHONE_SIGNUP=true
	ENABLE_PHONE_AUTOCONFIRM=true
	
	
	############
	# Studio - Configuration for the Dashboard
	############
	
	STUDIO_DEFAULT_ORGANIZATION=Default Organization
	STUDIO_DEFAULT_PROJECT=Default Project
	
	STUDIO_PORT=3000
	# replace if you intend to use Studio outside of localhost
	SUPABASE_PUBLIC_URL=http://localhost:8000
	
	# Enable webp support
	IMGPROXY_ENABLE_WEBP_DETECTION=true
	
	# Add your OpenAI API key to enable SQL Editor Assistant
	OPENAI_API_KEY=
	
	############
	# Functions - Configuration for Functions
	############
	# NOTE: VERIFY_JWT applies to all functions. Per-function VERIFY_JWT is not supported yet.
	FUNCTIONS_VERIFY_JWT=false
	
	############
	# Logs - Configuration for Logflare
	# Please refer to https://supabase.com/docs/reference/self-hosting-analytics/introduction
	############
	
	LOGFLARE_LOGGER_BACKEND_API_KEY=your-super-secret-and-long-logflare-key
	
	# Change vector.toml sinks to reflect this change
	LOGFLARE_API_KEY=your-super-secret-and-long-logflare-key
	
	# Docker socket location - this value will differ depending on your OS
	DOCKER_SOCKET_LOCATION=/var/run/docker.sock
	
	# Google Cloud Project details
	GOOGLE_PROJECT_ID=GOOGLE_PROJECT_ID
	GOOGLE_PROJECT_NUMBER=GOOGLE_PROJECT_NUMBER
		
	```

### 4. pull 并启动容器
```zsh
# Pull the latest images  
docker compose pull  
# Start the services (in detached mode)  
docker compose up -d
```

### 5. 开放服务器端口
开放服务器端口用于访问 supadatabase 的 dashboard，默认为8000


## 常见操作
### 修改 ENV
这个方法会清除掉 docker 容器和缓存，以及之前部署存储的数据，谨慎使用
```zsh
# 进入目录
cd supabase/docker
# 清除docker 的所有没有启动的容器和镜像
docker system prune -a
# 删除掉之前挂载的卷
sudo rm -rf volumes/db/data
# 重新 pull 并 up
docker compose up -d
```

### Docker Pull 加速
添加国内的镜像源地址
```zsh
# 这里配置的腾讯云
# {
#    "registry-mirrors": [
#    "https://mirror.ccs.tencentyun.com"
#   ]
# }
vim /etc/docker/daemon.json


# 重启 docker
sudo systemctl restart docker
```