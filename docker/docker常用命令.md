# Docker常用命令

## 1. 通用操作

```shell
# 查看帮助文档
docker 具体命令 --help

# 查看docker磁盘占用情况
docker system df

# 可视化操作
docker run -d -p 8088:9000 --restart=always -v /var/run/docker.sock:/var/run/docker.sock --privileged=true portainer/portainer

# 查看实时的容器占用资源情况
docker stats
```

## 2. 镜像相关操作

```shell
# 查看本地镜像
# -a: 列出本地所有的镜像（含历史映像层）
# -q: 只显示镜像ID
docker images 

# 下载镜像
docker pull 镜像名字:版本号

# 删除镜像
docker rmi -f 镜像ID

# 删除全部镜像
docker rmi -f $(docker images -qa)
docker images -qa | xargs docker rmi -f
```

## 3. 容器相关操作

```shell
# 查看正在运行的镜像
# -a 查看正在运行和停止的全部镜像
# -q 只查看id
docker ps 

# 运行容器，并打开容器内部终端
docker run -it --name 自定义名称 -p 本机端口:容器端口 -v 本地路径:容器路径 镜像名称:tag /bin/bash

# 后台运行，要求容器内部有前台应用，不然会立刻终止
docker run -d 镜像名称:tag

# 退出容器，容器内部输入命令，容器会变为Exit状态
exec

# 退出容器但不停止容器
ctrl+p+q

# 启动容器
docker start 容器ID或者容器名

# 重启容器
docker restart 容器ID或者容器名

# 停止容器（比较温和，会等待容器内部应用停止）
docker stop 容器ID或者容器名

# 强制停止容器（直接停止）
docker kill 容器ID或容器名

# 删除容器
docker rm 容器ID

# 删除全部容器
docker rm -f $(docker ps -aq)
docker ps -aq | xargs docker rm -f

# 查看容器日志
docker logs 容器ID

# 查看容器内部进程
docker top 容器ID

# 重新进入容器（打开新的终端）
docker exec -it 容器ID /bash/shell

# 重新进入容器（打开上次的终端）
docker attach 容器ID

# 容器和本机之间拷贝文件
docker cp 容器ID:容器内路径 目的主机路径
docker cp 目的主机路径 容器ID:容器内路径

# 查看容器详细信息
docker inspect 容器名/容器id

# 提交容器为镜像
docker commit -m="提交的描述信息" -a="作者" 容器id 目标镜像:[TAG]  
```

![image.png](https://gitee.com/may1234/md-imgs/raw/master/202410242029188.png)


