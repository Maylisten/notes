# Docker安装

## Macos安装

安装Docker Desktop for Mac

https://docs.docker.com/desktop/install/mac-install/



## Centos7安装

```shell
# 检查内核， 需要在3.8以上
uname -a

# 安装前置依赖
yum install -y yum-utils device-mapper-persistent-data lvm2

# 添加yum源（阿里）
yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

# 查看docker版本
yum list docker-ce --showduplicates | sort -r

# 安装docker
yum install docker-ce -y

# 启动docker
systemctl start docker

# 设置开机启动
systemctl enable  docker

# 检查docker版本
docker version

# 卸载docker
yum erase docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-selinux \
                  docker-engine-selinux \
                  docker-engine \
                  docker-ce
```

