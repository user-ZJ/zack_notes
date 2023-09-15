#docker build -f Dockerfile --no-cache . -t zackwilliam/ubuntu:18-mybuild
#FROM ubuntu:18.04
FROM chxuan/ubuntu-vimplus:latest
MAINTAINER Docker zack <zackwilliam@docker.com>
#ENV LANG C.UTF-8
COPY sources.list /etc/apt/sources.list
# 指令合并，减少层数
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && \
	apt-get -y --no-install-recommends install zip unzip git cmake build-essential zlib1g-dev automake autoconf wget sox gfortran libtool subversion locales && rm -rf /var/lib/apt/lists/*
RUN locale-gen zh_CN
RUN locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8
#RUN useradd --create-home --no-log-init --shell /bin/bash zack
#RUN adduser zack sudo
#RUN echo 'zack:123456' | chpasswd
#USER zack
#WORKDIR /home/zack