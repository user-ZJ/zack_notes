docker使用笔记
=========================

docker-cpu
------------------------
Docker 属于 Linux 容器的一种封装，提供简单易用的容器使用接口。它是目前最流行的 Linux 容器解决方案。
Docker 将应用程序与该程序的依赖，打包在一个文件里面。运行这个文件，就会生成一个虚拟容器。
程序在这个虚拟容器里运行，就好像在真实的物理机上运行一样。有了 Docker，就不用担心环境问题。  
用户可以方便地创建和使用容器，把自己的应用放入容器。容器还可以进行版本管理、复制、分享、修改，就像管理普通的代码一样。 

VM是一个运行在宿主机之上的完整的操作系统，VM运行自身操作系统会占用较多的CPU、内存、硬盘资源。
Docker不同于VM，只包含应用程序以及依赖库，基于libcontainer运行在宿主机上，并处于一个隔离的环境中，这使得Docker更加轻量高效，
启动容器只需几秒钟之内完成。由于Docker轻量、资源占用少，使得Docker可以轻易的应用到构建标准化的应用中。

相关概念
`````````````````````
* Docker daemon：运行在宿主机上，Docker守护进程，用户通过Docker client(Docker命令)与Docker daemon交互  
* Docker client: Docker 命令行工具，是用户使用Docker的主要方式，Docker client与Docker daemon通信并将结果返回给用户，
  Docker client也可以通过socket或者RESTful api访问远程的Docker daemon  
* Docker image：镜像是只读的，镜像中包含有需要运行的文件。镜像用来创建container，一个镜像可以运行多个container；
  镜像可以通过Dockerfile创建，也可以从Docker hub/registry上下载。  
* Docker container：容器是Docker的运行组件，启动一个镜像就是一个容器，容器是一个隔离环境，多个容器之间不会相互影响，
  保证容器中的程序运行在一个相对安全的环境中。   
* Docker hub/registry: 共享和管理Docker镜像，用户可以上传或者下载上面的镜像，官方地址为https://registry.hub.docker.com/，
  也可以搭建自己私有的Docker registry。

Image(镜像)
`````````````````````
Docker 镜像可以看作是一个特殊的文件系统，除了提供容器运行时所需的程序、库、资源、配置等文件外，还包含了一些为运行时准备的一些配置参数(如匿名卷、环境变量、用户等)。  
镜像不包含任何动态数据，其内容在构建之后也不会被改变。镜像(Image)就是一堆只读层(read-only layer)的统一视角，也许这个定义有些难以理解，
下面的这张图能够帮助读者理解镜像的定义：    

.. image:: /images/docker/docker_1.png

从左边我们看到了多个只读层，它们重叠在一起。除了最下面一层，其他层都会有一个指针指向下一层。
这些层是 Docker 内部的实现细节，并且能够在主机的文件系统上访问到。  
统一文件系统(Union File System)技术能够将不同的层整合成一个文件系统，为这些层提供了一个统一的视角。  
这样就隐藏了多层的存在，在用户的角度看来，只存在一个文件系统。我们可以在图片的右边看到这个视角的形式。  

Container(容器)
`````````````````````````
容器(Container)的定义和镜像(Image)几乎一模一样，也是一堆层的统一视角，唯一区别在于容器的最上面那一层是可读可写的。  

.. image:: /images/docker/docker_2.png 

由于容器的定义并没有提及是否要运行容器，所以实际上，容器 = 镜像 + 读写层。  

Repository(仓库)
`````````````````````````
Docker 仓库是集中存放镜像文件的场所。镜像构建完成后，可以很容易的在当前宿主上运行。
但是， 如果需要在其他服务器上使用这个镜像，我们就需要一个集中的存储、分发镜像的服务，Docker Registry(仓库注册服务器)就是这样的服务。   
有时候会把仓库(Repository)和仓库注册服务器(Registry)混为一谈，并不严格区分。     

Docker 仓库的概念跟 Git 类似，注册服务器可以理解为 GitHub 这样的托管服务。   
实际上，一个 Docker Registry 中可以包含多个仓库(Repository)，每个仓库可以包含多个标签(Tag)，每个标签对应着一个镜像。   
所以说，镜像仓库是 Docker 用来集中存放镜像文件的地方，类似于我们之前常用的代码仓库。  
通常，一个仓库会包含同一个软件不同版本的镜像，而标签就常用于对应该软件的各个版本 。   
我们可以通过<仓库名>:<标签>的格式来指定具体是这个软件哪个版本的镜像。如果不给出标签，将以 Latest 作为默认标签。    
仓库又可以分为两种形式（和git类似）：   
Public(公有仓库)   
Private(私有仓库)  

Docker 的架构
`````````````````````
.. image:: /images/docker/docker_3.png
   
Docker 使用 C/S 结构，即客户端/服务器体系结构。Docker 客户端与 Docker 服务器进行交互，Docker服务端负责构建、运行和分发 Docker 镜像。  
Docker 客户端和服务端可以运行在一台机器上，也可以通过 RESTful 、 Stock 或网络接口与远程 Docker 服务端进行通信。

.. image:: /images/docker/docker_4.png

这张图展示了 Docker 客户端、服务端和 Docker 仓库(即 Docker Hub 和 Docker Cloud )，默认情况下 Docker 会在 Docker 中央仓库寻找镜像文件。   
这种利用仓库管理镜像的设计理念类似于 Git ，当然这个仓库是可以通过修改配置来指定的，甚至我们可以创建我们自己的私有仓库。

安装
`````````````````
windows安装docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
系统要求：windows10专业版

docker ce下载：[下载链接](https://store.docker.com/editions/community/docker-ce-desktop-windows )  

安装步骤：
  
1. 启用Hyper-V和容器   
    
    打开控制面板 - 程序和功能 - 启用或关闭Windows功能，勾选Hyper-V和容器功能（家庭版无此功能），然后点击确定，系统会重启完成配置  

2. 安装Docker  

    下载docker安装包，双击运行，按默认安装即可  
    安装完成后，点击Close and log out，重新启动系统  

3. 启动Docker  

    在桌面找到Docker for Windows快捷方式，双击启动即可！启动成功后托盘处会有一个小鲸鱼的图标。  
    打开命令行输入命令：docker version可以查看当前docker版本号(docker启动比较慢，如果docker version命令报错，可以等会儿再试)  

4. 更换镜像源地址  

    中国官方镜像源地址为：https://registry.docker-cn.com   
    点击托盘处docker图标右键选择-Settings，在Daemon->Registry mirrors中添加https://registry.docker-cn.com   
    点击Apply后会重启Docker。  

5. 载入测试镜像测试

	在cmd中输入命令“docker run hello-world”可以加载测试镜像来测试  
	docker会检查Unable to find image 'hello-world:latest' locally  
	然后从镜像源中下载hello-world   
	输出：Hello from Docker!  
	

ubuntu18.04 docker-ce安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
.. code-block:: shell

	#更换国内软件源，推荐中国科技大学的源，稳定速度快
	sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
	sudo sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
	sudo apt update
	#安装需要的包
	sudo apt install apt-transport-https ca-certificates software-properties-common curl
	#添加 GPG 密钥，并添加 Docker-ce 软件源，这里还是以中国科技大学的 Docker-ce 源为例
	curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
	sudo add-apt-repository "deb [arch=amd64] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
	#添加成功后更新软件包缓存
	sudo apt update
	#安装 Docker-ce
	sudo apt install docker-ce
	#设置开机自启动并启动 Docker-ce（安装成功后默认已设置并启动，可忽略）
	sudo systemctl enable docker
	sudo systemctl start docker
	#测试运行
	sudo docker run hello-world
	#添加当前用户到 docker 用户组，可以不用 sudo 运行 docker（可选）
	sudo groupadd docker
	sudo usermod -aG docker $USER
	#测试添加用户组（可选）
	docker run hello-world

ubuntu安装包下载：https://download.docker.com/linux/ubuntu/dists/xenial/pool/stable/amd64/
https://download.docker.com/linux/static/stable/aarch64/

docker centos安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

docker安装参考官方教程： `Get Docker CE for CentOS <https://link.zhihu.com/?target=https%3A//docs.docker.com/install/linux/docker-ce/centos/%23upgrade-docker-ce>`_ . 

从docker官网下载docker安装文件，上传至服务器，然后执行  

.. code-block:: shell 

	# 安装
	sudo yum install xxx.rpm 
	# 启动
	sudo systemctl start docker
	# 卸载 
	sudo yum remove docker-ce
	sudo rm -rf /var/lib/docker 

docker普通用户使用
------------------------------------
.. code-block:: shell

	# 创建 docker 用户组
	sudo groupadd docker
	# 添加你想用普通用户权限的用户名到 docker 用户组
	sudo usermod -aG docker $USER
	# 系统重启后就可以使用普通用户权限执行 docker, 如果不想重启，可以使用下面的命令更新并激活组权限
	newgrp docker
	sudo chmod 777 /var/run/docker.sock
	# 验证设置是否成功
	docker run hello-world


docker常用命令
-------------------------
.. code-block:: shell

    #拉取docker镜像
	docker pull image_name
	docker pull image_name:tag
	#查看宿主机上的镜像，Docker镜像保存在/var/lib/docker目录下:
	docker images ls
	#删除镜像
	docker rmi  docker.io/tomcat:7.0.77-jre7   
    docker rmi b39c68b7af30
	#查看当前有哪些容器正在运行
	docker ps
	docker container -ls
	#查看所有容器
	docker ps -a
	docker container ls --all
	#quiet mode运行的容器
	docker container ls -aq
	#启动、停止、重启容器命令：
	docker start container_name/container_id
	docker stop container_name/container_id
	docker restart container_name/container_id
	# 重命名容器
	docker rename old_container_name  new_container_name
	# 导出容器
	docker export container_id -o image_file #保存容器
	docker export container_id | gzip > image_name.tar.gz
	docker save -o centos72.tar centos72:v1  #保存镜像
	docker save centos72:v1 | gzip > centos72.tar.gz
	# 导入成镜像
	docker import image_file image_name
	zcat image_name.tar.gz | sudo docker import - image_name
	docker load -i <xxx.tar>
	docker load < <xxx.tar> 
	#后台启动一个容器后，如果想进入到这个容器，可以使用attach命令：
	docker attach container_name/container_id
	#删除容器的命令：
	docker rm container_name/container_id
	#查看当前系统Docker信息
	docker info
	#从Docker hub上下载某个镜像:
	docker pull centos:latest
	# 启动容器
	docker run  --name docker_nginx_v1  -d -p 80:80 nginx:v1
	# 用 nginx 镜像启动一个容器，命名为docker_nginx_v1，并且映射了 80 端口，这样我们可以用浏览器去访问这个 nginx 服务器   
	# -p 参数来发布容器端口到 host 的某个端口上，<host_port>:<container_port>
	# -w参数覆盖构建时所设置的工作目录
	# -u 参数来覆盖所指定的用户
	# -v 参数挂载目录
	# --shm-size=6g  设置共享内存的大小,默认情况下，Docker使用64MB的共享内存。
	
	# 修改容器内容
	# 容器启动后，需要对容器内的文件进行进一步的完善，可以使用docker exec -it xx bash命令再次进行修改  
	docker exec -it docker_nginx_v1  /bin/bash
	
	# 创建镜像
	docker build -t friendlyhello -f Dockerfile 	
	
	# Tag <image> for upload to registry
	docker tag <image> username/repository:tag
	
	docker push username/repository:tag # Upload tagged image to registry
	docker run username/repository:tag  # Run image from a registry
	
	# 容器宿主机之间文件拷贝
	docker cp /opt/test/file.txt mycontainer:/opt/testnew/
	docker cp mycontainer:/opt/testnew/file.txt /opt/test/


覆盖ENTRYPOINT
`````````````````````
.. code-block:: shell

	docker run -it --entrypoint /bin/bash [docker_image]


Dockerfile
---------------------
Dockerfile 是用来定义 **镜像**   
Dockerfile 是一个文本文件，其内包含了一条条的指令(Instruction)，每一条指令构建一层，因此每一条指令的内容，
就是描述该层应当如何构建。有了 Dockerfile，当我们需要定制自己额外的需求时，只需在 Dockerfile 上添加或者修改指令，
重新生成 image 即可，省去了敲命令的麻烦。

Dockerfile文件格式
```````````````````````````````
Dockerfile 分为四部分：基础镜像信息、维护者信息、镜像操作指令、容器启动执行指令    

.. code-block:: dockerfile

	# 1、第一行必须指定 基础镜像信息
	FROM ubuntu
	# 2、维护者信息
	MAINTAINER docker_user docker_user@email.com
	# 3、镜像操作指令
	RUN echo "deb http://archive.ubuntu.com/ubuntu/ raring main universe" >> /etc/apt/sources.list
	RUN apt-get update && apt-get install -y nginx
	RUN echo "\ndaemon off;" >> /etc/nginx/nginx.conf
	# 4、容器启动执行指令
	CMD /usr/sbin/nginx  


Dockerfile中指令
````````````````````
::

    RUN 执行命令并提交结果
    CMD 为执行容器提供默认值，Dockerfile只能有一条CMD指令。如果列出多个CMD，则只有最后一个CMD生效  
    ADD 向镜像中添加文件或目录，不复制目录本身，只复制其内容，ADD [--chown=<user>:<group>] <src>... <dest>
    COPY 复制文件或目录，COPY [--chown=<user>:<group>] <src>... <dest>
    ENV  ENV <key>=<value> 将环境变量<key>设置为该值 <value>
            为单个命令设置环境变量 RUN <key>=<value> <command>
    EXPOSE  Docker容器在运行时侦听指定的网络端口,EXPOSE 80/tcp  EXPOSE 80/udp
    FROM 
    LABEL 设置镜像标签，使用docker inspect ImageID查看
    STOPSIGNAL
    USER  USER <user>[:<group>]
    VOLUME  创建挂载点，VOLUME ["/data"]
    WORKDIR  工作目录 WORKDIR /path/to/workdir
    ENTRYPOINT  配置容器运行入口的可执行文件，如docker run -i -t --rm -p 80:80 nginx
    ARG docker build 参数
    SHELL  允许覆盖用于shell命令形式的默认shell

使用Dockerfile构建镜像
```````````````````````````````
docker build 命令会根据 Dockerfile 文件及上下文构建新 Docker 镜像。  
构建上下文是指 Dockerfile 所在的本地路径或一个URL（Git仓库地址）。
构建上下文环境会被递归处理，所以构建所指定的路径还包括了子目录，而URL还包括了其中指定的子模块。  

将当前目录做为构建上下文时，可以像下面这样使用docker build命令构建镜像：

docker build .
	
在构建上下文中使用的 Dockerfile 文件，是一个构建指令文件。为了提高构建性能，可以通过.dockerignore文件排除上下文目录下不需要的文件和目录  
Dockerfile 一般位于构建上下文的根目录下，也可以通过-f指定该文件的位置：

docker build -f /path/to/a/Dockerfile .

构建时，还可以通过-t参数指定构建成镜像的仓库、标签。

docker build -t ubuntu:v1 -f /path/to/a/Dockerfile --no-cache .


docker-compose
------------------------------
对docker容器进行编排

安装
```````````
https://docs.docker.com/compose/install/standalone/

.. code-block:: shell

	curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
	sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

.. literalinclude:: docker-compose.yaml
    :language: yaml


FAQ
-------------
删除镜像失败
`````````````````````
>rm -rf /var/lib/docker  
rm: cannot remove ‘/var/lib/docker/containers’: Device or resource busy  

删除不了的原因是: 在建立容器的时候做了相应目录的挂载，没有卸载，所以Device or resource busy

>cat /proc/mounts | grep "docker"    
>umount /var/lib/docker/containers  
>rm -rf /var/lib/docker   

容器中删除文件夹失败
`````````````````````````````
> rm /root/test
> can't remove : Directory not empty  

原因是centos7.2的overlay2文件系统驱动存在bug

> 修改docker存储驱动程序，从overlay2修改为devicemapper
> 具体配置是添加启动参数：–storage-driver=devicemapper  
> 在/etc/docker/daemon.json配置文件中添加"storage-driver": "devicemapper"
> systemctl restart docker重启docker
> docker info查看storage-driver配置是否修改为devicemapper


docker中配置jupyter notebook
-------------------------------------
下载镜像
```````````````````
docker pull jupyter/base-notebook

创建容器
`````````````````
docker run -it --name jupyter-notebook -p 7777:8888 jupyter/base-notebook:latest /bin/bash

将8888端口映射到本机的7777端口

测试
```````````````
在容器中运行jupyter notebook

可以看到一串类似 http://localhost:8888/?token=3c32ac9203dc507d0d6bbcc191c83c650c081308100eb397 的带 token 的 URL，将 8888 替换为我们的 7777 在浏览器中打开即可完成验证。

nvidia docker 2
-----------------------------

nvidia docker 2安装
```````````````````````````
1. 卸载旧版本
   
   .. code-block:: shell

        sudo yum remove docker \
	              docker-client \
	              docker-client-latest \
	              docker-common \
	              docker-latest \
	              docker-latest-logrotate \
	              docker-logrotate \
	              docker-engine
	
	    docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
	    sudo yum remove nvidia-docker

nvidia-docker命令
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nvidia-docker run -it  \--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 需要启动的docker名称 bash

docker run \--gpus \'\"device=1,2\"\' nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 nvidia-smi