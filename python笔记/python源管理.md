# python源列表
https://pypi.python.org/simple 默认源  
http://pypi.douban.com/simple/ 豆瓣  
http://mirrors.aliyun.com/pypi/simple/ 阿里  
http://pypi.hustunique.com/simple/ 华中理工大学  
http://pypi.sdutlinux.org/simple/ 山东理工大学  
http://pypi.mirrors.ustc.edu.cn/simple/ 中国科学技术大学  
https://pypi.tuna.tsinghua.edu.cn/simple 清华  

# 源设置方法
## 临时设置  
添加“-i”或“--index”参数，如：  
pip install -i http://pypi.douban.com/simple/ flask  

## windows配置
在C:\Users\用户名\AppData\Roaming  目录下创建pip目录，在pip目录下创建pip.ini配置文件，在配置文件内写入如下内容：  

	[global]    
	index-url=http://mirrors.aliyun.com/pypi/simple/    
	[install]    
	
	trusted-host=mirrors.aliyun.com  

## linux配置
创建~/.pip/pip.conf配置文件，在配置文件中写入如下内容：  

	[global]    
	index-url=http://mirrors.aliyun.com/pypi/simple/    
	[install]    
	
	trusted-host=mirrors.aliyun.com  

## pycharm配置
在file->setting->project interpreter->Manage Repositores中修改pip源配置  

## Anaconda国内源配置  

https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

在用户目录下创建配置文件.condarc，写入如下内容，Windows 用户无法直接创建名为 `.condarc` 的文件，可先执行 `conda config --set show_channel_urls yes` 生成该文件之后再修改

	channels:
	  - defaults
	show_channel_urls: true
	channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
	default_channels:
	  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
	  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
	  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
	  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
	  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
	custom_channels:
	  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
	  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
	  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
	  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
	  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
	  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

运行 `conda clean -i` 清除索引缓存，保证用的是镜像站提供的索引

## 卸载所有安装包

	pip freeze | grep -v "^-e" | xargs pip uninstall -y


## conda拷贝的方式创建env
	conda create -n tfcpu-py37 --clone base

## conda删除环境

```shell 
conda env remove -n ENV_NAME
```



## env关联jupyter notebook

```python
conda install nb_conda
```

## anaconda环境迁移

1. 在线方式

```shell
spec list:
# 需要在具有 相同操作系统 的计算机之间复制环境;
# 不包含pip安装的软件包
conda list --explicit > spec-list.txt
conda create  --name python-course --file spec-list.txt

Environment.yml
# 在 不同的平台和操作系统之间 复现项目环境
# 包含pip安装的软件包
conda env export > environment.yml
conda env create -f environment.yml
```

2. 离线方式

`Conda-pack` 是一个命令行工具，用于打包 conda 环境，其中包括该环境中安装的软件包的所有二进制文件。 
当您想在有限或没有网络访问的系统中重现环境时，此功能很有用。

**conda-pack 指定平台和操作系统，目标计算机必须具有与源计算机相同的平台和操作系统。**

要安装 conda-pack，请确保您位于 root 或 base 环境中，以便 conda-pack 在子环境中可用。
Conda-pack 可通过 conda-forge 或者 PyPI 安装

```shell
conda-forge: conda install -c conda-forge conda-pack
PyPI: pip install conda-pack
```

```shell
# 打包环境
# Pack environment my_env into my_env.tar.gz
conda pack -n my_env
# Pack environment my_env into out_name.tar.gz
# conda pack -f --ignore-missing-files -n my_env -o out_name.tar.gz 
conda pack -n my_env -o out_name.tar.gz
# Pack environment located at an explicit path into my_env.tar.gz
conda pack -p /explicit/path/to/my_env

# 重现环境
# Unpack environment into directory `my_env`
mkdir -p my_env
tar -xzf my_env.tar.gz -C my_env
# Use Python without activating or fixing the prefixes. Most Python
# libraries will work fine, but things that require prefix cleanups
# will fail.
./my_env/bin/python
# Activate the environment. This adds `my_env/bin` to your path
source my_env/bin/activate
# Run Python from in the environment
(my_env) $ python
# Cleanup prefixes from in the active environment.
# Note that this command can also be run without activating the environment
# as long as some version of Python is already installed on the machine.
(my_env) $ conda-unpack
```

