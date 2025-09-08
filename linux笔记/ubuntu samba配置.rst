ubuntu samba配置
===========================

安装samba
----------------------------
.. code-block:: bash

    sudo apt update
    sudo apt-get install samba samba-common

配置samba
----------------------------

添加用户
``````````````
.. code-block:: bash

    sudo usermod -aG sambashare [username]
    # 将用户添加到sambashare组中
    # [username]为你要添加的用户名
    sudo smbpasswd -a username
    # 输入密码，确认密码
    # username为你要添加的用户名

编辑samba配置文件

.. code-block:: bash

    sudo vim /etc/samba/smb.conf

在文件末尾添加以下内容

.. code-block:: txt

    [zack]
        comment = public anonymous access
        path = /home/zack/
        browsable = yes
        create mask = 0660
        directory mask = 0771
        writable = yes
        guest ok = yes

* [public]方括号内为share后显示的目录名
* path = /data/ 为用于share的本地路径
* browsable =yes 是否可以浏览
* create mask = 0660
* directory mask = 0771
* writable = yes是否可写
* guest ok = yes是否允许匿名访问

重启samba服务
``````````````````````
.. code-block:: bash

    sudo systemctl restart smbd nmbd
