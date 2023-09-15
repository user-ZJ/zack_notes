ubuntu服务自启动
========================

systemd(需要sudo权限)
--------------------------------
1. 检查/lib/systemd/system/rc-local.service，如果没有，自己新创建，文件内容为(如果文件存在，没有Install项，需要自己添加进去)

::

    [Unit]
    Description=/etc/rc.local Compatibility
    Documentation=man:systemd-rc-local-generator(8)
    ConditionFileIsExecutable=/etc/rc.local
    After=network.target

    [Service]
    Type=forking
    ExecStart=/etc/rc.local start
    TimeoutSec=0
    RemainAfterExit=yes
    GuessMainPID=no

    [Install]
    WantedBy=multi-user.target
    Alias=rc-local.service


2. 检查/etc/systemd/system/rc-local.service,同样做以上修改
3. 创建/etc/rc.local,并写入想要运行的脚本程序，以下是一个示例

.. code-block:: shell

    #! bin/bash
    mkdir /usr/local/temp
    echo "test auto bootstrap" > /usr/local/temp/1.log

4. 给/etc/rc.local添加可执行权限。chmod +x /etc/rc.local
5. 设置服务自启动

.. code-block:: shell

    sudo systemctl enable rc-local

6. 启动服务或重启机器验证效果

.. code-block:: shell

    sudo systemctl start rc-local.service
    sudo systemctl status rc-local.service

/usr/local/temp/1.log文件被创建，且写入对应的内容
