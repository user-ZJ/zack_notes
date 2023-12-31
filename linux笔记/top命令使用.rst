.. _TOP命令使用:

===========
TOP命令使用
===========

top是linux系统监控工具中的瑞士军刀。它善于将相当多的系统整体性能放在一个屏幕上。
显示内容还能以交互的方式进行改变。默认情况下top将占用CPU最多的进程按照降序排列，每3秒刷新一次。

top命令显示
==============
.. image:: /images/top.png

统计区域信息
--------------

+--------------------+-------------------------------+---------------------------------------------------------------------------------------------------------------------+
|         行         |           显示信息            |                                                        说明                                                         |
+====================+===============================+=====================================================================================================================+
| 第一行             | - 17:17:14                    | - 当前时间                                                                                                          |
|                    | - up 7 days,2:34              | - 表示系统运行时间                                                                                                  |
|                    | - 0 users                     | - 当前登录用户数                                                                                                    |
|                    | - load average:0.00 0.00 0.00 | - 系统负载，即任务队列的平均长度；三个数值分别为 1分钟、5分钟、15分钟前到现在的平均值(现代操作系统中任务单位为线程) |
+--------------------+-------------------------------+---------------------------------------------------------------------------------------------------------------------+
| 第二行             |                               | 总共有8个进程，1个正在运行，7个睡眠进程，0个停止的进程，0个僵尸进程                                                 |
| (进程信息)         |                               |                                                                                                                     |
+--------------------+-------------------------------+---------------------------------------------------------------------------------------------------------------------+
| 第三行(CPU信息)    | - us                          | - 用户空间占用CPU百分比                                                                                             |
|                    | - sy                          | - 内核空间占用CPU百分比                                                                                             |
|                    | - ni                          | - 用户进程空间内改变过优先级的进程占用CPU百分比                                                                     |
|                    | - id                          | - 空闲CPU百分比                                                                                                     |
|                    | - wa                          | - 等待输入输出的CPU时间百分比                                                                                       |
|                    | - hi                          | - 服务硬件中断所花费的时间                                                                                          |
|                    | - si                          | - 服务软件中断所花费的时间                                                                                          |
|                    | - st                          | - 虚拟 CPU 等待实际 CPU 的时间的百分比,高st值可能意味着主机供应商在服务器上过量地出售虚拟机                         |
+--------------------+-------------------------------+---------------------------------------------------------------------------------------------------------------------+
| 第四行(内存信息)   | - total                       | - 物理内存总量                                                                                                      |
|                    | - used                        | - 使用的物理内存总量                                                                                                |
|                    | - free                        | - 空闲内存总量                                                                                                      |
|                    | - buff/cache                  | - 用作内核缓存的内存量                                                                                              |
+--------------------+-------------------------------+---------------------------------------------------------------------------------------------------------------------+
| 第五行(交换区信息) | - total                       | - 交换区总量                                                                                                        |
|                    | - used                        | - 使用的交换区总量                                                                                                  |
|                    | - free                        | - 空闲交换区总量                                                                                                    |
|                    | - avail                       | - free+cache                                                                                                        |
+--------------------+-------------------------------+---------------------------------------------------------------------------------------------------------------------+

.. note:: 

    load avg统计的是系统中处于以下三种状态的进程数量：

    | 运行状态（Running）：正在使用CPU执行的进程数量。
    | 等待状态（Waiting）：正在等待某些事件（如IO操作）完成的进程数量。
    | 就绪状态（Ready）：已经准备好运行，但是还没有得到CPU时间片的进程数量。

    这三种状态的进程数量加起来就是load avg的值。因此，load avg可以反映系统中正在运行的进程数和等待运行的进程数的平均值，是衡量系统负载的一个重要指标。

进程信息区
-------------

**默认情况下仅显示比较重要的 PID、USER、PR、NI、VIRT、RES、SHR、S、%CPU、%MEM、TIME+、COMMAND 列。可通过 f 键可以选择显示的内容**

+----------+-------------------------------------------------------------------------+
| 显示信息 |                                  说明                                   |
+==========+=========================================================================+
| PID      | 进程ID                                                                  |
+----------+-------------------------------------------------------------------------+
| USER     | 用户                                                                    |
+----------+-------------------------------------------------------------------------+
| PR       | 优先级                                                                  |
+----------+-------------------------------------------------------------------------+
| NI       | nice值，负值表示高优先级，正值表示低优先级                              |
+----------+-------------------------------------------------------------------------+
| VIRT     | 进程使用的虚拟内存总量，单位kb。VIRT=SWAP+RES                           |
+----------+-------------------------------------------------------------------------+
| RES      | 进程使用的、未被换出的物理内存大小，单位kb。RES=CODE+DATA               |
+----------+-------------------------------------------------------------------------+
| SHR      | 共享内存大小，单位kb                                                    |
+----------+-------------------------------------------------------------------------+
| S        | 进程状态；D=不可中断的睡眠状态，R=运行，S=睡眠，T=跟踪/停止，Z=僵尸进程 |
+----------+-------------------------------------------------------------------------+
| %CPU     | 上次更新到现在的CPU时间占用百分比                                       |
+----------+-------------------------------------------------------------------------+
| %MEM     | 进程使用的物理内存百分比                                                |
+----------+-------------------------------------------------------------------------+
| TIME+    | 进程使用的CPU时间总计，单位1/100秒                                      |
+----------+-------------------------------------------------------------------------+
| COMMAND  | 命令名/命令行                                                           |
+----------+-------------------------------------------------------------------------+

命令行参数
---------------
    | top [-] [d] [p] [q] [c] [C] [S] [s] [n]

+----------------------+-------------------------------------------------------------------------------------------------------------------------+
|         参数         |                                                          说明                                                           |
+======================+=========================================================================================================================+
| -h /-v               | Help/Version，帮助信息和版本信息                                                                                        |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -b                   | 以批处理模型运行。通常top只显示单屏信息，超出该屏幕的进程不显示。该选项显示全部进程。                                   |
|                      | 如果你要将top的输出保存为文件或输出给另一个命令处理，那么该项是很有用的<br />可以配合-n一起使用，如top -b -n 1          |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -H                   | 线程模式。top 显示单个线程。 如果没有此命令行选项，则会显示每个进程中所有线程的总和。 稍后可以使用“H”交互命令更改此设置 |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -o fieldname         | 指定排序的列。<br />top -o +%CPU   按CPU占用率从高到底排序<br />top -o -%CPU    按CPU占用率从低到高排序                 |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -O                   | 输出所有列名，可以帮助-o选项选择对应的列                                                                                |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -u/-U number or name | 只显示指定用户的进程                                                                                                    |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -w number            | 指定每行的宽度                                                                                                          |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -d num               | 指定每两次屏幕信息刷新之间的时间间隔。当然用户可以使用s交互命令来改变                                                   |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -p pid1 pid2         | 监控指定pid的进程，最多可指定20个pid                                                                                    |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -q                   | 该选项将使top没有任何延迟的进行刷新。如果调用程序有超级用户权限，那么top将以尽可能高的优先级运行                        |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -S                   | 指定累计模式,累计时间，当累积时间模式为On时，每个进程都会列出它及其子进程使用的CPU时间                                  |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -s                   | 使top命令在安全模式中运行。这将去除交互命令所带来的潜在危险                                                             |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -i                   | 使top不显示任何闲置或者僵死进程                                                                                         |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -c                   | 显示整个命令行而不只是显示命令名                                                                                        |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+
| -n num               | 刷新次数，-n 1表示只显示一次就退出                                                                                      |
+----------------------+-------------------------------------------------------------------------------------------------------------------------+

交互命令
-----------

全局交互命令
`````````````````
+----------------+---------------------------------------------------------------------------------------------------------------------+
|      命令      |                                                        说明                                                         |
+================+=====================================================================================================================+
| Enter or Space | 刷新显示                                                                                                            |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| h或者?         | 显示帮助画面，给出一些简短的命令总结说明                                                                            |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| =              | 退出任务限制。取消对显示的筛选操作，如top -i                                                                        |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| 0(数字0)       | 零抑制，即不显示0                                                                                                   |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| A              | 交替显示模式。进程的另一种显示方式，其内容为各种系统资源最大的消耗者                                                |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| B              | 粗体禁用/启用                                                                                                       |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| S              | 切换到累计模式                                                                                                      |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| s              | 改变两次刷新之间的延迟时间。系统将提示用户输入新的时间，单位为s。如果有小数，就换算成m s。输入0值则系统将不断刷新   |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| E              | 修改summary区域内存显示尺度（K/M/G）                                                                                |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| e              | 修改task区域（进程）内存显示尺度（K/M/G）                                                                           |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| g              | 选择另一个窗口/字段组                                                                                               |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| H              | 线程模式                                                                                                            |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| I（大写i）     | Irix mode;top是否用系统中的CPU数量除以CPU使用率<br />                                                               |
|                | 例如一个系统中有两个CPU，如果一个进程占用了这两个CPU，那么这个选项将在top显示CPU使用率为100%或200%之间切换          |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| k              | 终止一个进程。系统将提示用户输入需要终止的进程PID，以及需要发送给该进程什么样的信号。                               |
|                | 一般的终止进程可以使用15信号；如果不能正常结束那就使用信号9强制结束该进程。默认值是信号15。在安全模式中此命令被屏蔽 |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| q              | 退出                                                                                                                |
+----------------+---------------------------------------------------------------------------------------------------------------------+
| r pid nice     | 调整进程的优先级（NI列），进程优先级为-20-19；-20最高，19最低。普通用户只能从高优先级往低优先级调整                 |
+----------------+---------------------------------------------------------------------------------------------------------------------+


SUMMARY区域交互命令
`````````````````````````
+-----------+--------------------------------------------------------------------------------+
|   命令    |                                      说明                                      |
+===========+================================================================================+
| l(小写L)  | 切换是否显示平均负载和启动时间信息。                                           |
+-----------+--------------------------------------------------------------------------------+
| t         | 修改task行和CPU行显示，在1,2,3,4之间循环切换<br />                             |
|           | 1. 按类别划分的详细百分比<br />2. 简略的CPU占用百分比+bar图形显示<br />        |
|           | 3. 简略的CPU占用百分比+block图形显示<br />4. 不显示CPU和TASK行                 |
+-----------+--------------------------------------------------------------------------------+
| m         | Memory/Swap的显示切换<br />                                                    |
|           | 1. 显示Memory/Swap详细信息<br />2. 显示内存占用百分比/总内存+bar图形显示<br /> |
|           | 3. 显示内存占用百分比/总内存+block图形显示<br />4. 关闭显示                    |
+-----------+--------------------------------------------------------------------------------+
| 1（数字） | 切换CPU使用率，是按照独立使用率显示还是按照总量显示                            |
+-----------+--------------------------------------------------------------------------------+

TASK区域交互命令
`````````````````````
+--------+-------------------------------------------------------------------------+
|  命令  |                                  说明                                   |
+========+=========================================================================+
| J/j    | 切换左、右对齐                                                          |
+--------+-------------------------------------------------------------------------+
| x      | 高亮排序的列                                                            |
+--------+-------------------------------------------------------------------------+
| c      | 是否显示进程的完整名称                                                  |
+--------+-------------------------------------------------------------------------+
| f/F    | 添加、删除要显示的列                                                    |
+--------+-------------------------------------------------------------------------+
| S      | 累计时间模式。<br />开启时，显示进程及dead子进程CPU占用时间之和的百分比 |
+--------+-------------------------------------------------------------------------+
| u/U    | 显示指定用户的进程                                                      |
+--------+-------------------------------------------------------------------------+
| V      | tree视图，显示父子进程之间的关系                                        |
+--------+-------------------------------------------------------------------------+
| i      | 忽略闲置和僵死进程。这是一个开关式命令                                  |
+--------+-------------------------------------------------------------------------+
| n / #  | 限制显示进程的个数                                                      |
+--------+-------------------------------------------------------------------------+
| < / >  | 左右切换需要排序的列；配合x高亮排序的列可以更明显的观察                 |
+--------+-------------------------------------------------------------------------+
| R      | 从大到小排序切换到从小到大排序                                          |
+--------+-------------------------------------------------------------------------+
| o或者O | 改变显示列的顺序                                                        |
+--------+-------------------------------------------------------------------------+
| M      | 根据驻留内存大小进行排序。                                              |
+--------+-------------------------------------------------------------------------+
| P      | 根据CPU使用百分比大小进行排序。                                         |
+--------+-------------------------------------------------------------------------+
| T      | 根据时间/累计时间进行排序。                                             |
+--------+-------------------------------------------------------------------------+

