Linux 性能优化与故障排查
===========================

本文是一份面向日常运维和应用开发的 Linux 性能排查手册。重点不是记住所有
工具参数，而是建立一条可重复的诊断链路：

``确认现象 → 建立基线 → 找到饱和资源 → 定位进程或内核路径 → 修改 → 对比验证``。

文中的命令以现代 64 位 Linux、systemd 和 cgroup v2 为主要环境。不同内核、
发行版和 sysstat/procps-ng 版本的字段可能略有差异，应同时参考本机手册页。

.. contents:: 目录
   :local:
   :depth: 3


性能分析的基本原则
------------------

先测量，再优化
~~~~~~~~~~~~~~

性能问题必须有可观测的目标，例如延迟、吞吐、错误率、CPU 时间、内存占用或
磁盘等待时间。不要仅凭单次 ``top`` 截图做结论。

建议遵循以下原则：

* 明确工作负载、数据量、并发度和性能目标。
* 同时记录业务指标和系统指标，避免只优化局部资源占用。
* 使用持续采样而不是单点值；至少覆盖问题发生前、发生时和恢复后三个阶段。
* 一次只改变一个主要变量，并保留修改前后的对照数据。
* 优先修复错误、排队和资源饱和，再考虑微观代码优化。
* 在生产环境使用跟踪工具前评估开销、权限和数据敏感性。


USE 方法
~~~~~~~~

USE（Utilization、Saturation、Errors）方法适合快速检查每一种硬件或受限资源：

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - 维度
     - 含义
     - 典型例子
   * - Utilization
     - 资源忙碌程度
     - CPU ``%usr/%sys``、磁盘 ``%util``、网卡吞吐
   * - Saturation
     - 超出处理能力后形成的排队或压力
     - 运行队列、IO 队列、PSI stall、swap 抖动
   * - Errors
     - 操作失败或需要重试
     - TCP 重传、磁盘错误、OOM、网卡 drop

利用率接近 100% 不一定代表异常。例如顺序 IO 可让磁盘长期繁忙但延迟仍然可控；
反之，低平均利用率也可能掩盖短时排队和尾延迟。因此必须联合观察利用率、延迟、
队列和错误。


平均负载与 PSI
~~~~~~~~~~~~~~

Linux load average 是过去 1、5、15 分钟处于以下状态的任务数量平均值：

* 正在运行或等待 CPU 的可运行任务；
* 不可中断睡眠任务，通常显示为 ``D`` 状态，常见于 IO 或内核等待。

因此，load 高不等于 CPU 一定繁忙。粗略判断时可以将 load 与可用逻辑 CPU 数比较，
但还应检查 ``vmstat`` 的 ``r/b``、CPU 利用率和 PSI。

.. code-block:: console

   $ uptime
   $ nproc
   $ vmstat 1
   $ cat /proc/pressure/cpu
   $ cat /proc/pressure/memory
   $ cat /proc/pressure/io

PSI（Pressure Stall Information）衡量任务因为 CPU、内存或 IO 资源不足而停顿的时间：

* ``some``：至少一个任务受到影响；
* ``full``：所有非空闲任务同时受到影响；CPU PSI 通常只有 ``some``；
* ``avg10/avg60/avg300``：最近 10、60、300 秒的停顿比例；
* ``total``：系统启动以来累计停顿微秒数。

PSI 比单纯利用率更接近“资源压力是否已经影响工作”。容器环境还可以读取对应
cgroup 目录中的 ``cpu.pressure``、``memory.pressure`` 和 ``io.pressure``。


开始排查前
~~~~~~~~~~

先记录时间、机器身份、内核、CPU、内存、块设备和资源限制，避免分析错机器或忽略
容器配额。

.. code-block:: console

   $ date -Ins
   $ hostnamectl
   $ uname -a
   $ lscpu
   $ free -h
   $ lsblk -o NAME,TYPE,SIZE,ROTA,FSTYPE,MOUNTPOINTS
   $ systemd-detect-virt
   $ cat /proc/self/cgroup

还应确认最近是否发生发布、流量变化、配置调整、内核升级、磁盘扩容、日志激增或
上游依赖故障。很多“系统性能问题”实际是工作负载或外部服务发生了变化。


5 到 10 分钟快速检查
--------------------

下面的顺序适合首次接触一台出现问题的机器。每条命令回答一个不同问题。

总体状态
~~~~~~~~

.. code-block:: console

   $ uptime
   $ free -h
   $ vmstat 1 10
   $ pidstat -durwt 1 10
   $ iostat -xz 1 10
   $ df -h
   $ df -i
   $ ss -s
   $ cat /proc/pressure/{cpu,memory,io}
   $ dmesg --level=err,warn --ctime

重点观察：

#. load 是突然升高还是长期升高；
#. ``vmstat r`` 是否持续大于可用 CPU，``b`` 是否持续不为零；
#. CPU 时间主要在 user、system、iowait、steal 还是 softirq；
#. ``free`` 的 ``available`` 是否很低，是否持续发生 swap in/out；
#. 磁盘 ``await``、队列长度和设备利用率是否同时升高；
#. 文件系统空间或 inode 是否耗尽；
#. 是否有 OOM、文件系统、块设备、网卡或硬件错误；
#. PSI 是否证明资源压力已实际阻塞任务。

找出责任进程
~~~~~~~~~~~~

.. code-block:: console

   $ top
   $ pidstat -u -r -d -w 1
   $ ps -eo pid,ppid,stat,ni,psr,%cpu,%mem,rss,vsz,comm --sort=-pcpu
   $ ps -eo pid,ppid,stat,%cpu,%mem,rss,comm --sort=-rss
   $ systemd-cgtop

``top`` 的详细交互方式参见 :ref:`TOP命令使用`。需要注意：进程可能受 cgroup 限制，
宿主机总体仍然空闲，因此容器问题不能只看宿主机总利用率。


CPU 与调度
----------

关键指标
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 指标
     - 解读
   * - ``us``
     - 用户态计算时间；高值常见于计算热点、忙循环或高请求量
   * - ``sy``
     - 内核态时间；高值可能来自系统调用、网络、锁、页错误或驱动
   * - ``wa``
     - CPU 空闲且至少有任务等待 IO 的时间；不是磁盘利用率
   * - ``st``
     - 虚拟机被宿主机抢占的时间
   * - ``hi/si``
     - 硬中断和软中断时间；网络高负载时常见 ``si`` 升高
   * - ``r``
     - 可运行任务数；持续超过可用 CPU 通常表示 CPU 饱和
   * - 上下文切换
     - 过多切换可能来自线程过量、锁竞争、频繁阻塞或过短任务

系统与进程级采样
~~~~~~~~~~~~~~~~

.. code-block:: console

   $ mpstat -P ALL 1
   $ pidstat -u -t 1
   $ pidstat -w -t 1
   $ sar -u ALL 1
   $ ps -eo pid,tid,psr,stat,ni,pri,rtprio,pcpu,comm --sort=-pcpu

如果只有少数 CPU 忙，检查线程并行度、CPU affinity、中断亲和性和 NUMA 分布。
``mpstat`` 中单核长期 100%，而其他核空闲，通常不是“机器 CPU 不够”，而是应用
无法扩展、线程被绑核或某类中断集中在单核。


CPU 高的排查路径
~~~~~~~~~~~~~~~~

#. 用 ``pidstat -u -t 1`` 找到高 CPU 进程和线程。
#. 用 ``perf top -p PID`` 快速查看实时热点。
#. 用 ``perf record`` 采样一段具有代表性的负载。
#. 根据热点判断是业务计算、锁、自旋、内存访问、系统调用还是内核路径。
#. 修改后使用相同工作负载重新采样，并比较吞吐和尾延迟。

.. code-block:: console

   $ sudo perf stat -p PID sleep 10
   $ sudo perf top -p PID
   $ sudo perf record -F 99 -g -p PID -- sleep 30
   $ sudo perf report

``perf stat`` 常用来观察 cycles、instructions、IPC、branch-misses 和
cache-misses。硬件计数器受 CPU 型号、虚拟化环境和权限限制，不能脱离工作负载
直接使用固定阈值。


load 高但 CPU 不高
~~~~~~~~~~~~~~~~~~

这种情况优先检查 D 状态任务、存储、网络文件系统、内存回收和锁等待。

.. code-block:: console

   $ vmstat 1
   $ ps -eo state,pid,ppid,wchan:32,comm | awk '$1 ~ /D/'
   $ cat /proc/pressure/io
   $ iostat -xz 1
   $ sudo perf sched timehist
   $ sudo dmesg --level=err,warn --ctime

``wchan`` 只提供当前内核等待点，采样可能瞬间变化。不要看到 ``D`` 状态就直接认定
磁盘故障，还应结合调用栈、IO 指标和内核日志。


调度优先级与 CPU 亲和性
~~~~~~~~~~~~~~~~~~~~~~~

普通任务的 nice 值范围是 -20 到 19。nice 值越小，调度权重越高；修改其他用户
进程或设置负 nice 通常需要额外权限。实时调度策略可能让普通任务长时间得不到 CPU，
不应作为普通性能优化手段。

.. code-block:: console

   $ ps -eo pid,cls,rtprio,pri,ni,psr,comm
   $ nice -n 10 command
   $ renice 5 -p PID
   $ taskset -cp PID
   $ chrt -p PID

绑核有助于减少迁移或隔离延迟敏感任务，但也可能造成负载不均。修改 affinity 前，
应确认 CPU 拓扑、IRQ 分布和 NUMA 节点。


锁与调度延迟
~~~~~~~~~~~~

.. code-block:: console

   $ sudo perf sched record -- sleep 10
   $ sudo perf sched latency
   $ sudo perf lock record -- command
   $ sudo perf lock report
   $ sudo bpftrace -e 'tracepoint:sched:sched_switch { @[comm] = count(); }'

生产环境运行 ``perf lock`` 或高频 BPF 程序前，应先在测试环境评估开销。


内存
----

正确理解内存占用
~~~~~~~~~~~~~~~~

Linux 会把空闲内存用于页缓存。``free`` 很小并不等于内存不足，通常应关注：

* ``MemAvailable``：无需大量 swap 即可提供给新工作负载的估算内存；
* RSS/PSS：进程实际驻留内存，以及共享页按比例分摊后的内存；
* major page fault：需要从磁盘读取页面的缺页；
* swap in/out：是否正在发生交换，而不只是 swap 已被使用；
* memory PSI：内存回收是否使任务停顿；
* OOM 和 cgroup ``memory.events``。

.. code-block:: console

   $ free -h
   $ grep -E 'MemAvailable|AnonPages|Cached|Slab|SReclaimable|Dirty|Writeback' /proc/meminfo
   $ vmstat 1
   $ pidstat -r -p ALL 1
   $ cat /proc/pressure/memory
   $ dmesg --ctime | grep -i -E 'oom|out of memory|killed process'

``vmstat si/so`` 持续非零比“swap 使用量不为零”更值得关注。长期存在但不活跃的
swap 页面可能只是冷数据。


定位进程内存
~~~~~~~~~~~~

.. code-block:: console

   $ ps -eo pid,ppid,rss,vsz,%mem,stat,comm --sort=-rss
   $ cat /proc/PID/status
   $ cat /proc/PID/smaps_rollup
   $ pmap -x PID
   $ pidstat -r -p PID 1

常见字段：

* ``VmRSS``：当前驻留物理内存，不包含已换出的页；
* ``VmSize``：虚拟地址空间，不等于实际物理内存；
* ``RssAnon/RssFile/RssShmem``：匿名页、文件页和共享内存；
* ``Pss``：共享页按映射进程数分摊后的占用；
* ``VmSwap``：该进程已换出的匿名内存；
* ``minflt/s`` 与 ``majflt/s``：次缺页和主缺页速率。

虚拟地址空间很大可能来自 mmap、预留堆或语言运行时，不应直接判断为泄漏。


内存持续增长
~~~~~~~~~~~~

#. 确认增长的是 RSS、PSS、page cache、slab 还是共享内存。
#. 将增长与请求量、缓存条目、连接数和队列长度关联。
#. 对原生程序使用 ASan、Valgrind Massif 或 heaptrack。
#. 对托管语言优先使用其运行时 profiler 和 heap dump。
#. 检查文件描述符、线程和 mmap 数量，避免把资源泄漏误认为纯堆泄漏。

.. code-block:: console

   $ watch -n 2 'grep -E "VmRSS|VmData|VmSwap|Threads" /proc/PID/status'
   $ ls /proc/PID/fd | wc -l
   $ wc -l /proc/PID/maps
   $ heaptrack command
   $ valgrind --tool=massif command

开发期可使用 AddressSanitizer：

.. code-block:: console

   $ clang++ -O1 -g -fsanitize=address -fno-omit-frame-pointer app.cc -o app
   $ ASAN_OPTIONS=detect_leaks=1 ./app

ASan 适合测试环境快速发现越界、use-after-free 和部分泄漏；Valgrind 无需重新编译，
但运行开销通常更高。二者都不应直接用于常规生产流量。


页缓存、脏页和内核内存
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ grep -E 'Cached|Dirty|Writeback|Slab|SReclaimable|SUnreclaim' /proc/meminfo
   $ sudo slabtop
   $ cat /proc/vmstat
   $ sysctl vm.dirty_background_ratio vm.dirty_ratio

不要为了“释放内存”定期执行 ``drop_caches``。这会清空有价值的缓存并造成后续 IO
抖动，只适合受控基准测试。修改 dirty 参数也必须结合存储延迟、写入模式和断电
安全要求验证。


NUMA
~~~~

多路服务器上，远端内存访问会增加延迟并消耗互联带宽。

.. code-block:: console

   $ numactl --hardware
   $ numastat
   $ numastat -p PID
   $ lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
   $ sudo perf mem record -p PID -- sleep 10
   $ sudo perf mem report

NUMA 优化通常包括线程与内存同节点放置、避免单节点内存耗尽、按节点拆分工作负载。
不要在不了解应用内存访问模式时盲目使用 ``numactl --interleave=all`` 或强制绑节点。


磁盘与文件系统 IO
------------------

从延迟和队列开始
~~~~~~~~~~~~~~~~

.. code-block:: console

   $ iostat -xz 1
   $ pidstat -d -p ALL 1
   $ sudo iotop -oPa
   $ cat /proc/pressure/io
   $ lsblk -o NAME,TYPE,SIZE,ROTA,SCHED,MOUNTPOINTS

现代 ``iostat -x`` 常见字段：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 解读
   * - ``r/s``、``w/s``
     - 每秒读写请求数
   * - ``rkB/s``、``wkB/s``
     - 每秒读写吞吐
   * - ``r_await``、``w_await``
     - 请求从提交到完成的平均延迟，包括排队时间
   * - ``aqu-sz``
     - 平均队列长度；需结合设备并行能力解释
   * - ``rareq-sz``、``wareq-sz``
     - 平均请求大小
   * - ``%util``
     - 设备至少有一个 IO 在处理的时间比例

不要使用已经废弃且容易误导的 ``svctm``。NVMe、RAID、云盘和虚拟块设备具有并行
队列，``%util=100`` 也不一定代表吞吐上限；应同时检查延迟、队列、IOPS、带宽和
厂商限额。


磁盘延迟高的排查路径
~~~~~~~~~~~~~~~~~~~~

#. 用 ``iostat -xz 1`` 确认设备、方向和延迟。
#. 用 ``pidstat -d`` 或 ``iotop`` 找到责任进程。
#. 用 ``lsof``、``/proc/PID/fd`` 或 ``strace`` 找到文件和访问模式。
#. 检查文件系统、块设备和硬件日志。
#. 检查是否受到 cgroup IO 限制、云盘额度或快照任务影响。
#. 用 BPF 延迟工具区分应用提交延迟与块设备完成延迟。

.. code-block:: console

   $ sudo lsof -p PID
   $ sudo strace -ttT -f -e trace=%file,%desc -p PID
   $ sudo filefrag -v /path/to/file
   $ findmnt -T /path/to/file
   $ journalctl -k -p warning
   $ sudo biolatency
   $ sudo biosnoop

``biolatency`` 和 ``biosnoop`` 通常来自 BCC 工具包，发行版中的命令名可能带
``-bpfcc`` 后缀。


IO 模式与优化方向
~~~~~~~~~~~~~~~~~

常见优化方向包括：

* 合并小 IO、批量写入，减少同步刷盘次数；
* 避免不必要的 ``fsync``，但不能牺牲持久性语义；
* 调整缓存、预读和并发深度，使其匹配实际访问模式；
* 将日志、临时文件和主数据放到独立故障域或设备；
* 修复随机 IO、目录热点、文件碎片和 inode 耗尽；
* 为数据库等应用按其文档选择文件系统、挂载参数和 IO 调度器。

``io_uring`` 可以通过批量提交、完成队列和减少部分系统调用开销改善某些异步 IO
工作负载，但不会让底层设备突破物理极限。是否获益取决于 IO 模式、内核版本、
库实现和安全策略。

.. code-block:: console

   $ sudo strace -f -e trace=io_uring_setup,io_uring_enter,io_uring_register command
   $ ls -l /proc/PID/fd | grep 'anon_inode:\[io_uring\]'


网络
----

分层定位
~~~~~~~~

网络问题应区分应用、套接字、TCP/IP 栈、网卡、交换网络和远端服务。先确认是吞吐
不足、连接失败、握手慢、重传、丢包，还是应用处理不及时。

.. code-block:: console

   $ ip -s link
   $ ss -s
   $ ss -lntup
   $ ss -ti
   $ nstat -az
   $ sar -n DEV,TCP,ETCP 1
   $ sudo ethtool eth0
   $ sudo ethtool -S eth0

重点指标包括：

* 网卡 RX/TX bytes、packets、errors、dropped；
* TCP retransmissions、timeouts、listen drops；
* 套接字 send-q/recv-q；
* RTT、拥塞窗口、重传计时器；
* softirq CPU、单核中断热点；
* conntrack、监听队列和临时端口使用量。

``ifconfig`` 和 ``netstat`` 已由 iproute2 的 ``ip`` 与 ``ss`` 取代。旧命令可能仍
可安装，但新脚本应优先使用现代接口。


TCP 重传或丢包
~~~~~~~~~~~~~~

.. code-block:: console

   $ nstat -az | grep -E 'Retrans|Timeout|Listen|Drop'
   $ ss -ti dst REMOTE_IP
   $ ip -s link show dev eth0
   $ sudo ethtool -S eth0 | grep -Ei 'drop|error|miss|timeout'
   $ sudo tcpdump -ni eth0 -s 128 -w /tmp/capture.pcap 'host REMOTE_IP'

网卡 drop 不一定都是链路问题，也可能来自 ring buffer、CPU/softirq 处理不及时、
驱动、虚拟交换机或容器网络。抓包应限制接口、主机、端口、包长和持续时间，避免
带来磁盘压力和敏感数据风险。


连接和端口问题
~~~~~~~~~~~~~~

.. code-block:: console

   $ ss -lnt
   $ ss -ant state time-wait | wc -l
   $ ss -ant state syn-recv
   $ sysctl net.core.somaxconn net.ipv4.ip_local_port_range
   $ cat /proc/sys/net/netfilter/nf_conntrack_count
   $ cat /proc/sys/net/netfilter/nf_conntrack_max

大量 TIME_WAIT 通常是连接生命周期的结果，不应直接通过危险的内核参数“消除”。
先检查连接复用、短连接模式、客户端端口范围、负载均衡和服务端处理能力。


网络 CPU 与中断
~~~~~~~~~~~~~~~

.. code-block:: console

   $ mpstat -P ALL 1
   $ watch -n 1 cat /proc/softirqs
   $ cat /proc/interrupts
   $ sudo ethtool -l eth0
   $ sudo ethtool -x eth0

如果 NET_RX softirq 集中在少数 CPU，检查 RSS/RPS/XPS、IRQ affinity、队列数和
应用绑核。调整前要考虑 NUMA：网卡、中断处理 CPU 和应用内存最好避免跨节点访问。


应用与进程
----------

从时间分解开始
~~~~~~~~~~~~~~

.. code-block:: console

   $ /usr/bin/time -v command
   $ pidstat -u -r -d -w -p PID 1
   $ strace -c -f command
   $ strace -ttT -f -p PID

``time`` 区分 wall time、user time 和 system time：

* wall time 远大于 CPU time：程序主要在等待、睡眠或排队；
* user time 高：优先分析用户态计算热点；
* system time 高：检查系统调用、缺页、网络、文件系统和锁；
* 大量 involuntary context switches：可能存在 CPU 竞争；
* 大量 major faults：可能有文件读取、内存压力或 mmap 冷页。

``strace`` 会改变被跟踪程序的时序，高系统调用频率程序尤其明显。优先短时、按 PID、
按系统调用类别跟踪，避免长时间记录全部调用。


perf 与火焰图
~~~~~~~~~~~~~

常见采样流程：

.. code-block:: console

   $ sudo perf record -F 99 -g -p PID -- sleep 30
   $ sudo perf report
   $ sudo perf script > perf.script
   $ stackcollapse-perf.pl perf.script > perf.folded
   $ flamegraph.pl perf.folded > flamegraph.svg

为了获得完整调用栈，应用应保留调试符号，并选择适合的 unwind 方式。编译原生程序时
通常建议加入 ``-g``，必要时使用 ``-fno-omit-frame-pointer``。JIT 语言需要对应的
perf map、运行时 profiler 或专用导出方式。

火焰图中横向宽度表示采样占比，不表示时间顺序；纵向表示调用栈深度。优化时应结合
业务吞吐、尾延迟和锁/IO 数据，不能只根据最宽函数做改动。


/proc 进程接口
~~~~~~~~~~~~~~

.. code-block:: console

   $ cat /proc/PID/status
   $ cat /proc/PID/sched
   $ cat /proc/PID/io
   $ cat /proc/PID/limits
   $ cat /proc/PID/cgroup
   $ ls -l /proc/PID/fd
   $ cat /proc/PID/smaps_rollup

这些文件是瞬时视图，读取不同文件之间状态可能已经变化。``/proc/PID/io`` 的统计还
可能受权限、缓存和子进程行为影响。


eBPF、bpftrace 与 BCC
~~~~~~~~~~~~~~~~~~~~~

eBPF 适合在低侵入条件下观察内核和应用事件，但仍需控制探针频率、聚合维度和 map
大小。优先使用 tracepoint 等稳定接口；依赖内核函数名的 kprobe 可能随版本变化。

常见 BCC 工具：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 工具
     - 用途
   * - ``execsnoop``
     - 跟踪新进程执行
   * - ``opensnoop``
     - 跟踪文件打开
   * - ``runqlat``
     - CPU 运行队列延迟分布
   * - ``offcputime``
     - 统计线程阻塞在 CPU 之外的调用栈
   * - ``biolatency``
     - 块 IO 延迟直方图
   * - ``biosnoop``
     - 按请求查看块 IO
   * - ``tcplife``
     - 跟踪 TCP 连接生命周期
   * - ``tcpconnect``
     - 跟踪主动 TCP 连接

简单 bpftrace 示例：

.. code-block:: console

   $ sudo bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm] = count(); }'
   $ sudo bpftrace -e 'tracepoint:sched:sched_process_exit { @[comm] = count(); }'
   $ sudo bpftrace -l 'tracepoint:block:*'

生产使用前，应确认内核 BTF、锁定内存限制、安全策略和工具版本。不要复制来源不明的
BPF 脚本直接在核心生产节点运行。


容器、cgroup v2 与 systemd
--------------------------

cgroup v2 常用接口
~~~~~~~~~~~~~~~~~~

先通过 ``/proc/PID/cgroup`` 找到进程所属 cgroup，再读取对应目录。以下路径中的
``CGROUP`` 需要替换为实际路径。

.. code-block:: console

   $ cat /proc/PID/cgroup
   $ cat /sys/fs/cgroup/CGROUP/cpu.stat
   $ cat /sys/fs/cgroup/CGROUP/cpu.max
   $ cat /sys/fs/cgroup/CGROUP/memory.current
   $ cat /sys/fs/cgroup/CGROUP/memory.events
   $ cat /sys/fs/cgroup/CGROUP/memory.pressure
   $ cat /sys/fs/cgroup/CGROUP/io.stat
   $ cat /sys/fs/cgroup/CGROUP/pids.current

重要语义：

* ``cpu.max``：CPU 带宽配额和周期；
* ``cpu.stat`` 中 ``nr_throttled``、``throttled_usec``：CPU 被限流情况；
* ``memory.current``：当前 cgroup 内存占用；
* ``memory.high``：内存软上限，超过后触发回收与分配限流；
* ``memory.max``：硬上限，可能触发 cgroup 内 OOM；
* ``memory.events``：high、max、oom、oom_kill 等事件计数；
* ``io.stat``：按设备记录读写字节数和 IO 次数；
* ``pids.max``：进程/线程数量限制。

容器内看到的 CPU 数、load、内存和宿主机视图可能不同。排查时必须同时看容器 cgroup
和宿主机物理资源。


systemd 资源控制
~~~~~~~~~~~~~~~~

.. code-block:: console

   $ systemctl status SERVICE
   $ systemctl show SERVICE -p CPUQuotaPerSecUSec -p MemoryCurrent -p MemoryMax
   $ systemctl show SERVICE -p IOReadBytes -p IOWriteBytes -p TasksCurrent
   $ systemd-cgtop
   $ systemd-cgls

常用 unit 配置包括 ``CPUQuota=``、``CPUWeight=``、``MemoryHigh=``、
``MemoryMax=``、``IOWeight=`` 和 ``TasksMax=``。修改后应使用
``systemctl show`` 和 cgroup 文件确认实际生效值。


容器性能排查
~~~~~~~~~~~~

#. 先判断问题只发生在单个容器，还是宿主机所有负载。
#. 检查 CPU throttling、memory events、IO 限制和 PID 限制。
#. 检查同节点其他容器是否产生 noisy neighbor 干扰。
#. 检查 overlay 文件系统、日志驱动、sidecar 和虚拟网络开销。
#. 在宿主机命名空间中定位真实 PID，再使用 perf/BPF。
#. 对 Kubernetes 同时检查 requests、limits、QoS、eviction 和节点压力。

CPU 使用率看似不高但应用延迟上升时，``cpu.stat`` 中持续增加的 throttling 是常见
原因。简单提高 limit 之前，还应检查线程数、峰值模式和调度周期。


典型故障场景
------------

CPU 持续 100%
~~~~~~~~~~~~~

.. code-block:: text

   top/pidstat 找进程与线程
     → mpstat 判断单核还是全机
     → perf top 快速看热点
     → perf record + 火焰图保留证据
     → 判断计算、系统调用、锁、自旋或中断
     → 修改后用同一负载验证


load 很高但 CPU 空闲
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   vmstat 查看 r/b 和 wa
     → ps 查 D 状态与 wchan
     → PSI 判断 IO/内存压力
     → iostat 和 pidstat 找设备与进程
     → dmesg/journalctl 检查设备、文件系统、NFS


内存持续增长或 OOM
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   free + memory PSI 确认系统压力
     → ps/smaps_rollup 找进程与内存类型
     → memory.events 判断 cgroup OOM
     → 检查 heap、mmap、线程、FD、缓存和共享内存
     → 使用运行时 profiler、ASan、heaptrack 或 Massif


磁盘延迟突然升高
~~~~~~~~~~~~~~~~

.. code-block:: text

   iostat 确认设备、方向、await 和队列
     → pidstat/iotop 找责任进程
     → lsof/strace 找文件和 IO 模式
     → biolatency/biosnoop 看延迟分布
     → 检查 cgroup、云盘额度、文件系统和内核日志


接口超时或网络吞吐下降
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   先比较应用处理时间和网络时间
     → ss 查看队列、RTT 和重传
     → nstat 查看 TCP 错误
     → ip/ethtool 查看网卡 drop
     → mpstat/softirqs 查看 CPU 处理能力
     → 必要时限制条件抓包并与远端联合分析


修改与验证
----------

建立可复现基准
~~~~~~~~~~~~~~

至少记录：

* 测试时间、版本、配置、硬件和内核；
* 输入数据、并发、预热时间和测试时长；
* 吞吐、平均延迟、P95/P99/P999 和错误率；
* CPU、内存、IO、网络、PSI 和 cgroup 指标；
* 环境噪声，例如其他任务、频率调节、NUMA 和虚拟化 steal。

基准测试应覆盖稳定状态，避免只比较冷启动或缓存完全不同的两次结果。


优化优先级
~~~~~~~~~~

#. 修复错误、重试风暴、OOM、设备故障和资源限制配置。
#. 消除不必要的工作，例如重复计算、无效 IO、过量日志和短连接。
#. 降低排队与锁竞争，控制并发和队列长度。
#. 改善算法、数据结构、批处理和缓存命中率。
#. 最后再考虑编译参数、绑核、内核参数和硬件扩容。

内核参数通常是全局策略，不应作为第一反应。修改 sysctl 前必须理解其语义、作用范围、
回滚方式和对其他工作负载的影响。


常用工具速查
------------

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - 目标
     - 首选工具
     - 深入工具
   * - 全局概览
     - ``uptime``、``vmstat``、``sar``
     - PSI、监控时序数据
   * - CPU
     - ``mpstat``、``pidstat``
     - ``perf``、``runqlat``、``offcputime``
   * - 内存
     - ``free``、``ps``、``smaps_rollup``
     - ``numastat``、heap profiler、ASan
   * - 磁盘 IO
     - ``iostat``、``pidstat``、``iotop``
     - ``biolatency``、``biosnoop``、``strace``
   * - 网络
     - ``ip``、``ss``、``nstat``
     - ``ethtool``、``tcpdump``、BPF
   * - 应用
     - ``time``、``strace``、``lsof``
     - ``perf``、火焰图、语言运行时 profiler
   * - 容器
     - cgroup v2 文件、``systemd-cgtop``
     - 宿主机 perf/BPF、编排平台指标


过时工具说明
------------

以下工具可能仍存在于旧系统或历史文档中，但不建议作为新排障流程的主线：

* OProfile：现代通用采样优先使用 ``perf``；
* ``gprof``：需要编译插桩，适用范围有限，优先考虑采样 profiler；
* ``procinfo``、memprof：维护和发行版支持有限；
* ``ifconfig``、``netstat``：优先使用 ``ip``、``ss``；
* iptraf、etherape：可视化或交互观察可用，但自动化诊断优先使用现代指标和抓包工具；
* ``svctm``：已从现代 sysstat 中移除，不应用于判断设备服务时间。


参考资料
--------

* `Linux kernel documentation <https://docs.kernel.org/>`__
* `PSI - Pressure Stall Information <https://docs.kernel.org/accounting/psi.html>`__
* `Control Group v2 <https://docs.kernel.org/admin-guide/cgroup-v2.html>`__
* `perf Wiki <https://perfwiki.github.io/main/>`__
* `Brendan Gregg: Linux Performance <https://www.brendangregg.com/linuxperf.html>`__
* `BPF Performance Tools <https://github.com/iovisor/bcc>`__
* `bpftrace documentation <https://bpftrace.org/docs/>`__
* `sysstat project <https://github.com/sysstat/sysstat>`__
* `procps-ng project <https://gitlab.com/procps-ng/procps>`__
* `systemd.resource-control <https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html>`__
* `iproute2 manual pages <https://man7.org/linux/man-pages/man8/ss.8.html>`__
* `FlameGraph <https://github.com/brendangregg/FlameGraph>`__
