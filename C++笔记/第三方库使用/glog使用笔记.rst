glog使用笔记
===============

示例
------------

.. code-block:: cpp

    //cat example.cc
    #include <glog/logging.h>

    int main(int argc, char* argv[]) {
        // Initialize Google’s logging library.
        google::InitGoogleLogging(argv[0]);

        // ...
        LOG(INFO) << "Found " << num_cookies << " cookies";
    }


.. code-block:: cmake

    include(FetchContent)
    set(FETCHCONTENT_QUIET off)
    set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/thirdpart)

    # third_party: glog
    FetchContent_Declare(glog
    URL      https://github.com/google/glog/archive/refs/tags/v0.5.0.zip
    URL_HASH SHA256=21bc744fb7f2fa701ee8db339ded7dce4f975d0d55837a97be7d46e8382dea5a
    )
    set(WITH_CUSTOM_PREFIX ON)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(glog)
    include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

    add_executable(example example.cpp)
    target_link_libraries(example glog)


执行：

.. code-block:: shell

    export GLOG_logtostderr=1
    export GLOG_v=2
    $ ./example

glog库说明
-------------------
glog 定义了一系列宏，这些宏简化了许多常见的日志记录任务。您可以按严重性级别记录消息，
从命令行控制日志记录行为，基于条件进行日志记录，在不满足预期条件时中止程序，
引入自己的详细日志记录级别，自定义附加到日志消息的前缀等。


日志级别
-------------
* INFO
* WARNING
* ERROR
* FATAL:打印日志后中断程序
* DFATAL:debug模式下的日志

日志文件命名
----------------------
glog日志文件命名方式：/tmp/<program name>.<hostname>.<user name>.log.<severity level>.<date>.<time>.<pid>

示例：/tmp/hello_world.example.com.hamaji.log.INFO.20080709-222411.10474


环境变量
-----------------
* GLOG_logtostderr=1(bool,默认为false)：将日志文件打印到 `stderr` 而不是打印到日志文件
* GLOG_stderrthreshold=N(int,默认为2) ：将日志级别大于等于N的日志拷贝到 `stderr` ;INFO,WARNING,ERROR,FATAL级别分别为0,1,2,3
* GLOG_minloglevel=N(int,默认为0): 进记录日志级别大于等于N的日志 ;INFO,WARNING,ERROR,FATAL级别分别为0,1,2,3
* GLOG_log_dir="/logs"(string,默认为""):指定日志文件存储路径
* GLOG_v=N(int,默认为0) ： 显示所有VLOG(m)的日志，当m小于等于N时


代码中修改Flags
-----------------

1. 在代码中添加
   
.. code-block:: cpp

    #include <glog/logging.h>
    
    int main(int argc, char* argv[]) {
        // Initialize Google’s logging library.
        FLAGS_log_dir=/path/to/your/logdir
        google::InitGoogleLogging(argv[0]);
    
        // ...
        LOG(INFO) << "Found " << num_cookies << " cookies";
    }

2. 命令行参数,需要再本地安装glog
   
.. code-block:: shell

   ./your_application --log_dir=/some/log/directory
   

3. 环境变量，在未安装glog的时候使用
   
.. code-block:: shell

   GLOG_log_dir=/some/log/directory ./your_application
   

条件日志
---------------------
.. code-block:: cpp

    // 当num_cookies > 10时打印日志
    LOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";
    // 第1st, 11th, 21st次执行输出该日志，google::COUNTER用来验证第几次执行
    LOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie";
    // IF 和 EVERY_N的联合使用
    LOG_IF_EVERY_N(INFO, (size > 1024), 10) << "Got the " << google::COUNTER
                                        << "th big cookie";
    // 值输出前N次的日志
    LOG_FIRST_N(INFO, 20) << "Got the " << google::COUNTER << "th cookie";
    // 每10ms输出一次日志
    LOG_EVERY_T(INFO, 0.01) << "Got a cookie";
    // Debug模式下输出日志
    DLOG(INFO) << "Found cookies";
    DLOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";
    DLOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie";


CHECK宏
------------------
CHECK宏用来检查程序运行条件，当条件不满足时，终止程序。
和assert不同，CHECK不受NDEBUG控制，所以总是会被执行。

.. code-block:: cpp

    // condition为false输出日志
    CHECK(condition)<<" failed";
    // 检查是否为空指针
    CHECK_NOTNULL(variable) << " failed";
    // val1 == val2
    CHECK_EQ(val1, val2)<< " failed";
    // val1 != val2
    CHECK_NE(val1, val2)<< " failed"; 
    // val1 > val2
    CHECK_GT(val1, val2)<< " failed"; 
    // val1 >= val2
    CHECK_GE(val1, val2)<< " failed"; 
    // val1 < val2
    CHECK_LT(val1, val2)<< " failed"; 
    // val1 <= val2
    CHECK_LE(val1, val2)<< " failed";
    // 比较C语言风格的字符串
    CHECK_STREQ(Foo().c_str(), Bar().c_str())<< " failed";
    CHECK_STRNE(Foo().c_str(), Bar().c_str())<< " failed";
    // 比较C语言风格的字符串,忽略大小写
    CHECK_STRCASEEQ(Foo().c_str(), Bar().c_str())<< " failed";
    CHECK_STRCASENE(Foo().c_str(), Bar().c_str())<< " failed";
    // 比较两个float类型是否相等，在一个小的margin内
    CHECK_DOUBLE_EQ(f1,f2)<<" failed";
    CHECK_NEAR(f1,f2)<<" failed";

VLOG
-----------------
* VLOG(2)<<"xxxx";
* VLOG_IS_ON(n);判断VLOG是否输出
* VLOG_IF
* VLOG_EVERY_N
* VLOG_IF_EVERY_N

  
日志文件自动清理
-------------------------

.. code-block:: cpp

    #include <glog/logging.h>

    int main(int argc, char* argv[]) {
        // Initialize Google’s logging library.
        FLAGS_log_dir=/path/to/your/logdir
        google::EnableLogCleaner(3); // keep your logs for 3 days
        google::InitGoogleLogging(argv[0]);

        // ...
        LOG(INFO) << "Found " << num_cookies << " cookies";
    }


异常信号处理
-----------------------
glog库提供了一个方便的信号处理程序，当程序在某些信号（如SIGSEGV）上崩溃时，它将转储有用的信息

信号处理程序可以通过google::InstallFailureSignalHandler()安装。


将INFO日志flush到文件
-----------------------------------------
在代码中添加flush

.. code-block:: cpp

    google::FlushLogFiles(google::GLOG_INFO);

https://stackoverflow.com/questions/35572073/logging-with-glog-is-not-working-properly
