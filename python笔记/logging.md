# logging

## python中日志使用
| 需要执行的任务 | 打印日志选择 |
| ------------- | ----------- |
| 对于命令行或程序的应用，结果显示在控制台 | print() |
| 在对程序的普通操作发生时提交事件报告 | logging.info(),logging.debug() |
| 提出一个警告信息基于一个特殊的运行时事件 | logging.warning() |
| 对一个特殊的运行时事件报告错误 | 引发异常 |
| 报告错误而不引发异常 | logging.error(), logging.exception(),logging.critical() |

## 基本使用示例
	import logging
	logging.basicConfig(filename='example.log',level=logging.DEBUG)
	#修改filemode重写日志，而不是默认的追加日志
	logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
	# 设置日志格式
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
	# 在日志中显示日期和时间
	logging.basicConfig(format='%(asctime)s %(message)s')
	# 控制日期/时间的格式
	logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	logging.debug('This message should go to the log file')
	logging.info('So should this')
	logging.warning('And this, too')

## 进阶
记录器（Logger）、处理程序（handler）、过滤器(filter)和格式化程序(format)。

### Logger
1. Logger.setLevel() :指定记录器将处理的最低严重性日志消息，其中 debug 是最低内置严重性级别， critical 是最高内置严重性级别。 例如，如果严重性级别为 INFO ，则记录器将仅处理 INFO 、 WARNING 、 ERROR 和 CRITICAL 消息，并将忽略 DEBUG 消息。  
2. Logger.addHandler() 和 Logger.removeHandler()：从记录器对象中添加和删除处理程序对象。  
3. Logger.addFilter() 和 Logger.removeFilter()：添加或移除记录器对象中的过滤器


	logger = logging.getLogger(__name__)
getLogger() 返回对具有指定名称的记录器实例的引用（如果已提供），或者如果没有则返回 root 。名称是以句点分隔的层次结构。多次调用 getLogger() 具有相同的名称将返回对同一记录器对象的引用。在分层列表中较低的记录器是列表中较高的记录器的子项。例如，给定一个名为 foo 的记录器，名称为 foo.bar 、 foo.bar.baz 和 foo.bam 的记录器都是 foo 子项。

### Handler 
Handler 对象负责将适当的日志消息分派给处理程序的指定目标。如：应用程序可能希望将所有日志消息发送到日志文件（FileHandler），将错误或更高的所有日志消息发送到标准输出（StreamHandler），以及将所有关键消息发送至一个邮件地址。

1. setLevel() ：指定将被分派到适当目标的最低严重性，和Logger中不同，记录器中设置的级别确定将传递给其处理程序的消息的严重性。每个处理程序中设置的级别确定处理程序将发送哪些消息。  
2. setFormatter() ：选择一个该处理程序使用的 Formatter 对象
3. addFilter() 和 removeFilter() ：分别在处理程序上配置和取消配置过滤器对象  


	handler = logging.StreamHandler()
handler列表：https://docs.python.org/zh-cn/3/howto/logging.html#useful-handlers

### Formatter
格式化程序对象配置日志消息的最终顺序、结构和内容。  

	fmt = "%(asctime)s [%(threadName)-10.10s] [%(levelname)-4.4s]  %(message)s"
	formatter = logging.Formatter(fmt)
	logger.addHandler(handler)

### filter
日志过滤器

### 示例
	def initialize_logger(output_dir):
	    logger = logging.getLogger()
	    logger.setLevel(logging.DEBUG)
	    fmt = "%(asctime)s [%(threadName)-10.10s] [%(levelname)-4.4s]  %(message)s"
	
	    handler = logging.StreamHandler()
	    handler.setLevel(logging.INFO)
	    formatter = logging.Formatter(fmt)
	    handler.setFormatter(formatter)
	    logger.addHandler(handler)
	
	    if output_dir is not None:
	        handler = logging.FileHandler(os.path.join(output_dir, "log_info.txt"))
	        handler.setLevel(logging.INFO)
	        formatter = logging.Formatter(fmt)
	        handler.setFormatter(formatter)
	        logger.addHandler(handler)
	
	        # create debug file handler and set level to debug
	        handler = logging.FileHandler(os.path.join(output_dir, "log_debug.txt"))
	        handler.setLevel(logging.DEBUG)
	        formatter = logging.Formatter(fmt)
	        handler.setFormatter(formatter)
	        logger.addHandler(handler)


参考：
https://docs.python.org/zh-cn/3/howto/logging.html

