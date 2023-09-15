# argparse

1. import argparse    导入模块  
2. parser = argparse.ArgumentParser（）    创建一个解析对象  
3. parser.add_argument()    向该对象中添加你要关注的命令行参数和选项  
4. parser.parse_args()    进行解析  

## ArgumentParser  
	argparse.ArgumentParser(prog=None, 
	                        usage=None, 
	                        epilog=None, 
	                        parents=[], 
	                        formatter_class=argparse.HelpFormatter, 
	                        prefix_chars='-',                            
	                        fromfile_prefix_chars=None,              
	                        argument_default=None,
	                        conflict_handler='error', 
	                        add_help=True)

	prog：默认情况下, ArgumentParser对象根据sys.argv[0]的值(不包括路径名)生成帮助信息中的程序名，也可以人为指定一个程序名，通过 %(prog)s可以引用程序名  
	usage:默认情况下，ArgumentParser对象可以根据参数自动生成用法信息,也可以指定usage  
	description:用于展示程序的简要介绍信息，通常包括:这个程序可以做什么、怎么做。在帮助信息中 description位于用法信息与参数说明之间  
	epilog:与description类似，程序的额外描述信息，位于参数说明之后  
	prefix_chars:一般情况下，我们使用’-‘作为选项前缀,ArgumentParser也支持自定义选项前缀  
	add_help:是否禁用-h –help选项  

## add_argument
	argumentParser.add_argument(name or flags...[,action][,nargs][,const][,default]
                           [,type][,choices][,required][,help][,metavar][,dest])  
	name 或 flags:指定一个可选参数或位置参数  
		>>> parser.add_argument('-f', '--foo')  #指定一个可选参数
		>>> parser.add_argument('bar')          #指定一个位置参数  
	action:指定应该如何处理命令行参数，预置的操作有以下几种: 
		’store’ 仅仅保存参数值，为action默认值
		’store_const’ 与store基本一致，但store_const只保存const关键字指定的值,parser.add_argument('--foo', action='store_const', const=42)  
		’store_true’或’store_false’ 与store_const一致，只保存True和False
		’append’ 将相同参数的不同值保存在一个list中  
		’count’ 统计该参数出现的次数
		’help’ 输出程序的帮助信息
		’version’ 输出程序版本信息
	nargs:通过指定 nargs可以将多个参数与一个action相关联  
	default:如果参数可以缺省，default指定命令行参数不存在时的参数值
	type:默认情况下，ArgumentParser对象将命令行参数保存为字符串。但通常命令行参数应该被解释为另一种类型，如 float或int。
		通过指定type,可以对命令行参数执行类型检查和类型转换。通用的内置类型和函数可以直接用作type参数的值:
		parser = argparse.ArgumentParser()
		>>> parser.add_argument('foo', type=int)
		>>> parser.add_argument('bar', type=open)
		>>> parser.parse_args('2 temp.txt'.split())
		Namespace(bar=<_io.TextIOWrapper name='temp.txt' encoding='UTF-8'>, foo=2)  
	choices:将命令行参数的值限定在一个范围内，超出范围则报错   
	required:指定命令行参数是否必需  
	dest:dest允许自定义ArgumentParser的参数属性名称  

## 实例

	import argparse
	parser = argparse.ArgumentParser(description='Search some files')
	
	parser.add_argument(dest='filenames',metavar='filename', nargs='*')
	parser.add_argument('-p', '--pat',metavar='pattern', required=True,
	                    dest='patterns', action='append',
	                    help='text pattern to search for')
	parser.add_argument('-v', dest='verbose', action='store_true',
	                    help='verbose mode')
	parser.add_argument('-o', dest='outfile', action='store',
	                    help='output file')
	parser.add_argument('--speed', dest='speed', action='store',
	                    choices={'slow','fast'}, default='slow',
	                    help='search speed')
	
	args = parser.parse_args()
	
	# Output the collected arguments
	print(args.filenames)
	print(args.patterns)
	print(args.verbose)
	print(args.outfile)
	print(args.speed)
	

	result1：
	python3 search.py -h
	usage: search.py [-h] [-p pattern] [-v] [-o OUTFILE] [--speed {slow,fast}]
	                 [filename [filename ...]]
	
	Search some files
	
	positional arguments:
	  filename
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -p pattern, --pat pattern
	                        text pattern to search for
	  -v                    verbose mode
	  -o OUTFILE            output file
	  --speed {slow,fast}   search speed
	

	result2：
	python3 search.py foo.txt bar.txt
	usage: search.py [-h] -p pattern [-v] [-o OUTFILE] [--speed {fast,slow}]
	                 [filename [filename ...]]
	search.py: error: the following arguments are required: -p/--pat


	result3：
	python3 search.py -v -p spam --pat=eggs foo.txt bar.txt
	filenames = ['foo.txt', 'bar.txt']
	patterns  = ['spam', 'eggs']
	verbose   = True
	outfile   = None
	speed     = slow

	result4：
	python3 search.py -v -p spam --pat=eggs foo.txt bar.txt -o results
	filenames = ['foo.txt', 'bar.txt']
	patterns  = ['spam', 'eggs']
	verbose   = True
	outfile   = results
	speed     = slow

	result5：
	python3 search.py -v -p spam --pat=eggs foo.txt bar.txt -o results \
             --speed=fast
	filenames = ['foo.txt', 'bar.txt']
	patterns  = ['spam', 'eggs']
	verbose   = True
	outfile   = results
	speed     = fast

	

参考：  
https://blog.csdn.net/guoyajie1990/article/details/76739977  
https://python3-cookbook.readthedocs.io/zh_CN/latest/c13/p03_parsing_command_line_options.html