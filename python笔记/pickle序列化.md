# pickle
python的pickle模块实现了基本的数据序列和反序列化。  
通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储。  
通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。 

和json序列化区别：json只能把常用的数据类型序列化（列表、字典、列表、字符串、数字、），比如日期格式、类对象！josn就不行了。而pickle可以序列化所有的数据类型，包括类，函数都可以序列化     

	pickle.dump(obj, file, protocol=None)
		obj表示将要封装的对象;
		file表示obj要写入的文件对象，file必须以二进制可写模式打开，即“wb”
		protocol表示告知pickler使用的协议，支持的协议有0,1,2,3，默认的协议是添加在Python 3中的协议3  
	pickle.load(file,*,fix_imports=True, encoding="ASCII", errors="strict")
		file必须以二进制可读模式打开，即“rb”，其他都为可选参数
	pickle.dumps(obj)
		以字节对象形式返回封装的对象，不需要写入文件中
	pickle.loads(bytes_object)
		 从字节对象中读取被封装的对象，并返回
	pickle模块可能出现三种异常:
		PickleError：封装和拆封时出现的异常类，继承自Exception
		PicklingError: 遇到不可封装的对象时出现的异常，继承自PickleError
		UnPicklingError: 拆封对象过程中出现的异常，继承自PickleError

示例：

	import pickle,pprint

	# 使用pickle模块将数据对象保存到文件
	data1 = {'a': [1, 2.0, 3, 4+6j],
	         'b': ('string', u'Unicode string'),
	         'c': None}
	
	selfref_list = [1, 2, 3]
	selfref_list.append(selfref_list)
	
	output = open('data.pkl', 'wb')
	
	# Pickle dictionary using protocol 0.
	pickle.dump(data1, output)
	
	# Pickle the list using the highest protocol available.
	pickle.dump(selfref_list, output, -1)
	
	output.close()

	#使用pickle模块从文件中重构python对象
	pkl_file = open('data.pkl', 'rb')
	
	data1 = pickle.load(pkl_file)
	pprint.pprint(data1)
	
	data2 = pickle.load(pkl_file)
	pprint.pprint(data2)
	
	pkl_file.close()


	
