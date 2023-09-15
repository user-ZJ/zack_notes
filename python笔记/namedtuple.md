# nametuple(命名元组)  
namedtuples是不可变的，高效的和优雅的。    
因为元组的局限性：不能为元组内部的数据进行命名，所以往往我们并不知道一个元组所要表达的意义，所以在这里引入了 collections.namedtuple 这个工厂函数，来构造一个带字段名的元组。  
命名元组的实例和普通元组消耗的内存一样多，因为字段名都被存在对应的类里面。  
namedtuple 对象的定义如以下格式：  

	collections.namedtuple(typename, field_names, verbose=False, rename=False)  
	typename：元组名称
	field_names: 元组中元素的名称
	rename: 如果元素名称中含有 python 的关键字，则必须设置为 rename=True
	verbose: 默认就好

使用示例:  

	import collections

	# 两种方法来给 namedtuple 定义方法名
	#User = collections.namedtuple('User', ['name', 'age', 'id'])
	User = collections.namedtuple('User', 'name age id')
	user = User('tester', '22', '464643123')
	
	print(user)   


参考：http://www.runoob.com/note/25726







# namedtupled  
namedtuples是不可变的，高效的和优雅的。  
namedtupled是一个轻量级的包装器，用于从嵌套的dicts，lists，json和yaml递归创建namedtuples。  

https://github.com/brennv/namedtupled  