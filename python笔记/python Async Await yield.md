# Python Async/Await/yield

* 普通函数
		# type(function) is types.FunctionType
		def function():
	    	return 1
* 生成器函数
		# type(generator()) is types.GeneratorType
		def generator():
		    yield 1
* 异步函数（协程）
		# type(async_function()) is types.CoroutineType
		async def async_function():
		    return 1
* 异步生成器
		# type(async_generator()) is types.AsyncGeneratorType
		async def async_generator():
		    yield 1

## async
async修饰将普通函数和生成器函数包装成异步函数和异步生成器。  
异步函数(协程)需要通过其他方式来驱动，因此可以使用这个协程对象的send方法给协程发送一个  
print(async_function().send(None))  
不幸的是，如果通过上面的调用会抛出一个异常：  
StopIteration: 1    
因为生成器/协程在正常返回退出时会抛出一个StopIteration异常，而原来的返回值会存放在StopIteration对象的value属性中  

	try:
	    async_function().send(None)
	except StopIteration as e:
	    print(e.value)
	# 1

通过以下捕获可以获取协程真正的返回值：  

	def run(coroutine):
	    try:
	        coroutine.send(None)
	    except StopIteration as e:
	        return e.value
  

## await
await语法只能出现在通过async修饰的函数中，否则会报SyntaxError错误
在协程函数中，可以通过await语法来挂起自身的协程，并等待另一个协程完成直到返回结果  
await后面的对象需要是一个Awaitable，或者实现了相关的协议(简单来说，await后面接async修饰的函数)  

## yield
yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器  

* 可迭代对象：当你建立了一个列表，你可以逐项地读取这个列表，这叫做一个可迭代对象  

		mylist = [x*x for x in range(3)]
		for i in mylist :
			print(i)

* 生成器：生成器是可以迭代的，但是你**只可以读取它一次**，因为它并不把所有的值放在内存中，它是实时地生成数据  

		mygenerator = (x*x for x in range(3))
		for i in mygenerator :
			print(i)  
看起来除了把 [] 换成 () 外没什么不同。但是，你不可以再次使用 for i in mygenerator , 因为生成器只能被迭代一次  
* yield关键字  
yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器  

		def createGenerator() :
		   mylist = range(3)
		   for i in mylist :
		       yield i*i
		
		mygenerator = createGenerator() # create a generator
		print(mygenerator) # mygenerator is an object!
		#<generator object createGenerator at 0xb7555c34>
		for i in mygenerator:
			print(i)  
当你调用这个函数的时候，函数内部的代码并不立马执行 ，这个函数只是返回一个生成器对象，当你使用for进行迭代的时候才真正执行。  