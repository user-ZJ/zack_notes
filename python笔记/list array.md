# list和array区别
1. list中的数据类不必相同的，而array的中的类型必须全部相同
2. list中的数据类型保存的是数据的存放的地址，简单的说就是指针，并非数据  
3. list中的元素之间用逗号隔开，而数组中的数据是没有逗号隔开的  
4. range(start, end, step)，返回一个list对象，只能创建int型list；arange(start, end, step)，与range()类似，也不含终止值。但是返回一个array对象。需要导入numpy模块（from numpy import*），并且arange可以使用float型数据 
5. array相当于Java的数组，定义时需要指定长度，list相当于链表，可变长度 


# list和array相互转化
	array = np.array(list, dtype = int)
	list = array.tolist()

# list

## 列表排序reverse sort sorted
	# reverse
	x = [1,5,2,3,4]
	print(x.reverse())
	# [4, 3, 2, 5, 1]
	 
	# sort
	a = [5,7,6,3,4,1,2]
	b = a.sort()
	print(a)
	print(b)
	# [1, 2, 3, 4, 5, 6, 7]
	# None
	 
	# sorted
	a = [5,7,6,3,4,1,2]
	b = sorted(a)
	print(a)
	print(b)
	# [5, 7, 6, 3, 4, 1, 2]
	# [1, 2, 3, 4, 5, 6, 7]

## list遍历
	foo = ['a', 'b', 'c', 'd', 'e']
	for d in foo:
		print(d)
	#遍历时加序号
	for i,d in enumerate(list)：
		print(i,d)
	#设置遍历开始初始位置，只改变了起始序号
	for i,d in enumerate(list,2)：
		print(i,d)

## 随机取 list 中的元素
	#随机取一组数据
	import random
	a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	b = random.sample(a, 5)
	>[2, 7, 4, 5, 8]
	random.choices(a,k=2)
	
	#随机取一个数据
	import random
	foo = ['a', 'b', 'c', 'd', 'e']
	print(random.choice(foo))
	
	foo = ['a', 'b', 'c', 'd', 'e']
	from random import randrange
	random_index = randrange(0,len(foo))
	print foo[random_index]

## list拼接

```python
aList = [123, 'xyz', 'zara', 'abc', 123];
bList = [2009, 'manni'];
aList.extend(bList)
```

## list去重

```python
ids = [1,4,3,3,4,2,3,4,5,6,1]
ids = list(set(ids))
```



# array

## 取array中指定元素

```python 
import numpy as np
input=np.array([1,2,3,4,5,6,7,8,9,10])
index=np.array([0,1,0,0,0,0,1,0,0,1])
output = input[index.astype(np.bool)]
output = input[np.where(index)[0]]
```

## 生成指定mask

```python
for i in range(16):
    mask = "{:0>4b}".format(i)
    mask = np.array(list(mask),dtype=int)
    print(mask)
```



## array增加列和删除列

	# 增加一列
	np.expand_dims(tensor, axis=0)
	 
	# 删除一列
	dataset=[[1,2,3],[2,3,4],[4,5,6]]
	dataset = np.delete(dataset, -1, axis=1)
	print(dataset)
	#array([[1, 2],
	       [2, 3],
	       [4, 5]])
	 
	# 删除多列
	arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
	arr = np.delete(arr, [1,2], axis=1)
	print(arr)
	#array([[ 1,  4],
	       [ 5,  8],
	       
	#删除内容为空的维度
	data = np.random.random((1,512))
	print(data.shape)
	#data = np.squeeze(data)   #两种方法均可
	data = data.squeeze()
	print(data.shape) 

## numpy自动生成数组

	# 1. np.arange()：通过指定开始值，终值和步长来创建表示等差数列的一维数组，不包含终值
	>>> np.arange(10)
	array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	>>> np.arange(0,1,0.1)
	array([ 0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	 
	# 2. np.linspace():通过指定开始值，终值和元素个数来创建表示等差数列的一维数组，含有参数endpoint布尔值，默认为True表示包含终值，设定为False表示不包含终值。
	>>> np.linspace(0,1,10)
	array([ 0.    , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
	    0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.    ])
	>>> np.linspace(0,1,10,endpoint = False)
	array([ 0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	 
	# 3. np.logspace():通过指定开始值，终值和元素个数来创建表示等等比列的一维数组，基数默认为10，可以使用base参数指定基数
	>>> np.logspace(0,4,5)
	array([ 1.00000000e+00,  1.00000000e+01,  1.00000000e+02,
	     1.00000000e+03,  1.00000000e+04])
	>>> np.logspace(0,3,5,base = 2)
	array([ 1. , 1.68179283, 2.82842712, 4.75682846, 8. ])
	起点为2^0 = 1，终点为2^3 = 8,一共按照等比数列生成5个点，这样公比q = 2^(3/4)
	 
	# 4. np.zeros(),np.ones(),np.empty(),创建指定的形状和类型数组，np.empty只分配数组所使用的内存，不对数据初始化起作用
	# 5. np.full():生成初始化为指定值的数组
	>>> np.empty((2,3),np.int32)
	array([[ 8078112, 37431728, 8078112],
	    [47828800, 47828712,    10]])
	>>> np.ones(4)
	array([ 1., 1., 1., 1.])
	>>> np.ones((2,3))
	array([[ 1., 1., 1.],
	    [ 1., 1., 1.]])
	>>> np.ones(4,dtype = np.bool)
	array([ True, True, True, True], dtype=bool)
	>>> np.zeros(4,dtype = np.bool)
	array([False, False, False, False], dtype=bool)
	>>> np.zeros(4)
	array([ 0., 0., 0., 0.])
	>> np.full(4,np.pi)
	array([ 3.14159265, 3.14159265, 3.14159265, 3.14159265])
	>>> np.full((2,3),np.pi)
	array([[ 3.14159265, 3.14159265, 3.14159265],
	    [ 3.14159265, 3.14159265, 3.14159265]])
	 
	# 6. np.zeros_like()，np.ones_like():创建于参数形状相同的数组即np.zeros_like(a)与np.zeros(a.shape,dtype = a.type)相同
	>>> a = np.arange(10).reshape(2,5)
	>>> np.zeros_like(a)
	array([[0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0]])
	 
	# 7. np.fromfunction():从指定的函数中生成数组，第一个参数是函数名称，第二个参数是数组形状
	>>> np.fromfunction(lambda a,b:a == b,(3,3))
	array([[ True, False, False],
	    [False, True, False],
	    [False, False, True]], dtype=bool)
	>>> np.fromfunction(lambda i:i%7 +1,(10,))
	array([ 1., 2., 3., 4., 5., 6., 7., 1., 2., 3.])
	 
	# 8. xrange():用法与 range 完全相同，所不同的是生成的不是一个数组，而是一个生成器。
	xrange(start, stop, step)
	from six.moves import xrange
	>> xrange(8)
	xrange(8)
	>>> list(xrange(8))
	[0, 1, 2, 3, 4, 5, 6, 7]
	>>> list(xrange(3,5))
	[3, 4]
	>>> list(xrange(0,6,2))
	[0, 2, 4]
	
	# 9. random方法随机生成多维数组
	np.random.random((145,20))  # 生成0-1之间的矩阵
	np.random.randint(0, 10, (4,3)) #生成0-10之间的4x3矩阵
	np.ones((3, 2))  #生成全1的矩阵
	np.zeros((2, 3)) #生成全0的矩阵
	np.eye(3)#3维单位矩阵
	y = np.array([4, 5, 6])
	np.diag(y)#以y为主对角线创建矩阵

## 矩阵拼接
	p = np.ones([2, 3], int)
	np.hstack([p, 2*p])#水平拼接
	np.vstack([p, 2*p])#竖直拼接
	横向拼接：hstack, vstack, concatenate等。横向拼接的意思是指，拼接不会产生更高的维度。
	扩维拼接：dstack, stack等。拼接后会产生更高的维度，两个(2, 2)的二维矩阵拼接会产生(2, 2, 2)的三维矩阵

## transpose(转置)

```python
x = np.arange(4).reshape((2,2))
np.transpose(x)
x = np.ones((1, 2, 3))
np.transpose(x, (1, 0, 2)).shape
x.transpose((1,0,2)).shape
```



## 打印数组中所有内容

	print array时array中间是省略号没有输出全部
	在开头加入：
	import numpy as np
	np.set_printoptions(threshold=np.inf)

## numpy.ndarray.strides
	遍历数组时每个维度中的字节数
	如：
	aaa = np.random.randint(0, 10, (20,10))
	print(aaa.strides)
	> (40, 4)  
	解释为每个int数据占4个Byte,没行10个数据，所以就为(40, 4)

## 对数组进行块分割
	numpy.lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)
	使用给定的形状和步幅创建数组视图
	
	shape:分割后的数组shape
	strides:shape各位对应的偏移量，将原始数组x的strides修改为进行分割
	
	例如：
	# 步幅为3从20x10的数组中取4个10x10的数组
	aaa = np.random.randint(0, 10, (20,10))
	np.lib.stride_tricks.as_strided(aaa,(4,10,10),(120,40,4),writeable=False)
	# 将9x9数组分割为9个3x3的九宫格
	aaa = np.random.randint(0, 10, (9,9))
	# 大行之间隔 27 个元素，大列之间隔 3 个元素
	# 小行之间隔 9 个元素，小列之间隔 1 个元素
	#strides = aaa.itemsize * np.array([27, 3, 9, 1]) —> (108,12,36,4)
	np.lib.stride_tricks.as_strided(aaa,(3,3,3,3),(108,12,36,4))

## numpy array 保存meta数据
```python 
t = numpy.array([1,2,3,4,5])
t.tofile("test.meta")
```
