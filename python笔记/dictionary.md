# 字典

## 字典创建方法
1. 传统的文字表达式

		d={'name':'Allen','age':21,'gender':'male'}

2. 动态分配键值

		d={}
		d['name']='Allen'

3. 字典键值表

		c = dict(name='Allen', age=14, gender='male')
		>> {'gender': 'male', 'name': 'Allen', 'age': 14}

4. 字典键值元组表

		e=dict([('name','Allen'),('age',21),('gender','male')])

5. 所有键的值都相同或者赋予初始值

		f=dict.fromkeys(['height','weight'],'normal')
		>> {'weight': 'normal', 'height': 'normal'}

## 增、删、改、查
	dict['Age'] = 8 # 更新
	dict['School'] = "RUNOOB" # 添加
	dict['Age']  #查
	del dict['Name'] #删


## 字典的value转化为list
		list(data.values())


## 字典遍历
	for key,value in a.items()
		print(key,value)
	for key in a.keys()：
		print(key,a[key]