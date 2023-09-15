# pandas


## Series 
Series 是pandas两大数据结构中（DataFrame，Series）的一种   
Series是一种类似于一维数组的对象，它由一组数据（各种NumPy数据类型）以及一组与之相关的数据标签（即索引）组成     
Series对象本质上是一个NumPy的数组，因此NumPy的数组处理函数可以直接对Series进行处理。但是Series除了可以使用位置作为下标存取元素之外，还可以使用标签下标存取元素，这一点和字典相似。每个Series对象实际上都由两个数组组成：   
index: 它是从NumPy数组继承的Index对象，保存标签信息。   
values: 保存值的NumPy数组。     

	创建series
	Series([1,2,'3',4,'a'])  Series创建后会自动生成索引，默认从0开始
	series_4.index=['a','b','c']  指定和修改索引
	Series(data, index)  
	Series({'a':1,'b':2,'c':3})  通过字典创建Series
	增加Series元素：append()
	删除元素：drop(index)
	修改元素：series[index]=data
	通过索引查单值:series[index]
	通过索引序列查多值:series[[index1,index2]]
	通过布尔类型索引筛选:series[series>2]
	通过位置切片和标签切片查询数据:series[index1:index2]   

### Serise遍历
	for index,value in s.iteritems():
		print index,value

## Dataframe
定义 二维、表格型的数组结构，可存储许多不同类型的数据，且每个轴都有标签，可当作一个series的字典；  

### 遍历dataframe

	indexs = top_stocks.index
	columns = top_stocks.columns 
	for index in indexs:
	    topns = top_stocks.loc[index].nlargest(top_n)
	    top_stocks.loc[index] = np.zeros(len(columns),dtype=np.int64)
	    for comp in topns.index:
	        top_stocks.loc[index,comp] = 1

### 两个dataframe差集

```python
import pandas as pd
data_a={'state':[1,1,2],'pop':['a','b','c']}
data_b={'state':[1,2,3],'pop':['b','c','d']}
a=pd.DataFrame(data_a)
b=pd.DataFrame(data_b)
c = a.append(b)
c = c.drop_duplicates(subset=['state','pop'],keep=False)
```

### dataframe新增一列做统计

```python
data['len'] = data['label'].map(lambada x:len(x))
```

### 筛选

```python
data = data[['id','label']][(data.len>2)&(data.len)<8]
data = data[['id','label']][(data.len<2)|(data.len)>8]
```



## 读入csv文件

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

重要参数：

**sep**：分隔符，默认为“,”

**header**:int, list of int, default ‘infer’，推断列名使用

**names**：自动定义列名列表

**dtype**：指定列数据类型

index_col=False强制pandas不使用第一列作为索引

```python
data = pd.read_csv('diamonds.csv',dtype=object,index_col=0)
data.head()
data = pd.read_csv('diamonds.csv',dtype={'carat': np.float64,'depth': np.float64,'table':np.float64})
data.dtypes
# 丢弃原来标题，重命名标题
pd.read_csv(StringIO(data),names=['foo','bar','baz'],header=0)
```

## 写入csv文件

```python
dt.to_csv('C:/Users/think/Desktop/Result1.csv',float_format='%.2f') #保留两位小数
dt.to_csv('C:/Users/think/Desktop/Result.csv',columns=['name']) #保存索引列和name列
dt.to_csv('C:/Users/think/Desktop/Result.csv',header=0) #不保存列名
dt.to_csv('C:/Users/think/Desktop/Result1.csv',index=0) #不保存行索引
```



## date_range
	返回固定频率DatetimeIndex。
	start : str or datetime-like, optional
	Left bound for generating dates.表示日期的起点
	
	end : str or datetime-like, optional
	Right bound for generating dates.表示日期的终点
	
	periods : integer, optional
	Number of periods to generate.
	表示你要从这个函数产生多少个日期索引值；如果是None的话，那么start和end必须不能为None。
	
	freq : str or DateOffset, default ‘D’
	Frequency strings can have multiples, e.g. ‘5H’. See here for a list of frequency aliases.
	指定计时单位，例如“5H”，表示每隔5个小时计算一次
	
	tz : str or tzinfo, optional
	Time zone name for returning localized DatetimeIndex, for example ‘Asia/Hong_Kong’. By default, the resulting DatetimeIndex is timezone-naive.
	返回本地化DatetimeIndex的时区名称，例如“Asia / Hong_Kong”。
	
	normalize : bool, default False
	Normalize start/end dates to midnight before generating date range.
	在生成日期范围之前将开始/结束日期标准化为午夜0点
	
	name : str, default None
	Name of the resulting DatetimeIndex.
	给返回的时间索引指定一个名字
	
	closed : {None, ‘left’, ‘right’}, optional
	Make the interval closed with respect to the given frequency to the ‘left’, ‘right’, or both sides (None, the default).
	表示start和end这个区间端点是否包含在区间内，可以有三个值，’left’表示左闭右开区间，’right’表示左开右闭区间，None表示两边都是闭区间
	
	pd.date_range(start='1/1/2018', periods=8)
	DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
	           '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
	          dtype='datetime64[ns]', freq='D')



## resample
	对常规时间序列数据重新采样和频率转换的便捷的方法
	freq:
	B	business day frequency
	C	custom business day frequency
	D	calendar day frequency
	W	weekly frequency
	M	month end frequency
	SM	semi-month end frequency (15th and end of month)
	BM	business month end frequency
	CBM	custom business month end frequency
	MS	month start frequency
	SMS	semi-month start frequency (1st and 15th)
	BMS	business month start frequency
	CBMS	custom business month start frequency
	Q	quarter end frequency
	BQ	business quarter end frequency
	QS	quarter start frequency
	BQS	business quarter start frequency
	A, Y	year end frequency
	BA, BY	business year end frequency
	AS, YS	year start frequency
	BAS, BYS	business year start frequency
	BH	business hour frequency
	H	hourly frequency
	T, min	minutely frequency
	S	secondly frequency
	L, ms	milliseconds
	U, us	microseconds
	N	nanoseconds
	
	close_prices.resample('M').first() 每月的第一个数据  
	close_prices.resample('M').last()  每月的最后一个数据
	close_prices.resample('M').max()   每月的最大数据
	close_prices.resample('M').min()   每月的最小数据

## shift
	shift函数是对数据进行移动的操作
	df.shift(1) 数据向右移动一个单位
	df.shift(-1) 数据向左移动一个单位

## loc
	按标签或布尔数组访问一组行和列。  
	df.loc[0,'name']  
	df.loc[0:2, ['name','age']]

## iloc
	按标签索引值访问一组行和列。
	df.iloc[:16]
	df.iloc[[6, 7, 13, 14]]

## nlargest(n, columns, keep='first')
	返回按列降序排列的前n行。

## groupby
	根据一个或多个键（可以是函数、数组或DataFrame列名）拆分pandas对象，计算分组摘要统计，如计数、平均值、标准差，或用户自定义函数。
	df.groupby(['Animal']).mean()

## DataFrame 增删改查
https://blog.csdn.net/zhangchuang601/article/details/79583551

## DataFrame.median
	返回行/列数据的中位数

## DataFrame.pivot
	返回由给定行/列重新生成的DataFrame。
	price_df.pivot(index='date', columns='ticker', values='open')

## copy
	copy(deep=True)
	复制dataframe内容，deep参数表示是直接引用还是完全拷贝到一块新的内存空间  

## timestamp

```python
# int 转datatime
t = pd.to_datetime(df["ts"])	
time_data = pd.concat([t,t.dt.hour,t.dt.day,t.dt.weekofyear,t.dt.month,t.dt.year,t.dt.weekday],axis=1)
```




​	