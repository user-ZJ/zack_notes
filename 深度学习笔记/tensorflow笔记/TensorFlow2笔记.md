TensorFlow1是符号式编程，先创建计算图后运行的编程方式

tensorflow2 支持动态图优先模式，在计算时可以同时获得计算图与数值结果，可以在代码中调试并实时打印数据，搭建网络也像搭积木一样层层堆叠，符合软件开发思维

## 1 tensorflow2安装及使用

```shell
# 使用国内清华源安装TensorFlow CPU 版本
pip install -U tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
# 使用清华源安装TensorFlow GPU 版本,"-U"参数指定如果已安装此包，则执行升级命令
pip install -U tensorflow-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
# 测试
import tensorflow as tf
tf.test.is_gpu_available()
tf.__version__

# 创建名为tf2 的虚拟环境，并根据预设环境名tensorflow-gpu
# 自动安装CUDA,cuDNN,TensorFlow GPU 等
conda create -n tf2 tensorflow-gpu
# 激活tf2 虚拟环境
conda activate tf2
```

```python
# 设置GPU显存使用方式
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # 设置GPU为增长式占用
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True) 
  except RuntimeError as e:
    # 打印异常
    print(e)
```

```py
with tf.device('/cpu:0'):
	cpu_a=tf.random.normal([1,n])
	cpu_b=tf.random.normal([n,1])
	print(cpu_a.device,cpu_b,device)
	c=tf.matmul(cpu_a,cpu_b)
with tf.device('/gpu:0'):
	gpu_a=tf.random.normal([1,n])
	gpu_b=tf.random.normal([n,1])
	print(gpu_a.device,gpu_b.device)
	c=matmul(gpu_a,gpu_b)
```

```py
#自动梯度
import tensorflow as tf
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)
with tf.GradientTape as tape: #构建梯度环境
	tape.watch([w]) #将w加入梯度跟踪列表
	#构建计算过程，函数表达式
	y = a * w**2 + b * w + c
#自动求导
[dy_dw] = tape.gradient(y,[w])
print(dy_dw)
```

## 2 实例

### 2.1 简单手写数字实例

```python
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,datasets

# 设置GPU显存使用方式
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # 设置GPU为增长式占用
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True) 
  except RuntimeError as e:
    # 打印异常
    print(e)

(x,y),(x_val,y_val)=datasets.mnist.load_data() #加载mnist数据集
x = 2*tf.convert_to_tensor(x,dtype=tf.float32)/255.-1 #标准化数据，缩放到[-1,1]
y = tf.convert_to_tensor(y,dtype=tf.int32) 
y = tf.one_hot(y,depth=10)
print(x.shape,y.shape)
train_dataset = tf.data.from_tensor_slices((x,y)) #构建数据集对象
train_dataset = train_dataset.batch(32).repeat(30) #批量训练,batch_size=32,epoch=30
#构建模型
model = keras.Sequential(
	layers.Desen(256,activation='relu'),
	layers.Desen(128,activation='relu'),
	layers.Desen(10))
model.build(input_shape=(4, 28*28))
model.summary()

optimizer = optimizers.SGD(lr=0.01) #创建优化器
acc_meter = metrics.Accuracy()  # 创建准确率测量器

for step, (x,y) in enumerate(db):
	with tf.GradientTape() as tape: #构建梯度记录环境
		x = tf.reshape(x,(-1,28*28))
		out = model(x)
		y_onehot = tf.onehot(y,depth=10)
		loss = tf.square(out-y_onehot)
		loss = tf.reduce_sum(loss)/x.shape(0) 
	acc_meter.update_state(tf.argmax(out, axis=1), y)  # 记录采样的数据
	grads = tape.gradient(loss,model.trainable_variables)  #计算梯度
	optimizer.apply_gradient(zip(grads,model.trainable_variables)) #更新参数
	if step % 200==0:
        # 打印统计期间的平均准确率
        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy()) 
        acc_meter.reset_states()
```



## 3 数据类型

TensorFlow中的基本数据类型包含**数值类型**，**字符串类型**和**布尔类型**

### 3.1 数值类型

数值类型可以区分为：

**标量（Scalar）**：单个的实数；

**向量**：n个实数的有序集合；

**矩阵**：n行m列实数的有序集合。

在TensorFlow中为了表达方便，一般把标量、向量、矩阵统称为**张量**，不做区分，需要根据张量的维度数或形状自行判断

```python
a = tf.constant(1.2)
b = tf.constant([1.2,3.3])
c = tf.constant([[1,2],[3,4]])
c = tf.constant([[[1,2],[3,4]],
				[[5,6],[7,8]]])
loss = float(a)  #通过float()函数将张量转换为普通数值
c.numpy() #张量转换为python数据
```

> 张量的numpy()方法可以放回Numpy.array类型的数据，方便导出数据到系统其它模块

### 3.2 字符串类型

在表示图片数据是，可以先记录图片的路径字符串，在通过预处理函数根据路径读取图片张量

tf.string模块中提供了常见的字符串类型的工具函数，如lower(),join(),length(),split()等

```py
a = tf.constant('hello world')
```

### 3.3 布尔类型

为了方便表达比较运算操作的结果，TensorFlow支持布尔类型的张量

```py
a = tf.constant(True)
b = tf.constant([True,False]) #创建布尔类型向量
```

TensorFlow的布尔类型和Python语言的布尔类型并不等价，不能通用

```py
a = tf.constant(True)
a is True  #结果为False
a == True #仅数值比较，结果为<tf.Tensor: id=8,shape=(),dtype=bool,numpy=True>
```

### 3.4 类型转换

tf中常用的精度类型有tf.int16、tf.int32、tf.int64、tf.float16、tf.float32、tf.float64 等，其中tf.float64 即为tf.double

对于大部分深度学习算法，一般使用tf.int32 和tf.float32 可满足大部分场合的运算精度要求，部分对精度要求较高的算法，如强化学习某些算法，可以选择使用tf.int64 和tf.float64 精度保存张量。

```python
a = tf.constant(np.pi, dtype=tf.float16) # 创建tf.float16 低精度张量
a.dtype
tf.cast(a, tf.double) # 转换为高精度张量
a = tf.constant([True, False])
tf.cast(a, tf.int32) # 布尔类型转整型
a = tf.constant([-1, 0, 1, 2])
tf.cast(a, tf.bool) # 整型转布尔类型,非0 数字都视为True
```

### 3.5 待优化张量（Variable）

为了**区分需要计算梯度信息的张量和不需要计算梯度信息的张量**，TensorFlow增加了一种专门的数据类型来支持梯度信息的记录：tf.Variable。tf.Variable类型在普通的张量类型基础上添加了name，trainable等属性来支持计算图的构建。由于梯度运算会消耗大连的计算资源，而且会自动更新相关参数，对于不需要优化的张量，如神经网络的输入X，不需要通过tf.Variable封装，对于需要计算梯度并优化的张量，如神经网络层的W和b，需要通过tf.Variable包裹以便TensorFlow跟踪相关梯度信息。

```
#普通张量转换为待优化张量
a = tf.constant([-1,0,1,2])
aa = tf.Variable(a)
print(aa.name,aa.trainable)
# 直接创建Variable 张量
a = tf.Variable([[1,2],[3,4]]) 
```

name和trainable属性是Variable特有的属性，name属性用于命名计算图中的变量，这套命名体系是TensorFlow内部维护的，一般不需要用户关注name属性，trainable属性表征当前张量是否需要被优化，创建Variable 对象时是**默认启用优化标志**，可以设置trainable=False 来设置张量不需要优化。  

**待优化张量可视为普通张量的特殊类型，普通张量其实也可以通过GradientTape.watch()方法临时加入跟踪梯度信息的列表，从而支持自动求导功能**。    

## 4. 数据操作

### 4.1 创建张量

```
tf.constant() tf.convert_to_tensor() 自动的把Numpy 数组或者Python列表数据类型转化为Tensor 类型
tf.zeros()和tf.ones()即可创建任意形状，且内容全0 或全1 的张量  
tf.zeros_like, tf.ones_like 可以方便地新建与某个张量shape 一致，且内容为全0 或全1 的张量。  
tf.fill(shape, value)可以创建全为自定义数值value 的张量，形状由shape 参数指定。   
tf.random.normal(shape, mean=0.0, stddev=1.0)可以创建形状为shape，均值为mean，标准差为stddev 的正态分布𝒩(mean, stddev2)  
tf.random.truncated_normal([784, 256], stddev=0.1)使用截断的正太分布初始化权值张量
tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32)可以创建采样自
[minval, maxval)区间的均匀分布的张量,如果需要均匀采样整形类型的数据，必须指定采样区间的最大值maxval 参数，同时指定数据类型为tf.int*型  
eg:tf.random.uniform([2,2]) # 创建采样自[0,1)均匀分布的矩阵
tf.range(limit, delta=1)可以创建[0, limit)之间，步长为delta 的整型序列，不包含limit 本身
tf.range(start, limit, delta=1)可以创建[start, limit)，步长为delta 的序列，不包含limit 本身  
```

### 4.2 索引和切片

通过索引与切片操作可以提取张量的部分数据  

**索引**：在 TensorFlow 中，支持标准下标索引方式，也支持通过逗号分隔索引号的索引方式

```python
x = tf.random.normal([4,32,32,3]) # 创建4D 张量
x[0] # 取第 1 张图片的数据
x[0][1] #取第 1 张图片的第2 行
x[0][1][2] #取第 1 张图片，第2 行，第3 列的数据
x[2][1][0][1] #取第 3 张图片，第2 行，第1 列的像素，B 通道(第2 个通道)颜色强度值
x[1,9,2]  # 取第 2 张图片，第10 行，第3 列的数据
```

**切片**：通过start: end: step切片方式可以方便地提取一段数据，其中start 为开始读取位置的索
引，end 为结束读取位置的索引(不包含end 位)，step 为采样步长  

```python
x = tf.random.normal([4,32,32,3]) # 创建4D 张量
x[1:3] #读取第2，3张图片
# start: end: step切片方式有很多简写方式，其中start、end、step 3 个参数可以根据需要选择性地省略，全部省略时即为::，表示从最开始读取到最末尾，步长为1
# 从第一个元素读取时start 可以省略，即start=0 是可以省略，取到最后一个元素时end 可以省略，步长为1 时step 可以省略  
x[0,::] # 表示读取第1 张图片的所有行，其中::表示在行维度上读取所有行，它等价于x[0]的写法
#为了更加简洁，::可以简写为单个冒号:
x[:,0:28:2,0:28:2,:] #读取所有图片、隔行采样、隔列采样，、读取所有通道数据，相当于在图片的高宽上各缩放至原来的50%
#step 可以为负数，考虑最特殊的一种例子，当step = −1时，start: end: −1表示从start 开始，逆序读取至end 结束(不包含end)，索引号𝑒𝑛𝑑 ≤ 𝑠𝑡𝑎𝑟𝑡
x[::-1] # 逆序全部元素
x[0,::-2,::-2] # 行、列逆序间隔采样
# 当张量的维度数量较多时，不需要采样的维度一般用单冒号:表示采样所有元素，此时有可能出现大量的:出现。继续考虑[4,32,32,3]的图片张量，当需要读取G 通道上的数据时，前面所有维度全部提取，此时需要写为 
x[:,:,:,1]
# 为了避免出现像 [: , : , : ,1]这样过多冒号的情况，可以使用⋯符号表示取多个维度上所有的数据，其中维度的数量需根据规则自动推断：当切片方式出现⋯符号时，⋯符号左边的维度将自动对齐到最左边，⋯符号右边的维度将自动对齐到最右边，此时系统再自动推断⋯符号代表的维度数量
x[0:2,...,1:] # 高宽维度全部采集
x[2:,...] # 高、宽、通道维度全部采集，等价于x[2:]
x[...,:2] # 所有样本，所有高、宽的前2 个通道
```

### 4.3 维度变换

基本的维度变换操作函数包含了改变视图reshape、插入新维度expand_dims，删除维度squeeze、交换维度transpose、复制数据tile 等函数

**reshape**:reshape只会改变数据视图（即数据读取方式），不会改变数据的存储，在 TensorFlow 中，可以通过张量的ndim 和shape 成员属性获得张量的维度数和形状。  

参数−1表示当前轴上长度需要根据张量总元素不变的法则自动推导，从而方便用户书写

**expand_dims**：增加一个长度为1 的维度相当于给原有的数据添加一个新维度的概念，维度长度为1，故数据并不需要改变，仅仅是改变数据的理解方式，因此它其实可以理解为改变视图的一种特殊方式  

```python
x = tf.random.uniform([28,28],maxval=10,dtype=tf.int32)
#通过tf.expand_dims(x, axis)可在指定的axis 轴前可以插入一个新的维度
x = tf.expand_dims(x,axis=2)
```

tf.expand_dims 的axis 为正时，表示在当前维度之前插入一个新维度；为负时，表示从后往前数，在所在位置插入一个维度

**squeeze**：是增加维度的逆操作，与增加维度一样，删除维度只能删除长度为1 的维度，也不会改变张量的存储。

通过tf.squeeze(x, axis)函数，axis 参数为待删除的维度的索引号，如果不指定维度参数axis，即tf.squeeze(x)，那么它会默认删除所有长度为1 的维度

**transpose**：改变视图、增删维度都不会影响张量的存储，交换维度(Transpose)会调整存储顺序。交换维度操作是非常常见的，比如在TensorFlow 中，图片张量的默认存储格式是通道后行格式：[𝑏, ℎ, , 𝑐]，但是部分库的图片格式是通道先行格式：[𝑏, 𝑐, ℎ, ]，因此需要完成[𝑏, ℎ, , 𝑐]到[𝑏, 𝑐, ℎ, ]维度交换运算，此时若简单的使用改变视图函数reshape，则新视图的存储方式需要改变，因此使用改变视图函数是不合法的。  

tf.transpose(x, perm)函数完成维度交换操作，其中参数perm表示新维度的顺序List

```python
x = tf.random.normal([2,32,32,3])
tf.transpose(x,perm=[0,3,1,2]) # 交换维度
```

需要注意的是，通过tf.transpose 完成维度交换后，张量的存储顺序已经改变，视图也随之改变，后续的所有操作必须基于新的存续顺序和视图进行  

**tile** :当通过增加维度操作插入新维度后，可能希望在新的维度上面复制若干份数据，满足后续算法的格式要求。可以通过tf.tile(x, multiples)函数完成数据在指定维度上的复制操作，multiples 分别指定了每个维度上面的复制倍数，对应位置为1 表明不复制，为2 表明新长度为原来长度的2 倍，即数据复制一份，以此类推

```python
b = tf.constant([1,2])
b = tf.expand_dims(b, axis=0)
b = tf.tile(b, multiples=[2,1]) # 样本维度上复制一份
```

需要注意的是，tf.tile 会创建一个新的张量来保存复制后的张量，由于复制操作涉及大量数据的读写IO 运算，计算代价相对较高。  

**Broadcasting**:Broadcasting 称为广播机制(或自动扩展机制)，它是一种轻量级的张量复制手段，在逻辑上扩展张量数据的形状，但是只会在需要时才会执行实际存储复制操作。对于大部分场景，Broadcasting 机制都能通过优化手段避免实际复制数据而完成逻辑运算，从而相对于tf.tile 函数，减少了大量计算代价。  

对于所有长度为1 的维度，Broadcasting 的效果和tf.tile 一样，都能在此维度上逻辑复制数据若干份，区别在于tf.tile 会创建一个新的张量，执行复制IO 操作，并保存复制后的张量数据，而Broadcasting 并不会立即复制数据，它会在逻辑上改变张量的形状，使得视图上变成了复制后的形状。Broadcasting 会通过深度学习框架的优化手段避免实际复制数据而完成逻辑运算，至于怎么实现的用户不必关心，对于用户来说，Broadcasting 和tf.tile 复
制的最终效果是一样的，操作对用户透明，但是Broadcasting 机制节省了大量计算资源，建议在运算过程中尽可能地利用Broadcasting 机制提高计算效率

tf.broadcast_to(x, new_shape)

操作符+在遇到shape 不一致的2 个张量时，会自动考虑将2 个张量自动扩展到一致的shape，然后再调用tf.add 完成张量相加运算

### 4.4 数学运算

**加、减、乘、除**是最基本的数学运算，分别通过tf.add, tf.subtract, tf.multiply, tf.divide函数实现，TensorFlow 已经重载了+、− 、∗ 、/运算符，一般推荐直接使用运算符来完成加、减、乘、除运算

整除和余除也是常见的运算之一，分别通过//和%运算符实现。

**乘方运算**：

tf.pow(x, a)可以方便地完成𝑦 = x^𝑎的乘方运算，也可以通过运算符**实现 x∗∗ 𝑎运算  

设置指数为1/𝑎形式，即可实现根号运算。  

```python
x=tf.constant([1.,4.,9.])
x**(0.5) #平方根
```

对于常见的平方和平方根运算，可以使用tf.square(x)和tf.sqrt(x)实现

**指数和对数运算**：

tf.pow(a, x)或者**运算符也可以方便地实现指数运算𝑎^𝑥

tf.exp(x)实现自然指数e^𝑥

tf.math.log(x)实现自然对数log_e(x)

如果希望计算其它底数的对数，可以根据对数的换底公式间接地通过tf.math.log(x)实现。如log_10(x)可以写为

tf.math.log(x)/tf.math.log(10.) # 换底公式

**矩阵相乘运算**：

通过@运算符可以方便的实现矩阵相乘，还可以通过tf.matmul(a, b)函数实现。

需要注意的是，TensorFlow 中的矩阵相乘可以使用批量方式，也就是张量𝑨和𝑩的维度数可以大于2。当张量𝑨和𝑩维度数大于2 时，TensorFlow 会选择𝑨和𝑩的最后两个维度进行矩阵相乘，前面所有的维度都视作Batch 维度。

```python
a = tf.random.normal([4,3,28,32])
b = tf.random.normal([4,3,32,2])
a@b # 批量形式的矩阵相乘,shape=(4, 3, 28, 2)
#矩阵相乘函数同样支持自动Broadcasting 机制
a = tf.random.normal([4,28,32])
b = tf.random.normal([32,16])
tf.matmul(a,b) #先自动扩展再矩阵相乘，shape=(4,28,16)
```

### 4.5 合并和分割

**合并**：张量的合并可以使用拼接(Concatenate)和堆叠(Stack)操作实现，拼接操作并不会产生新
的维度，仅在现有的维度上合并，而堆叠会创建新维度。

> tf.concat(tensors, axis)函数拼接张量其中参数tensors 保存了所有需要合并的张量List，axis 参数指定需要合并的维度索引,从语法上来说，拼接合并操作可以在任意的维度上进行，唯一的约束是非合并维度的长度必须一致
>
> tf.stack(tensors, axis)可以堆叠方式合并多个张量，通过tensors 列表表示，参数axis 指定新维度插入的位置，axis 的用法与tf.expand_dims 的一致，当axis ≥ 0时，在axis之前插入；当axis < 0时，在axis 之后插入新维度。需要所有待合并的张量shape 完全一致才可合并

```python
a = tf.random.normal([4,35,8]) # 模拟成绩册A
b = tf.random.normal([6,35,8]) # 模拟成绩册B
tf.concat([a,b],axis=0) # 拼接合并成绩册
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.stack([a,b],axis=0) # 堆叠合并为2 个班级，班级维度插入在最前
tf.stack([a,b],axis=-1) # 在末尾插入班级维度
```

**分割**：将一个张量分拆为多个张量

> tf.split(x, num_or_size_splits, axis)可以完成张量的分割操作
>
> x 参数：待分割张量。
> num_or_size_splits 参数：切割方案。当num_or_size_splits 为单个数值时，如10，表示等长切割为10 份；当num_or_size_splits 为List 时，List 的每个元素表示每份的长度，如[2,4,2,2]表示切割为4 份，每份的长度依次是2、4、2、2。
> axis 参数：指定分割的维度索引号。
>
> tf.unstack(x,axis),在某个维度上全部按长度为1 的方式分割,这种方式是tf.split 的一种特殊情况，切割长度固定为1，只需要指定切割维度
> 的索引号即可

```python
x = tf.random.normal([10,35,8])
# 等长切割为10 份
result = tf.split(x, num_or_size_splits=10, axis=0)
# 自定义长度的切割，切割为4 份，返回4 个张量的列表result
result = tf.split(x, num_or_size_splits=[4,2,2,2] ,axis=0)
result = tf.unstack(x,axis=0) # Unstack 为长度为1 的张量
```

### 4.6 数据统计

**向量范数**：向量范数(Vector Norm)是表征向量“长度”的一种度量方法，它可以推广到张量上。在神经网络中，常用来表示张量的权值大小，梯度大小等。常用的向量范数有

* L1 范数，定义为向量𝒙的所有元素绝对值之和

* L2 范数，定义为向量𝒙的所有元素的平方和，再开根号

* ∞ −范数，定义为向量𝒙的所有元素绝对值的最大值

对于矩阵和张量，同样可以利用向量范数的计算公式，等价于将矩阵和张量打平成向量后计算

tf.norm(x, ord)求解张量的L1、L2、∞等范数，其中参数ord 指定为1、2 时计算L1、L2 范数，指定为np.inf 时计算∞ −范数

```python
x = tf.ones([2,2])
tf.norm(x,ord=1) # 计算L1 范数
tf.norm(x,ord=2) # 计算L2 范数
tf.norm(x,ord=np.inf) # 计算∞范数
```

**最值、均值、和**:通过 tf.reduce_max、tf.reduce_min、tf.reduce_mean、tf.reduce_sum 函数可以求解张量在某个维度上的最大、最小、均值、和，也可以求全局最大、最小、均值、和信息   

```python
x = tf.random.normal([4,10]) # 模型生成概率
tf.reduce_max(x,axis=1) # 统计概率维度上的最大值
tf.reduce_min(x,axis=1) # 统计概率维度上的最小值
tf.reduce_mean(x,axis=1) # 统计概率维度上的均值
tf.reduce_sum(x,axis=-1) # 求最后一个维度的和
```

除了希望获取张量的最值信息，还希望获得最值所在的位置索引号，例如分类任务的标签预测，就需要知道概率最大值所在的位置索引号，一般把这个位置索引号作为预测类别  

tf.argmax(x, axis)和tf.argmin(x, axis)可以求解在axis 轴上，x 的最大值、最小值所在的索引号

```python
pred = tf.argmax(out, axis=1) # 选取概率最大的位置
pred = tf.argmin(out, axis=1) # 选取概率最小的位置
```

### 4.7 张量比较

通过tf.equal(a, b)(或tf.math.equal(a,b)，两者等价)函数可以比较这2 个张量是否相等

tf.equal()函数返回布尔类型的张量比较结果，只需要统计张量中True 元素的个数，即可知道预测正确的个数

tf.math.greater : a>b

tf.math.less : a<b

tf.math.greater_equal : a>=b

tf.math.less_equal : a<=b

tf.math.not_equal : a!=b

tf.math.is_nan : a=nan

```python
out = tf.random.normal([100,10])
out = tf.nn.softmax(out, axis=1) # 输出转换为概率
pred = tf.argmax(out, axis=1) # 计算预测值
y = tf.random.uniform([100],dtype=tf.int64,maxval=10)
out = tf.equal(pred,y) # 预测值与真实值比较，返回布尔类型的张量
out = tf.cast(out, dtype=tf.float32) # 布尔型转int 型
correct = tf.reduce_sum(out) # 统计True 的个数
```

### 4.8 填充与复制

tf.pad(x, paddings)函数实现填充操作,参数paddings 是包含了多个[Left Padding, Right Padding]的嵌套方案List，如[[0,0], [2,1], [1,2]]表示第一个维度不填充，第二个维度左边(起始处)填充两个单元，右边(结束处)填充一个单元，第三个维度左边填充一个单元，右边填充两个单元。

```python
total_words = 10000 # 设定词汇量大小
max_review_len = 80 # 最大句子长度
embedding_len = 100 # 词向量长度
# 加载IMDB 数据集
(x_train, y_train), (x_test, y_test) =
keras.datasets.imdb.load_data(num_words=total_words)
# 将句子填充或截断到相同长度，设置为末尾填充和末尾截断方式
x_train = keras.preprocessing.sequence.pad_sequences(x_train,
maxlen=max_review_len,truncating='post',padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test,
maxlen=max_review_len,truncating='post',padding='post')
print(x_train.shape, x_test.shape) # 打印等长的句子张量形状
```

tf.tile 函数可以在任意维度将数据重复复制多份，如shape 为[4,32,32,3]的数据，复制方案为multiples=[2,3,3,1]，即通道数据不复制，高和宽方向分别复制2 份，图片数再复制1 份

```python
x = tf.random.normal([4,32,32,3])
tf.tile(x,[2,3,3,1]) # 数据复制
```

### 4.9数据限幅

tf.maximum(x, a)实现数据的下限幅，即𝑥 ∈ [𝑎, +∞)；

tf.minimum(x, a)实现数据的上限幅，即𝑥 ∈ (−∞, 𝑎]，

tf.clip_by_value(x,2,7)实现上下限幅

tf.clip_by_norm(a,5) 按L2范数方式裁剪

tf.clip_by_global_norm  缩放整体网络梯度𝑾的L2范数

```
tf.minimum(tf.maximum(x,2),7) # 限幅为2~7
tf.clip_by_value(x,2,7) # 限幅为2~7
```

## ## 4.10 高级操作

**tf.gather**：根据索引号收集数据，适合索引号没有规则的场合，其中索引号可以乱序排列，此时收集的数据也是对应顺序

```python
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32) # 成绩册张量
tf.gather(x,[0,1],axis=0) # 在班级维度收集第1~2 号班级成绩册
tf.gather(x,[0,3,8,11,12,26],axis=1) # 收集第1,4,9,12,13,27 号同学成绩
tf.gather(x,[2,4],axis=2) # 第3，5 科目的成绩
```

**tf.gather_nd**：通过指定每次采样点的多维坐标来实现采样多个点的目的

```python
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32) # 成绩册张量
tf.gather_nd(x,[[1,1],[2,2],[3,3]])
tf.gather_nd(x,[[1,1,2],[2,2,3],[3,3,4]]) # 根据多维度坐标收集数据
```

**tf.boolean_mask**：通过给定掩码(Mask)的方式进行采样,注意掩码的长度必须与对应维度的长度一致，

```python
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32) # 成绩册张量
tf.boolean_mask(x,mask=[True, False,False,True],axis=0)
x = tf.random.uniform([2,3,8],maxval=100,dtype=tf.int32)
tf.gather_nd(x,[[0,0],[0,1],[1,1],[1,2]]) # 多维坐标采集
tf.boolean_mask(x,[[True,True,False],[False,True,True]]) # 多维掩码采样
```

**tf.where**：tf.where(cond, a, b)操作可以根据cond 条件的真假从参数𝑨或𝑩中读取数据，

当参数a=b=None 时，即a 和b 参数不指定，tf.where 会返回cond 张量中所有True 的元素的索引坐标。那么这有什么用途呢？考虑一个场景，我们需要提取张量中所有正数的数据和索引。首先构造张量a，并通过比较运算得到所有正数的位置掩码

```python
a = tf.ones([3,3]) # 构造a 为全1 矩阵
b = tf.zeros([3,3]) # 构造b 为全0 矩阵
cond =tf.constant([[True,False,False],[False,True,False],[True,True,False]])
tf.where(cond,a,b) # 根据条件从a,b 中采样
tf.where(cond) # 获取cond 中为True 的元素索引

x = tf.random.normal([3,3])
mask=x>0
indices=tf.where(mask) # 提取所有大于0 的元素索引
tf.gather_nd(x,indices) # 提取正数的元素值
tf.boolean_mask(x,mask) # 通过掩码提取正数的元素值
```

**scatter_nd**：tf.scatter_nd(indices, updates, shape)函数可以高效地刷新张量的部分数据，但是这个函数只能在全0 的白板张量上面执行刷新操作，因此可能需要结合其它操作来实现现有张量的数据刷新功能。  

白板的形状通过shape 参数表示，需要刷新的数据索引号通过indices 表示，新数据为updates。根据indices 给出的索引位置将updates 中新的数据依次写入白板中，并返回更新后的结果张量  

```python
# 构造需要刷新数据的位置参数，即为4、3、1 和7 号位置
indices = tf.constant([[4], [3], [1], [7]])
# 构造需要写入的数据，4 号位写入4.4,3 号位写入3.3，以此类推
updates = tf.constant([4.4, 3.3, 1.1, 7.7])
# 在长度为8 的全0 向量上根据indices 写入updates 数据
tf.scatter_nd(indices, updates, [8])

# 构造写入位置，即2 个位置
indices = tf.constant([[1],[3]])
updates = tf.constant([# 构造写入数据，即2 个矩阵
[[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]],
[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
])
# 在shape 为[4,4,4]白板上根据indices 写入updates
tf.scatter_nd(indices,updates,[4,4,4])
```

**meshgrid**：tf.meshgrid 函数可以方便地生成二维网格的采样点坐标，方便可视化等应用场合

```python
points = [] # 保存所有点的坐标列表
for x in range(-8,8,100): # 循环生成x 坐标，100 个采样点
for y in range(-8,8,100): # 循环生成y 坐标，100 个采样点
z = sinc(x,y) # 计算每个点(x,y)处的sinc 函数值
points.append([x,y,z]) # 保存采样点

x = tf.linspace(-8.,8,100) # 设置x 轴的采样点
y = tf.linspace(-8.,8,100) # 设置y 轴的采样点
x,y = tf.meshgrid(x,y) # 生成网格点，并内部拆分后返回
x.shape,y.shape # 打印拆分后的所有点的x,y 坐标张量shape
# tf.meshgrid 会返回在axis=2 维度切割后的2 个张量𝑨和𝑩，其中张量𝑨
# 包含了所有点的x 坐标，𝑩包含了所有点的y 坐标，shape 都为[100,100]
z = tf.sqrt(x**2+y**2)
z = tf.sin(z)/z # sinc 函数实现
```

## 5. 网络操作

TensorFlow底层接口为tf.nn，高层接口为tf.keras.layers，tf.keras.layers 命名空间中提供了大量常见网络层的类，如全连接层、激活函数层、池化层、卷积层、循环神经网络层等。

### 5.1 激活函数

一般来说，激活函数类并不是主要的网络运算层，不计入网络的层数

sigmoid:tf.nn.sigmoid

relu:tf.nn.relu或layers.ReLU()

leaky_relu:tf.nn.leaky_relu或layers.LeakyReLU(alpha)

tanh:tf.nn.tanh

```python
x = tf.linspace(-6.,6.,10)
tf.nn.sigmoid(x) # 通过Sigmoid 函数
tf.nn.relu(x) # 通过ReLU 激活函数
layers.ReLU()
tf.nn.leaky_relu(x, alpha=0.1) # 通过LeakyReLU 激活函数
layer.LeakyReLU(alpha)
tf.nn.tanh(x) # 通过tanh 激活函数
```

### 5.2 分类函数

tf.nn.softmax  layers.Softmax(axis)

```python
z = tf.constant([2.,1.,0.1])
tf.nn.softmax(z) # 通过Softmax 函数
ayers.Softmax(axis=-1)
```

### 5.3 误差函数

tf.keras.losses.MSE/keras.losses.MeanSquaredError()：均方差(Mean Squared Error，简称MSE)误差函数,欧式距离(准确地说是欧式距离的平方)

tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)，交叉熵，其中y_true 代表了
One-hot 编码后的真实标签，y_pred 表示网络的预测值，当from_logits 设置为True 时，
y_pred 表示须为未经过Softmax 函数的变量z；当from_logits 设置为False 时，y_pred 表示
为经过Softmax 函数的输出。为了数值计算稳定性，一般设置from_logits 为True，此时
tf.keras.losses.categorical_crossentropy 将在内部进行Softmax 函数计算，所以不需要在模型
中显式调用Softmax 函数

tf.keras.losses.BinaryCrossentropy():二分类的交叉熵损失函数

tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec): 对输入的logits先通过sigmoid函数计算，再计算它们的交叉熵，但是它对交叉熵的计算方式进行了优化，使得的结果不至于溢出。 适用：每个类别相互独立但互不排斥的情况：例如一幅图可以同时包含一条狗和一只大象 

```python
o = tf.random.normal([2,10]) # 构造网络输出
y_onehot = tf.constant([1,3]) # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10)
loss = keras.losses.MSE(y_onehot, o) # 计算均方差
# 创建MSE 类
criteon = keras.losses.MeanSquaredError()
loss = criteon(y_onehot,o) # 计算batch 均方差
# 计算交叉熵
loss = tf.losses.categorical_crossentropy(y_onehot, logits,from_logits=True)
# 创建Softmax 与交叉熵计算类，输出层的输出z 未使用softmax
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot,z) # 计算损失
# 2 分类的交叉熵损失函数
loss = tf.losses.BinaryCrossentropy(y_onehot, logits)
```

### 5.4 网络层

全连接层、激活函数层、池化层、卷积层、循环神经网络层

#### 5.4.1 全连接层

layers.Dense(units, activation)

```python
fc = layers.Dense(256, activation='relu')
fc.kernel # 获取Dense 类的权值矩阵
fc.bias # 获取Dense 类的偏置向量
fc.trainable_variables  # 返回待优化参数列表
fc.variables # 返回所有参数列表
```

#### 5.4.2 卷积层

卷积包含普通卷积，空洞卷积和转置卷积

```python
tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[0,0],[0,0],[0,0]])  
# padding参数为[[0,0],[上,下],[左,右],[0,0]]，或者设置为'SAME'、‘VALID’
layers.Conv2D(4,kernel_size=3,strides=1,padding='SAME')
# 3 × 4大小的卷积核，竖直方向移动步长𝑠ℎ = 2，水平方向移动步长𝑠𝑤 =1 
layers.Conv2D(4,kernel_size=(3,4),strides=(2,1),padding='SAME') 
# 当dilation_rate 参数设置为默认值1 时，使用普通卷积方式进行运算；当dilation_rate 参数大于1 时，采样空# 洞卷积方式进行计算。
layers.Conv2D(1,kernel_size=3,strides=1,dilation_rate=2)  #空洞卷积，1 个3x3 的卷积核
# 普通卷积的输出作为转置卷积的输入，进行转置卷积运算,输出的高宽为5x5
tf.nn.conv2d_transpose(out, w, strides=2,padding='VALID',output_shape=[1,5,5,1])
layers.Conv2DTranspose(1,kernel_size=3,strides=1,padding='VALID')
```

#### 5.4.3 池化层

```python
layers.GlobalAveragePooling2D()
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
layers.AveragePool2D(pool_size=[2, 2], strides=2, padding='same')
```

#### 5.4.4 BN层

```python
network = Sequential([ # 网络容器
layers.Conv2D(6,kernel_size=3,strides=1),
# 插入BN 层
layers.BatchNormalization(),
layers.MaxPooling2D(pool_size=2,strides=2),
layers.ReLU(),
layers.Conv2D(16,kernel_size=3,strides=1),
# 插入BN 层
layers.BatchNormalization(),
layers.MaxPooling2D(pool_size=2,strides=2),
layers.ReLU(),
layers.Flatten(),
layers.Dense(120, activation='relu'),
# 此处也可以插入BN 层
layers.Dense(84, activation='relu'),
# 此处也可以插入BN 层
layers.Dense(10)
])
# 在训练阶段，需要设置网络的参数training=True 以区分BN 层是训练还是测试模型
with tf.GradientTape() as tape:
    # 插入通道维度
	x = tf.expand_dims(x,axis=3)
	# 前向计算，设置计算模式，[b, 784] => [b, 10]
	out = network(x, training=True)
# 在测试阶段，需要设置training=False，避免BN 层采用错误的行为
for x,y in db_test: # 遍历测试集
	# 插入通道维度
	x = tf.expand_dims(x,axis=3)
	# 前向计算，测试模式
	out = network(x, training=False)
```

#### 5.4.5 dropout层

dropout同BN，需要使用is_training进行控制

```python
tf.nn.dropout(x, rate=0.5)
layers.Dropout(rate=0.5)
```

#### 5.4.6 词向量层(embeding)

```python
layers.Embedding(total_words, embedding_len,input_length=max_review_len)
x = tf.range(10) # 生成10 个单词的数字编码
x = tf.random.shuffle(x) # 打散
# 创建共10 个单词，每个单词用长度为4 的向量表示的层
net = layers.Embedding(10, 4)
out = net(x) # 获取词向量
# 从预训练模型中加载词向量表
embed_glove = load_embed('glove.6B.50d.txt')
# 直接利用预训练的词向量表初始化Embedding 层
net.set_weights([embed_glove])
# 经过预训练的词向量模型初始化的Embedding 层可以设置为不参与训练：net.trainable
# = False，那么预训练的词向量就直接应用到此特定任务上；如果希望能够学到区别于预训
# 练词向量模型不同的表示方法，那么可以把Embedding 层包含进反向传播算法中去，利用
# 梯度下降来微调单词表示方法。


# 预训练词向量
print('Indexing word vectors.')
embeddings_index = {} # 提取单词及其向量，保存在字典中
# 词向量模型文件存储路径
GLOVE_DIR = r'C:\Users\z390\Downloads\glove6b50dtxt'
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding='utf-8') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))
# GloVe.6B 版本共存储了40 万个词汇的向量表。根据词汇的数字编码表依次从GloVe 模型中获取其词向量，并写入对应位置。
num_words = min(total_words, len(word_index))
embedding_matrix = np.zeros((num_words, embedding_len)) #词向量表
for word, i in word_index.items():
	if i >= MAX_NUM_WORDS:
		continue # 过滤掉其他词汇
	embedding_vector = embeddings_index.get(word) # 从GloVe 查询词向量
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector # 写入对应位置
print(applied_vec_count, embedding_matrix.shape)
# 创建Embedding 层
self.embedding = layers.Embedding(total_words, embedding_len,input_length=max_review_len,
	trainable=False)#不参与梯度更新
self.embedding.build(input_shape=(None, max_review_len))
# 利用GloVe 模型初始化Embedding 层
self.embedding.set_weights([embedding_matrix])#初始化
```

#### 5.4.7 循环神经网络层

循环神经网络层包含RNN，LSTM，GRU

```python
layers.SimpleRNNCell(3) # 创建RNN Cell，内存向量长度为3,需要自己维护状态向量
layers.SimpleRNNCell(units, dropout=0.5)  #使用dropout 技术防止过拟合
layers.SimpleRNN(64) # 创建状态向量长度为64 的SimpleRNN 层
layers.SimpleRNN(64,return_sequences=True) # 创建RNN 层时，设置返回所有时间戳上的输出
layers.LSTMCell(64) # 创建LSTM Cell
layers.LSTM(64, return_sequences=True)
layers.GRUCell(64) # 新建GRU Cell，向量长度为64
layers.GRU(64, return_sequences=True)
net = keras.Sequential([ # 构建2 层RNN 网络
    #除最末层外，都需要返回所有时间戳的输出，用作下一层的输入
	layers.SimpleRNN(64, return_sequences=True), 
	layers.SimpleRNN(64),
])
```

#### 5.4.8 onehot

```python
tf.one_hot(x,depth=10)
```

#### 5.4.9 文本处理

```python
# 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_len)
```

## 6. 自定义网络

### 6.1 自定义网络层

在创建自定义网络层类时，需要继承自layers.Layer 基类  

自定义的网络层，至少需要实现初始化__init__方法和前向传播逻辑call 方法

```python
# 一个没有偏置向量的全连接层,同时固定激活函数为ReLU 函数
class MyDense(layers.Layer):
	# 自定义网络层
	def __init__(self, inp_dim, outp_dim):
		super(MyDense, self).__init__()
		# 创建权值张量并添加到类管理列表中，设置为需要优化
		self.kernel = self.add_variable('w', [inp_dim, outp_dim],trainable=True)
    def call(self, inputs, training=None):
        #training 参数用于指定模型的状态：training 为True 时执
		#行训练模式，training 为False 时执行测试模式，默认参数为None，即测试模式。由于全连
		#接层的训练模式和测试模式逻辑一致，此处不需要额外处理。对于部份测试模式和训练模
		#式不一致的网络层，需要根据training 参数来设计需要执行的逻辑
		# 实现自定义类的前向计算逻辑
		# X@W
		#out = inputs @ self.kernel
		# 执行激活函数运算
		out = tf.nn.relu(out)
		return out
```

### 6.3 网络容器(Sequential)

对于常见的网络，需要手动调用每一层的类实例完成前向传播运算，当网络层数变得较深时，这一部分代码显得非常臃肿。可以通过Keras 提供的网络容器Sequential 将多个网络层封装成一个大网络模型，只需要调用网络模型的实例一次即可完成数据从第一层到最末层的顺序传播运算  

Sequential 容器也可以通过add()方法继续追加新的网络层，实现动态创建网络的功能

```python
# 导入Sequential 容器
from tensorflow.keras import layers, Sequential
network = Sequential([ # 封装为一个网络
layers.Dense(3, activation=None), # 全连接层，此处不使用激活函数
layers.ReLU(),#激活函数层
layers.Dense(2, activation=None), # 全连接层，此处不使用激活函数
layers.ReLU() #激活函数层
])
x = tf.random.normal([4,3])
out = network(x) # 输入从第一层开始，逐层传播至输出层，并返回输出层的输出

layers_num = 2 # 堆叠2 次
network = Sequential([]) # 先创建空的网络容器
for _ in range(layers_num):
	network.add(layers.Dense(3)) # 添加全连接层
	network.add(layers.ReLU())# 添加激活函数层
network.build(input_shape=(4, 4)) # 创建网络参数
network.summary()
for p in network.trainable_variables:
	print(p.name, p.shape) # 参数名和形状
```

### 6.3 自定义网络

创建自定义的网络类时，需要继承自keras.Model 基类

Sequential 容器方便地封装成一个网络模型,Sequential 容器适合于数据按序从第一层传播到第二层，再从第二层传播到第三层，以此规律传播的网络模型。

对于复杂的网络结构，例如第三层的输入不仅是第二层的输出，还有第一层的输出，此时使用自定义网络更加灵活。  

```python
class MyModel(keras.Model):
	# 自定义网络类，继承自Model 基类
	def __init__(self):
		super(MyModel, self).__init__()
		# 完成网络内需要的网络层的创建工作
		self.fc1 = MyDense(28*28, 256)
		self.fc2 = MyDense(256, 128)
		self.fc3 = MyDense(128, 64)
		self.fc4 = MyDense(64, 32)
		self.fc5 = MyDense(32, 10)
	def call(self, inputs, training=None):
		# 自定义前向运算逻辑
		x = self.fc1(inputs)
		x = self.fc2(x)
		x = self.fc3(x)
		x = self.fc4(x)
		x = self.fc5(x)
		return x
model = Network() # 创建网络类实例
# 通过build 函数完成内部张量的创建，其中4 为任意设置的batch 数量，9 为输入特征长度
model.build(input_shape=(4, 9))
model.summary() # 打印网络信息
```

## 7. pipeline

### 7.1 数据处理

#### 7.1.1 Dataset

Dateset主要功能是将numpy转换为Dateset(from_tensor_slices)、随机打散（shuffle）、产生批数据（batch）、预处理（map）、循环产生数据（repeat）。调用Dataset提供的这些工具函数会返回新的Dataset 对象，可以通过db = db. step1(). step2(). step3 ()方式按序完成所有的数据处理步骤

```python
train_db = tf.data.Dataset.from_tensor_slices((x, y)).map(preprocess).
		shuffle(10000).batch(128).repeat(30)
```

**转换成Dataset 对象**:数据加载进入内存后，需要转换成Dataset 对象，才能利用TensorFlow 提供的各种便捷功能。通过Dataset.from_tensor_slices 可以将训练部分的数据图片x 和标签y 都转换成Dataset 对象：

```python
train_db = tf.data.Dataset.from_tensor_slices((x, y)) # 构建Dataset 对象
```

**随机打散**：Dataset.shuffle(buffer_size)

Dataset.shuffle(buffer_size)工具可以设置Dataset 对象随机打散数据之间的顺序，防止每次训练时数据按固定顺序产生，从而使得模型尝试“记忆”住标签信息，buffer_size 参数指定缓冲池的大小，一般设置为一个较大的常数即可

```python
train_db = train_db.shuffle(10000) # 随机打散样本，不会打乱样本与标签映射关系
```

**批训练**：为了利用显卡的并行计算能力，一般在网络的计算过程中会同时计算多个样本，我们把这种训练方式叫做批训练，其中一个批中样本的数量叫做Batch Size。为了一次能够从Dataset 中产生Batch Size 数量的样本，需要设置Dataset 为批训练方式

```python
train_db = train_db.batch(128) # 设置批训练，batch size 为128
#构建数据集，打散，批量，并丢掉最后一个不够batchsz 的batch
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
```

**预处理**:从 keras.datasets 中加载的数据集的格式大部分情况都不能直接满足模型的输入要求，因此需要根据用户的逻辑自行实现预处理步骤。Dataset 对象通过提供map(func)工具函数，可以非常方便地调用用户自定义的预处理逻辑，它实现在func 函数里。例如，下方代码调用名为preprocess 的函数完成每个样本的预处理：

```python
# 预处理函数实现在preprocess 函数中，传入函数名即可
train_db = train_db.map(preprocess)
def preprocess(x, y): # 自定义的预处理函数
	# 调用此函数时会自动传入x,y 对象，shape 为[b, 28, 28], [b]
	# 标准化到0~1
	x = tf.cast(x, dtype=tf.float32) / 255.
	x = tf.reshape(x, [-1, 28*28]) # 打平
	y = tf.cast(y, dtype=tf.int32) # 转成整型张量
	y = tf.one_hot(y, depth=10) # one-hot 编码
	# 返回的x,y 将替换传入的x,y 参数，从而实现数据的预处理功能
	return x,y
```

**循环训练**：

```python
for epoch in range(20): # 训练Epoch 数
	for step, (x,y) in enumerate(train_db): # 迭代Step 数
	# training...

# 也可以通过设置Dataset 对象，使得数据集对象内部遍历多次才会退出
train_db = train_db.repeat(20) # 数据集迭代20 遍才终止
```

#### 7.1.2 经典数据集

在 TensorFlow 中，keras.datasets 模块提供了常用经典数据集的自动下载、管理、加载与转换功能，并且提供了tf.data.Dataset 数据集对象，方便实现多线程(Multi-threading)、预处理(Preprocessing)、随机打散(Shuffle)和批训练(Training on Batch)等常用数据集的功能。

> 常用的经典数据集
>
> * Boston Housing，波士顿房价趋势数据集，用于回归模型训练与测试。
> * CIFAR10/100，真实图片数据集，用于图片分类任务。
> * MNIST/Fashion_MNIST，手写数字图片数据集，用于图片分类任务。
> * IMDB，情感分类任务数据集，用于文本分类任务。

datasets.xxx.load_data()函数即可实现经典数据集的自动加载，其中xxx 代表具体的数据集名称，如“CIFAR10”、“MNIST”。TensorFlow 会默认将数据缓存在用户目录下的.keras/datasets 文件夹，如果当前
数据集不在缓存中，则会自动从网络下载、解压和加载数据集；如果已经在缓存中，则自动完成加载。

对于图片数据集**MNIST、CIFAR10 等，会返回2 个tuple**，第一个tuple 保存了用于训练的数据x 和y 训练集对象；第2 个tuple 则保存了用于测试的数据x_test 和y_test 测试集对象，所有的数据都用Numpy 数组容器保存

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets # 导入经典数据集加载模块
# 加载MNIST 数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:',y_test)
```

#### 7.1.3 自定义数据集

**创建编码表**

样本的类别一般以字符串类型的类别名标记，但是对于神经网络来说，首先需要将类别名进行**数字编码**，然后在合适的时候再转换成One-hot 编码或其他编码格式。

类别名与数字的映射关系称为编码表，一旦创建后，一般不能变动

编码表的现有键值对数量作为类别的标签映射数字，并保存进name2label 字典对象。

```python
def load_pokemon(root, mode='train'):
	# 创建数字编码表
	name2label = {} # 编码表字典，"sq...":0
	# 遍历根目录下的子文件夹，并排序，保证映射关系固定
	for name in sorted(os.listdir(os.path.join(root))):
		# 跳过非文件夹对象
		if not os.path.isdir(os.path.join(root, name)):
			continue
		# 给每个类别编码一个数字
		name2label[name] = len(name2label.keys())
```

**创建样本和标签表格**

编码表确定后，我们需要根据实际数据的存储方式获得每个样本的存储路径以及它的标签数字，分别表示为images 和labels 两个List 对象。其中images List 存储了每个样本的路径字符串，labels List 存储了样本的类别数字，两者长度一致，且对应位置的元素相互关联。

每行的第一个元素保存了当前样本的存储路径，第二个元素保存了样本的类别数字。

```python
def load_csv(root, filename, name2label):
	# 从csv 文件返回images,labels 列表
	# root:数据集根目录，filename:csv 文件名， name2label:类别名编码表
	if not os.path.exists(os.path.join(root, filename)):
        # 如果csv 文件不存在，则创建
		images = []
		for name in name2label.keys(): # 遍历所有子目录，获得所有的图片
			# 只考虑后缀为png,jpg,jpeg 的图片：'pokemon\\mewtwo\\00001.png
			images += glob.glob(os.path.join(root, name, '*.png'))
			images += glob.glob(os.path.join(root, name, '*.jpg'))
			images += glob.glob(os.path.join(root, name, '*.jpeg'))
        # 打印数据集信息：1167, 'pokemon\\bulbasaur\\00000000.png'
		print(len(images), images)
		random.shuffle(images) # 随机打散顺序
		# 创建csv 文件，并存储图片路径及其label 信息
		with open(os.path.join(root, filename), mode='w', newline='') as f:
			writer = csv.writer(f)
			for img in images: # 'pokemon\\bulbasaur\\00000000.png'
				name = img.split(os.sep)[-2]
				label = name2label[name]
				# 'pokemon\\bulbasaur\\00000000.png', 0
				writer.writerow([img, label])
			print('written into csv file:', filename)
	
    # 此时已经有csv 文件在文件系统上，直接读取
	images, labels = [], []
	with open(os.path.join(root, filename)) as f:
		reader = csv.reader(f)
		for row in reader:
			# 'pokemon\\bulbasaur\\00000000.png', 0
			img, label = row
			label = int(label)
			images.append(img)
			labels.append(label)
	# 返回图片路径list 和标签list
	return images, labels
```

**数据集划分**

数据集的划分需要根据实际情况来灵活调整划分比率。当数据集样本数较多时，可以选择80%-10%-10%的比例分配给训练集、验证集和测试集;对于小型的数据集，尽管样本数量较小，但还是需要适当增加验证集和测试集的比例，以保证获得准确的测试结果。

```python
def load_pokemon(root, mode='train'):
	# 读取Label 信息
	# [file1,file2,], [3,1]
	images, labels = load_csv(root, 'images.csv', name2label)
	# 数据集划分
	if mode == 'train': # 60%
		images = images[:int(0.6 * len(images))]
		labels = labels[:int(0.6 * len(labels))]
	elif mode == 'val': # 20% = 60%->80%
		images = images[int(0.6 * len(images)):int(0.8 * len(images))]
		labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
	else: # 20% = 80%->100%
		images = images[int(0.8 * len(images)):]
		labels = labels[int(0.8 * len(labels)):]
	return images, labels, name2label
```

需要注意的是，每次运行时的数据集划分方案需固定，防止使用测试集的样本训练，导致模型泛化性能不准确

**创建 Dataset 对象**

```python
# 加载pokemon 数据集，指定加载训练集
# 返回训练集的样本路径列表，标签数字列表和编码表字典
images, labels, table = load_pokemon('pokemon', 'train')
print('images:', len(images), images)
print('labels:', len(labels), labels)
print('table:', table)
# images: string path
# labels: number
db = tf.data.Dataset.from_tensor_slices((images, labels))
db = db.shuffle(1000).map(preprocess).batch(32)
```

**数据预处理**

上面我们在构建数据集时通过调用.map(preprocess)函数来完成数据的预处理工作。由于目前我们的images 列表只是保存了所有图片的路径信息，而不是图片的内容张量，因此需要在预处理函数中完成图片的读取以及张量转换等工作。

对于预处理函数(x,y) = preprocess(x,y)，它的传入参数需要和创建Dataset 时给的参数的格式保存一致，返回参数也需要和传入参数的格式保存一致。

```python
def preprocess(x,y): # 预处理函数
	# x: 图片的路径，y：图片的数字编码
	x = tf.io.read_file(x) # 根据路径读取图片
	x = tf.image.decode_jpeg(x, channels=3) # 图片解码，忽略透明通道
	x = tf.image.resize(x, [244, 244]) # 图片缩放为略大于224 的244
	# 数据增强，这里可以自由组合增强手段
	# x = tf.image.random_flip_up_down(x)
	x= tf.image.random_flip_left_right(x) # 左右镜像
	x = tf.image.random_crop(x, [224, 224, 3]) # 随机裁剪为224
	# 转换成张量，并压缩到0~1 区间
	# x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
	# 0~1 => D(0,1)
	x = normalize(x) # 标准化
	y = tf.convert_to_tensor(y) # 转换成张量
	return x, y
```

将0~255 范围的像素值缩放到0~1 范围，并通过标准化函数normalize 实现数据的标准化运算，将像素映射为0 周围分布，有利于网络的优化

标准化后的数据适合网络的训练及预测，但是在进行可视化时，需要将数据映射回0~1 的范围,实现标准化和标准化的逆过程如下

```python
# 这里的mean 和std 根据真实的数据计算获得，比如ImageNet
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])
def normalize(x, mean=img_mean, std=img_std):
	# 标准化函数
	# x: [224, 224, 3]
	# mean: [224, 224, 3], std: [3]
	x = (x - mean)/std
	return x
def denormalize(x, mean=img_mean, std=img_std):
	# 标准化的逆过程函数
	x = x * std + mean
	return x
```

### 7.2 网络构建

#### 7.2.1 内置模型使用

在keras.applications 模块中实现了常用的网络模型，如VGG 系列、ResNet 系列、DenseNet 系列、MobileNet 系列等等，只需要一行代码即可创建这些模型网络。

```python
# 加载ImageNet 预训练网络模型，并去掉最后一层
resnet = keras.applications.ResNet50(weights='imagenet',include_top=False)
resnet.summary()
# 测试网络的输出
x = tf.random.normal([4,224,224,3])
out = resnet(x) # 获得子网络的输出
out.shape
global_average_layer = layers.GlobalAveragePooling2D()
fc = layers.Dense(100)
mynet = Sequential([resnet, global_average_layer, fc])
mynet.summary()
#通过设置resnet.trainable = False 可以选择冻结ResNet 部分的网络参数，只训练新建的
#网络层，从而快速、高效完成网络模型的训练。当然也可以在自定义任务上更新网络的全部参数。
```

#### 7.2.2 迁移学习

对于卷积神经网络，一般认为它能够逐层提取特征，越末层的网络的抽象特征提取能力越强，输出层一般使用与类别数相同输出节点的全连接层，作为分类网络的概率分布预测。对于相似的任务A 和B，如果它们的特征提取方法是相近的，则网络的前面数层可以重用，网络后面的数层可以根据具体的任务设定从零开始训练

```python
# 加载DenseNet 网络模型，并去掉最后一层全连接层，最后一个池化层设置为max pooling
# 并使用预训练的参数初始化
net = keras.applications.DenseNet121(weights='imagenet', include_top=False,pooling='max')
# 设计为不参与优化，即DenseNet 这部分参数固定不动
net.trainable = False
newnet = keras.Sequential([
	net, # 去掉最后一层的DenseNet121
	layers.Dense(1024, activation='relu'), # 追加全连接层
	layers.BatchNormalization(), # 追加BN 层
	layers.Dropout(rate=0.5), # 追加Dropout 层，防止过拟合
	layers.Dense(5) # 根据宝可梦数据的任务，设置最后一层输出节点数为5
])
newnet.build(input_shape=(4,224,224,3))
newnet.summary()
```

#### 7.2.3 自定义网络使用

参考6.2

### 7.3 模型训练

#### 7.3.1 常规训练

```python
optimizer = tf.keras.optimizers.RMSprop(0.001) # 创建优化器，指定学习率
for epoch in range(200): # 200 个Epoch
	for step, (x,y) in enumerate(train_db): # 遍历一次训练集
		# 梯度记录器，训练时需要使用它
		with tf.GradientTape() as tape:
			out = model(x,training=True) # 通过网络获得输出
			loss = tf.reduce_mean(losses.MSE(y, out)) # 计算MSE
			mae_loss = tf.reduce_mean(losses.MAE(y, out)) # 计算MAE
		if step % 10 == 0: # 间隔性地打印训练误差
			print(epoch, step, float(loss))
		# 计算梯度，并更新
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # 在测试阶段，需要设置training=False，避免BN 层采用错误的行为
	for x,y in test_db: # 遍历测试集
		# 前向计算，测试模式
		out = network(x, training=False)
```

#### 7.3.2 装配训练

装配：指定网络使用的优化器对象、损失函数类型，评价指标等

```python
network.compile(optimizer=optimizers.Adam(lr=0.01),
loss=losses.CategoricalCrossentropy(from_logits=True),
metrics=['accuracy'] # 设置测量指标为准确率
)
```

训练:送入待训练的数据集和验证用的数据集，这一步称为模型训练

```python
# 指定训练集为train_db，验证集为val_db,训练5 个epochs，每2 个epoch 验证一次
# 返回训练轨迹信息保存在history 对象中
history = network.fit(train_db, epochs=5, validation_data=val_db,validation_freq=2,callbacks=[early_stopping])
# train_db 为tf.data.Dataset 对象，也可以传入Numpy Array 类型的数据；epochs 参数指
# 定训练迭代的Epoch 数量；validation_data 参数指定用于验证(测试)的数据集和验证的频率validation_freq。
# fit 函数会返回训练过程的数据记录history，其中history.history 为字典对象，包含了训练过程中的loss、
# 测量指标等记录项，我们可以直接查看这些训练数据，
# 创建Early Stopping 类，连续3 次不上升则终止训练
early_stopping = EarlyStopping(monitor='val_accuracy',
			min_delta=0.001,patience=3)
```

测试、预测：前向计算

```python
# 加载一个batch 的测试数据
x,y = next(iter(db_test))
print('predict x:', x.shape) # 打印当前batch 的形状
# predict计算一次
out = network.predict(x) # 模型预测，预测结果保存在out 中
# evaluate计算整个数据集
network.evaluate(db_test) # 模型测试，测试在db_test 上的性能表现
```

### 7.4 模型保存和加载

**张量方式**：只保存模型参数，不保存模型结构，需要使用相同的网络结构才能够正确恢复网络状态，因此一般
在拥有网络源文件的情况下使用。  

```python
network.save_weights('weights.ckpt')
network.load_weights('weights.ckpt')
```

**网络方式**:保存模型结构和模型参数，可直接恢复出网络模型

```python
network.save('model.h5')
print('saved total model.')
del network
# 从文件恢复网络结构与网络参数
network = keras.models.load_model('model.h5')
```

**SavedModel 方式**：将模型保存为pb，在移动端运行

```python
tf.saved_model.save(network, 'model-savedmodel')
# 从文件恢复网络结构与网络参数
network = tf.saved_model.load('model-savedmodel')
```

### 7.5 可视化

可视化（tensorboard）

```python
# 创建监控类，监控数据将写入log_dir 目录
summary_writer = tf.summary.create_file_writer(log_dir)
with summary_writer.as_default(): # 写入环境
	# 当前时间戳step 上的数据为loss，写入到名为train-loss 数据库中
	tf.summary.scalar('train-loss', float(loss), step=step)
    # 可视化测试用的图片，设置最多可视化9 张图片
	tf.summary.image("val-onebyone-images:", val_images,max_outputs=9, step=step)
    # 可视化真实标签的直方图分布
	tf.summary.histogram('y-hist',y, step=step)
	# 查看文本信息
	tf.summary.text('loss-text',str(float(loss)))
```

## 8. 模型调优

### 8.1 过拟合和欠拟合

#### 8.1.1 过拟合

在训练集上面表现较好，但是在未见的样本上表现不，也就是模型泛化能力偏弱

缓解过拟合方法：

降低网络的层数、降低网络的参数量、添加正则化手段、添加假设空间的约束

对于神经网络，即使网络结构超参数保持不变(即网络最大容量固定)，模型依然可能会出现过拟合的现象，这是因为神经网络的有效容量和网络参数的状态息息相关，神经网络的有效容量可以很大，也可以通过稀疏化参数、添加正则化等手段降低有效容量

1. 提前停止

一般把对训练集中的一个Batch 运算更新一次叫做一个Step，对训练集的所有样本循环迭代一次叫做一个Epoch。验证集可以在数次Step 或数次Epoch 后使用，计算模型的验性能。当发现验证准确率连续𝑛个Epoch 没有下降时，可以预测可能已经达到了最适合的Epoch 附近，从而提前终止训练

2. 模型设计

对于神经网络来说，网络的层数和参数量是网络容量很重要的参考指标，通过减少网络的层数，并减少每层中网络参数量的规模，可以有效降低网络的容量。反之，如果发现模型欠拟合，需要增大网络的容量，可以通过增加层数，增大每层的参数量等方式实现。

3. 正则化

通过设计不同层数、大小的网络模型可以为优化算法提供初始的函数假设空间，但是模型的实际容量可以随着网络参数的优化更新而产生变化。通过限制网络参数的稀疏性，可以来约束网络的实际容量。

这种约束一般通过在损失函数上添加额外的参数稀疏性惩罚项实现，一般地，参数𝜃的稀疏性约束通过约束参数
𝜃的𝐿范数实现，过大的𝜆参数有可能导致网络不收敛，需要根据实际任务调节

​			𝑚𝑖𝑛 ℒ(𝑓𝜃( ), 𝑦) + 𝜆 ∙ 𝛺(𝜃), ( , 𝑦) ∈ 𝔻train

L0 正则化：非零元素的个数，通过约束L0范数的大小可以迫使网络中的连接权值大部分为0，从而降低网络的实际参数量和网络容量，L0 范数并不可导，不能利用梯度下降算法进行优化，在神经网络中使用的并不多

L0 正则化：绝对值之和

L2 正则化：平方和

4. Dropout

Dropout 通过随机断开神经网络的连接，减少每次训练时实际参与计算的模型的参数量；但是在测试时，Dropout 会恢复所有的连接，保证模型测试时获得最好的性能。

5. 数据增强

TensorFlow 中提供了常用图片的处理函数，位于tf.image 子模块中。

```python
tf.image.resize(x, [244, 244])
# 图片逆时针旋转180 度
x = tf.image.rot90(x,2)
# 随机水平翻转
x = tf.image.random_flip_left_right(x)
# 随机竖直翻转
x = tf.image.random_flip_up_down(x)
# 图片先缩放到稍大尺寸
x = tf.image.resize(x, [244, 244])
# 再随机裁剪到合适尺寸
x = tf.image.random_crop(x, [224,224,3])
```

6. 生成数据

#### 8.1.2 欠拟合

训练集上表现不佳，同时在未见的样本上表现也不佳

模型在训练集上误差一直维持较高的状态，很难优化减少，同时在测试集上也表现不佳时，我们可以考虑是否出现了欠拟合的现象。

可以通过增加神经网络的层数、增大中间维度的大小等手段，比较好的解决欠拟合的问题

加深网络的层数、增加网络的参数量，尝试更复杂的网络结构。

### 8.2 梯度弥散和梯度消失

#### 8.2.1 梯度弥散

梯度弥散(Gradient Vanishing)：梯度值接近于0 的现象叫做梯度弥散

对于梯度弥散现象，可以通过增大学习率、减少网络深度、添加Skip Connection 等一系列的措施抑制。

#### 8.2.2 梯度消失

梯度爆炸(Gradient Exploding)：梯度值远大于1 的现象叫做梯度爆炸(Gradient Exploding)

梯度爆炸可以通过梯度裁剪(Gradient Clipping)的方式在一定程度上的解决。梯度裁剪与张量限幅非常类似，也是通过将梯度张量的数值或者范数限制在某个较小的区间内，从而将远大于1 的梯度值减少，避免出现梯度爆炸

tf.clip_by_value(a,0.4,0.6) # 梯度值裁剪

tf.clip_by_norm(a,5) 按范数方式裁剪

tf.clip_by_global_norm  缩放整体网络梯度𝑾的范数

```python
with tf.GradientTape() as tape:
	logits = model(x) # 前向传播
	loss = criteon(y, logits) # 误差计算
# 计算梯度值
grads = tape.gradient(loss, model.trainable_variables)
grads, _ = tf.clip_by_global_norm(grads, 25) # 全局梯度裁剪
# 利用裁剪后的梯度张量更新参数
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

## 9 网络实现实例

### 9.1 VGG13

```python
conv_layers = [ # 先创建包含多网络层的列表
# Conv-Conv-Pooling 单元1
# 64 个3x3 卷积核, 输入输出同大小
layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
# 高宽减半
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
# Conv-Conv-Pooling 单元2,输出通道提升至128，高宽大小减半
layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
# Conv-Conv-Pooling 单元3,输出通道提升至256，高宽大小减半
layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
# Conv-Conv-Pooling 单元4,输出通道提升至512，高宽大小减半
layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
# Conv-Conv-Pooling 单元5,输出通道提升至512，高宽大小减半
layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]
# 利用前面创建的层列表构建网络容器
conv_net = Sequential(conv_layers)
# 创建3 层全连接层子网络
fc_net = Sequential([
layers.Dense(256, activation=tf.nn.relu),
layers.Dense(128, activation=tf.nn.relu),
layers.Dense(10, activation=None),
])
# build2 个子网络，并打印网络参数信息
conv_net.build(input_shape=[4, 32, 32, 3])
fc_net.build(input_shape=[4, 512])
conv_net.summary()
fc_net.summary()
# 列表合并，合并2 个子网络的参数
variables = conv_net.trainable_variables + fc_net.trainable_variables
# 对所有参数求梯度
grads = tape.gradient(loss, variables)
# 自动更新
optimizer.apply_gradients(zip(grads, variables))
```

### 9.2 resnet

```python
class BasicBlock(layers.Layer):
	# 残差模块类
	def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
		# f(x)包含了2 个普通卷积层，创建卷积层1
		self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
		self.bn1 = layers.BatchNormalization()
		self.relu = layers.Activation('relu')
		# 创建卷积层2
		self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
		self.bn2 = layers.BatchNormalization()
        #当ℱ(𝒙)的形状与𝒙不同时，无法直接相加，我们需要新建identity(𝒙)卷积层，来完成𝒙的形状转换
    	if stride != 1: # 插入identity 层
			self.downsample = Sequential()
			self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
		else: # 否则，直接连接
			self.downsample = lambda x:x
    def call(self, inputs, training=None):
		# 前向传播函数
		out = self.conv1(inputs) # 通过第一个卷积层
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out) # 通过第二个卷积层
		out = self.bn2(out)
		# 输入通过identity()转换
		identity = self.downsample(inputs)
		# f(x)+x 运算
		output = layers.add([out, identity])
		# 再通过激活函数并返回
		output = tf.nn.relu(output)
		return output
```

```python
def build_resblock(self, filter_num, blocks, stride=1):
	# 辅助函数，堆叠filter_num 个BasicBlock
	res_blocks = Sequential()
	# 只有第一个BasicBlock 的步长可能不为1，实现下采样
	res_blocks.add(BasicBlock(filter_num, stride))
	for _ in range(1, blocks):#其他BasicBlock 步长都为1
		res_blocks.add(BasicBlock(filter_num, stride=1))
	return res_blocks
class ResNet(keras.Model):
	# 通用的ResNet 实现类
	def __init__(self, layer_dims, num_classes=10): # [2, 2, 2, 2]
		super(ResNet, self).__init__()
		# 根网络，预处理
		self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
			layers.BatchNormalization(),
			layers.Activation('relu'),
			layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1),padding='same')
			])
		# 堆叠4 个Block，每个block 包含了多个BasicBlock,设置步长不一样
		self.layer1 = self.build_resblock(64, layer_dims[0])
		self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
		self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
		self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
		# 通过Pooling 层将高宽降低为1x1
		self.avgpool = layers.GlobalAveragePooling2D()
		# 最后连接一个全连接层分类
		self.fc = layers.Dense(num_classes)
	def call(self, inputs, training=None):
		# 前向计算函数：通过根网络
		x = self.stem(inputs)
		# 一次通过4 个模块
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		# 通过池化层
		x = self.avgpool(x)
		# 通过全连接层
		x = self.fc(x)
		return x
def resnet18():
	# 通过调整模块内部BasicBlock 的数量和配置实现不同的ResNet
	return ResNet([2, 2, 2, 2])
def resnet34():
	# 通过调整模块内部BasicBlock 的数量和配置实现不同的ResNet
	return ResNet([3, 4, 6, 3])
```

## 10 测量工具

Keras 的测量工具的使用方法一般有4 个主要步骤：新建测量器，写入数据，读取统计数据和清零测量器。

在 keras.metrics 模块中，提供了较多的常用测量器类，如统计平均值的Mean 类，统计准确率的Accuracy 类，统计余弦相似度的CosineSimilarity 类等。  

```python
acc_meter = metrics.Accuracy() # 创建准确率测量器
# 新建平均测量器，适合Loss 数据
loss_meter = metrics.Mean()
# 记录采样的数据，通过float()函数将张量转换为普通数值
loss_meter.update_state(float(loss))
# 打印统计期间的平均loss
print(step, 'loss:', loss_meter.result())
# 测量器会统计所有历史记录的数据，因此在启动新一轮统计时，有必要清除历史状态,
if step % 100 == 0:
	# 打印统计的平均loss
	print(step, 'loss:', loss_meter.result())
	loss_meter.reset_states() # 打印完后，清零测量器
```



# 参考

https://github.com/dragen1860/Deep-Learning-with-TensorFlow-book


