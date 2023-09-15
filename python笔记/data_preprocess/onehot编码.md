# one-hot编码
One-Hot编码，又称为一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效。  


## sklearn实现
	from numpy import array
	from numpy import argmax
	from sklearn.preprocessing import LabelEncoder
	from sklearn.preprocessing import OneHotEncoder
	data=['cold','cold','warm','cold','hot','hot','warm','cold','warm','hot','test']
	values=array(data)
	print(values)
	# integer encode
	label_encoder=LabelEncoder()
	integer_encoded=label_encoder.fit_transform(values)
	print(integer_encoded)
	# binary encode
	onehot_encoder=OneHotEncoder(sparse=False)
	integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
	onehot_encoded=onehot_encoder.fit_transform(integer_encoded)
	print(onehot_encoded)
	# invert first example
	inverted=label_encoder.inverse_transform([argmax(onehot_encoded[0,:])])
	print(inverted)
	#output 
	'''
	[[1. 0. 0. 0.]
	 [1. 0. 0. 0.]
	 [0. 0. 0. 1.]
	 [1. 0. 0. 0.]
	 [0. 1. 0. 0.]
	 [0. 1. 0. 0.]
	 [0. 0. 0. 1.]
	 [1. 0. 0. 0.]
	 [0. 0. 0. 1.]
	 [0. 1. 0. 0.]
	 [0. 0. 1. 0.]]
	['cold']
	'''

## tensorflow实现
	#tf.one_hot(indices,depth,on_value=None,off_value=None,axis=None,dtype=None,name=None)
	indices = [0, 1, 2]
	depth = 3
	tf.one_hot(indices, depth)  # output: [3 x 3]
	# [[1., 0., 0.],
	#  [0., 1., 0.],
	#  [0., 0., 1.]]
	
	indices = [0, 2, -1, 1]
	depth = 3
	tf.one_hot(indices, depth,
	           on_value=5.0, off_value=0.0,
	           axis=-1)  # output: [4 x 3]
	# [[5.0, 0.0, 0.0],  # one_hot(0)
	#  [0.0, 0.0, 5.0],  # one_hot(2)
	#  [0.0, 0.0, 0.0],  # one_hot(-1)
	#  [0.0, 5.0, 0.0]]  # one_hot(1)  

## keras实现
	#keras.utils.to_categorical(y, num_classes=None, dtype='float32')
	array([0, 2, 1, 2, 0])
	to_categorical(labels)
	#output
	array([[ 1.,  0.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.]], dtype=float32)


## 缺点
1. 矩阵的每一维长度都是字典的长度，比如字典包含10000个单词，那么每个单词对应的one-hot向量就是1X10000的向量，而这个向量只有一个位置为1，其余都是0，浪费空间，不利于计算。  
2. one-hot矩阵相当于简单的给每个单词编了个号，但是单词和单词之间的关系则完全体现不出来。  


# onehot转换为int
	np.argmax(a, axis=1)
	tf.argmax(a, axis=1)