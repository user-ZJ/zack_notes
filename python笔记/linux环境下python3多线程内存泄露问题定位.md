# linux环境下python3多线程内存泄露问题定位

## 1. 环境说明
centos7  
python3.6.8

## 2. 使用工具
本次定位中主要使用一下两个工具监听内存。

**top**

	top -d 1
	top查看各个进程CPU和内存占用信息，主要关注以下三个内存相关的字段  
	VIRT：进程占用的虚拟内存
	RES：进程占用的物理内存
	SHR：进程使用的共享内存

**free**
	
	free -h
	free查看整个系统内存使用情况，主要关注free和cache使用情况  

## 3. 初始代码段
		tfrecords_list = []
        count_file = 0 #统计处理文件的数量
        count_sample = 0
        pbar = tqdm(total=len(files), desc="Creating: {} tfrecord".format(flag))
        executor = ThreadPoolExecutor(max_workers=self.processes)
        examples_feature = {executor.submit(self.get_examples, i): i for i in files}
        for feature in as_completed(examples_feature):
            if count_file % PER_SAMPLES_NUM == 0: #每PER_SAMPLES_NUM个文件写入一个tfrecord
                filename = os.path.join(self.tfrecords_dir,flag+"_"+str(count_file)+".tfrecord")
                tfrecords_list.append(filename)
                writer = tf.python_io.TFRecordWriter(filename)
            i = examples_feature[feature]
            try:
                examples = feature.result()
            except Exception as exc:
                del examples_feature[feature]
                print('%r generated an exception: %s' % (self.id_filename_dict[i], exc))
            else:
                del examples_feature[feature]
                for example in examples:
                    writer.write(example)
                    count_sample += 1
                count_file +=1
                if count_file % PER_SAMPLES_NUM ==0:
                    writer.close()
            pbar.update()
        writer.close()
        executor.shutdown()
        pbar.close()

## 4. 问题描述
该段代码功能为提取音频文件特征，并将提取后的特征写入多个tfrecord文件。在windows系统python3.6环境下，处理VCTK数据集内存稳定，不存在内存泄露。但在linux环境下运行相同的代码，大约一个小时，python进程内存会从初始的3G升高到20G，存在明显的内存泄露。

## 5. 定位过程

### 5.1 线程池需要分批次submit?
**参考方案**   
参考[stackoverflow](https://stackoverflow.com/questions/34770169/using-concurrent-futures-without-running-out-of-ram  )内容，将音频文件分批次提交到线程池处理，处理完一批次后再提交下一批次。    
  
**代码修改**    

		tfrecords_list = []
        count_file = 0 #统计处理文件的数量
        count_sample = 0
        pbar = tqdm(total=len(files), desc="Creating: {} tfrecord".format(flag))
        executor = ThreadPoolExecutor(max_workers=self.processes)
        seg_num = len(files)//PER_SAMPLES_NUM+1 if len(files)%PER_SAMPLES_NUM else len(files)//PER_SAMPLES_NUM
        for x in range(seg_num):  #每PER_SAMPLES_NUM个文件写入一个tfrecord
            left = x*PER_SAMPLES_NUM
            right = (x+1)*PER_SAMPLES_NUM if (x+1)*PER_SAMPLES_NUM<len(files) else len(files)
            tfrecord_filename = os.path.join(self.tfrecords_dir,flag + "_" + str(x) + ".tfrecord")
            tfrecords_list.append(tfrecord_filename)
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)
            examples_feature = {executor.submit(self.get_examples, i): i for i in files[left:right]}
            for feature in as_completed(examples_feature):
                file_id = examples_feature[feature]
                try:
                    examples = feature.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (self.id_filename_dict[file_id], exc))
                    del examples_feature[feature]
                else:
                    for example in examples:
                        writer.write(example)
                        count_sample += 1
                    count_file +=1
                    del examples_feature[feature]
                pbar.update()
            writer.close()
        executor.shutdown()
        pbar.close()

**测试结果**    
运行修改后的代码，1个小时后python占用内存从初始的3G升高到20G，说明修改无效。

### 5.2 注释线程池的后处理部分代码，定位线程池是否存在内存泄露
**修改5.1部分代码**  

			for feature in as_completed(examples_feature):
				del examples_feature[feature]
                #file_id = examples_feature[feature]
                #try:
                #    examples = feature.result()
                #except Exception as exc:
                #    print('%r generated an exception: %s' % (self.id_filename_dict[file_id], exc))
                #    del examples_feature[feature]
                #else:
                #    for example in examples:
                #        writer.write(example)
                #        count_sample += 1
                #    count_file +=1
                #    del examples_feature[feature]
                pbar.update()

**测试结果**  
压测修改后的python代码，内存在1个小时内从3G增长到10G，说明存在明显的内存泄露。  

**修改方案**  
通过google发现，python线程池本身存在bug。    
   
参考该[issue](https://bugs.python.org/issue27144)说明，修改代码如下：  

			for feature in as_completed(examples_feature):
				del examples_feature[feature]
				del feature
                #file_id = examples_feature[feature]
                #try:
                #    examples = feature.result()
                #except Exception as exc:
                #    print('%r generated an exception: %s' % (self.id_filename_dict[file_id], exc))
                #    del examples_feature[feature]
                #else:
                #    for example in examples:
                #        writer.write(example)
                #        example = None
                #        count_sample += 1
                #    count_file +=1
                #    del examples_feature[feature]
                pbar.update()	

**修改后测试结果**
对修改后的python代码进行压测，内存占用稳定在3G左右，说明修改有效。  

### 打开获取线程执行结果代码，确定是否有内存泄露
**修改5.1部分代码** 

            for feature in as_completed(examples_feature):
                file_id = examples_feature[feature]
                try:
                    examples = feature.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (self.id_filename_dict[file_id], exc))
                    del examples_feature[feature]
					del feature
                else:
                #    for example in examples:
                #        writer.write(example)
                #        count_sample += 1
                #    count_file +=1
                    del examples_feature[feature]
					del feature
                pbar.update()

**测试结果**   
压测修改后的python代码，内存在1个小时内从3G增长到10G，说明存在明显的内存泄露。   

**修改方案**
对比5.2部分修改，仅增加了examples = feature.result()部分代码，存在内存泄露，则在examples使用后将examples=None  

            for feature in as_completed(examples_feature):
                file_id = examples_feature[feature]
                try:
                    examples = feature.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (self.id_filename_dict[file_id], exc))
                    del examples_feature[feature]
					del feature
					examples = None
                else:
                #    for example in examples:
                #        writer.write(example)
                #        count_sample += 1
                #    count_file +=1
                    del examples_feature[feature]
					del feature
					examples = None
                pbar.update()

**修改后测试结果**
对修改后的python代码进行压测，内存占用稳定在3G左右，说明修改有效。  

### 打开写tfrecord文件代码，确定是否有内存泄露
**修改5.1部分代码**   

		tfrecords_list = []
        count_file = 0 #统计处理文件的数量
        count_sample = 0
        pbar = tqdm(total=len(files), desc="Creating: {} tfrecord".format(flag))
        executor = ThreadPoolExecutor(max_workers=self.processes)
        seg_num = len(files)//PER_SAMPLES_NUM+1 if len(files)%PER_SAMPLES_NUM else len(files)//PER_SAMPLES_NUM
        for x in range(seg_num):  #每PER_SAMPLES_NUM个文件写入一个tfrecord
            left = x*PER_SAMPLES_NUM
            right = (x+1)*PER_SAMPLES_NUM if (x+1)*PER_SAMPLES_NUM<len(files) else len(files)
            tfrecord_filename = os.path.join(self.tfrecords_dir,flag + "_" + str(x) + ".tfrecord")
            tfrecords_list.append(tfrecord_filename)
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)
            examples_feature = {executor.submit(self.get_examples, i): i for i in files[left:right]}
            for feature in as_completed(examples_feature):
                file_id = examples_feature[feature]
                try:
                    examples = feature.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (self.id_filename_dict[file_id], exc))
                    del examples_feature[feature]
					del feature
					examples = None
                else:
                    for example in examples:
                        writer.write(example)
                        count_sample += 1
                    count_file +=1
                    del examples_feature[feature]
					del feature
					examples = None
                pbar.update()
            writer.close()
        executor.shutdown()
        pbar.close()

**测试结果**   
压测修改后的python代码，内存在1个小时内从3G增长到10G，说明存在明显的内存泄露。

**修改方案**
参考[blog](https://blog.csdn.net/nirendao/article/details/44426201/),在上述代码writer.close()后添加gc.collect()

**修改后测试结果**    
运行修改后的代码，1个小时后python占用内存从初始的3G升高到20G，说明修改无效。

**新修改方案**  
通过google查找发现，tensorflow [issues 23733](https://github.com/tensorflow/tensorflow/issues/23733)存在相同的问题，使用tensorflow gpu版本，在部分环境下存在写tfrecord文件之后内存不回收问题，改用tensorflow cpu版本可以规避该问题。  
在本次定位中，将原来的tensorflow_gpu==1.9.0更新为tensorflow==1.13.0版本。  

**修改后测试结果**
对修改后的python代码进行压测，内存占用稳定在3G左右，说明修改有效。 

### 终版无内存泄露版本

		tfrecords_list = []
        count_file = 0 #统计处理文件的数量
        count_sample = 0
        pbar = tqdm(total=len(files), desc="Creating: {} tfrecord".format(flag))
        executor = ThreadPoolExecutor(max_workers=self.processes)
        seg_num = len(files)//PER_SAMPLES_NUM+1 if len(files)%PER_SAMPLES_NUM else len(files)//PER_SAMPLES_NUM
        for x in range(seg_num):  #每PER_SAMPLES_NUM个文件写入一个tfrecord
            left = x*PER_SAMPLES_NUM
            right = (x+1)*PER_SAMPLES_NUM if (x+1)*PER_SAMPLES_NUM<len(files) else len(files)
            tfrecord_filename = os.path.join(self.tfrecords_dir,flag + "_" + str(x) + ".tfrecord")
            tfrecords_list.append(tfrecord_filename)
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)
            examples_feature = {executor.submit(self.get_examples, i): i for i in files[left:right]}
            for feature in as_completed(examples_feature):
                file_id = examples_feature[feature]
                try:
                    examples = feature.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (self.id_filename_dict[file_id], exc))
                    del examples_feature[feature]
					del feature
					examples = None
                else:
                    for example in examples:
                        writer.write(example)
						example = None
                        count_sample += 1
                    count_file +=1
                    del examples_feature[feature]
					del feature
					examples = None
                pbar.update()
            writer.close()
			writer = None
			examples_feature = None
        executor.shutdown()
        pbar.close()