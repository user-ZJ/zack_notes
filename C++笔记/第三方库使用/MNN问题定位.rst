MNN问题定位
========================

调试信息
----------------------
MNN版本：1.1.7

::

    tools/converter/source/MNNConverter.cpp:22
    tools/converter/source/optimizer/PostConverter.cpp:483
    tools/converter/source/optimizer/PostConverter.cpp:161
    tools/converter/source/optimizer/PostConverter.cpp:107
    tools/converter/source/optimizer/PostConverter.cpp:128
    tools/converter/source/optimizer/onnxextra/OnnxConvolutionMerge.cpp:106
    tools/converter/source/optimizer/TemplateMerge.cpp:14
    tools/converter/source/optimizer/TemplateMerge.cpp:21
    tools/converter/source/optimizer/TemplateMerge.cpp:23
    tools/converter/source/optimizer/TemplateMerge.cpp:25
    tools/converter/source/optimizer/onnxextra/OnnxExtraManager.cpp:35
    tools/converter/source/optimizer/onnxextra/OnnxExtraManager.cpp:47
    tools/converter/source/optimizer/onnxextra/OnnxExtraManager.cpp:53
    tools/converter/source/optimizer/onnxextra/OnnxExtraManager.cpp:58
    source/shape/ShapeBinaryOp.cpp:38
    source/shape/ShapeBinaryOp.cpp:59
    express/Expr.cpp:250
    express/Expr.cpp:278
    express/Expr.cpp:289
    express/Expr.cpp:503
    express/Expr.cpp:585
    express/Expr.cpp:637
    express/Executor.cpp:188
    tools/converter/source/optimizer/onnxextra/OnnxEinsum.cpp:19
    tools/converter/source/optimizer/onnxextra/OnnxEinsum.cpp:232
    source/shape/SizeComputer.cpp:79
    tools/converter/source/optimizer/Program.cpp:105
    source/geometry/GeometryComputerUtils.cpp:188
    source/core/Pipeline.cpp:139
    source/core/Pipeline.cpp:289



    op->name()->c_str()
    op->type()

    op->inputs()
    op->inputs()[0]->printShape()


GlobalAvgPooling在推理的过程报错
------------------------------------------
MNN版本：1.1.7

模型中包含GlobalAvgPooling层时，转换为MNN可以成功，但在模型加载运行时会报错；
定位发现报错位置在source/geometry/GeometryPooling3D.cpp中第23行会检查输入的维度必须是5（MNN_ASSERT(input->dimensions() == 5);）
实际上在代码的后半段支持维度小于5的情况

.. image:: /images/MNN1.png

解决方法：

将source/geometry/GeometryPooling3D.cpp中第23行MNN_ASSERT(input->dimensions() == 5);注释掉重新编译库即可
Gather和GatherND问题

Gather和GatherND问题
---------------------------------
在使用Gather算子时，如果使用了batch_dims参数，转换为onnx的时候会使用GatherND算子替换，因为onnx中的Gather算子没有batch_dims参数，GatherND有此参数，所有会使用GatherND替换
在使用了GatherND且包含batch_dims参数时，onnx可以顺利转换为MNN模型，但是在运行是会报错，原因是：MNN中的GatherND没有batch_dims参数，导致输出维度计算错误。

总结：在使用Gather或GatherND算子时不要使用batch_dims参数。

Conv1d不支持，需要转化为Conv2d
-----------------------------------------
+------+------------------------------------------------------------+--------------------------------------------------------------------+
|      |                           Conv1d                           |                               Conv2d                               |
+======+============================================================+====================================================================+
| 输入 | [batch,in_channel,seq_len]                                 | [batch,in_channel,1,seq_len]                                       |
+------+------------------------------------------------------------+--------------------------------------------------------------------+
| 定义 | nn.Conv1d(in_channel,out_channel,kernel_size=k, padding=p) | nn.Conv2d(in_channel,out_channel,kernel_size=(1,k), padding=(0,p)) |
+------+------------------------------------------------------------+--------------------------------------------------------------------+
| 输出 | [batch,out_channel,out_len]                                | [batch,out_channel,1,out_len]                                      |
+------+------------------------------------------------------------+--------------------------------------------------------------------+

转换和推理的时候不支持dropout，需要自己实现dropout
-----------------------------------------------------
自己实现dropout层代码如下: https://github.com/alibaba/MNN/pull/1528

循环解码问题
---------------------
在循环调用TTS decoder模型时，直接将输出的rnn_h0_out和rnn_h1_out赋值给run_h0_in和run_h1_out时导致decoder结果不正确。
定位发现，在拷贝rnn_h0_out给rnn_h0_in时导致rnn_h1_out的数据发生改变。

解决方案：使用vector将rnn_h0_out和rnn_h1_out数据拷贝出来，再对rnn_h0_in和rnn_h1_in进行赋值，结果正常

输出shape和elementSize大小不一致
------------------------------------------
在调用TTS hifigan的时候，输入1x80x234，输出1x1x59904，输出elementSize为239616，elementSize和shape不对齐
查看数据发现每4个数据里面只有一个有值，其余的数据都是0

解决方案：新创建一个tensor，将hifigan的输出复制到创建的tensor，mnn会对数据进行重新对齐

自定义算子导致推理结果不正确
-------------------------------------
添加Dropout自定义算子之后，导致TTS生成的音频不正常，调试发现不能调用到Dropout算子backend infer的代码

原因：在MNN/source/geometry/GeometryComputerUtils.cpp中，计算完算子的所有输出的Shape后，会校验输出是否存在为0的维度，如果存在，则该算子不能参与计算，导致输出是随机数。

.. image:: /images/MNN2.png

Dropout 算子有3个输入（data,ratio,training_mode）两个输出（data,mask）。
由于在实现算子的时候没有计算mask的shape，导致Dropout算子在infer的时候被断开，导致计算结果错误。

解决方案：添加mask的shape计算

onnx opset=13 转化出的onnx再使用MNNConvert转换报错
----------------------------------------------------------
当前版本的MNNConvert对于onnx opset=13的部分算子不兼容，需要使用opset=12进行转换





