.. _onnx使用笔记:

onnx使用笔记
===========================

onnx结构
---------------------
ONNX是一种表示深度学习模型的跨平台开放格式，可以在不同的深度学习框架之间转换模型。ONNX文件是包含序列化模型的二进制文件，具有以下基本结构：

* ModelProto对象: ONNX文件的根对象，表示整个模型。包含了所有的图(GraphProto)、初始参数(Initializer)、元数据(Metadata)等信息。
* GraphProto对象: 表示模型的计算图，包含了所有的节点(NodeProto)、边(Edge)以及变量(Input/Output)信息。
* NodeProto对象: 表示模型中的一个节点，每个节点对应于模型中的一个计算操作，如卷积、池化、全连接等。每个节点包含了操作的名称、输入、输出、属性等信息。
* InputProto和OutputProto对象: 表示模型的输入和输出变量，包含了名称、数据类型、维度、形状等信息。
* TensorProto对象: 表示模型中的张量，即包含多个数值的多维数组。包含了数据类型、维度、形状、数据等信息。
* Initializer对象: 表示模型中的初始参数，如权重、偏置等。包含了参数名称、数据类型、维度、形状、数据等信息。
* ValueInfoProto对象: 表示节点的输入和输出的名称、数据类型、维度、形状等信息。
* AttributeProto对象: 表示节点的属性，如卷积核大小、步长等。包含了属性的名称、数据类型、数据等信息。

以上是ONNX文件的基本结构和对象，当然，ONNX文件还可以包含其他的元数据信息，如描述符、版本信息、作者信息等。理解ONNX文件的结构和对象非常重要，可以帮助我们有效地读取、修改和转换ONNX模型。

示例
------------
.. code-block:: python 

    #!/usr/bin/env python
    # coding=utf-8
    import onnx
    import onnxruntime
    import numpy as np



    sess = onnxruntime.InferenceSession("test.onnx")
    input_names = []
    output_names = []
    for input in sess.get_inputs():
        print(input.name,input.shape,input.type)
        input_names.append(input.name)
    for output in sess.get_outputs():
        print(output.name,output.shape,output.type)
        output_names.append(output.name)


    x = np.random.random((1,120))
    x = x.astype(np.int64)

    outputs = sess.run(output_names, {input_names[0]: x})
    for o in outputs:
        print(o.shape)
    outputs = sess.run(None, {input_names[0]: x})

onnx-sim
-----------------
静态输入优化：

python3 -m onnxsim encoder.onnx encoder1.onnx 

动态输入优化

python3 -m onnxsim encoder.onnx encoder1.onnx --dynamic-input-shape  --input-shape 1,120

输出onnx中所有节点的运行结果
-------------------------------------
.. code-block:: python

    #!/usr/bin/env python
    # coding=utf-8
    import onnx
    import onnxruntime as ort
    import numpy as np

    model = onnx.load("xxx.onnx")

    del model.graph.output[:]  # clear old output
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    outputs_node = [x.name for x in ort_session.get_outputs()]

    x = np.random.random((1,120))
    x = x.astype(np.int64)

    outputs = ort_session.run(outputs_node, {input_name: x})  
    print(outputs)

对onnx进行裁剪
---------------------------------------------------------------------------------------------------------------------
适合调试运行出错的onnx，将onnx运行出错部分裁剪掉，查看前半部分运行结果查找运行出错原因

.. code-block:: python 

    #!/usr/bin/env python
    # coding=utf-8
    import onnx
    import onnxruntime as ort
    import numpy as np

    model = onnx.load("xxx.onnx")

    oldnodes = [n for n in model.graph.node]
    newnodes = oldnodes[0:103] # or whatever
    del model.graph.node[:] # clear old nodes
    model.graph.node.extend(newnodes)
    # 裁剪之后需要重新指定输入节点，要不会运行失败。输出节点可以指定为调试的观察节点
    del model.graph.output[:]
    graph.output.extend([onnx.ValueInfoProto(name="70")])
    onnx.save(model, "XXX_split.onnx")

    ort_session = ort.InferenceSession(model.SerializeToString())
    outputs_node = [x.name for x in ort_session.get_outputs()]

    x = np.random.random((1,120))
    x = x.astype(np.int64)

    outputs = ort_session.run(outputs_node, {input_name: x})  
    print(res)


查看onnx输入、输出、模型参数
------------------------------------
.. code-block:: python 

    import onnx
    model = onnx.load('xxx.onnx')
    output = model.graph.output

    input_all = model.graph.input
    input_initializer = model.graph.initializer
    net_feed_input = set(input_all)  - (input_initializer)
    print(len(input_initializer))
    for ini in input_initializer:
        print(ini.name,ini.dims,ini.data_type)


将一个onnx拆分为两个onnx
------------------------------------
将一个onnx模型拆分为两个onnx模型有两种思路：

1. 拷贝一份原始的ModelProto，删除不需要的节点和initializer等
2. 创建两个ModelProto,设置节点、initializer、input、output

方式1：

.. code-block:: python

    import onnx
    from onnx import shape_inference,helper,numpy_helper,TensorProto
    import copy
    # 加载原始模型
    model = onnx.load('xxx.onnx')
    graph = model.graph

    sub_nodes1 = graph.node[0:10]
    sub_nodes2 = graph.node[10:]

    submodel1 = copy.deepcopy(model)
    submodel2 = copy.deepcopy(model)                                                                                                                                                    

    del submodel1.graph.node[:]
    del submodel1.graph.initializer[:]
    del submodel1.graph.input[:]
    del submodel1.graph.output[:]
    for node in sub_nodes1:
        for i in node.input:
            for ini in graph.initializer:
                if i==ini.name:
                    submodel1.graph.initializer.append(ini)
    submodel1.graph.node.extend(sub_nodes1)
    submodel1.graph.input.extend(graph.input[:1])
    output_1 = helper.make_tensor_value_info(sub_nodes1[-1].output[0], onnx.TensorProto.FLOAT, [1,256])
    submodel1.graph.output.extend([output_1])

    del submodel2.graph.node[:]
    del submodel2.graph.initializer[:]
    del submodel2.graph.input[:]
    del submodel2.graph.output[:]
    for node in sub_nodes2:
        for i in node.input:
            for ini in graph.initializer:
                if i==ini.name:
                    submodel2.graph.initializer.append(ini)
    submodel2.graph.node.extend(sub_nodes2)
    input_1 = helper.make_tensor_value_info(sub_nodes2[0].input[0], onnx.TensorProto.FLOAT, [1,256])
    submodel2.graph.input.append(input_1)
    submodel2.graph.input.extend(graph.input[1:])
    submodel2.graph.output.extend(graph.output)

    onnx.save(submodel1, "submodel1.onnx")
    onnx.save(submodel2, "submodel2.onnx")
    onnx.checker.check_model("submodel1.onnx")
    onnx.checker.check_model("submodel2.onnx")







方式2：

.. code-block:: python 

    import onnx
    from onnx import shape_inference
    from onnx import helper,numpy_helper
    from onnx import TensorProto

    model = onnx.load('xxx.onnx')
    graph = model.graph

    sub_nodes1 = graph.node[0:10]
    sub_nodes2 = graph.node[10:]

    submodel1 = onnx.ModelProto()
    submodel2 = onnx.ModelProto()

    for node in sub_nodes1:
    for i in node.input:
        for ini in graph.initializer:
            if i==ini.name:
                submodel1.graph.initializer.append(ini)
    submodel1.graph.name = model.graph.name
    submodel1.graph.node.extend(sub_nodes1)
    submodel1.graph.input.extend(graph.input[:1])
    #构建output
    output_1 = helper.make_tensor_value_info(sub_nodes1[-1].output[0], onnx.TensorProto.FLOAT, [1,256])
    submodel1.graph.output.extend([output_1])
    submodel1.ir_version = model.ir_version
    submodel1.producer_name = model.producer_name
    submodel1.producer_version = model.producer_version
    submodel1.domain = model.domain
    submodel1.model_version = model.model_version
    submodel1.doc_string = model.doc_string
    submodel1.opset_import.extend(model.opset_import)
    onnx.save(submodel1, "submodel1.onnx")
    onnx.checker.check_model("submodel1.onnx")

    #model2使用同样的操作
    for node in sub_nodes2:
    for i in node.input:
        for ini in graph.initializer:
            if i==ini.name:
                submodel2.graph.initializer.append(ini)

    submodel2.graph.name = model.graph.name
    submodel2.ir_version = model.ir_version
    submodel2.producer_name = model.producer_name
    submodel2.producer_version = model.producer_version
    submodel2.domain = model.domain
    submodel2.model_version = model.model_version
    submodel2.doc_string = model.doc_string
    submodel2.opset_import.extend(model.opset_import)

    submodel2.graph.node.extend(sub_nodes2)
    # 构建input
    input_1 = helper.make_tensor_value_info(sub_nodes2[0].input[0], onnx.TensorProto.FLOAT, [1,256])
    submodel2.graph.input.append(input_1)
    submodel2.graph.input.extend(graph.input[1:])
    submodel2.graph.output.extend(graph.output)
    onnx.save(submodel2, "submodel2.onnx")
    onnx.checker.check_model("submodel2.onnx")

方式3：

.. code-block:: python

    import onnx
    from onnx import shape_inference,helper,numpy_helper,TensorProto
    from onnx import NodeProto, GraphProto, TensorProto
    import onnxruntime as ort
    import numpy as np
    from onnxsim import simplify
    import copy

    # onnx裁剪思路
    # 1. 在指定节点添加输入/输出
    # 2. 删除中间某个节点，将不需要的图和主图断开连接
    # 3. 使用onnx-simplifier清理不需要的图（删除不需要的输出节点即可删除链路上的所有节点）
    # 4. 删除不需要的输入和输出节点

    model = onnx.load('/home/zack/Documents/models/nlp/paimon_sentiment.onnx')
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    graph = model.graph


    sub_nodes1 = graph.node
    sub_nodes2 = graph.node

    submodel1 = copy.deepcopy(model)
    submodel2 = copy.deepcopy(model)

    # model1
    dynamic_dim = onnx.TensorProto.UNDEFINED
    value_info = onnx.ValueInfoProto()
    # 注意，添加输出时，名称必须是某个节点的输出名称，可以在simplifier之后再修改名称
    value_info.name = '/bert/embeddings/Add_1_output_0'  # 张量的名称
    value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT  # 元素类型，这里是浮点数
    value_info.type.tensor_type.shape.dim.extend([
        onnx.TensorShapeProto.Dimension(dim_value=dynamic_dim,dim_param='batch_size'),
        onnx.TensorShapeProto.Dimension(dim_value=dynamic_dim,dim_param='sequence_length'),
        onnx.TensorShapeProto.Dimension(dim_value=768)   # 第三维度是静态的，值为1
    ])
    del submodel1.graph.output[:]
    submodel1.graph.output.extend([value_info])
    print(submodel1.graph.output)
    del submodel1.graph.node[:]
    submodel1.graph.node.extend(sub_nodes1)
    submodel1, check = simplify(submodel1)
    assert check, "Simplified ONNX model could not be validated"
    #修改输出节点名称
    for node in submodel1.graph.node:
        if node.name == '/bert/embeddings/Add_1':
            print(node)
            node.output[0] = "feature"
            print(node)
    submodel1.graph.output[0].name = "feature"
    #删除多余的输入节点
    print(submodel1.graph.input)
    submodel1.graph.input.pop(1)
    onnx.save(submodel1, "embedding.onnx")
    onnx.checker.check_model("embedding.onnx")

    feature = onnx.ValueInfoProto()
    feature.name = 'feature'  # 张量的名称
    feature.type.tensor_type.elem_type = onnx.TensorProto.FLOAT  # 元素类型，这里是浮点数
    feature.type.tensor_type.shape.dim.extend([
        onnx.TensorShapeProto.Dimension(dim_value=dynamic_dim,dim_param='batch_size'), #动态维度
        onnx.TensorShapeProto.Dimension(dim_value=dynamic_dim,dim_param='sequence_length'),
        onnx.TensorShapeProto.Dimension(dim_value=768)   # 第三维度是静态的，值为768
    ])
    submodel2.graph.input.append(feature)

    for node in sub_nodes2:
        if node.name == '/bert/embeddings/LayerNorm/ReduceMean':
            node.input[0] = 'feature'
            print(node)
        if node.name == '/bert/embeddings/LayerNorm/Sub':
            node.input[0] = 'feature'
            print(node)
    for node in sub_nodes2:
        if node.name == '/bert/embeddings/Add_1':
            sub_nodes2.remove(node)

    del submodel2.graph.node[:]
    submodel2.graph.node.extend(sub_nodes2)
    submodel2, check = simplify(submodel2)
    assert check, "Simplified ONNX model could not be validated"
    submodel2.graph.input.pop(0)
    onnx.save(submodel2, "sentiment.onnx")
    onnx.checker.check_model("sentiment.onnx")

onnx不支持算子
--------------------------
adaptive_avg_pool1d  https://github.com/pytorch/pytorch/issues/61172
